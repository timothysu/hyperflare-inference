/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Qwen image profile, learned position interpolation and multimodal prompt positions.

use super::*;

#[derive(Clone)]
pub(super) struct QwenProcessor {
    pub(super) patches: patches::PatchProcessor,
    pub(super) position_side: usize,
    pub(super) image_token_id: u32,
    pub(super) video_token_id: u32,
    pub(super) position_scheme: QwenPositionScheme,
}

impl QwenProcessor {
    pub(super) fn load(settings: &QwenImageSettings, limits: ImageLimits) -> Result<Self> {
        if settings.min_pixels == 0
            || settings.max_pixels < settings.min_pixels
            || settings.position_side != 48
            || settings.image_token_id == settings.video_token_id
        {
            return Err("invalid prepared Qwen image settings".into());
        }
        let normalization = match (settings.image_mean, settings.image_std) {
            (None, None) => patches::Normalization::Symmetric,
            (Some(mean), Some(std))
                if mean.iter().all(|v| v.is_finite())
                    && std.iter().all(|v| v.is_finite() && *v > 0.) =>
            {
                patches::Normalization::Channel { mean, std }
            }
            _ => return Err("invalid prepared Qwen image settings".into()),
        };
        let resize = match settings.resize_kernel {
            QwenResizeKernel::TorchvisionBicubic => resize::resize,
            QwenResizeKernel::PillowBicubic => resize::pillow,
        };
        Ok(Self {
            patches: patches::PatchProcessor {
                limits,
                patch_size: PATCH,
                min_pixels: settings.min_pixels,
                max_pixels: settings.max_pixels,
                normalization,
                resize,
            },
            position_side: settings.position_side,
            image_token_id: settings.image_token_id,
            video_token_id: settings.video_token_id,
            position_scheme: settings.position_scheme,
        })
    }

    pub fn process(&self, images: &[RgbImage]) -> Result<ProcessedImages> {
        let mut result = self.patches.process(images)?;
        let side = self.position_side;
        let mut start = 0;
        result
            .interpolation_indices
            .reserve_exact(result.position_ids.len());
        result
            .interpolation_weights
            .reserve_exact(result.position_ids.len());
        for geometry in &result.geometry {
            let grid_h = geometry.height / PATCH;
            let grid_w = geometry.width / PATCH;
            for &[row, col] in &result.position_ids[start..start + geometry.patches] {
                let (yi, yw) = interpolation_axis(row as usize, grid_h, side);
                let (xi, xw) = interpolation_axis(col as usize, grid_w, side);
                result.interpolation_indices.push([
                    (yi[0] * side + xi[0]) as i64,
                    (yi[0] * side + xi[1]) as i64,
                    (yi[1] * side + xi[0]) as i64,
                    (yi[1] * side + xi[1]) as i64,
                ]);
                result.interpolation_weights.push([
                    yw[0] * xw[0],
                    yw[0] * xw[1],
                    yw[1] * xw[0],
                    yw[1] * xw[1],
                ]);
            }
            start += geometry.patches;
        }
        Ok(result)
    }

    /// Expands the image placeholders in already rendered chat tokens, in image order.
    /// Inputs contain no padding; the scheduler adds padding outside logical lengths.
    pub fn expand_prompt(
        &self,
        frontend: &Frontend,
        prompt_token_ids: &[u32],
        images: &ProcessedImages,
    ) -> Result<ImagePrompt> {
        if frontend.encode(IMAGE_TOKEN, false)? != [self.image_token_id] {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image token ID does not match tokenizer",
            ))));
        }
        let placeholder_count = prompt_token_ids
            .iter()
            .filter(|id| **id == self.image_token_id)
            .count();
        if placeholder_count != images.geometry.len() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image placeholder count does not match prepared images",
            ))));
        }
        if prompt_token_ids.contains(&self.video_token_id) {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "video inputs are not supported by the image processor",
            ))));
        }
        let mut token_ids = Vec::new();
        let mut geometry = images.geometry.iter();
        for &id in prompt_token_ids {
            if id == self.image_token_id {
                let tokens = geometry.next().expect("placeholder count was checked").tokens;
                token_ids.extend(std::iter::repeat_n(self.image_token_id, tokens));
            } else {
                token_ids.push(id);
            }
        }
        let mm_token_type_ids: Vec<i32> = token_ids
            .iter()
            .map(|id| i32::from(*id == self.image_token_id))
            .collect();
        let mut positions: [Vec<i64>; 4] =
            std::array::from_fn(|_| Vec::with_capacity(token_ids.len()));
        positions[0].extend((0..token_ids.len()).map(|index| index as i64));
        let (mut offset, mut current, mut image_index) = (0, 0i64, 0);
        while offset < token_ids.len() {
            let modality = mm_token_type_ids[offset];
            let end = offset
                + mm_token_type_ids[offset..]
                    .iter()
                    .take_while(|value| **value == modality)
                    .count();
            if modality == 0 {
                for row in &mut positions[1..] {
                    row.extend((0..end - offset).map(|index| current + index as i64));
                }
                current += (end - offset) as i64;
            } else {
                let geometry = images.geometry.get(image_index).ok_or_else(|| {
                    Error::frontend(FrontendError::Input(Failure::new(
                        "unexpected image token group",
                    )))
                })?;
                if end - offset != geometry.tokens {
                    return Err(Error::frontend(FrontendError::Input(Failure::new(
                        "image token group does not match image grid",
                    ))));
                }
                let (height, width) = (
                    geometry.grid_thw[1] / MERGE as i64,
                    geometry.grid_thw[2] / MERGE as i64,
                );
                let (h_center, w_center) = match self.position_scheme {
                    QwenPositionScheme::CornerMax => (0, 0),
                    QwenPositionScheme::CenteredStep => ((height - 1) / 2, (width - 1) / 2),
                };
                for y in 0..height {
                    for x in 0..width {
                        positions[1].push(current);
                        positions[2].push(current + y - h_center);
                        positions[3].push(current + x - w_center);
                    }
                }
                current += match self.position_scheme {
                    QwenPositionScheme::CornerMax => height.max(width),
                    QwenPositionScheme::CenteredStep => 1,
                };
                image_index += 1;
            }
            offset = end;
        }
        if image_index != images.geometry.len() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "prepared image was not consumed by prompt",
            ))));
        }
        Ok(ImagePrompt {
            rope_delta: current - token_ids.len() as i64,
            token_ids,
            mm_token_type_ids,
            position_ids: positions,
        })
    }
}

fn interpolation_axis(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let source = index as f32 * (side - 1) as f32 / (size - 1).max(1) as f32;
    let floor = source.floor();
    let distance = source - floor;
    (
        [floor as usize, (floor as usize + 1).min(side - 1)],
        [1.0 - distance, 1.0 - (distance - 1.0).abs()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::{Asset, FrontendAssets};
    use sha2::{Digest, Sha256};

    fn test_frontend() -> Frontend {
        let root = std::env::temp_dir().join(format!(
            "helios-qwen-centered-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let write = |name: &str, bytes: &[u8]| {
            let path = root.join(name);
            std::fs::write(&path, bytes).unwrap();
            Asset {
                path,
                sha256: format!("{:x}", Sha256::digest(bytes)),
                range: None,
            }
        };
        let tokenizer = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "a": 0, "b": 1, "<|image_pad|>": 157157, "<|video_pad|>": 156909
                },
                "unk_token": "a"
            },
            "added_tokens": [
                {"id": 157157, "content": "<|image_pad|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 156909, "content": "<|video_pad|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ]
        });
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: write("tokenizer.json", tokenizer.to_string().as_bytes()),
                tokens: Default::default(),
                chat_template: Some(write(
                    "chat_template.jinja",
                    b"{% mark %}{{ messages[0].content }}",
                )),
            },
            Default::default(),
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&root);
        frontend
    }

    #[test]
    fn centered_step_position_scheme_centers_spatial_axes_and_advances_by_one() {
        let frontend = test_frontend();
        let processor = QwenProcessor::load(
            &QwenImageSettings {
                min_pixels: 4096,
                max_pixels: 65_536,
                position_side: 48,
                image_token_id: 157157,
                video_token_id: 156909,
                image_mean: Some([0.48145466, 0.4578275, 0.40821073]),
                image_std: Some([0.26862954, 0.26130258, 0.27577711]),
                position_scheme: QwenPositionScheme::CenteredStep,
                resize_kernel: QwenResizeKernel::PillowBicubic,
            },
            ImageLimits::default(),
        )
        .unwrap();
        let image = RgbImage::new(128, 128, vec![128; 128 * 128 * 3]).unwrap();
        let processed = processor.process(&[image]).unwrap();
        // 128x128 -> 8x8 patches -> 4x4 merged image tokens (h_center = w_center = 1).
        // Prompt: [image_pad, 0, 1] starts at current = 0 so y=0, x=0 produces negative h=w=-1.
        let expanded = processor
            .expand_prompt(&frontend, &[157157, 0, 1], &processed)
            .unwrap();
        assert_eq!(expanded.token_ids.len(), 18);
        assert_eq!(expanded.position_ids[1][0], 0);
        assert_eq!(expanded.position_ids[2][0], -1);
        assert_eq!(expanded.position_ids[3][0], -1);
        assert_eq!(expanded.position_ids[1][15], 0);
        assert_eq!(expanded.position_ids[2][15], 2);
        assert_eq!(expanded.position_ids[3][15], 2);
        // Trailing text tokens advance from current = 0 + 1 = 1.
        assert_eq!(expanded.position_ids[1][16], 1);
        assert_eq!(expanded.position_ids[2][16], 1);
        assert_eq!(expanded.position_ids[3][16], 1);
        assert_eq!(expanded.position_ids[1][17], 2);
        assert_eq!(expanded.rope_delta, 3 - 18);
    }

    #[test]
    fn pillow_bicubic_resize_kernel_selects_pillow_resampler() {
        let base = QwenImageSettings {
            min_pixels: 4096,
            max_pixels: 65_536,
            position_side: 48,
            image_token_id: 157157,
            video_token_id: 156909,
            image_mean: None,
            image_std: None,
            position_scheme: QwenPositionScheme::CornerMax,
            resize_kernel: QwenResizeKernel::TorchvisionBicubic,
        };
        let tv = QwenProcessor::load(&base, ImageLimits::default()).unwrap();
        let pil = QwenProcessor::load(
            &QwenImageSettings {
                resize_kernel: QwenResizeKernel::PillowBicubic,
                ..base
            },
            ImageLimits::default(),
        )
        .unwrap();
        // High-contrast non-multiple-of-32 image exercises horizontal and vertical down/up sampling.
        let width = 101;
        let height = 73;
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 37 + (i / 3) * 19) % 256) as u8)
            .collect();
        let image = RgbImage::new(width, height, pixels).unwrap();
        let (tv_geom, tv_rgb) = tv.patches.resized_rgb(&image).unwrap();
        let (pil_geom, pil_rgb) = pil.patches.resized_rgb(&image).unwrap();
        assert_eq!(tv_geom, pil_geom);
        assert_eq!(tv_rgb, resize::resize(&image, tv_geom.width, tv_geom.height));
        assert_eq!(pil_rgb, resize::pillow(&image, pil_geom.width, pil_geom.height));
        assert_ne!(tv_rgb, pil_rgb);
    }
}
