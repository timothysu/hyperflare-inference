/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Dynamic-aspect row-major patch profile with learned position interpolation and row newlines.

use super::*;

pub(super) const PATCH_SIZE: usize = 16;
pub(super) const MERGE_SIZE: usize = 2;
pub(super) const FEATURES: usize = 3 * PATCH_SIZE * PATCH_SIZE;

#[derive(Clone)]
pub(super) struct RowNewlineProcessor {
    pub(super) limits: ImageLimits,
    pub(super) min_pixels: usize,
    pub(super) max_pixels: usize,
    pub(super) patch_size: usize,
    pub(super) merge_size: usize,
    pub(super) position_side: usize,
    pub(super) image_token_id: u32,
    pub(super) newline_token_id: u32,
    pub(super) video_token_id: u32,
}

impl RowNewlineProcessor {
    pub(super) fn load(settings: &RowNewlineImageSettings, limits: ImageLimits) -> Result<Self> {
        if settings.min_pixels == 0
            || settings.max_pixels < settings.min_pixels
            || settings.patch_size != PATCH_SIZE
            || settings.merge_size != MERGE_SIZE
            || settings.position_side == 0
            || settings.image_token_id == settings.newline_token_id
            || settings.image_token_id == settings.video_token_id
            || settings.newline_token_id == settings.video_token_id
        {
            return Err("invalid prepared row-newline image settings".into());
        }
        Ok(Self {
            limits,
            min_pixels: settings.min_pixels,
            max_pixels: settings.max_pixels,
            patch_size: settings.patch_size,
            merge_size: settings.merge_size,
            position_side: settings.position_side,
            image_token_id: settings.image_token_id,
            newline_token_id: settings.newline_token_id,
            video_token_id: settings.video_token_id,
        })
    }

    pub(super) fn tokens(&self) -> (u32, u32, u32) {
        (self.image_token_id, self.newline_token_id, self.video_token_id)
    }

    pub(super) fn geometry(&self, width: usize, height: usize) -> Result<ImageGeometry> {
        check_input_extent(self.limits, width, height)?;
        if width.max(height) as f64 / width.min(height) as f64 > 200.0 {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image aspect ratio exceeds 200",
            ))));
        }
        let factor = (self.patch_size * self.merge_size) as f64;
        let (h, w) = (height as f64, width as f64);
        let mut target_h = (h / factor).round_ties_even() * factor;
        let mut target_w = (w / factor).round_ties_even() * factor;
        if target_h * target_w > self.max_pixels as f64 {
            let beta = (h * w / self.max_pixels as f64).sqrt();
            target_h = ((h / beta / factor).floor() * factor).max(factor);
            target_w = ((w / beta / factor).floor() * factor).max(factor);
        } else if target_h * target_w < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / (h * w)).sqrt();
            target_h = (h * beta / factor).ceil() * factor;
            target_w = (w * beta / factor).ceil() * factor;
        }
        let height = target_h.max(factor) as usize;
        let width = target_w.max(factor) as usize;
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let patches = grid_h * grid_w;
        check_image_patches(self.limits, patches)?;
        Ok(ImageGeometry {
            height,
            width,
            grid_thw: [1, grid_h as i64, grid_w as i64],
            patches,
            tokens: patches / (self.merge_size * self.merge_size),
        })
    }

    pub(super) fn resized_rgb(&self, image: &RgbImage) -> Result<(ImageGeometry, Vec<u8>)> {
        let geometry = self.geometry(image.width, image.height)?;
        Ok((geometry, resize::resize(image, geometry.width, geometry.height)))
    }

    pub(super) fn process(&self, images: &[RgbImage]) -> Result<ProcessedImages> {
        self.limits.check_batch(images.len(), 0)?;
        let geometry: Vec<_> = images
            .iter()
            .map(|image| self.geometry(image.width, image.height))
            .collect::<Result<_>>()?;
        let total: usize = geometry
            .iter()
            .try_fold(0usize, |sum, image| sum.checked_add(image.patches))
            .ok_or_else(|| {
                Error::frontend(FrontendError::Input(Failure::new("patch count overflow")))
            })?;
        self.limits.check_batch(images.len(), total)?;
        let mut result = ProcessedImages {
            geometry,
            cu_seqlens: vec![0],
            ..Default::default()
        };
        result.reserve_outputs(total, FEATURES, true, images.len(), images.len() + 1)?;
        let normalized: [f32; 256] =
            std::array::from_fn(|value| (value as f32 * (1.0 / 255.0) - 0.5) / 0.5);
        let side = self.position_side;
        for (image, geometry) in images.iter().zip(&result.geometry) {
            let pixels = resize::resize(image, geometry.width, geometry.height);
            let grid_h = geometry.height / self.patch_size;
            let grid_w = geometry.width / self.patch_size;
            let y_axis: Vec<_> = (0..grid_h)
                .map(|row| interpolation_axis(row, grid_h, side))
                .collect();
            let x_axis: Vec<_> = (0..grid_w)
                .map(|col| interpolation_axis(col, grid_w, side))
                .collect();
            // Patches follow row-major spatial order without temporal duplication.
            for (row, &(yi, yw)) in y_axis.iter().enumerate() {
                for (col, &(xi, xw)) in x_axis.iter().enumerate() {
                    result.position_ids.push([row as i64, col as i64]);
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
                    for channel in 0..3 {
                        for y in 0..self.patch_size {
                            for x in 0..self.patch_size {
                                let byte = pixels[((row * self.patch_size + y) * geometry.width
                                    + col * self.patch_size
                                    + x)
                                    * 3
                                    + channel];
                                result.pixel_values.push(normalized[byte as usize]);
                            }
                        }
                    }
                }
            }
            result
                .cu_seqlens
                .push(result.cu_seqlens.last().unwrap() + geometry.patches as i32);
        }
        Ok(result)
    }
}

/// Bilinear coordinates matching `F.interpolate(..., scale_factor=(size + 0.1) / side,
/// mode="bilinear", align_corners=False)`. Under ATen's `area_pixel_compute_scale`,
/// `output_size == input_size` resolves to `scale = 1.0` per axis, sampling each index
/// directly without interpolation.
fn interpolation_axis(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    if size == side {
        return ([index, index], [1.0, 0.0]);
    }
    let scale_factor = (size as f64 + 0.1) / side as f64;
    let scale = (1.0 / scale_factor) as f32;
    let source = (scale * (index as f32 + 0.5) - 0.5).max(0.0);
    let lo = (source as usize).min(side - 1);
    if lo >= side - 1 {
        ([side - 1, side - 1], [1.0, 0.0])
    } else {
        let frac = source - lo as f32;
        ([lo, lo + 1], [1.0 - frac, frac])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_settings() -> RowNewlineImageSettings {
        RowNewlineImageSettings {
            min_pixels: 4096,
            max_pixels: 4_194_304,
            patch_size: 16,
            merge_size: 2,
            position_side: 128,
            image_token_id: 120120,
            newline_token_id: 120121,
            video_token_id: 120124,
        }
    }

    #[test]
    fn settings_validation_rejects_invalid_geometry_or_colliding_tokens() {
        let base = default_settings();
        for bad in [
            RowNewlineImageSettings { min_pixels: 0, ..base.clone() },
            RowNewlineImageSettings { max_pixels: 1024, ..base.clone() },
            RowNewlineImageSettings { patch_size: 14, ..base.clone() },
            RowNewlineImageSettings { merge_size: 3, ..base.clone() },
            RowNewlineImageSettings { position_side: 0, ..base.clone() },
            RowNewlineImageSettings { newline_token_id: base.image_token_id, ..base.clone() },
            RowNewlineImageSettings { video_token_id: base.newline_token_id, ..base.clone() },
        ] {
            assert!(RowNewlineProcessor::load(&bad, ImageLimits::default()).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn geometry_and_row_major_patches_match_profile_contract() {
        let processor = RowNewlineProcessor::load(
            &default_settings(),
            ImageLimits {
                max_patches_per_image: 16384,
                max_total_patches: 16384,
                patches_per_feature: 4,
                ..Default::default()
            },
        )
        .unwrap();
        let small = processor.geometry(37, 29).unwrap();
        assert_eq!(small.width, 96);
        assert_eq!(small.height, 64);
        assert_eq!(small.grid_thw, [1, 4, 6]);
        assert_eq!(small.patches, 24);
        assert_eq!(small.tokens, 6);

        let full = processor.geometry(2048, 2048).unwrap();
        assert_eq!(full.grid_thw, [1, 128, 128]);
        assert_eq!(full.patches, 16384);
        assert_eq!(full.tokens, 4096);

        let image = RgbImage::new(
            64,
            64,
            (0..64 * 64 * 3).map(|i| (i % 256) as u8).collect(),
        )
        .unwrap();
        let processed = processor.process(&[image]).unwrap();
        assert_eq!(processed.geometry[0].grid_thw, [1, 4, 4]);
        assert_eq!(processed.pixel_values.len(), 16 * FEATURES);
        assert_eq!(processed.position_ids[0], [0, 0]);
        assert_eq!(processed.position_ids[1], [0, 1]);
        assert_eq!(processed.position_ids[4], [1, 0]);
        assert_eq!(processed.cu_seqlens, [0, 16]);
        for (indices, weights) in processed
            .interpolation_indices
            .iter()
            .zip(&processed.interpolation_weights)
        {
            assert!(indices.iter().all(|i| (0..128 * 128).contains(i)));
            assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        // Identity axis (size == side) maps each patch directly to its learned position.
        let ([lo, hi], [w0, w1]) = interpolation_axis(73, 128, 128);
        assert_eq!((lo, hi, w0, w1), (73, 73, 1.0, 0.0));
    }

    #[test]
    fn interpolation_axis_and_resized_rgb_match_pytorch_and_resize_reference() {
        // Exact source positions verified against PyTorch
        // F.interpolate(grid, scale_factor=(size + 0.1) / 128.0, mode="bilinear", align_corners=False)
        // on linear coordinate ramps (side=128).
        let eval = |index: usize, size: usize| -> f32 {
            let ([lo, hi], [w0, w1]) = interpolation_axis(index, size, 128);
            lo as f32 * w0 + hi as f32 * w1
        };
        // (128, 128) identity axis and (128, 64) mixed identity + scaled axes.
        for i in [0, 1, 63, 127] {
            assert_eq!(eval(i, 128), i as f32);
        }
        // size=64 (matches (128, 64) width axis and (64, 64) both axes).
        assert!((eval(0, 64) - 0.498_440).abs() < 1e-5, "got {}", eval(0, 64));
        assert!((eval(1, 64) - 2.495_320).abs() < 1e-5, "got {}", eval(1, 64));
        assert!((eval(63, 64) - 126.301_865).abs() < 1e-4, "got {}", eval(63, 64));
        // size=48 and size=32 (matches (48, 32) rectangular image).
        assert!((eval(0, 48) - 0.830_561).abs() < 1e-5, "got {}", eval(0, 48));
        assert!((eval(1, 48) - 3.491_684).abs() < 1e-5, "got {}", eval(1, 48));
        assert!((eval(47, 48) - 125.903_320).abs() < 1e-4, "got {}", eval(47, 48));
        assert!((eval(0, 32) - 1.493_770).abs() < 1e-5, "got {}", eval(0, 32));
        assert!((eval(31, 32) - 125.107_483).abs() < 1e-4, "got {}", eval(31, 32));

        // Non-uniform rectangular RGB image checked against independent Transformers
        // HunYuanVLImageProcessor.resize / PyTorch uint8 F.interpolate(mode="bicubic",
        // align_corners=False, antialias=True) reference output for 101x73 -> 96x64.
        let processor = RowNewlineProcessor::load(
            &default_settings(),
            ImageLimits {
                max_patches_per_image: 16384,
                max_total_patches: 16384,
                patches_per_feature: 4,
                ..Default::default()
            },
        )
        .unwrap();
        let (w, h) = (101, 73);
        let data: Vec<u8> = (0..w * h * 3).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let image = RgbImage::new(w, h, data).unwrap();
        let (geom, resized) = processor.resized_rgb(&image).unwrap();
        assert_eq!((geom.width, geom.height), (96, 64));
        assert_eq!(resized.len(), 96 * 64 * 3);
        assert_eq!(
            &resized[..12],
            &[39, 77, 81, 131, 148, 182, 215, 43, 52, 94, 148, 144]
        );
        assert_eq!(
            &resized[resized.len() - 12..],
            &[78, 109, 105, 163, 201, 210, 34, 71, 105, 128, 172, 176]
        );
        use sha2::{Digest, Sha256};
        assert_eq!(
            format!("{:x}", Sha256::digest(&resized)),
            "ff29d69b81a1c12431a3b60bb15729a565cc3455a61b7c07b5f000bf189d5d9c"
        );
    }
}
