/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Expand each image placeholder into merged spatial rows terminated by a row-newline token,
//! with 3D M-RoPE positions advancing by `max(H, W + 1)` per image.

use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{
        RenderedPrompt,
        images::{ImageDescription, ImageProcessor},
    },
    model::{PreparedInput, PromptData, contract::ImageTokenLayout},
};

pub(crate) fn prepare_input(
    image_tokens: Option<&ImageTokenLayout>,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: &[ImageDescription],
) -> Result<PreparedInput> {
    let tokens_spec = processor.and_then(ImageProcessor::row_newline_tokens);
    if let (Some((image, _, _)), Some(layout)) = (tokens_spec, image_tokens)
        && image != layout.feature_token
    {
        return Err("row-newline image_token_id differs from contract feature_token".into());
    }
    let image_token = tokens_spec
        .map(|(image, _, _)| image)
        .or_else(|| image_tokens.map(|layout| layout.feature_token))
        .ok_or("row-newline image layout requires prepared image token settings")?;
    if let Some((_, newline_token, video_token)) = tokens_spec {
        if rendered.token_ids.contains(&video_token) {
            return Err("video inputs are not supported by the image processor".into());
        }
        if rendered.token_ids.contains(&newline_token) {
            return Err("prompt contains an unexpanded image newline token".into());
        }
    }
    let placeholder_count = rendered
        .token_ids
        .iter()
        .filter(|id| **id == image_token)
        .count();
    if placeholder_count != images.len() {
        return Err("image placeholder count does not match input images".into());
    }
    if images.is_empty() {
        let len = rendered.token_ids.len();
        return Ok(PreparedInput {
            tokens: rendered.token_ids.clone(),
            token_expansions: vec![],
            data: PromptData {
                positions: (0..len).map(|pos| [pos as i64; 3]).collect(),
                vision_token_indices: vec![-1; len],
                spans: vec![],
            },
            conditioning: vec![],
            conditioned_inputs: vec![],
        });
    }
    let processor = processor.ok_or("image processing is unavailable")?;
    let (_, newline_token, _) = tokens_spec
        .ok_or("row-newline image layout requires a row-newline image processor")?;
    let processed = processor.describe_inputs(images)?;
    let mut tokens = Vec::with_capacity(rendered.token_ids.len());
    let mut positions = Vec::with_capacity(rendered.token_ids.len());
    let mut vision_token_indices = Vec::with_capacity(rendered.token_ids.len());
    let mut conditioning = Vec::with_capacity(images.len());
    let mut token_expansions = Vec::with_capacity(images.len());
    let mut current: i64 = 0;
    let mut image_index = 0usize;
    let mut feature_offset = 0usize;

    for (source_index, &token) in rendered.token_ids.iter().enumerate() {
        if token != image_token {
            tokens.push(token);
            positions.push([current, current, current]);
            vision_token_indices.push(-1);
            current += 1;
            continue;
        }
        let geometry = &processed.geometry[image_index];
        let grid_h = usize::try_from(geometry.grid_thw[1]).map_err(|_| "invalid vision height")?;
        let grid_w = usize::try_from(geometry.grid_thw[2]).map_err(|_| "invalid vision width")?;
        if geometry.grid_thw[0] != 1
            || grid_h == 0
            || grid_w == 0
            || grid_h % 2 != 0
            || grid_w % 2 != 0
        {
            return Err("row-newline image grid must have positive even spatial dimensions".into());
        }
        let height = grid_h / 2;
        let width = grid_w / 2;
        if height.checked_mul(width) != Some(geometry.tokens) {
            return Err("row-newline image grid does not match feature count".into());
        }
        let start = tokens.len();
        conditioning.push(ConditioningEvent {
            position: start as u64,
            identity: images[image_index].identity,
        });
        for y in 0..height {
            for x in 0..width {
                tokens.push(image_token);
                positions.push([current, current + y as i64, current + x as i64]);
                vision_token_indices.push((feature_offset + y * width + x) as i32);
            }
            tokens.push(newline_token);
            positions.push([current, current + y as i64, current + width as i64]);
            vision_token_indices.push(-1);
        }
        let expanded = height * (width + 1);
        current += height.max(width + 1) as i64;
        feature_offset += geometry.tokens;
        image_index += 1;
        token_expansions.push((source_index, expanded - 1));
    }

    Ok(PreparedInput {
        tokens,
        token_expansions,
        data: PromptData {
            positions,
            vision_token_indices,
            spans: vec![],
        },
        conditioning,
        conditioned_inputs: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::images::{
        ImageLimits, ImageSettings, RgbImage, RowNewlineImageSettings,
    };

    #[test]
    fn row_newline_expansion_emits_newlines_and_advances_3d_positions() {
        let processor = ImageProcessor::load(
            ImageSettings::RowNewline(RowNewlineImageSettings {
                min_pixels: 4096,
                max_pixels: 4_194_304,
                patch_size: 16,
                merge_size: 2,
                position_side: 128,
                image_token_id: 120120,
                newline_token_id: 120121,
                video_token_id: 120124,
            }),
            ImageLimits {
                max_patches_per_image: 16384,
                max_total_patches: 16384,
                patches_per_feature: 4,
                ..Default::default()
            },
        )
        .unwrap();
        // 64x64 -> 4x4 patches -> H=2, W=2 merged -> 2 rows x (2 image + 1 newline) = 6 tokens.
        let image = RgbImage::new(64, 64, vec![42; 64 * 64 * 3]).unwrap();
        let desc = processor.describe_image(&image).unwrap();
        assert_eq!(desc.geometry.tokens, 4);
        let rendered = RenderedPrompt {
            text: String::new(),
            token_ids: vec![10, 120118, 120120, 120119, 11],
            message_boundaries: vec![],
            image_sources: vec![],
            image_checkpoint_hints: vec![],
            ordinary: vec![],
        };
        let prepared = prepare_input(None, Some(&processor), &rendered, &[desc]).unwrap();
        assert_eq!(
            prepared.tokens,
            vec![10, 120118, 120120, 120120, 120121, 120120, 120120, 120121, 120119, 11]
        );
        assert_eq!(
            prepared.data.vision_token_indices,
            vec![-1, -1, 0, 1, -1, 2, 3, -1, -1, -1]
        );
        // Text tokens at 0, 1; image block starts at 2 with H=2, W+1=3 (max=3) -> next text at 5.
        assert_eq!(
            prepared.data.positions,
            vec![
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 3],
                [2, 2, 4],
                [2, 3, 2],
                [2, 3, 3],
                [2, 3, 4],
                [5, 5, 5],
                [6, 6, 6],
            ]
        );
        assert_eq!(prepared.token_expansions, vec![(2, 5)]);
        assert_eq!(prepared.conditioning.len(), 1);
        assert_eq!(prepared.conditioning[0].position, 2);

        // Text-only prompt emits 1D-equivalent 3D positions and rejects stray newline/video tokens.
        let text_only = RenderedPrompt {
            text: String::new(),
            token_ids: vec![10, 11, 12],
            message_boundaries: vec![],
            image_sources: vec![],
            image_checkpoint_hints: vec![],
            ordinary: vec![],
        };
        let plain = prepare_input(None, Some(&processor), &text_only, &[]).unwrap();
        assert_eq!(plain.data.positions, vec![[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
        assert_eq!(plain.data.vision_token_indices, vec![-1, -1, -1]);

        for bad_token in [120120, 120121, 120124] {
            let bad = RenderedPrompt {
                text: String::new(),
                token_ids: vec![10, bad_token],
                message_boundaries: vec![],
                image_sources: vec![],
                image_checkpoint_hints: vec![],
                ordinary: vec![],
            };
            assert!(prepare_input(None, Some(&processor), &bad, &[]).is_err());
        }

        // Mismatch between RowNewlineImageSettings.image_token_id and contract feature_token is rejected.
        let matching_layout = ImageTokenLayout {
            feature_token: 120120,
            span: None,
        };
        assert!(prepare_input(Some(&matching_layout), Some(&processor), &text_only, &[]).is_ok());
        let mismatched_layout = ImageTokenLayout {
            feature_token: 999999,
            span: None,
        };
        match prepare_input(Some(&mismatched_layout), Some(&processor), &text_only, &[]) {
            Ok(_) => panic!("expected feature_token mismatch error"),
            Err(err) => assert!(err.to_string().contains("feature_token")),
        }
    }
}
