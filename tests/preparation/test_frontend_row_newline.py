# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Ignition's row_newline image profile validation."""

from __future__ import annotations

import copy
import pytest

from ignition.preparation.frontend import image_settings


def _row_newline_configs():
    processor = {
        "processor_class": "HYV4FlashProcessor",
        "image_processor": {
            "image_processor_type": "HYV4VLImageProcessor",
            "patch_size": 16,
            "temporal_patch_size": 1,
            "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "rescale_factor": 1 / 255,
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
            "do_convert_rgb": True,
            "resample": 1,
            "min_pixels": 4096,
            "max_pixels": 4_194_304,
        },
    }
    model = {
        "model_type": "hy_v4_flash",
        "image_token_id": 120120,
        "image_newline_token_id": 120121,
        "video_token_id": 120124,
        "vision_config": {
            "patch_size": 16,
            "spatial_merge_size": 2,
            "num_channels": 3,
            "max_image_size": 2048,
        },
    }
    return processor, model


@pytest.mark.parametrize("resample", [None, 1, 3])
def test_row_newline_accepts_documented_bicubic_and_unused_lanczos_resample(resample):
    processor, model = _row_newline_configs()
    if resample is None:
        processor["image_processor"].pop("resample")
    else:
        processor["image_processor"]["resample"] = resample
    result = image_settings(processor, model, "row_newline")
    assert result == {
        "row_newline": {
            "min_pixels": 4096,
            "max_pixels": 4_194_304,
            "patch_size": 16,
            "merge_size": 2,
            "position_side": 128,
            "image_token_id": 120120,
            "newline_token_id": 120121,
            "video_token_id": 120124,
        }
    }


@pytest.mark.parametrize("bad_resample", [0, 2, 4, True, "bicubic"])
def test_row_newline_rejects_conflicting_resample_kernel(bad_resample):
    processor, model = _row_newline_configs()
    processor["image_processor"]["resample"] = bad_resample
    with pytest.raises(ValueError, match="resample"):
        image_settings(processor, model, "row_newline")


def test_row_newline_rejects_invalid_area_and_token_ids():
    processor, model = _row_newline_configs()
    bad_area = copy.deepcopy(processor)
    bad_area["image_processor"]["min_pixels"] = bad_area["image_processor"]["max_pixels"] + 1
    with pytest.raises(ValueError, match="resize area"):
        image_settings(bad_area, model, "row_newline")

    bad_token = copy.deepcopy(model)
    bad_token["image_newline_token_id"] = -1
    with pytest.raises(ValueError, match="image_newline_token_id"):
        image_settings(processor, bad_token, "row_newline")
