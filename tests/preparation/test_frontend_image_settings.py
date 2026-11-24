# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for parameterized Qwen image settings extraction and Protobuf binding."""

import json
import math
import pytest

from google.protobuf.json_format import ParseDict

from ignition.preparation import runtime_core_pb2 as pb
from ignition.preparation.frontend import (
    QwenPositionScheme,
    QwenResizeKernel,
    image_settings,
    prepare_frontend,
)


def _base_qwen_model(model_type="qwen4_exp", image_token_id=248056, video_token_id=248057):
    return {
        "model_type": model_type,
        "image_token_id": image_token_id,
        "video_token_id": video_token_id,
        "vision_config": {
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_channels": 3,
            "num_position_embeddings": 2304,
        },
    }


def _default_qwen_processor():
    return {
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast",
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
    }


def test_default_qwen_image_settings_omit_extended_fields():
    result = image_settings(_default_qwen_processor(), _base_qwen_model(), "qwen")
    assert result == {
        "qwen": {
            "min_pixels": 65536,
            "max_pixels": 16777216,
            "position_side": 48,
            "image_token_id": 248056,
            "video_token_id": 248057,
        }
    }
    proto = ParseDict(result, pb.ImageSettings())
    assert proto.qwen.position_scheme == pb.QWEN_POSITION_SCHEME_CORNER_MAX
    assert proto.qwen.resize_kernel == pb.QWEN_RESIZE_KERNEL_TORCHVISION_BICUBIC
    assert list(proto.qwen.image_mean) == []
    assert list(proto.qwen.image_std) == []


def test_parameterized_qwen_image_settings_validate_and_serialize(tmp_path):
    preproc = {
        "processor_class": "CustomVLProcessor",
        "image_processor_type": "CustomVLImageProcessor",
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "min_pixels": 4096,
        "max_pixels": 4194304,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "image_token": "<|image_pad|>",
        "video_token": "<|video_pad|>",
    }
    model = _base_qwen_model(
        model_type="custom_vl",
        image_token_id=157157,
        video_token_id=156909,
    )
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"clean_up_tokenization_spaces": False}))
    (tmp_path / "config.json").write_text(json.dumps(model))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(preproc))

    frontend = prepare_frontend(
        tmp_path,
        eos_token_ids=[1, 2],
        image_profile="qwen",
        qwen_model_type="custom_vl",
        qwen_processor_class="CustomVLProcessor",
        qwen_image_processor_type="CustomVLImageProcessor",
        qwen_channel_normalization=True,
        qwen_position_scheme=QwenPositionScheme.CENTERED_STEP,
        qwen_resize_kernel=QwenResizeKernel.PILLOW_BICUBIC,
        qwen_tokens={"image_token": "<|image_pad|>", "video_token": "<|video_pad|>"},
    )
    assert frontend["image"] == {
        "qwen": {
            "min_pixels": 4096,
            "max_pixels": 4194304,
            "position_side": 48,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "position_scheme": int(QwenPositionScheme.CENTERED_STEP),
            "resize_kernel": int(QwenResizeKernel.PILLOW_BICUBIC),
            "image_token_id": 157157,
            "video_token_id": 156909,
        }
    }
    proto = ParseDict(frontend, pb.FrontendSettings())
    assert proto.image.qwen.position_scheme == pb.QWEN_POSITION_SCHEME_CENTERED_STEP
    assert proto.image.qwen.resize_kernel == pb.QWEN_RESIZE_KERNEL_PILLOW_BICUBIC
    assert list(proto.image.qwen.image_mean) == pytest.approx([0.48145466, 0.4578275, 0.40821073])
    assert list(proto.image.qwen.image_std) == pytest.approx([0.26862954, 0.26130258, 0.27577711])


@pytest.mark.parametrize(
    "override,match",
    [
        ({"resample": 2}, "resample"),
        ({"do_resize": False}, "do_resize"),
        ({"min_pixels": 8192, "max_pixels": 4096}, "resize area"),
        ({"image_mean": [math.nan, 0.5, 0.5]}, "image_mean"),
        ({"image_std": [0.25, 0.0, 0.25]}, "image_std"),
        ({"image_std": [0.25, math.inf, 0.25]}, "image_std"),
        ({"image_token": "<wrong>"}, "image_token"),
    ],
)
def test_parameterized_qwen_image_settings_reject_invalid_config(override, match):
    preproc = {
        "processor_class": "CustomVLProcessor",
        "image_processor_type": "CustomVLImageProcessor",
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "min_pixels": 4096,
        "max_pixels": 4194304,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "image_token": "<|image_pad|>",
        "video_token": "<|video_pad|>",
        **override,
    }
    with pytest.raises(ValueError, match=match):
        image_settings(
            preproc,
            _base_qwen_model(model_type="custom_vl"),
            "qwen",
            qwen_model_type="custom_vl",
            qwen_processor_class="CustomVLProcessor",
            qwen_image_processor_type="CustomVLImageProcessor",
            qwen_channel_normalization=True,
            qwen_position_scheme=QwenPositionScheme.CENTERED_STEP,
            qwen_resize_kernel=QwenResizeKernel.PILLOW_BICUBIC,
            qwen_tokens={"image_token": "<|image_pad|>", "video_token": "<|video_pad|>"},
        )


def test_parameterized_qwen_rejects_untyped_enum_values():
    with pytest.raises(ValueError, match="qwen_position_scheme"):
        image_settings(
            _default_qwen_processor(),
            _base_qwen_model(),
            "qwen",
            qwen_position_scheme=1,
        )
    with pytest.raises(ValueError, match="qwen_resize_kernel"):
        image_settings(
            _default_qwen_processor(),
            _base_qwen_model(),
            "qwen",
            qwen_resize_kernel=1,
        )


def test_parameterized_qwen_tokens_rejects_geometry_override():
    with pytest.raises(ValueError, match="qwen_tokens"):
        image_settings(
            {**_default_qwen_processor(), "patch_size": 14},
            _base_qwen_model(),
            "qwen",
            qwen_tokens={"patch_size": 14},
        )
    with pytest.raises(ValueError, match="qwen_tokens"):
        image_settings(
            _default_qwen_processor(),
            _base_qwen_model(),
            "qwen",
            qwen_tokens={"image_token": ""},
        )
