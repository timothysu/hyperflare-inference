# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Extract and validate the structural frontend fields consumed by Helios."""

import json
import hashlib
import math
from enum import IntEnum
from pathlib import Path


class PromptEncoder(IntEnum):
    PROMPT_JINJA = 0
    DEEPSEEK_V4 = 1
    DEEPSEEK_V41 = 2
    TML_V0 = 3


class QwenPositionScheme(IntEnum):
    CORNER_MAX = 0
    CENTERED_STEP = 1


class QwenResizeKernel(IntEnum):
    TORCHVISION_BICUBIC = 0
    PILLOW_BICUBIC = 1


TOKEN_FIELDS = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")


def annotated_template_asset(source: bytes, patch_path: Path):
    """Insert message marks into exact checkpoint bytes, without bundling that source."""
    patch = json.loads(patch_path.read_text())
    digest = hashlib.sha256(source).hexdigest()
    if digest != patch["source_sha256"]:
        raise ValueError(f"checkpoint chat template does not match {patch_path}: its SHA-256 is {digest},"
                         f" and the patch applies to {patch['source_sha256']}")
    parts, previous = [], 0
    for offset in patch["insertions"]:
        if type(offset) is not int or not previous <= offset <= len(source):
            raise ValueError(f"invalid template insertion offset in {patch_path}")
        parts.extend((source[previous:offset], b"{%- mark -%}"))
        previous = offset
    contents = b"".join((*parts, source[previous:]))
    if hashlib.sha256(contents).hexdigest() != patch["result_sha256"]:
        raise ValueError(f"patched chat template digest does not match {patch_path}")
    return {"text": contents.decode("utf-8"), "sha256": patch["result_sha256"]}


def downloaded_template_asset(patch_path: Path, *, cache_dir=None):
    """Fetch a canonical template at its pinned HF revision, then annotate it."""
    from huggingface_hub import hf_hub_download

    from ignition.errors import FetchError

    source = json.loads(patch_path.read_text())["source"]
    try:
        path = hf_hub_download(source["repository"], source["filename"],
                               revision=source["revision"], cache_dir=cache_dir)
    except Exception as error:  # noqa: BLE001 - Network, authorization, and disk failures share one exit code.
        raise FetchError(
            f"cannot read {source['filename']} of {source['repository']} at revision {source['revision']}: {error}."
            f" This chat template is not part of the checkpoint snapshot, so preparation needs it in the hub cache."
            f" On a host without network access, run `hf download {source['repository']} {source['filename']}"
            f" --revision {source['revision']} --cache-dir <hub cache>` on a connected host and copy the"
            f" downloaded cache entries to the hub cache of this host.") from error
    return annotated_template_asset(Path(path).read_bytes(), patch_path)


def _token(value):
    if isinstance(value, dict):
        value = value.get("content")
    if not isinstance(value, str):
        raise ValueError("special token must be a string or an object with string content")
    return value


def tokenizer_settings(config):
    if not isinstance(config, dict):
        raise ValueError("tokenizer configuration must be an object")
    if config.get("clean_up_tokenization_spaces") is True:
        raise ValueError("frontend requires tokenizer clean_up_tokenization_spaces=false")
    result = {name: _token(config[name]) for name in TOKEN_FIELDS if config.get(name) is not None}
    additional = config.get("additional_special_tokens")
    if additional is not None:
        if not isinstance(additional, list):
            raise ValueError("additional_special_tokens must be a list")
        result["additional_special_tokens"] = {"values": [_token(value) for value in additional]}
    return result


def _expected(config, values, label):
    for key, expected in values.items():
        value = config.get(key)
        if value != expected or (type(expected) in (bool, int) and type(value) is not type(expected)):
            raise ValueError(f"unsupported {label} {key}")


def _validated_resize_and_normalization(config):
    result = dict(min_pixels=config.get("min_pixels"), max_pixels=config.get("max_pixels"),
                  image_mean=config.get("image_mean"), image_std=config.get("image_std"))
    if (any(type(result[key]) is not int for key in ("min_pixels", "max_pixels"))
            or not 0 < result["min_pixels"] <= result["max_pixels"]):
        raise ValueError("unsupported image processor resize area")
    for key in ("image_mean", "image_std"):
        values = result[key]
        if (not isinstance(values, list) or len(values) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or key == "image_std" and any(v <= 0 for v in values)):
            raise ValueError(f"unsupported image processor {key}")
        result[key] = [float(v) for v in values]
    return result


def image_settings(
    config,
    model,
    profile,
    qwen_model_type="qwen4_exp",
    *,
    qwen_processor_class="Qwen3VLProcessor",
    qwen_image_processor_type="Qwen2VLImageProcessorFast",
    qwen_channel_normalization=False,
    qwen_position_scheme=QwenPositionScheme.CORNER_MAX,
    qwen_resize_kernel=QwenResizeKernel.TORCHVISION_BICUBIC,
    qwen_tokens=None,
):
    """Resolve the image settings the native frontend consumes for ``profile``.

    For the ``qwen`` profile, recipes declare the checkpoint's expected ``model_type``,
    processor classes, normalization mode, 3D position scheme, resize kernel, and any
    required image/video placeholder strings explicitly.
    """
    if profile == "padded_grid":
        # Inkling 2 ships the processor in processor_config.json; `size` is the token cell, not a resize target.
        _expected(config, {"processor_class": "Inkling2Processor"}, "padded-grid processor")
        image = config.get("image_processor")
        expected = {"image_processor_type": "Inkling2ImageProcessor", "size": {"height": 40, "width": 40},
            "image_mean": [0.5] * 3, "image_std": [0.5] * 3, "rescale_factor": 1 / 255,
            "do_rescale": True, "do_normalize": True, "do_convert_rgb": True}
        if not isinstance(image, dict) or set(image) != set(expected):
            raise ValueError("unsupported padded-grid image processor fields")
        _expected(image, expected, "padded-grid image processor")
        if model.get("model_type") not in ("inkling2", "inkling2_loop"):
            raise ValueError("unsupported padded-grid model configuration model_type")
        _expected(model["vision_config"], {"native_patch_size": 20, "adapter_merge_size": 2,
            "posemb_grid_size": 32, "temporal_patch_size": 2}, "padded-grid vision configuration")
        return {"padded_grid": {}}
    if profile == "glm":
        _expected(config, {"processor_class": "Glm5NextProcessor"}, "GLM processor")
        image = config["image_processor"]
        _expected(image, {"image_processor_type": "Glm5NextImageProcessor", "patch_size": 14,
            "temporal_patch_size": 2, "merge_size": 2, "patch_expand_factor": 1,
            "min_image_tokens": 16, "max_image_tokens": 8000,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711]}, "GLM image processor")
        optional = {"resample": 3, "rescale_factor": 1 / 255,
            "do_resize": True, "do_rescale": True, "do_normalize": True, "do_convert_rgb": True}
        _expected(image, {key: value for key, value in optional.items() if key in image}, "GLM image processor")
        _expected(model, {"model_type": "glm5_next", "image_token_id": 154854, "video_token_id": 154855},
                  "GLM model configuration")
        _expected(model["vision_config"], {"patch_size": 14, "temporal_patch_size": 2,
            "spatial_merge_size": 2, "in_channels": 3}, "GLM vision configuration")
        return {"glm": {}}
    if profile == "radio":
        _expected(config, {"image_processor_type": "NemotronH_Omni_Reasoning_V3ImageProcessor",
            "patch_size": 16, "downsample_ratio": .5, "max_model_len": 16384,
            "min_num_patches": 1024, "max_num_patches": 13312,
            "norm_mean": [0.48145466, 0.4578275, 0.40821073],
            "norm_std": [0.26862954, 0.26130258, 0.27577711]}, "RADIO image processor")
        _expected(model, {"patch_size": 16, "downsample_ratio": .5, "ps_version": "v2",
            "img_context_token_id": 18}, "RADIO image configuration")
        _expected(model["vision_config"], {"max_img_size": 2048, "patch_size": 16}, "RADIO vision configuration")
        return {"radio": {}}
    if profile == "moonvit":
        _expected(config["media_proc_cfg"], {"in_patch_limit": 65536, "patch_size": 14,
            "image_mean": [0.5] * 3, "image_std": [0.5] * 3, "merge_kernel_size": 2,
            "fixed_output_tokens": None, "patch_limit_on_one_side": 512,
            "transparent_bg_fill_stage": "after_resize", "transparent_bg_config": {
                "pattern": "chessboard", "chessboard_square_size": 8,
                "chessboard_square_on_top_left": True, "chessboard_white_value": 255,
                "chessboard_gray_value": 180}}, "MoonViT image processor")
        _expected(model["vision_config"], {"patch_size": 14, "init_pos_emb_height": 64,
            "init_pos_emb_width": 64, "pos_emb_interpolation_mode": "bilinear",
            "merge_kernel_size": [2, 2]}, "MoonViT vision configuration")
        return {"moonvit": {}}
    if profile == "deepseek_row_major":
        vision = model["vision_config"]
        _expected(vision, {"patch_size": 14, "downsample_ratio": 3, "max_image_tokens": 1024,
            "min_pixels": 295936, "max_wh_ratio": None}, "DeepSeek row-major vision configuration")
        return {profile: dict(min_pixels=vision["min_pixels"], max_tokens=vision["max_image_tokens"])}
    if profile == "deepseek4":
        _expected(config, {"image_processor_type": "DeepSeekV4ImageProcessor", "patch_size": 14,
            "merge_size": 3, "max_image_tokens": 384, "min_pixels": 147456,
            "max_width_height_ratio": 8}, "DeepSeek image processor")
        _expected(model, {"vision_patch_size": 14, "vision_downsample_ratio": 3,
            "vision_max_n_token": 384, "vision_min_pixels": 147456,
            "vision_max_wh_ratio": 8}, "DeepSeek vision configuration")
        return {"deepseek4": {}}
    if profile not in ("qwen", "mimo"):
        raise ValueError("unsupported image preprocessing profile")
    qwen = profile == "qwen"
    if qwen:
        if not isinstance(qwen_position_scheme, QwenPositionScheme):
            raise ValueError("qwen_position_scheme must be a QwenPositionScheme")
        if not isinstance(qwen_resize_kernel, QwenResizeKernel):
            raise ValueError("qwen_resize_kernel must be a QwenResizeKernel")
        if type(qwen_channel_normalization) is not bool:
            raise ValueError("qwen_channel_normalization must be a boolean")
        expected_processor = {
            "processor_class": qwen_processor_class,
            "image_processor_type": qwen_image_processor_type,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
        }
        if not qwen_channel_normalization:
            expected_processor["image_mean"] = [0.5] * 3
            expected_processor["image_std"] = [0.5] * 3
        if qwen_tokens is not None:
            if (
                not isinstance(qwen_tokens, dict)
                or not set(qwen_tokens).issubset({"image_token", "video_token"})
                or any(type(v) is not str or not v for v in qwen_tokens.values())
            ):
                raise ValueError(
                    "qwen_tokens may only specify non-empty image_token and video_token strings"
                )
            expected_processor.update(qwen_tokens)
        _expected(config, expected_processor, "image processor")
    else:
        _expected(config, {"processor_class": "Qwen2_5_VLProcessor",
            "image_processor_type": "Qwen2VLImageProcessor",
            "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2}, "image processor")
    # Processor subclasses whose class defaults omit these keys from preprocessor_config.json
    # are pinned by `image_processor_type`; any explicit override that conflicts is rejected.
    optional = {"resample": 3, "rescale_factor": 1 / 255,
        "do_resize": True, "do_rescale": True, "do_normalize": True, "do_convert_rgb": True}
    _expected(config, {key: value for key, value in optional.items() if key in config}, "image processor")
    _expected(model, {"model_type": qwen_model_type if qwen else "mimo_v2"}, "model configuration")
    vision = model["vision_config"]
    _expected(vision, {"patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2}, "vision configuration")
    if qwen:
        _expected(vision, {"in_channels": 3, "num_position_embeddings": 2304}, "vision configuration")
        if qwen_channel_normalization:
            result = _validated_resize_and_normalization(config)
            result["position_side"] = 48
        else:
            _expected(config["size"], {"shortest_edge": 65536, "longest_edge": 16777216}, "resize area")
            result = dict(min_pixels=65536, max_pixels=16777216, position_side=48)
        if qwen_position_scheme != QwenPositionScheme.CORNER_MAX:
            result["position_scheme"] = int(qwen_position_scheme)
        if qwen_resize_kernel != QwenResizeKernel.TORCHVISION_BICUBIC:
            result["resize_kernel"] = int(qwen_resize_kernel)
        for name in ("image_token_id", "video_token_id"):
            value = model[name]
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"invalid model {name}")
            result[name] = value
    else:
        # MiMo checkpoints publish their own resize area and normalization; the native frontend
        # consumes the packaged values, so preparation validates their form rather than fixing them.
        _expected(model, {"image_token_id": 151655}, "model configuration")
        result = _validated_resize_and_normalization(config)
    return {profile: result}


def _native_image_placeholder(snapshot, model, encoder):
    from tokenizers import Tokenizer

    token = "<｜deepseek_image｜>"
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    token_id = tokenizer.token_to_id(token)
    if (token_id != 129264 or token_id >= tokenizer.get_vocab_size()
            or tokenizer.encode(token, add_special_tokens=False).ids != [token_id]):
        raise ValueError("DeepSeek image placeholder must be atomic vocabulary token 129264")
    vocabulary_size = model.get("vocab_size")
    if vocabulary_size is not None and (type(vocabulary_size) is not int or token_id >= vocabulary_size):
        raise ValueError("DeepSeek image placeholder is outside the model vocabulary")
    if encoder == PromptEncoder.DEEPSEEK_V41:
        _expected(model, {"image_token_id": token_id}, "DeepSeek image configuration")
    return token_id


def prepare_frontend(
    snapshot,
    *,
    eos_token_ids,
    image_profile=None,
    preprocessor=None,
    prompt_encoder=PromptEncoder.PROMPT_JINJA,
    qwen_model_type="qwen4_exp",
    qwen_processor_class="Qwen3VLProcessor",
    qwen_image_processor_type="Qwen2VLImageProcessorFast",
    qwen_channel_normalization=False,
    qwen_position_scheme=QwenPositionScheme.CORNER_MAX,
    qwen_resize_kernel=QwenResizeKernel.TORCHVISION_BICUBIC,
    qwen_tokens=None,
):
    """Resolve checkpoint structure during preparation; sampling is recipe-owned."""
    if not isinstance(prompt_encoder, PromptEncoder):
        raise ValueError("prompt_encoder must be a PromptEncoder")
    snapshot = Path(snapshot)
    if type(eos_token_ids) is int:
        eos_token_ids = [eos_token_ids]
    eos_token_ids = list(eos_token_ids)
    if (not eos_token_ids or len(eos_token_ids) > 8 or len(set(eos_token_ids)) != len(eos_token_ids)
            or any(type(value) is not int or not 0 <= value < 2**32 for value in eos_token_ids)):
        raise ValueError("EOS token IDs must be one through eight distinct uint32 values")
    result = {"tokenizer": tokenizer_settings(json.loads((snapshot / "tokenizer_config.json").read_text())),
              "eos_token_ids": eos_token_ids}
    if prompt_encoder != PromptEncoder.PROMPT_JINJA:
        result["prompt_encoder"] = int(prompt_encoder)
    if image_profile is not None:
        expected_profile = {PromptEncoder.DEEPSEEK_V4: "deepseek4",
                            PromptEncoder.DEEPSEEK_V41: "deepseek_row_major"}.get(prompt_encoder)
        if expected_profile is not None:
            if image_profile != expected_profile:
                raise ValueError("image profile differs from the prompt encoder")
            result["image_placeholder_token_id"] = _native_image_placeholder(
                snapshot, json.loads((snapshot / "config.json").read_text()), prompt_encoder)
        path = snapshot / "preprocessor_config.json" if preprocessor is None else Path(preprocessor)
        result["image"] = image_settings(
            {} if image_profile == "deepseek_row_major" else json.loads(path.read_text()),
            json.loads((snapshot / "config.json").read_text()),
            image_profile,
            qwen_model_type=qwen_model_type,
            qwen_processor_class=qwen_processor_class,
            qwen_image_processor_type=qwen_image_processor_type,
            qwen_channel_normalization=qwen_channel_normalization,
            qwen_position_scheme=qwen_position_scheme,
            qwen_resize_kernel=qwen_resize_kernel,
            qwen_tokens=qwen_tokens,
        )
    return result
