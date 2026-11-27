# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent startup outputs remain separate from serving inputs and capacities."""

import pytest
import torch

from ignition.preparation.execution import GraphSignature, PreparedExecution, TunedBinding, TuningGraph


def execution(**changes):
    values = dict(device=0, max_active=4, graphs={"decode": object()},
        signatures={"decode": GraphSignature("decode", 1, 1, ("tokens",))},
        buffers={"frame_index": torch.zeros(1, dtype=torch.int32),
                 "submission_sequence": torch.zeros(1, dtype=torch.int64),
                 "completion_records": torch.zeros((2, 4, 32), dtype=torch.int64),
                 "tokens": torch.zeros(1, dtype=torch.int64),
                 "selector": torch.ones(1, dtype=torch.int32),
                 "status": torch.ones(1, dtype=torch.int32)},
        tuning_graphs={"choose": TuningGraph(object(), ("selector",), "status")},
        tuned_bindings={"decode": (TunedBinding("selector", "choose"),)})
    return PreparedExecution(**(values | changes))


def test_manifest_has_separate_tuners_without_host_uploads(monkeypatch):
    bundle = execution()
    spans = [(4096 + index * 4096, tensor.numel() * tensor.element_size())
             for index, tensor in enumerate(bundle.buffers.values())]
    monkeypatch.setattr(bundle, "_validate_resources", lambda: (9, [11, 12], spans))
    manifest = bundle.manifest()
    assert manifest["abi_version"] == 6
    assert set(bundle.signatures) == set(bundle.estimated_durations_ns) == {"decode"}
    assert set(bundle.graphs) == {"decode", "choose"}
    assert manifest["tuning_graphs"] == [dict(name="choose", handle=12,
                                              outputs=["selector"], status_buffer="status")]
    assert manifest["graphs"][0]["tuned_bindings"] == [dict(buffer="selector", tuning_graph="choose")]
    assert {binding["buffer_name"] for binding in manifest["graphs"][0]["input_bindings"]} == {
        "frame_index", "submission_sequence", "tokens"}


def test_invalid_startup_dependencies_and_uploads_are_rejected():
    with pytest.raises(ValueError, match="output of its tuning graph"):
        execution(tuned_bindings={"decode": (TunedBinding("tokens", "choose"),)})
    with pytest.raises(ValueError, match="host uploads"):
        execution(signatures={"decode": GraphSignature("decode", 1, 1, ("selector",))})
    with pytest.raises(ValueError, match="int32"):
        execution(tuning_graphs={"choose": TuningGraph(object(), ("selector",), "tokens")})
    with pytest.raises(ValueError, match="distinct"):
        execution(tuning_graphs={"decode": TuningGraph(object(), ("selector",), "status")})


def test_manifest_emits_explicit_empty_startup_catalog(monkeypatch):
    bundle = execution(tuning_graphs={}, tuned_bindings={})
    spans = [(4096 + index * 4096, tensor.numel() * tensor.element_size())
             for index, tensor in enumerate(bundle.buffers.values())]
    monkeypatch.setattr(bundle, "_validate_resources", lambda: (9, [11], spans))
    manifest = bundle.manifest()
    assert manifest["tuning_graphs"] == []
    assert manifest["graphs"][0]["tuned_bindings"] == []
