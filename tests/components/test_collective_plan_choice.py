# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared reductions preserve the selected surface's capture and ownership contract."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("surface", ["PCIeTwoShotBF16.all_reduce", "OneshotAllReduce.all_reduce"])
def test_preparation_uses_selected_surface_and_retains_storage(monkeypatch, surface):
    from b12x.comm.pcie import _twoshot_preparation
    from b12x.preparation import PreparedCall
    from ignition.components import collectives

    source = torch.full((32, 5120), 3, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    callbacks, calls, selected = [], [], []
    plan = SimpleNamespace(query=SimpleNamespace(surface=surface))
    plan.request = lambda **kwargs: callbacks.append(kwargs) or object()
    runtime = SimpleNamespace(algorithm="oneshot", world_size=4, single_channel=True,
                              exchange_group=object(), for_stream=lambda: SimpleNamespace(),
                              plan=lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1, 2, 3])

    def two_call(state, *, payload, out):
        calls.append("two")
        assert payload is source and out is output
        call = PreparedCall(run=lambda: out.copy_(payload), output=out, capture_safe=False,
                            owners=(state, payload, out))
        selected.append(call)
        return call

    monkeypatch.setattr(_twoshot_preparation, "prepared_call", two_call)
    operation = collectives.PreparedAllReduce(source, output, runtime=runtime,
        session=object(), coordinator=object(), name="hidden.reduce")
    state = SimpleNamespace(query=plan.query,
        bind_input=lambda tensor: calls.append("bind") or None,
        run_plain=lambda inp, out: out.copy_(inp))
    prepared = callbacks[0]["prepare_call"](state)
    if selected:
        assert prepared is selected[0]
    else:
        assert prepared.owners == (runtime,)
    assert prepared.capture_safe is (surface != "PCIeTwoShotBF16.all_reduce")
    prepared.run()
    assert torch.equal(source, output)
    assert calls == (["two"] if surface == "PCIeTwoShotBF16.all_reduce" else ["bind"])
    assert operation.runtime is runtime and operation.plan is plan


@pytest.mark.parametrize("world", [2, 4])
def test_single_host_varlen_gather_plans_the_pcie_counted_byte_surface(monkeypatch, world):
    from b12x.comm import pcie
    from ignition.components import collectives
    from ignition.preparation.collectives import VARLEN_BANK_BYTES

    queries, requests = [], []
    plan = SimpleNamespace(request=lambda **kwargs: requests.append(kwargs) or object())
    runtime = SimpleNamespace(exchange_group=object(), world_size=world)
    monkeypatch.setattr(pcie, "query_from_runtime",
                        lambda runtime, *, surface, call: queries.append((surface, call)) or "query")
    monkeypatch.setattr(pcie, "plan", lambda query, *, runtime: plan if query == "query" else None)
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: list(range(world)))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    output = torch.zeros((world, 528 * 16), dtype=torch.uint8)
    counts = torch.zeros(world, dtype=torch.int64)
    operation = collectives.PreparedVarlenGather(output, counts, runtime=runtime, session=None,
                                                 coordinator=None, name="records")
    assert operation.plan is plan
    assert [(surface, call["bank_bytes"]) for surface, call in queries] == [
        ("DcpAllToAll.all_gather_varlen", VARLEN_BANK_BYTES)]
    assert queries[0][1]["out"] is output and queries[0][1]["counts"] is counts
    assert [request["name"] for request in requests] == ["records"]


@pytest.mark.parametrize("surface", ["OneshotAllReduce.all_reduce", "PCIeTwoShotBF16.all_reduce"])
def test_static_prepared_all_reduce_enables_fused_addend(monkeypatch, surface):
    from b12x.comm.pcie import _twoshot_preparation
    from b12x.preparation import PreparedCall
    from ignition.components import collectives

    plan_calls, callbacks, run_calls = [], [], []
    plan = SimpleNamespace(query=SimpleNamespace(surface=surface))
    plan.request = lambda **kwargs: callbacks.append(kwargs) or object()
    channel = SimpleNamespace(_eager_ptrs=((1,), (2,)))
    runtime = SimpleNamespace(algorithm="oneshot", world_size=4, single_channel=True,
                              exchange_group=object(), for_stream=lambda: channel,
                              plan=lambda *_args, **kwargs: plan_calls.append(kwargs) or plan)
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1, 2, 3])

    def two_call(state, *, payload, out, addend=None):
        run_calls.append(("two", addend is not None))
        return PreparedCall(run=lambda: out.copy_(payload + addend), output=out, capture_safe=False,
                            owners=(state, payload, out))

    monkeypatch.setattr(_twoshot_preparation, "prepared_call", two_call)
    source = torch.full((4, 64), 2.0, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    operation = collectives.PreparedAllReduce(source, output, runtime=runtime,
        session=object(), coordinator=object(),
        name="mlp.reduce", has_addend=True)
    assert operation.fused_addend is True
    assert plan_calls == [dict(out=output, has_addend=True)]

    state = SimpleNamespace(query=plan.query,
        bind_input=lambda tensor: run_calls.append(("bind", tensor is source)),
        run_plain=lambda inp, out, addend=None: (run_calls.append(("plain", addend is not None)), out.copy_(inp + addend)))
    prepared = callbacks[0]["prepare_call"](state)
    prepared.run()
    assert torch.equal(output, source)
    assert run_calls == ([("two", True)] if surface == "PCIeTwoShotBF16.all_reduce"
                         else [("bind", True), ("plain", True)])


def test_static_prepared_all_gather_enables_gate(monkeypatch):
    from b12x.comm import pcie
    from ignition.components import collectives

    queries, callbacks, runs = [], [], []
    plan = SimpleNamespace(request=lambda **kwargs: callbacks.append(kwargs) or object())
    runtime = SimpleNamespace(exchange_group=object(), world_size=4, device=torch.device("cpu"),
                              max_batch_size=8, heads_per_rank=1, total_heads=4, query_head_dim=64)
    monkeypatch.setattr(pcie, "query_from_runtime",
                        lambda rt, *, surface, call: queries.append((surface, dict(call))) or "query")
    monkeypatch.setattr(pcie, "plan", lambda query, *, runtime: plan if query == "query" else None)
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1, 2, 3])

    source = torch.zeros((2, 64), dtype=torch.bfloat16)
    output = torch.zeros((2, 256), dtype=torch.bfloat16)
    operation = collectives.PreparedAllGather(source, output, runtime=runtime,
        session=object(), coordinator=object(),
        name="logprobs.exchange", has_gate=True)
    assert operation.gated is True
    assert queries[0][0] == "DcpAllToAll.all_gather_heads"
    assert queries[0][1]["has_gate"] is True
    state = SimpleNamespace(run=lambda **kwargs: runs.append(kwargs))
    prepared = callbacks[0]["prepare_call"](state)
    prepared.run()
    assert "has_gate" not in runs[0]
    assert runs[0]["gate"].dtype == torch.int32 and runs[0]["gate"].item() == 1
