# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared graph capture and compiled warmup order indivisible allocations consistently across ranks."""

from concurrent.futures import Future
from contextlib import nullcontext
from datetime import timedelta
import multiprocessing
from pathlib import Path
import pickle
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from ignition.preparation import aot, compile_pool, inductor_tuning
from ignition.preparation.capture import _capture_order


def test_capture_orders_largest_request_before_largest_total_and_preserves_ties():
    names = ("init", "vision", "prefill", "decode", "decode-other", "restore")
    compiled = {name: SimpleNamespace(capture_allocation_sizes=sizes) for name, sizes in (
        ("vision", (1600, 128)), ("prefill", (1603,)), ("decode", (16,)), ("decode-other", (16,)))}
    assert _capture_order(names, compiled, None) == (
        "prefill", "vision", "decode", "decode-other", "init", "restore")
    assert _capture_order(names, {}, None) == names


def _rank_order(rank, store, output, mismatched):
    import torch.distributed as dist

    try:
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2,
                                timeout=timedelta(seconds=15))
        names = ("first", "second", "init")
        if mismatched and rank == 1:
            names = ("first", "second")
        compiled = {name: SimpleNamespace(capture_allocation_sizes=(size,))
                    for name, size in zip(names[:2], ((64, 32) if rank == 0 else (32, 128)))}
        coordinator = SimpleNamespace(ranks=(0, 1), group=dist.group.WORLD)
        try:
            result = _capture_order(names, compiled, coordinator)
        except ValueError as error:
            result = str(error)
        output.put((rank, result))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("mismatched", [False, True])
def test_capture_order_agrees_across_tensor_parallel_ranks(tmp_path, mismatched):
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [context.Process(target=_rank_order, args=(rank, tmp_path / "store", output, mismatched))
                 for rank in range(2)]
    for process in processes:
        process.start()
    try:
        results = dict(output.get(timeout=30) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        expected = ("tensor-parallel capture requires matching forward catalogs" if mismatched
                    else ("second", "first", "init"))
        assert results == {0: expected, 1: expected}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def _saved_forward_artifact(sizes):
    entry = SimpleNamespace(compiled_fw=SimpleNamespace(result=SimpleNamespace(
        _flare_pool_sizes=tuple(("cuda", size) for size in sizes))))
    signature = (((4,), (1,), 0, torch.float32, torch.device("cuda:0"), 0),)
    owner = SimpleNamespace(
        artifact=SimpleNamespace(serialize=lambda: pickle.dumps((entry, {}, {}))),
        inputs=[], invocations=[], signature=signature, _serialized_artifact=None,
        _cuda_allocation_sizes=None, _tuning_frozen=False)
    payload = aot.StorageBackend.serialize_compile_artifacts(owner)
    return {"compiled_fn": (aot.StorageBackend.deserialize_compile_artifacts, payload)}


class _CpuForward:
    def __init__(self):
        self.value = torch.ones(4, dtype=torch.float32)

    def __call__(self):
        return self.value + 1.0


def _write_real_aot_artifact(path, sizes=None, *, compiled=None):
    if compiled is None:
        compiled = aot.aot_forward(_CpuForward(), options={"compile_threads": 1})
    compiled._artifacts.compiled_fn._cuda_allocation_sizes = sizes
    compiled.save_compiled_function(str(path))
    return compiled


def test_allocation_sizes_reads_real_saved_aot_artifact_via_unpickler(tmp_path, monkeypatch):
    from torch._dynamo import aot_compile

    cache = aot.ForwardCache.__new__(aot.ForwardCache)
    default_path = tmp_path / "cpu_default.bin"
    compiled = _write_real_aot_artifact(default_path, sizes=None)
    assert cache._allocation_sizes(default_path) == ()

    planned_path = tmp_path / "cuda_planned.bin"
    _write_real_aot_artifact(planned_path, sizes=(4096, 1024), compiled=compiled)

    unpickler_calls = 0
    real_unpickler = aot_compile.AOTCompileUnpickler

    class TrackingUnpickler(real_unpickler):
        def __init__(self, external_data, file):
            nonlocal unpickler_calls
            unpickler_calls += 1
            super().__init__(external_data, file)

    monkeypatch.setattr(aot_compile, "AOTCompileUnpickler", TrackingUnpickler)
    assert cache._allocation_sizes(planned_path) == (4096, 1024)
    assert unpickler_calls == 1

    wrong_backend = tmp_path / "wrong_backend.bin"
    wrong_backend.write_bytes(pickle.dumps({"compiled_fn": (object, b"")}))
    with pytest.raises(TypeError, match="does not contain a Flare storage backend"):
        cache._allocation_sizes(wrong_backend)


def test_storage_backend_serializes_cuda_allocation_sizes_for_warmup_ordering(tmp_path, monkeypatch):
    entry = SimpleNamespace(compiled_fw=SimpleNamespace(result=SimpleNamespace(
        _flare_pool_sizes=(("cpu", 128), ("cuda", 4096), ("cuda", 1024)))))
    signature = (((4,), (1,), 0, torch.float32, torch.device("cuda:0"), 0),)
    owner = SimpleNamespace(
        artifact=SimpleNamespace(serialize=lambda: pickle.dumps((entry, {}, {}))),
        inputs=[], invocations=[], signature=signature, _serialized_artifact=None,
        _cuda_allocation_sizes=None, _tuning_frozen=False)

    payload = aot.StorageBackend.serialize_compile_artifacts(owner)
    assert owner._cuda_allocation_sizes == (4096, 1024)

    monkeypatch.setattr(aot, "_load_inductor", lambda data, device, *, frozen: (SimpleNamespace(), None))
    restored = aot.StorageBackend.deserialize_compile_artifacts(payload)
    assert restored._cuda_allocation_sizes == (4096, 1024)
    bound = aot.BoundForward(SimpleNamespace(_artifacts=SimpleNamespace(compiled_fn=restored)), object(), True)
    assert bound.capture_allocation_sizes == (4096, 1024)

    path = tmp_path / "forward.bin"
    path.write_bytes(pickle.dumps({"compiled_fn": (aot.StorageBackend.deserialize_compile_artifacts, payload)}))
    cache = aot.ForwardCache.__new__(aot.ForwardCache)
    assert cache._allocation_sizes(path) == (4096, 1024)


@pytest.mark.parametrize("deferred", [False, True])
def test_prepare_many_warms_compiled_forwards_by_largest_allocation_first(tmp_path, monkeypatch, deferred):
    sizes = {"small": (16,), "large": (64,), "medium": (32,)}
    warmed = []

    class Pool:
        workers = 2

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, owner, path):
            path.write_bytes(pickle.dumps(_saved_forward_artifact(sizes[owner.name])))
            future = Future()
            future.set_result(None)
            return future

    class Selections:
        def __init__(self, *args, **kwargs):
            pass

        def release(self, bound):
            pass

    def fake_bound(name, *, cache_hit):
        backend = aot.StorageBackend.__new__(aot.StorageBackend)
        backend._cuda_allocation_sizes = sizes[name]
        artifact = SimpleNamespace(_artifacts=SimpleNamespace(compiled_fn=backend))
        bound = aot.BoundForward(artifact, SimpleNamespace(name=name), cache_hit)
        bound.prepare_capture = lambda: 0
        return bound

    def fake_restore(self, forward, *, name, signature, deferred=False, report=None, before_run=None):
        path = tmp_path / f"{name}.bin"
        if not deferred and name == "small":
            return fake_bound(name, cache_hit=True), path
        return None, path

    def fake_load(self, forward, path, *, required=False, before_run=None):
        return fake_bound(forward.name, cache_hit=True)

    def fake_warm(self, bound, path, *, name, **kwargs):
        warmed.append(name)
        return bound

    monkeypatch.setattr(compile_pool, "CompilePool", Pool)
    monkeypatch.setattr(inductor_tuning, "InductorSelections", Selections)
    monkeypatch.setattr(aot, "_guard_code_cache", lambda path: nullcontext())
    monkeypatch.setattr(aot.ForwardCache, "_restore", fake_restore)
    monkeypatch.setattr(aot.ForwardCache, "_load", fake_load)
    monkeypatch.setattr(aot.ForwardCache, "_warm", fake_warm)
    monkeypatch.setattr(aot.ForwardCache, "_path", lambda self, forward, *, name, signature: tmp_path / f"{name}.bin")

    cache = aot.ForwardCache.__new__(aot.ForwardCache)
    cache.root, cache.device, cache.headroom_bytes = tmp_path, "cpu", 0
    cache.options, cache.compile_workers, cache.lock_gpu_clocks = MappingProxyType({}), 2, False
    functions = {name: SimpleNamespace(name=name) for name in ("small", "large", "medium")}
    signatures = {name: ("decode", 1, 1) for name in functions}

    compiled = cache._prepare_many(
        functions, signatures, progress=None, coordinator=None,
        materialize=(lambda: None) if deferred else None, warmup=nullcontext, before_run=None)
    assert warmed == ["large", "medium", "small"]
    assert tuple(compiled) == ("large", "medium", "small")


def _rank_warmup_order(rank, store, root, output):
    import torch.distributed as dist

    try:
        rank_sizes = {"first": (64,), "second": (32,)} if rank == 0 else {"first": (32,), "second": (128,)}
        warmed = []

        class Selections:
            def __init__(self, *args, **kwargs):
                pass

            def register(self, bound):
                pass

            def _check(self, failure):
                assert failure is None, failure

            def warm(self, bound, warmup, before_run=None):
                warmed.append(bound.owner.name)

            def freeze(self, bound):
                pass

            def release(self, bound):
                pass

        def make_bound(forward):
            backend = aot.StorageBackend.__new__(aot.StorageBackend)
            backend._cuda_allocation_sizes = rank_sizes[forward.name]
            artifact = SimpleNamespace(
                _artifacts=SimpleNamespace(compiled_fn=backend),
                save_compiled_function=lambda path: None)
            bound = aot.BoundForward(artifact, forward, True)
            bound.prepare_capture = lambda: 0
            return bound

        inductor_tuning.InductorSelections = Selections
        compile_pool.compiler_resources = lambda *args, **kwargs: compile_pool.CompilerResources(
            1, 40 << 30, 16, 32 << 30, 8 << 30)
        aot._guard_code_cache = lambda path: nullcontext()
        aot.ForwardCache._restore = lambda self, forward, *, name, signature, deferred=False, before_run=None: (
            make_bound(forward), Path(root) / f"{forward.name}.{forward.compilation_variant}.bin")
        aot.ForwardCache._load = lambda self, forward, path, required=False, before_run=None: make_bound(forward)
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2,
                                timeout=timedelta(seconds=15))
        cache = aot.ForwardCache.__new__(aot.ForwardCache)
        cache.root, cache.device, cache.headroom_bytes = Path(root), "cpu", 0
        cache.options, cache.compile_workers, cache.lock_gpu_clocks = MappingProxyType({}), 1, False
        functions = {name: SimpleNamespace(name=name, compilation_variant=f"rank{rank}")
                     for name in ("first", "second")}
        coordinator = SimpleNamespace(
            group=dist.group.WORLD, ranks=(0, 1), hosts=("host", "host"), spans_hosts=False,
            barrier=lambda key, ranks: dist.monitored_barrier(group=dist.group.WORLD))
        try:
            cache._prepare_ranks(functions, {name: ("decode", 1, 1) for name in functions}, coordinator,
                                 progress=None, materialize=lambda: None, warmup=nullcontext, before_run=None)
            output.put((rank, ("ok", warmed)))
        except Exception as error:
            output.put((rank, ("error", f"{type(error).__name__}: {error}")))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_prepare_ranks_warms_compiled_forwards_in_agreed_largest_allocation_order(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    compiled = aot.aot_forward(_CpuForward(), options={"compile_threads": 1})
    for name, variant, sizes in (
        ("first", "rank0", (64,)),
        ("second", "rank0", (32,)),
        ("first", "rank1", (32,)),
        ("second", "rank1", (128,)),
    ):
        _write_real_aot_artifact(cache_root / f"{name}.{variant}.bin", sizes=sizes, compiled=compiled)

    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [context.Process(target=_rank_warmup_order,
                                 args=(rank, tmp_path / "store", cache_root, output))
                 for rank in range(2)]
    for process in processes:
        process.start()
    try:
        results = dict(output.get(timeout=30) for _ in processes)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert results == {0: ("ok", ["second", "first"]), 1: ("ok", ["second", "first"])}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def test_prepare_ranks_propagates_allocation_size_failure_without_hanging(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    compiled = aot.aot_forward(_CpuForward(), options={"compile_threads": 1})
    for name, variant, sizes in (
        ("first", "rank0", (64,)),
        ("first", "rank1", (32,)),
        ("second", "rank1", (128,)),
    ):
        _write_real_aot_artifact(cache_root / f"{name}.{variant}.bin", sizes=sizes, compiled=compiled)
    # Corrupt one rank's saved artifact so _allocation_sizes fails only on rank 0.
    (cache_root / "second.rank0.bin").write_bytes(b"truncated")

    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [context.Process(target=_rank_warmup_order,
                                 args=(rank, tmp_path / "store", cache_root, output))
                 for rank in range(2)]
    for process in processes:
        process.start()
    try:
        results = dict(output.get(timeout=10) for _ in processes)
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        for rank in (0, 1):
            status, message = results[rank]
            assert status == "error"
            assert "RuntimeError: compiled forward materialization failed:" in message
            assert "None" in message
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
