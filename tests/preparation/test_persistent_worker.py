# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Ignition PersistentWorker export metadata, validation, and reachability."""

import struct
from types import SimpleNamespace

import pytest
import torch

from ignition.components.completion import VECTOR_TOKENS_FIELD
from ignition.preparation import runtime_core_pb2
from ignition.preparation.execution import GraphSignature, PersistentWorker, PreparedExecution, TuningGraph
from ignition.preparation.model import PreparedModel
from ignition.preparation.runtime_core import Allocation
from ignition.preparation.tensor_metadata import (
    NumericStorage,
    live_metadata,
    merge_ranges,
    record_numeric_tensor,
    subtract_ranges,
)


class _FakeGraph:
    def __init__(self, handle):
        self._handle = handle

    def raw_cuda_graph_exec(self):
        return self._handle

    def reset(self):
        pass


def _base_buffers(max_active=4, token_capacity=8):
    return {
        "frame_index": torch.zeros((1,), dtype=torch.int32),
        "submission_sequence": torch.zeros((1,), dtype=torch.int64),
        "completion_records": torch.zeros((2, max_active, VECTOR_TOKENS_FIELD + token_capacity), dtype=torch.int64),
        "worker_ctrl": torch.zeros((128,), dtype=torch.uint8),
        "tune_out": torch.zeros((1,), dtype=torch.int32),
        "tune_status": torch.zeros((1,), dtype=torch.int32),
    }


def _patch_bundle(monkeypatch, execution):
    def fake_validate(*, export_graphs=False):
        handles = [100 + i for i in range(len(execution.graphs))]
        spans = [(0x800_0000_0000 + i * 0x10000, t.numel() * t.element_size())
                 for i, t in enumerate(execution.buffers.values())]
        return (0x1234, handles, spans)

    monkeypatch.setattr(execution, "_validate_resources", fake_validate)


def test_persistent_worker_dataclass_validation():
    g = _FakeGraph(1)
    pw = PersistentWorker(
        name="worker_launch",
        graph=g,
        worker_sm_count=8,
        stop_flag_buffer="worker_ctrl",
        stop_flag_offset=0,
    )
    assert pw.worker_sm_count == 8
    assert pw.stop_flag_buffer == "worker_ctrl"
    assert pw.stop_flag_offset == 0

    with pytest.raises(ValueError, match="nonempty graph name"):
        PersistentWorker(name=" ", graph=g, worker_sm_count=8, stop_flag_buffer="worker_ctrl")
    with pytest.raises(ValueError, match="positive uint32"):
        PersistentWorker(name="worker_launch", graph=g, worker_sm_count=0, stop_flag_buffer="worker_ctrl")
    with pytest.raises(ValueError, match="positive uint32"):
        PersistentWorker(name="worker_launch", graph=g, worker_sm_count=True, stop_flag_buffer="worker_ctrl")
    with pytest.raises(ValueError, match="positive uint32"):
        PersistentWorker(name="worker_launch", graph=g, worker_sm_count=1 << 32, stop_flag_buffer="worker_ctrl")
    with pytest.raises(ValueError, match="stop_flag_buffer"):
        PersistentWorker(name="worker_launch", graph=g, worker_sm_count=8, stop_flag_buffer=" ")
    with pytest.raises(ValueError, match="4-byte-aligned"):
        PersistentWorker(
            name="worker_launch", graph=g, worker_sm_count=8, stop_flag_buffer="worker_ctrl", stop_flag_offset=2
        )
    with pytest.raises(ValueError, match="4-byte-aligned"):
        PersistentWorker(
            name="worker_launch", graph=g, worker_sm_count=8, stop_flag_buffer="worker_ctrl", stop_flag_offset=-4
        )
    with pytest.raises(ValueError, match="4-byte-aligned"):
        PersistentWorker(
            name="worker_launch", graph=g, worker_sm_count=8, stop_flag_buffer="worker_ctrl", stop_flag_offset=False
        )


def test_prepared_execution_persistent_workers_manifest_and_validation(monkeypatch):
    decode_graph = _FakeGraph(10)
    tune_graph = _FakeGraph(15)
    worker_graph = _FakeGraph(20)
    sig = GraphSignature(kind="decode", batch_capacity=4, token_capacity=1)
    pw = PersistentWorker(
        name="worker_launch",
        graph=worker_graph,
        worker_sm_count=8,
        stop_flag_buffer="worker_ctrl",
        stop_flag_offset=16,
    )
    exe = PreparedExecution(
        device=0,
        graphs={"decode": decode_graph},
        signatures={"decode": sig},
        buffers=_base_buffers(),
        max_active=4,
        persistent_workers=(pw,),
    )
    _patch_bundle(monkeypatch, exe)
    assert "worker_launch" in exe.graphs
    assert "worker_launch" not in exe.serving_graphs

    live = exe.manifest()
    assert live["persistent_workers"] == [
        {
            "worker_sm_count": 8,
            "graph": "worker_launch",
            "handle": 101,
            "stop_flag": {"allocation": "worker_ctrl", "offset": 16},
        }
    ]

    exported = exe.export_manifest()
    assert exported["persistent_workers"] == [
        {
            "worker_sm_count": 8,
            "graph": "worker_launch",
            "stop_flag": {"allocation": "worker_ctrl", "offset": 16},
        }
    ]

    # Empty persistent_workers serializes as an empty list.
    empty_exe = PreparedExecution(
        device=0,
        graphs={"decode": decode_graph},
        signatures={"decode": sig},
        buffers=_base_buffers(),
        max_active=4,
    )
    _patch_bundle(monkeypatch, empty_exe)
    assert empty_exe.manifest()["persistent_workers"] == []
    assert empty_exe.export_manifest()["persistent_workers"] == []

    # PreparedModel forwards persistent_workers cleanly for export_core callers.
    model = PreparedModel(
        device=0,
        graphs={"decode": decode_graph},
        signatures={"decode": sig},
        buffers=_base_buffers(),
        max_active=4,
        persistent_workers=(pw,),
        server_config={},
        memory_report={},
        session=None,
    )
    _patch_bundle(monkeypatch, model)
    assert model.export_manifest()["persistent_workers"] == exported["persistent_workers"]

    # Cardinality > 1 rejected explicitly.
    pw_second = PersistentWorker(
        name="worker_launch_2",
        graph=_FakeGraph(21),
        worker_sm_count=8,
        stop_flag_buffer="worker_ctrl",
        stop_flag_offset=32,
    )
    with pytest.raises(ValueError, match="at most one persistent worker"):
        PreparedExecution(
            device=0,
            graphs={"decode": decode_graph},
            signatures={"decode": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_workers=(pw, pw_second),
        )

    # Bare PersistentWorker or wrong element type rejected.
    with pytest.raises(TypeError, match="sequence of PersistentWorker"):
        PreparedExecution(
            device=0,
            graphs={"decode": decode_graph},
            signatures={"decode": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_workers=pw,
        )
    with pytest.raises(TypeError, match="PersistentWorker objects"):
        PreparedExecution(
            device=0,
            graphs={"decode": decode_graph},
            signatures={"decode": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_workers=(object(),),
        )

    # Name collision with serving or tuning graph rejected.
    with pytest.raises(ValueError, match="distinct from serving and tuning"):
        PreparedExecution(
            device=0,
            graphs={"worker_launch": decode_graph},
            signatures={"worker_launch": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_workers=(pw,),
        )
    with pytest.raises(ValueError, match="distinct from serving and tuning"):
        PreparedExecution(
            device=0,
            graphs={"decode": decode_graph},
            signatures={"decode": sig},
            buffers=_base_buffers(),
            max_active=4,
            tuning_graphs={"worker_launch": TuningGraph(tune_graph, ("tune_out",), "tune_status")},
            persistent_workers=(pw,),
        )

    # Unregistered stop_flag_buffer rejected.
    bad_missing = PersistentWorker(
        name="worker_launch",
        graph=worker_graph,
        worker_sm_count=8,
        stop_flag_buffer="missing_ctrl",
        stop_flag_offset=0,
    )
    with pytest.raises(ValueError, match="registered execution buffer"):
        PreparedExecution(
            device=0,
            graphs={"decode": decode_graph},
            signatures={"decode": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_workers=(bad_missing,),
        )

    # Out-of-bounds stop_flag_offset rejected.
    bad_oob = PersistentWorker(
        name="worker_launch",
        graph=worker_graph,
        worker_sm_count=8,
        stop_flag_buffer="worker_ctrl",
        stop_flag_offset=128,
    )
    with pytest.raises(ValueError, match="exceeds its registered buffer"):
        PreparedExecution(
            device=0,
            graphs={"decode": decode_graph},
            signatures={"decode": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_workers=(bad_oob,),
        )

    # Host-upload buffer rejected for stop flag.
    bad_upload = PersistentWorker(
        name="worker_launch",
        graph=worker_graph,
        worker_sm_count=8,
        stop_flag_buffer="frame_index",
        stop_flag_offset=0,
    )
    with pytest.raises(ValueError, match="cannot receive host uploads"):
        PreparedExecution(
            device=0,
            graphs={"decode": decode_graph},
            signatures={"decode": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_workers=(bad_upload,),
        )


def test_prepared_execution_and_heliopause_reject_in_process_persistent_worker(monkeypatch):
    import flare_heliopause

    decode_graph = _FakeGraph(10)
    worker_graph = _FakeGraph(20)
    sig = GraphSignature(kind="decode", batch_capacity=4, token_capacity=1)
    pw = PersistentWorker(
        name="worker_launch",
        graph=worker_graph,
        worker_sm_count=8,
        stop_flag_buffer="worker_ctrl",
        stop_flag_offset=0,
    )
    exe = PreparedExecution(
        device=0,
        graphs={"decode": decode_graph},
        signatures={"decode": sig},
        buffers=_base_buffers(),
        max_active=4,
        persistent_workers=(pw,),
    )
    _patch_bundle(monkeypatch, exe)
    with pytest.raises(ValueError, match="green-context runtime control"):
        exe.start_executor()
    with pytest.raises(ValueError, match="green-context runtime control"):
        exe.serve_rank(SimpleNamespace(fileno=lambda: 0), rank=0, world=1)

    model = PreparedModel(
        device=0,
        graphs={"decode": decode_graph},
        signatures={"decode": sig},
        buffers=_base_buffers(),
        max_active=4,
        persistent_workers=(pw,),
        server_config={},
        memory_report={},
        session=None,
    )
    with pytest.raises(ValueError, match="green-context runtime control"):
        model.start_server()

    # Exercise the compiled flare_heliopause boundary directly.
    live_manifest = exe.manifest()
    with pytest.raises(ValueError, match="green-context runtime control"):
        flare_heliopause.GraphExecutor(object(), live_manifest)
    with pytest.raises(ValueError, match="green-context runtime control"):
        flare_heliopause.serve_tensor_parallel_rank(object(), live_manifest, 0, 0, 1)


def test_runtime_core_pb2_persistent_workers_round_trip(monkeypatch):
    decode_graph = _FakeGraph(10)
    worker_graph = _FakeGraph(20)
    sig = GraphSignature(kind="decode", batch_capacity=4, token_capacity=1)
    pw = PersistentWorker(
        name="worker_launch",
        graph=worker_graph,
        worker_sm_count=8,
        stop_flag_buffer="worker_ctrl",
        stop_flag_offset=64,
    )
    exe = PreparedExecution(
        device=0,
        graphs={"decode": decode_graph},
        signatures={"decode": sig},
        buffers=_base_buffers(),
        max_active=4,
        persistent_workers=(pw,),
    )
    _patch_bundle(monkeypatch, exe)
    exported_worker = exe.export_manifest()["persistent_workers"][0]

    pb_exec = runtime_core_pb2.Execution(
        abi_version=6,
        max_active=4,
        input_frame_bytes=16,
        persistent_workers=[
            runtime_core_pb2.PersistentWorker(
                worker_sm_count=exported_worker["worker_sm_count"],
                graph=exported_worker["graph"],
                stop_flag=runtime_core_pb2.Reference(
                    allocation="host-0",
                    offset=exported_worker["stop_flag"]["offset"],
                ),
            )
        ],
    )
    decoded = runtime_core_pb2.Execution()
    decoded.ParseFromString(pb_exec.SerializeToString())
    assert len(decoded.persistent_workers) == 1
    assert decoded.persistent_workers[0].worker_sm_count == 8
    assert decoded.persistent_workers[0].graph == "worker_launch"
    assert decoded.persistent_workers[0].stop_flag.allocation == "host-0"
    assert decoded.persistent_workers[0].stop_flag.offset == 64


def test_export_metadata_retains_mapped_host_planes_via_descriptor_scan(monkeypatch):
    """Pointer-bearing descriptor tensors stay non-numeric so mapped-host planes remain reachable."""
    import ignition.preparation.tensor_metadata as tensor_metadata

    host_planes = [torch.zeros((256,), dtype=torch.uint8) for _ in range(4)]
    worker_control = torch.zeros((128,), dtype=torch.uint8)
    scalar_table = torch.zeros((64,), dtype=torch.int32)
    worker_descs = torch.zeros((4,), dtype=torch.int64)
    control_table = torch.zeros((6,), dtype=torch.int64)

    for idx, plane in enumerate(host_planes):
        worker_descs[idx] = plane.data_ptr()
        control_table[1 + idx] = plane.data_ptr()

    with NumericStorage() as numeric_storage:
        for plane in host_planes:
            record_numeric_tensor(plane)
        record_numeric_tensor(scalar_table)
        # Descriptor tables carry raw pointers to mapped-host planes and are intentionally not recorded numeric.

    host_allocs = [
        Allocation(f"host-{idx}", plane.data_ptr(), plane.nbytes, alignment=4096,
                   preserve_address=True, memory="mapped_host")
        for idx, plane in enumerate(host_planes)
    ]
    host_allocs.append(
        Allocation("host-control", worker_control.data_ptr(), worker_control.nbytes, alignment=4096,
                   preserve_address=True, memory="mapped_host")
    )
    device_tensors = {
        "worker_descs": worker_descs,
        "control_table": control_table,
        "scalar_table": scalar_table,
    }
    device_allocs = [
        Allocation(name, tensor.data_ptr(), tensor.nbytes, alignment=256, preserve_address=True)
        for name, tensor in device_tensors.items()
    ]
    all_allocs = [*host_allocs, *device_allocs]
    tracked_objects = [*device_tensors.values(), *host_planes, worker_control]

    monkeypatch.setattr(tensor_metadata.gc, "get_objects", lambda: tracked_objects)
    monkeypatch.setattr(torch.Tensor, "device", property(lambda _: SimpleNamespace(type="cuda")))
    numeric_ranges, _ = live_metadata(all_allocs, numeric_storage, {"worker_control": worker_control})
    monkeypatch.undo()

    def scan_reached_mapped_host(tensor, excluded_ranges):
        base = tensor.data_ptr()
        raw = bytes(tensor.view(torch.uint8).reshape(-1).tolist())
        reached = set()
        for span_start, span_end in subtract_ranges([(base, base + len(raw))], excluded_ranges):
            chunk = raw[span_start - base:span_end - base]
            for off in range(0, max(0, len(chunk) - 7)):
                ptr = struct.unpack_from("<Q", chunk, off)[0]
                for alloc in host_allocs:
                    if alloc.address <= ptr < alloc.address + alloc.nbytes:
                        reached.add(alloc.name)
        return reached

    expected_planes = {f"host-{idx}" for idx in range(4)}
    assert expected_planes <= scan_reached_mapped_host(worker_descs, numeric_ranges)
    assert expected_planes <= scan_reached_mapped_host(control_table, numeric_ranges)

    # Numeric leaves and execution I/O buffers remain classified as numeric.
    for tensor in (*host_planes, scalar_table, worker_control):
        start, end = tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes
        assert any(lo <= start and end <= hi for lo, hi in numeric_ranges)

    # Classifying pointer-bearing descriptor tables as numeric drops every mapped-host plane.
    for desc in (worker_descs, control_table):
        poisoned = merge_ranges(numeric_ranges + [(desc.data_ptr(), desc.data_ptr() + desc.nbytes)])
        assert not (expected_planes & scan_reached_mapped_host(desc, poisoned))
