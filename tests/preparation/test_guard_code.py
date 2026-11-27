# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist guard source while binding each process's closure values."""

import io
import pickle
import subprocess
import sys

import pytest
import torch
from torch._dynamo import guards
from torch._dynamo.guards import _Missing

from ignition.preparation.aot import _GuardPickler, _guard_code_cache, _guard_serialization


class GuardedForward:
    def __init__(self):
        self.value = torch.ones(2)
        self.offset = 2

    def __call__(self):
        return self.value + self.offset


def _execute(forward):
    return forward()


def test_loaded_guards_preserve_predicates_without_construction_stacks(tmp_path):
    from torch._dynamo import package
    from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable

    forward = GuardedForward()
    compiled = torch.compile(_execute,
        backend=lambda graph, inputs: GraphModuleSerializableCallable(graph),
        fullgraph=True, dynamic=False).aot_compile(((forward,), {}))
    data = type(compiled).serialize(compiled).serialized_data
    reference = type(compiled).deserialize(data)
    original = package.load_guard_manager
    with _guard_code_cache(tmp_path):
        loaded = type(compiled).deserialize(data)
    assert package.load_guard_manager is original
    torch.testing.assert_close(loaded(forward), reference(forward))
    for value, offset, valid in ((torch.ones(2), 2, True), (torch.ones(2), 3, False),
                                 (torch.ones(3), 2, False), (torch.ones(2, dtype=torch.float64), 2, False)):
        forward.value, forward.offset = value, offset
        assert loaded.guard_check(forward) is valid
        assert reference.guard_check(forward) is valid
        if not valid:
            reason = str(loaded._artifacts.guard_manager.check_verbose({"forward": forward}))
            assert "forward" in reason
    forward.value, forward.offset = torch.ones(2), 2
    torch.testing.assert_close(loaded(forward), reference(forward))


def test_cached_guard_source_binds_fresh_closures_in_another_process(tmp_path):
    parts = ["len(L['values']) == expected", "L['enabled'] == enabled"]
    expected = guards.build_guard_function(parts, 'expected, enabled')
    original = guards.build_guard_function
    with _guard_code_cache(tmp_path):
        assert guards.build_guard_function(parts, 'expected, enabled') == expected
    assert guards.build_guard_function is original

    subprocess.run([sys.executable, '-c', '''
import sys
from pathlib import Path
from torch._dynamo import guards
from ignition.preparation.aot import _guard_code_cache

def forbid_build(*args, **kwargs):
    raise AssertionError('cached guard source must not be regenerated')

guards.build_guard_function = forbid_build
with _guard_code_cache(Path(sys.argv[1])):
    _, source = guards.build_guard_function(
        ["len(L['values']) == expected", "L['enabled'] == enabled"], 'expected, enabled')
scope = {}
exec(source, scope)
for size, enabled in ((3, True), (1, False)):
    check = scope['___make_guard_fn'](size, enabled)
    assert check(dict(values=list(range(size)), enabled=enabled))
    assert not check(dict(values=list(range(size + 1)), enabled=enabled))
    assert not check(dict(values=list(range(size)), enabled=not enabled))
assert guards.build_guard_function is forbid_build
''', str(tmp_path)], check=True)


def _roundtrip_guard(value, *, guard_tree_values=None, empty_values=None, missing_values=None):
    buffer = io.BytesIO()
    pickler = _GuardPickler(
        {} if guard_tree_values is None else guard_tree_values,
        {} if empty_values is None else empty_values,
        {} if missing_values is None else missing_values,
        buffer,
    )
    pickler.dump(value)
    return pickle.loads(buffer.getvalue())


@pytest.mark.parametrize("use_fake", [False, True])
def test_guard_pickler_preserves_guarded_tensor_user_attributes_only(use_fake):
    from torch._subclasses import FakeTensorMode

    if use_fake:
        mode = FakeTensorMode(allow_non_fake_inputs=True)
        tensor = mode.from_tensor(torch.ones((4, 8), dtype=torch.float32))
        guarded_scale = mode.from_tensor(torch.ones((4, 1), dtype=torch.float32))
    else:
        tensor = torch.ones((4, 8), dtype=torch.float32)
        guarded_scale = torch.ones((4, 1), dtype=torch.float32)
    unguarded_owner = object()
    tensor.scale = guarded_scale
    tensor.unguarded = unguarded_owner

    restored = _roundtrip_guard(
        tensor,
        guard_tree_values={id(tensor): tensor, id(guarded_scale): guarded_scale},
    )
    assert tuple(restored.shape) == (4, 8)
    assert tuple(restored.scale.shape) == (4, 1)
    assert isinstance(restored.unguarded, _Missing)

    omitted = _roundtrip_guard(tensor, guard_tree_values={})
    assert isinstance(omitted, _Missing)


class _EmptySentinel:
    def __init__(self, tag):
        self.tag = tag


class _CustomGuardTensor(torch.Tensor):
    pass


class _IgnitionInt(int):
    __module__ = "ignition.test_builtin_subclass"

    def __getattr__(self, name):
        raise AttributeError(name)


def test_guard_pickler_honors_empty_and_missing_values_before_custom_overrides():
    device = torch.device("cpu")
    tensor = torch.ones(4)
    sentinel = _EmptySentinel("state")
    restored_missing = _roundtrip_guard(
        device,
        guard_tree_values={id(device): device},
        missing_values={id(device): device},
    )
    assert isinstance(restored_missing, _Missing)
    restored_missing_tensor = _roundtrip_guard(
        tensor,
        guard_tree_values={id(tensor): tensor},
        missing_values={id(tensor): tensor},
    )
    assert isinstance(restored_missing_tensor, _Missing)
    restored_empty = _roundtrip_guard(
        sentinel,
        guard_tree_values={id(sentinel): sentinel},
        empty_values={id(sentinel): sentinel},
    )
    assert type(restored_empty) is _EmptySentinel
    assert not hasattr(restored_empty, "tag")
    restored_empty_tensor = _roundtrip_guard(
        tensor,
        guard_tree_values={id(tensor): tensor},
        empty_values={id(tensor): tensor},
    )
    assert type(restored_empty_tensor) is torch.Tensor
    assert restored_empty_tensor.numel() == 0


def test_guard_pickler_delegates_sparse_and_custom_tensor_subclasses(monkeypatch):
    sparse = torch.sparse_coo_tensor([[0, 1], [1, 0]], [1.0, 2.0], (2, 2))
    custom = torch.ones(4).as_subclass(_CustomGuardTensor)
    quantized = torch.quantize_per_tensor(torch.ones(4), 0.1, 0, torch.quint8)

    delegated = []
    original_super = guards.GuardsStatePickler.reducer_override

    def spy_super(self, obj):
        if any(obj is target for target in (sparse, custom, quantized)):
            delegated.append(obj)
            if obj is quantized:
                return _Missing, ("delegated quantized",)
        return original_super(self, obj)

    monkeypatch.setattr(guards.GuardsStatePickler, "reducer_override", spy_super)
    restored_sparse = _roundtrip_guard(sparse, guard_tree_values={id(sparse): sparse})
    restored_custom = _roundtrip_guard(custom, guard_tree_values={id(custom): custom})
    restored_quantized = _roundtrip_guard(quantized, guard_tree_values={id(quantized): quantized})

    assert delegated == [sparse, custom, quantized]
    assert tuple(restored_sparse.shape) == (2, 2)
    assert restored_custom.pytype is _CustomGuardTensor
    assert isinstance(restored_quantized, _Missing)


def test_guard_pickler_preserves_ignition_enum_and_builtin_subclass_identity(monkeypatch):
    import types
    from ignition.preparation.frontend import PromptEncoder

    member = PromptEncoder.PROMPT_JINJA
    restored = _roundtrip_guard(member, guard_tree_values={id(member): member})
    assert restored is member

    module = types.ModuleType("ignition.test_builtin_subclass")
    module._IgnitionInt = _IgnitionInt
    monkeypatch.setitem(sys.modules, "ignition.test_builtin_subclass", module)
    value = _IgnitionInt(42)
    value.note = "keep"
    restored_int = _roundtrip_guard(value, guard_tree_values={id(value): value})
    assert type(restored_int) is _IgnitionInt
    assert int(restored_int) == 42


class _AnnotatedTensorForward:
    def __init__(self, scale_value=2.0):
        self.weight = torch.ones((2, 4), dtype=torch.float32)
        self.weight.scale = torch.full((2, 1), scale_value, dtype=torch.float32)
        self.weight.unused = lambda: None

    def __call__(self):
        return self.weight * self.weight.scale


def test_guard_serialization_roundtrips_tensor_user_attributes():
    from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable

    forward = _AnnotatedTensorForward(scale_value=3.0)
    with _guard_serialization():
        compiled = torch.compile(
            _execute,
            backend=lambda graph, inputs: GraphModuleSerializableCallable(graph),
            fullgraph=True,
            dynamic=False,
        ).aot_compile(((forward,), {}))
        data = type(compiled).serialize(compiled).serialized_data
    restored = type(compiled).deserialize(data)
    torch.testing.assert_close(restored(forward), torch.full((2, 4), 3.0))
    wrong_scale = _AnnotatedTensorForward(scale_value=3.0)
    wrong_scale.weight.scale = torch.ones((2, 2), dtype=torch.float32)
    assert restored.guard_check(forward)
    assert not restored.guard_check(wrong_scale)
