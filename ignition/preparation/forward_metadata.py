# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Serialize prepared Python forwards without tensor data or CUDA resources."""

import ctypes
import enum
import importlib
import importlib.util
import inspect
import io
import pickle
import socket
import sys
import types
from contextvars import ContextVar
from importlib.machinery import ExtensionFileLoader
from threading import Lock, RLock, local

import torch
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.utils.weak import WeakIdKeyDictionary

_MODE = ContextVar('flare_forward_metadata_mode')
_TENSOR_KINDS = ContextVar('flare_forward_tensor_kinds', default=None)
_LOCK_TYPES = (type(Lock()), type(RLock()))
_FAKE_TENSOR_ATTRS = (
    frozenset(vars(FakeTensor(FakeTensorMode(), torch.empty(0, device='meta'), torch.device('cpu'))))
    | frozenset(name for cls in FakeTensor.__mro__ for name in getattr(cls, '__annotations__', ()))
    | {'_is_param', '_debug_context', '_debug_trace', 'item_memo', 'nested_int_memo', 'fake_device'}
)
_NON_OBJECT_NEW_TYPES = (
    torch.Tensor, torch.nn.Module, enum.Enum,
    int, float, str, bytes, dict, list, tuple, set, frozenset,
)


def _uses_custom_object_state(value):
    return (
        hasattr(value, '__dict__')
        and type(value).__module__.startswith(('ignition.', 'b12x.'))
        and not inspect.isclass(value)
        and not isinstance(value, _NON_OBJECT_NEW_TYPES)
    )


def _tensor_user_attributes(tensor):
    if not hasattr(tensor, '__dict__') or not vars(tensor):
        return {}
    if isinstance(tensor, FakeTensor):
        return {key: value for key, value in vars(tensor).items() if key not in _FAKE_TENSOR_ATTRS}
    return dict(vars(tensor))


def _is_importable_module(name, parent_name=''):
    search_path = None
    if parent_name:
        parent = sys.modules.get(parent_name)
        search_path = getattr(parent, '__path__', None)
        if search_path is None:
            try:
                parent_spec = importlib.util.find_spec(parent_name)
            except (ImportError, ValueError):
                return False
            search_path = None if parent_spec is None else parent_spec.submodule_search_locations
        if search_path is None:
            return False
    for finder in sys.meta_path:
        find_spec = getattr(finder, 'find_spec', None)
        if find_spec is not None and find_spec(name, search_path) is not None:
            return True
    return False


def _op_registration(overload):
    from torch._library.custom_ops import _maybe_get_opdef

    definition = _maybe_get_opdef(overload)
    if definition is None:
        return None, False
    init_fn = definition._init_fn
    if '<locals>' in getattr(init_fn, '__qualname__', ''):
        return None, True
    module_name = getattr(init_fn, '__module__', None)
    if module_name is None:
        return None, False
    module = sys.modules.get(module_name)
    spec = None if module is None else getattr(module, '__spec__', None)
    if spec is not None and isinstance(spec.loader, ExtensionFileLoader):
        if not _is_importable_module(module.__name__, spec.parent):
            return None, True
    return module_name, False


def _restore_torch_op(namespace, opname, overload, registration):
    try:
        if registration is not None:
            importlib.import_module(registration)
        packet = getattr(getattr(torch.ops, namespace), opname)
        return getattr(packet, overload)
    except Exception:
        return _Unavailable(f'Torch operator binding {namespace}.{opname}.{overload}')


def _restore_torch_op_packet(namespace, opname, registration):
    try:
        if registration is not None:
            importlib.import_module(registration)
        return getattr(getattr(torch.ops, namespace), opname)
    except Exception:
        return _Unavailable(f'Torch operator binding {namespace}.{opname}')


def _restore_torch_op_namespace(name):
    prefix = f'{name}::'
    if name == 'higher_order' or any(op.startswith(prefix) for op in torch._C._dispatch_get_all_op_names()):
        return getattr(torch.ops, name)
    return _Unavailable(f'Torch operator binding {name}')


class _Unavailable:
    def __init__(self, description):
        self.description = description

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f'forward tracing called process-local resource {self.description}')

    def __getattr__(self, name):
        # Unpickling probes special methods such as __setstate__ before restoring the description.
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        description = self.__dict__.get('description', 'unknown')
        raise RuntimeError(f'forward tracing accessed process-local resource {description}.{name}')


def _storage(nbytes):
    return torch.UntypedStorage(nbytes, device='meta')


class _Storage:
    def __init__(self, nbytes):
        self.nbytes = nbytes

    def __reduce__(self):
        return _storage, (self.nbytes,)


def _tensor(storage, shape, stride, offset, dtype, device, requires_grad, kind):
    with torch._C._DisableTorchDispatch():
        meta = torch.empty(0, dtype=dtype, device='meta').set_(storage, offset, shape, stride)
        meta.requires_grad_(requires_grad)
    value = FakeTensor(_MODE.get(), meta, device)
    if kind is torch.nn.Parameter:
        value._is_param = True
    _TENSOR_KINDS.get()[value] = kind
    return value


def _object(cls):
    return object.__new__(cls)


def _set_state(value, state):
    value.__dict__.update(state)


def _mapping_proxy(values):
    return types.MappingProxyType(values)


class _ForwardPickler(pickle.Pickler):
    def __init__(self, file):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self.storages = {}

    def reducer_override(self, value):
        from triton.runtime import JITFunction
        from triton.compiler import CompiledKernel
        from cutlass.base_dsl.jit_executor import JitCompiledFunction
        from cutlass.cute.runtime import _Tensor as CuTeRuntimeTensor
        from b12x._lib.compile_plan import CompiledCuTeProgram, DeferredCuTeKernel, DeferredTritonKernel
        from b12x.preparation import PreparationSession
        from b12x.preparation.device import DetectedDevice
        from b12x.preparation.types import _PreparedSlot, _Prepared
        from torch._dynamo.aot_compile_types import _deserialize_triton_kernel, _serialize_triton_kernel
        from .device_binding import restore_device, restore_prepared_device
        from .core_storage import CoreStorage
        from .peer_storage import PeerStorage

        if isinstance(value, PreparationSession):
            return _Unavailable, ("b12x preparation session",)
        if isinstance(value, _Unavailable):
            return _Unavailable, (value.description,)
        # A worker may trace another local rank's forward on its own GPU; the artifact relocates when bound.
        if isinstance(value, torch.device):
            return restore_device, (value.type, value.index)
        if isinstance(value, DetectedDevice):
            return restore_prepared_device, (type(value), dict(vars(value)))
        if isinstance(value, torch.Tensor):
            if type(value) not in (torch.Tensor, torch.nn.Parameter, FakeTensor):
                raise TypeError(f'unsupported forward tensor subclass: {type(value)}')
            storage = value.untyped_storage()
            owner = self.storages.setdefault(storage._cdata, _Storage(storage.nbytes()))
            args = (owner, tuple(value.shape), value.stride(), value.storage_offset(),
                    value.dtype, value.device, value.requires_grad, type(value))
            user_attrs = _tensor_user_attributes(value)
            if user_attrs:
                return _tensor, args, user_attrs, None, None, _set_state
            return _tensor, args
        if isinstance(value, _PreparedSlot):
            # Tracing needs selected layouts; normal plan pickling clears prepared state.
            return _object, (type(value),), dict(vars(value)), None, None, _set_state
        if isinstance(value, _Prepared):
            state = dict(vars(value))
            for name in ('retained', 'owners', 'closers', 'users'):
                state[name] = _Unavailable(f'b12x prepared {name}')
            return _object, (type(value),), state, None, None, _set_state
        if isinstance(value, JITFunction):
            return _deserialize_triton_kernel, (_serialize_triton_kernel(value),)
        if isinstance(value, types.MappingProxyType):
            return _mapping_proxy, (dict(value),)
        if isinstance(value, torch._ops.OpOverload):
            registration, local_only = _op_registration(value)
            if local_only:
                return _Unavailable, (f'Torch operator binding {value}',)
            return _restore_torch_op, (value.namespace, value._opname, value._overloadname, registration)
        if isinstance(value, torch._ops.OpOverloadPacket):
            qualified = getattr(value, '_qualified_op_name', '')
            if '::' not in qualified:
                return _Unavailable, (f'Torch operator binding {value}',)
            namespace, opname = qualified.split('::', 1)
            registration, local_only = None, False
            for overload_name in value.overloads():
                registration, local_only = _op_registration(getattr(value, overload_name))
                if registration is not None or local_only:
                    break
            if local_only:
                return _Unavailable, (f'Torch operator binding {value}',)
            return _restore_torch_op_packet, (namespace, opname, registration)
        if isinstance(value, torch._ops._OpNamespace):
            return _restore_torch_op_namespace, (value.name,)
        from tvm_ffi.core import CObject

        if isinstance(value, CObject) or (not isinstance(value, type) and '<locals>' in type(value).__qualname__):
            # FlashInfer's TVM-FFI kernel libraries and its locally defined kernel runners are process-local.
            return _Unavailable, (f'{type(value).__module__}.{type(value).__qualname__}',)
        if isinstance(value, (torch.cuda.Stream, torch.cuda.Event, torch.cuda.CUDAGraph, torch.cuda.MemPool,
                              torch.distributed.ProcessGroup, socket.socket, ctypes.CDLL, ctypes._Pointer,
                              ctypes._CFuncPtr, ctypes.Structure, ctypes.Array, ctypes._SimpleCData,
                              CompiledKernel, JitCompiledFunction, CompiledCuTeProgram, CuTeRuntimeTensor, local,
                              DeferredCuTeKernel, DeferredTritonKernel, PreparationSession, CoreStorage,
                              PeerStorage, *_LOCK_TYPES)):
            # Preparation sessions, allocators and peer storage retain process-local contexts, not forward
            # metadata; collective runtimes keep peer storage for teardown. CuTe runtime tensors describe
            # device pointers, and their __class__ is not their type, so default pickling rejects them.
            return _Unavailable, (f'{type(value).__module__}.{type(value).__qualname__}',)
        if not isinstance(value, type) and (hasattr(type(value), '__del__')
                or type(value).__module__.startswith('cutlass._mlir.')):
            return _Unavailable, (f'{type(value).__module__}.{type(value).__qualname__}',)
        if isinstance(value, types.BuiltinMethodType) and (
                isinstance(value.__self__, (*_LOCK_TYPES, local))
                or not value.__name__
                or value.__name__ == '<lambda>'):
            owner = type(value).__qualname__ if value.__self__ is None else type(value.__self__).__qualname__
            return _Unavailable, (f'{owner}.{value.__name__ or "<anonymous>"}',)
        if isinstance(value, types.FunctionType) and (
                '<locals>' in value.__qualname__ or value.__name__ == '<lambda>'):
            return _Unavailable, (value.__qualname__,)
        if isinstance(value, types.ModuleType):
            spec = value.__spec__
            if spec is not None and isinstance(spec.loader, ExtensionFileLoader):
                if not _is_importable_module(value.__name__, spec.parent):
                    # Preparation-built extensions are owned by the rank, not compiler workers.
                    return _Unavailable, (f'native extension {value.__name__}',)
            return importlib.import_module, (value.__name__,)
        if _uses_custom_object_state(value) and hasattr(type(value), '__getattr__'):
            return _object, (type(value),), dict(vars(value)), None, None, _set_state
        return NotImplemented


class _ForwardUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('transformers_modules.'):
            from transformers.dynamic_module_utils import init_hf_modules
            init_hf_modules()
        return super().find_class(module, name)


def dumps(forward):
    buffer = io.BytesIO()
    _ForwardPickler(buffer).dump(forward)
    return buffer.getvalue()


def loads(payload):
    token = _MODE.set(FakeTensorMode(allow_non_fake_inputs=True))
    _TENSOR_KINDS.set(WeakIdKeyDictionary())
    try:
        return _ForwardUnpickler(io.BytesIO(payload)).load()
    finally:
        _MODE.reset(token)
