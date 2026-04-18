# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
L2 Cache Access Policy Control for CUDA tensors.

Uses the CUDA L2 Access Policy Window API (compute capability 8.0+) to control
how memory accesses interact with L2 cache. This is useful for matmul where:
- Reads from A/B benefit from L2 caching (reused across iterations)
- Writes to C are one-shot and should NOT pollute L2

By marking C's memory as "streaming", writes evict from L2 quickly,
leaving more L2 space for A/B read data.
"""

import ctypes
import torch

# CUDA enum values
_cudaStreamAttributeAccessPolicyWindow = 1
_cudaAccessPropertyNormal = 0
_cudaAccessPropertyStreaming = 1
_cudaAccessPropertyPersisting = 2

# Cache the CUDA runtime library handle
_cuda_rt = None


def _get_cuda_runtime():
    global _cuda_rt
    if _cuda_rt is not None:
        return _cuda_rt
    for lib_name in ('libcudart.so', 'libcudart.so.13', 'libcudart.so.12'):
        try:
            _cuda_rt = ctypes.cdll.LoadLibrary(lib_name)
            return _cuda_rt
        except OSError:
            continue
    raise RuntimeError("Cannot find libcudart.so. Ensure CUDA toolkit is installed.")


class _cudaAccessPolicyWindow(ctypes.Structure):
    _fields_ = [
        ("base_ptr", ctypes.c_void_p),
        ("num_bytes", ctypes.c_size_t),
        ("hitRatio", ctypes.c_float),
        ("hitProp", ctypes.c_int),
        ("missProp", ctypes.c_int),
    ]


class _cudaStreamAttrValue(ctypes.Union):
    _fields_ = [
        ("accessPolicyWindow", _cudaAccessPolicyWindow),
        # Padding to ensure the union is large enough (cudaStreamAttrValue is 64 bytes).
        ("_pad", ctypes.c_char * 64),
    ]


def set_l2_policy_streaming(tensor: torch.Tensor, stream: torch.cuda.Stream = None):
    """
    Mark a tensor's memory region as 'streaming' in L2 cache.

    Accesses to this memory region will be preferentially evicted from L2 cache,
    reducing L2 pollution. Ideal for write-only tensors like output matrix C in matmul.

    Args:
        tensor: A CUDA tensor whose memory should use streaming L2 policy.
        stream: CUDA stream to apply the policy to. Defaults to current stream.
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on a CUDA device")

    cuda_rt = _get_cuda_runtime()
    if stream is None:
        stream = torch.cuda.current_stream()

    attr_val = _cudaStreamAttrValue()
    attr_val.accessPolicyWindow.base_ptr = tensor.data_ptr()
    attr_val.accessPolicyWindow.num_bytes = tensor.nelement() * tensor.element_size()
    attr_val.accessPolicyWindow.hitRatio = 1.0
    attr_val.accessPolicyWindow.hitProp = _cudaAccessPropertyStreaming
    attr_val.accessPolicyWindow.missProp = _cudaAccessPropertyStreaming

    err = cuda_rt.cudaStreamSetAttribute(
        ctypes.c_void_p(stream.cuda_stream),
        ctypes.c_int(_cudaStreamAttributeAccessPolicyWindow),
        ctypes.byref(attr_val),
    )
    if err != 0:
        raise RuntimeError(f"cudaStreamSetAttribute failed with CUDA error {err}")


def set_l2_policy_persisting(tensor: torch.Tensor, hit_ratio: float = 1.0,
                             stream: torch.cuda.Stream = None):
    """
    Mark a tensor's memory region as 'persisting' in L2 cache.

    Accesses to this memory region will be preferentially retained in L2 cache.
    Ideal for read-heavy tensors like input matrices A, B in matmul.

    Note: Only one access policy window can be active per stream at a time.

    Args:
        tensor: A CUDA tensor whose memory should use persisting L2 policy.
        hit_ratio: Fraction of accesses that get the persisting property (0.0-1.0).
        stream: CUDA stream to apply the policy to. Defaults to current stream.
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on a CUDA device")

    cuda_rt = _get_cuda_runtime()
    if stream is None:
        stream = torch.cuda.current_stream()

    attr_val = _cudaStreamAttrValue()
    attr_val.accessPolicyWindow.base_ptr = tensor.data_ptr()
    attr_val.accessPolicyWindow.num_bytes = tensor.nelement() * tensor.element_size()
    attr_val.accessPolicyWindow.hitRatio = hit_ratio
    attr_val.accessPolicyWindow.hitProp = _cudaAccessPropertyPersisting
    attr_val.accessPolicyWindow.missProp = _cudaAccessPropertyStreaming

    err = cuda_rt.cudaStreamSetAttribute(
        ctypes.c_void_p(stream.cuda_stream),
        ctypes.c_int(_cudaStreamAttributeAccessPolicyWindow),
        ctypes.byref(attr_val),
    )
    if err != 0:
        raise RuntimeError(f"cudaStreamSetAttribute failed with CUDA error {err}")


def reset_l2_policy(stream: torch.cuda.Stream = None):
    """
    Reset the L2 access policy window on a stream to default (no policy).

    Should be called after the kernel completes, to avoid affecting subsequent
    operations on the same stream.

    Args:
        stream: CUDA stream to reset. Defaults to current stream.
    """
    cuda_rt = _get_cuda_runtime()
    if stream is None:
        stream = torch.cuda.current_stream()

    attr_val = _cudaStreamAttrValue()
    attr_val.accessPolicyWindow.base_ptr = 0
    attr_val.accessPolicyWindow.num_bytes = 0
    attr_val.accessPolicyWindow.hitRatio = 0.0
    attr_val.accessPolicyWindow.hitProp = _cudaAccessPropertyNormal
    attr_val.accessPolicyWindow.missProp = _cudaAccessPropertyNormal

    err = cuda_rt.cudaStreamSetAttribute(
        ctypes.c_void_p(stream.cuda_stream),
        ctypes.c_int(_cudaStreamAttributeAccessPolicyWindow),
        ctypes.byref(attr_val),
    )
    if err != 0:
        raise RuntimeError(f"cudaStreamSetAttribute failed with CUDA error {err}")


def get_max_persisting_l2_cache_size() -> int:
    """
    Query the maximum L2 cache size that can be reserved for persisting data.

    Returns:
        Maximum persisting L2 cache size in bytes.
    """
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.persistingL2CacheMaxSize


def set_persisting_l2_cache_size(size_bytes: int):
    """
    Reserve a portion of L2 cache for persisting data.

    The requested size is automatically clamped to the device's maximum
    allowed persisting L2 cache size.

    Args:
        size_bytes: Number of bytes of L2 cache to reserve for persisting data.
                    Pass 0 to reset to default behavior.
    """
    if size_bytes > 0:
        max_size = get_max_persisting_l2_cache_size()
        if max_size == 0:
            print("Warning: Device does not support L2 persisting cache. Skipping.")
            return
        size_bytes = min(size_bytes, max_size)
        print(f"Setting persisting L2 cache size to {size_bytes} bytes "
              f"(max: {max_size} bytes)")

    cuda_rt = _get_cuda_runtime()
    _cudaLimitPersistingL2CacheSize = 0x06
    err = cuda_rt.cudaDeviceSetLimit(
        ctypes.c_int(_cudaLimitPersistingL2CacheSize),
        ctypes.c_size_t(size_bytes),
    )
    if err != 0:
        raise RuntimeError(f"cudaDeviceSetLimit failed with CUDA error {err}")


def reset_persisting_l2_cache():
    """
    Reset all persisting L2 cache lines to normal.

    Forces all persisting lines to become eligible for normal eviction.
    Useful when transitioning between kernels with different access patterns.
    """
    cuda_rt = _get_cuda_runtime()
    err = cuda_rt.cudaCtxResetPersistingL2Cache()
    if err != 0:
        raise RuntimeError(f"cudaCtxResetPersistingL2Cache failed with CUDA error {err}")
