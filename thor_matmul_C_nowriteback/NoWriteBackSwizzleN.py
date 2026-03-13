# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
No-WriteBack variant of NormalSwizzleN kernel.

This kernel performs the FULL matrix multiplication computation (load A, load B,
MMA accumulate) but does NOT write the full C tile back to global memory.

To prevent the compiler / cuTile DCE from optimizing away the computation,
we reduce the accumulator tile to a single scalar checksum (ct.sum) and store
only that one value per block into a small 1-D checksum buffer.

This lets us measure the pure compute + read cost WITHOUT the C write-back cost.
"""

import cuda.tile as ct

ConstInt = ct.Constant[int]


def swizzle_2d_from_bid_n(M, N, tm, tn, GROUP_SIZE_N, bid):
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_N * num_bid_m
    group_id = bid // num_bid_in_group
    first_bid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_bid_n - first_bid_n, GROUP_SIZE_N)
    bid_n = first_bid_n + ((bid - group_id * num_bid_in_group) % group_size_n)
    bid_m = (bid % num_bid_in_group) // group_size_n
    return bid_m, bid_n


def swizzle_2d_n(M, N, tm, tn, GROUP_SIZE_N):
    bid = ct.bid(0)
    return swizzle_2d_from_bid_n(M, N, tm, tn, GROUP_SIZE_N, bid)


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel_n_no_writeback(A, B, checksum,
                                  tm: ConstInt,
                                  tn: ConstInt,
                                  tk: ConstInt):
    """
    No-writeback matmul kernel (N-swizzle variant).

    Performs the full C = A @ B tiled computation but instead of writing
    the tm×tn accumulator tile back to global memory, it:
      1. Reduces the accumulator to a single scalar via ct.sum (axis=None).
      2. Stores that scalar into checksum[block_linear_id].

    This prevents the compiler from eliminating the computation (the sum
    result is consumed by ct.store), while making the global memory write
    negligible (1 element per block instead of tm×tn elements).

    Args:
        A: Input matrix A (M x K).
        B: Input matrix B (K x N).
        checksum: 1-D buffer of shape (num_blocks,) in float32.
                  Each block writes one scalar checksum to prevent DCE.
        tm, tn, tk: Tile sizes along M, N, K dimensions.
    """
    GROUP_SIZE_N = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d_n(M, N, tm, tn, GROUP_SIZE_N)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # Accumulator in fp32 for precision
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # ---- Full MMA computation (same as normal kernel) ----
    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)

    # ---- Anti-DCE: reduce to scalar checksum and store ----
    tile_checksum = ct.sum(accumulator, axis=None)

    num_bid_n = ct.cdiv(N, tn)
    block_linear_id = bidx * num_bid_n + bidy

    tile_checksum_1d = ct.full((1,), tile_checksum.item(), dtype=ct.float32)
    ct.store(checksum, index=(block_linear_id,), tile=tile_checksum_1d)
