# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct

ConstInt = ct.Constant[int]


def swizzle_2d_from_bid_m(M, N, tm, tn, group_size_m, bid):
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = group_size_m * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * group_size_m
    active_group_size_m = min(num_bid_m - first_bid_m, group_size_m)
    bid_m = first_bid_m + ((bid - group_id * num_bid_in_group) % active_group_size_m)
    bid_n = (bid % num_bid_in_group) // active_group_size_m
    return bid_m, bid_n


def swizzle_2d_m(M, N, tm, tn, group_size_m):
    return swizzle_2d_from_bid_m(M, N, tm, tn, group_size_m, ct.bid(0))


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel_m_no_writeback(
    A,
    B,
    checksum,
    tm: ConstInt,
    tn: ConstInt,
    tk: ConstInt,
):
    group_size_m = 8
    M = A.shape[0]
    N = B.shape[1]
    bid_m, bid_n = swizzle_2d_m(M, N, tm, tn, group_size_m)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    input_dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for tile_k in range(num_tiles_k):
        a_tile = ct.load(
            A, index=(bid_m, tile_k), shape=(tm, tk), padding_mode=zero_pad
        ).astype(input_dtype)
        b_tile = ct.load(
            B, index=(tile_k, bid_n), shape=(tk, tn), padding_mode=zero_pad
        ).astype(input_dtype)
        accumulator = ct.mma(a_tile, b_tile, accumulator)

    checksum_scalar = ct.sum(accumulator, axis=None)
    num_bid_n = ct.cdiv(N, tn)
    checksum_index = bid_m * num_bid_n + bid_n
    checksum_tile = ct.full((1,), checksum_scalar.item(), dtype=ct.float32)
    ct.store(checksum, index=(checksum_index,), tile=checksum_tile)
