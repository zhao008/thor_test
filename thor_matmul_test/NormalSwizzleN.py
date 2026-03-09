# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
from math import ceil  # Required for host-side grid calculation


ConstInt = ct.Constant[int]

def swizzle_2d_from_bid_n(M, N, tm, tn, GROUP_SIZE_N, bid):
    # Get the global IDs of a given block in a 1D grid (swizzle in N direction).
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
    # Get the global IDs of the current block in a 1D grid (swizzle in N direction).
    bid = ct.bid(0)
    return swizzle_2d_from_bid_n(M, N, tm, tn, GROUP_SIZE_N, bid)


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel_n(A, B, C,
                  tm: ConstInt,         # Tile size along M dimension (rows of C)
                  tn: ConstInt,         # Tile size along N dimension (columns of C)
                  tk: ConstInt):        # Tile size along K dimension (inner product dimension)
    """
    cuTile kernel for performing matrix multiplication C = A @ B.

    This kernel uses a tiled approach, where each block
    computes a `tm` x `tn` tile of the output matrix C. The computation
    involves iterating over the K-dimension in chunks of `tk`.

    Args:
        A: Input matrix A (M x K).
        B: Input matrix B (K x N).
        C: Output matrix C (M x N).
        tm (ConstInt): The height of the output tile computed by this block.
                       Corresponds to rows of A and C.
        tn (ConstInt): The width of the output tile computed by this block.
                       Corresponds to columns of B and C.
        tk (ConstInt): The depth of the inner loop (K-dimension) tile size.
                       Corresponds to columns of A and rows of B.
    """
    GROUP_SIZE_N = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d_n(M, N, tm, tn, GROUP_SIZE_N)

    # Calculate the total number of tiles along the K-dimension that need to be processed.
    # `ct.num_tiles(A, axis=1, shape=(tm, tk))` means:
    #   "View A as an MxK tensor tiled by (tm, tk), and return the number of tiles along
    #    axis 1 (the K dimension)."
    # We pass shape=(tm, tk) to describe the 2D tiling, only `tk` matters for axis=1.
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # Initialize an accumulator for the current output tile (tm x tn).
    # It's common practice to use `float32` for accumulation even with `float16` inputs
    # to maintain higher precision during the sum-reduction of the matrix multiplication.
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # K-dimension loop: Iterate over the K-dimension in chunks of 'tk'.
    # In each iteration, a `tm` x `tk` tile from A and a `tk` x `tn` tile from B
    # are loaded, multiplied, and accumulated.
    for k in range(num_tiles_k):
        # Load tile from matrix A.
        # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
        # from global memory A. `shape=(tm, tk)` defines the size of this tile.
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)

        # Load tile from matrix B.
        # The `index=(k_tile_idx, bidy)` specifies which (K-tile, N-tile) to load
        # from global memory B. `shape=(tk, tn)` defines the size of this tile.
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)

        # Perform Matrix Multiplication for the current tiles.
        # `ct.mma` computes the product of the two loaded tiles and accumulates the result.
        accumulator = ct.mma(a, b, accumulator)

    # Convert the final accumulated result to the desired output data type (C.dtype).
    # This might downcast from float32 to float16 if the output is float16.
    accumulator = ct.astype(accumulator, C.dtype)

    # Store the computed tile to the global memory of the output matrix C.
    # The `(bidx, bidy)` directly corresponds to the tile's position in the 2D output matrix.
    ct.store(C, index=(bidx, bidy), tile=accumulator)