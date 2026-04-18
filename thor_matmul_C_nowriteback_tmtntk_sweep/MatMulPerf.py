import argparse
import cuda.tile as ct
import torch
from math import ceil

import NoWriteBackSwizzleM
import NoWriteBackSwizzleN
from L2CacheCtrl import (set_l2_policy_persisting, set_persisting_l2_cache_size,
                         reset_l2_policy, reset_persisting_l2_cache)


def cutile_matmul_no_writeback(A: torch.Tensor, B: torch.Tensor, matmul_kernel,
                               l2_persist_input: str = None,
                               tile_m: int = None,
                               tile_n: int = None,
                               tile_k: int = None) -> torch.Tensor:
    """
    Performs matrix multiplication A @ B using a cuTile kernel, but does not
    write the full C matrix back to global memory.

    Each block reduces its accumulator tile to one fp32 scalar checksum and
    stores that checksum into a 1-D output buffer. This keeps the MMA work alive
    while minimizing writeback traffic.
    """
    # --- Input Validation ---
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible matrices: K dimension of A ({A.shape[1]}) "
                         f"must match K dimension of B ({B.shape[0]})")
    if A.device != B.device:
        raise ValueError("Input tensors must be on the same device.")
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("Input tensors must be on a CUDA device.")

    # --- Determine Tile Shapes based on Data Type for Optimization ---
    if A.dtype.itemsize == 2:  # Likely torch.float16 or torch.bfloat16
        tm, tn, tk = 256, 256, 64
        if tile_m is not None:
            tm = tile_m
        if tile_n is not None:
            tn = tile_n
        if tile_k is not None:
            tk = tile_k
    else:  # Likely torch.float32 or other
        tm, tn, tk = 32, 32, 32

    # --- Get Matrix Dimensions ---
    m, _ = A.shape
    _, n = B.shape

    # --- Calculate Grid Dimensions for Kernel Launch (1D Grid) ---
    grid_x = ceil(m / tm)
    grid_y = ceil(n / tn)
    grid_size = grid_x * grid_y
    grid = (grid_size, 1, 1)

    # --- Create Output Tensor: one checksum per block ---
    checksum = torch.empty((grid_size,), device=A.device, dtype=torch.float32)

    # --- L2 Cache Policy: mark input tensor as persisting to reduce read amplification ---
    if l2_persist_input is not None:
        persist_tensor = B if l2_persist_input == 'B' else A
        persist_bytes = persist_tensor.nelement() * persist_tensor.element_size()
        set_persisting_l2_cache_size(persist_bytes)
        set_l2_policy_persisting(persist_tensor)

    # --- Launch the cuTile Kernel ---
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel, (A, B, checksum, tm, tn, tk))

    # --- Reset L2 policy after launch to avoid affecting subsequent operations ---
    if l2_persist_input is not None:
        reset_l2_policy()
        reset_persisting_l2_cache()
        set_persisting_l2_cache_size(0)

    return checksum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the checksum correctness of the results",
    )
    parser.add_argument(
        "--l2-persist-input",
        action="store_true",
        help="Enable L2 persisting policy for input tensors to reduce read amplification. "
             "Persists B for M-swizzle kernel, A for N-swizzle kernel.",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=256,
        help="M dimension of the matrix (rows of A and C)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=256,
        help="N dimension of the matrix (columns of B and C)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=768,
        help="K dimension of the matrix (columns of A, rows of B)",
    )
    parser.add_argument(
        "--tile-m",
        type=int,
        choices=[64, 128, 256, 512],
        default=256,
        help="Override tm for fp16/bf16 experiments",
    )
    parser.add_argument(
        "--tile-n",
        type=int,
        choices=[64, 128, 256, 512],
        default=256,
        help="Override tn for fp16/bf16 experiments",
    )
    parser.add_argument(
        "--tile-k",
        type=int,
        choices=[64],
        default=64,
        help="Fixed tk for fp16/bf16 experiments",
    )
    args = parser.parse_args()

    # --- Running no-writeback cuTile Matrix Multiplication Examples ---
    print("--- Running no-writeback cuTile Matrix Multiplication Examples (2D Grid) ---")
    if args.l2_persist_input:
        print("*** L2 persist input enabled: input tensors will use persisting L2 policy ***")
    print(f"*** Tile config for fp16/bf16: tm={args.tile_m}, tn={args.tile_n}, tk={args.tile_k} ***")
    print("*** No-writeback mode: full C is not stored; each block writes one fp32 checksum ***")

    # Define common matrix dimensions for the examples
    M_dim = args.M
    N_dim = args.N
    K_dim = args.K

    # --- Test Case 1: Swizzle M ---
    print("\n--- Test Case 1: Matrix Multiplication with float16 (Half-Precision) swizzle M ---")
    A_m_fp16 = torch.randn(M_dim, K_dim, dtype=torch.float16, device='cuda')
    B_m_fp16 = torch.randn(K_dim, N_dim, dtype=torch.float16, device='cuda')
    print(f"Input A shape: {A_m_fp16.shape}, dtype: {A_m_fp16.dtype}")
    print(f"Input B shape: {B_m_fp16.shape}, dtype: {B_m_fp16.dtype}")

    checksum_m_fp16 = cutile_matmul_no_writeback(
        A_m_fp16, B_m_fp16,
        matmul_kernel=NoWriteBackSwizzleM.matmul_kernel_m_no_writeback,
        l2_persist_input='B' if args.l2_persist_input else None,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k)
    print(f"cuTile checksum buffer shape: {checksum_m_fp16.shape}, dtype: {checksum_m_fp16.dtype}")
    if args.correctness_check:
        checksum_ref_m = (A_m_fp16 @ B_m_fp16).float().sum()
        torch.testing.assert_close(
            checksum_m_fp16.sum(),
            checksum_ref_m,
            rtol=1e-2,
            atol=1e-1,
        )
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 2: Swizzle N ---
    print("\n--- Test Case 2: Matrix Multiplication with float16 (Half-Precision) swizzle N ---")
    A_n_fp16 = torch.randn(M_dim, K_dim, dtype=torch.float16, device='cuda')
    B_n_fp16 = torch.randn(K_dim, N_dim, dtype=torch.float16, device='cuda')
    print(f"Input A shape: {A_n_fp16.shape}, dtype: {A_n_fp16.dtype}")
    print(f"Input B shape: {B_n_fp16.shape}, dtype: {B_n_fp16.dtype}")

    checksum_n_fp16 = cutile_matmul_no_writeback(
        A_n_fp16, B_n_fp16,
        matmul_kernel=NoWriteBackSwizzleN.matmul_kernel_n_no_writeback,
        l2_persist_input='A' if args.l2_persist_input else None,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k)
    print(f"cuTile checksum buffer shape: {checksum_n_fp16.shape}, dtype: {checksum_n_fp16.dtype}")
    if args.correctness_check:
        checksum_ref_n = (A_n_fp16 @ B_n_fp16).float().sum()
        torch.testing.assert_close(
            checksum_n_fp16.sum(),
            checksum_ref_n,
            rtol=1e-2,
            atol=1e-1,
        )
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- All no-writeback cuTile matrix multiplication perf test done. ---")
