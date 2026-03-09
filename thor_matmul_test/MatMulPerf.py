import argparse
import cuda.tile as ct
import torch
from math import ceil
import NormalSwizzleM
import NormalSwizzleN
from L2CacheCtrl import (set_l2_policy_persisting, set_persisting_l2_cache_size,
                         reset_l2_policy, reset_persisting_l2_cache)

def cutile_matmul(A: torch.Tensor, B: torch.Tensor, matmul_kernel,
                  l2_persist_input: str = None) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B using a cuTile kernel with a 2D grid.

    This wrapper function handles input validation, determines appropriate
    tile sizes based on data type, calculates the necessary grid dimensions,
    and launches the `matmul_kernel`.

    Args:
        A (torch.Tensor): The first input matrix (M x K). Must be on a CUDA device.
        B (torch.Tensor): The second input matrix (K x N). Must be on a CUDA device
                          and have its K dimension match A's K dimension.
        matmul_kernel: The cuTile kernel function to use for matrix multiplication.
        l2_persist_input (str): Which input tensor to mark as 'persisting' in L2.
                                - 'B': Persist B — best for M-swizzle kernels where
                                  groups of blocks share B column tiles.
                                - 'A': Persist A — best for N-swizzle kernels where
                                  groups of blocks share A row tiles.
                                - None: No L2 policy control (default).

    Returns:
        torch.Tensor: The resulting matrix C (M x N) on the CUDA device.

    Raises:
        ValueError: If matrices are incompatible (K dimensions don't match),
                    or if they are not on a CUDA device.
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
        tm, tn, tk = 256, 256, 64  # Balanced tiles for better memory efficiency
    else:  # Likely torch.float32 or other
        tm, tn, tk = 32, 32, 32   # Smaller, more general tiles

    # --- Get Matrix Dimensions ---
    m, k_a = A.shape
    k_b, n = B.shape

    # --- Calculate Grid Dimensions for Kernel Launch (1D Grid) ---
    grid_x = ceil(m / tm)
    grid_y = ceil(n / tn)
    grid_size = grid_x * grid_y
    grid = (grid_size, 1, 1)

    # --- Create Output Tensor C ---
    C = torch.empty((m, n), device=A.device, dtype=A.dtype)

    # --- L2 Cache Policy: mark input tensor as persisting to reduce read amplification ---
    if l2_persist_input is not None:
        persist_tensor = B if l2_persist_input == 'B' else A
        persist_bytes = persist_tensor.nelement() * persist_tensor.element_size()
        set_persisting_l2_cache_size(persist_bytes)
        set_l2_policy_persisting(persist_tensor)

    # --- Launch the cuTile Kernel ---
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel, (A, B, C, tm, tn, tk))

    # --- Reset L2 policy after launch to avoid affecting subsequent operations ---
    if l2_persist_input is not None:
        reset_l2_policy()
        reset_persisting_l2_cache()
        set_persisting_l2_cache_size(0)

    return C

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
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
    args = parser.parse_args()

    # --- Running cuTile Matrix Multiplication Examples ---
    print("--- Running cuTile Matrix Multiplication Examples (2D Grid) ---")
    if args.l2_persist_input:
        print("*** L2 persist input enabled: input tensors will use persisting L2 policy ***")

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

    C_m_fp16_cutile = cutile_matmul(A_m_fp16, B_m_fp16,
                                     matmul_kernel=NormalSwizzleM.matmul_kernel_m,
                                     l2_persist_input='B' if args.l2_persist_input else None)
    print(f"cuTile Output C shape: {C_m_fp16_cutile.shape}, dtype: {C_m_fp16_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(C_m_fp16_cutile, A_m_fp16 @ B_m_fp16)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")


    # --- Test Case 2: Swizzle N ---
    print("\n--- Test Case 2: Matrix Multiplication with float16 (Half-Precision) swizzle N ---")
    A_n_fp16 = torch.randn(M_dim, K_dim, dtype=torch.float16, device='cuda')
    B_n_fp16 = torch.randn(K_dim, N_dim, dtype=torch.float16, device='cuda')
    print(f"Input A shape: {A_n_fp16.shape}, dtype: {A_n_fp16.dtype}")
    print(f"Input B shape: {B_n_fp16.shape}, dtype: {B_n_fp16.dtype}")

    C_n_fp16_cutile = cutile_matmul(A_n_fp16, B_n_fp16,
                                     matmul_kernel=NormalSwizzleN.matmul_kernel_n,
                                     l2_persist_input='A' if args.l2_persist_input else None)
    print(f"cuTile Output C shape: {C_n_fp16_cutile.shape}, dtype: {C_n_fp16_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(C_n_fp16_cutile, A_n_fp16 @ B_n_fp16)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")


    print("\n--- All cuTile matrix multiplication perf test done. ---")
