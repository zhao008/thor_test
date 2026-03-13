import argparse
import cuda.tile as ct
import torch
from math import ceil
import NoWriteBackSwizzleM
import NoWriteBackSwizzleN


def cutile_matmul_no_writeback(A: torch.Tensor, B: torch.Tensor,
                                matmul_kernel) -> tuple:
    """
    Performs matrix multiplication C = A @ B using a cuTile kernel, but does NOT
    write back the full C result. Instead, each block reduces its accumulator tile
    to a single scalar checksum (via ct.sum) and stores only that value, preventing
    the compiler / cuTile DCE from optimizing away the computation while keeping
    the global memory write negligible.

    This is used to measure the pure compute + read cost WITHOUT C write-back overhead.

    Args:
        A (torch.Tensor): The first input matrix (M x K). Must be on a CUDA device.
        B (torch.Tensor): The second input matrix (K x N). Must be on a CUDA device.
        matmul_kernel: The no-writeback cuTile kernel function.

    Returns:
        tuple: (checksum,) where checksum is a 1-D float32 tensor containing
               one scalar per block — the sum of each block's accumulator tile.
    """
    # --- Input Validation ---
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible matrices: K dimension of A ({A.shape[1]}) "
                         f"must match K dimension of B ({B.shape[0]})")
    if A.device != B.device:
        raise ValueError("Input tensors must be on the same device.")
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("Input tensors must be on a CUDA device.")

    # --- Determine Tile Shapes based on Data Type ---
    if A.dtype.itemsize == 2:  # float16 / bfloat16
        tm, tn, tk = 256, 256, 64
    else:  # float32 or other
        tm, tn, tk = 32, 32, 32

    # --- Get Matrix Dimensions ---
    m, _ = A.shape
    _, n = B.shape

    # --- Calculate Grid Dimensions ---
    grid_x = ceil(m / tm)
    grid_y = ceil(n / tn)
    grid_size = grid_x * grid_y
    grid = (grid_size, 1, 1)

    # --- Create Checksum Buffer: 1 scalar per block, fp32 ---
    checksum = torch.empty((grid_size,), device=A.device, dtype=torch.float32)

    # --- Launch the no-writeback kernel ---
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel,
              (A, B, checksum, tm, tn, tk))

    return checksum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="No-WriteBack MatMul Performance Test: computes C = A @ B but does NOT "
                    "write C back to global memory. A scalar checksum per block is stored "
                    "instead to prevent compiler DCE."
    )
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Compare kernel checksum against torch reference to verify computation",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=256,
        help="M dimension of the matrix (rows of A)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=256,
        help="N dimension of the matrix (columns of B)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=768,
        help="K dimension of the matrix (columns of A, rows of B)",
    )
    args = parser.parse_args()

    # --- Running No-WriteBack Matrix Multiplication ---
    print("--- No-WriteBack MatMul Performance Test ---")
    print("C will NOT be written back; scalar checksum per block prevents compiler DCE.")

    M_dim = args.M
    N_dim = args.N
    K_dim = args.K

    # --- Test Case 1: Swizzle M ---
    print(f"\n--- Test Case 1: No-WriteBack MatMul float16, swizzle M ---")
    A_m = torch.randn(M_dim, K_dim, dtype=torch.float16, device='cuda')
    B_m = torch.randn(K_dim, N_dim, dtype=torch.float16, device='cuda')
    print(f"Input A shape: {A_m.shape}, dtype: {A_m.dtype}")
    print(f"Input B shape: {B_m.shape}, dtype: {B_m.dtype}")

    cksum_m = cutile_matmul_no_writeback(
        A_m, B_m,
        matmul_kernel=NoWriteBackSwizzleM.matmul_kernel_m_no_writeback)
    print(f"Checksum buffer shape: {cksum_m.shape}, "
          f"sample values: {cksum_m[:min(4, len(cksum_m))]}")

    if args.correctness_check:
        C_ref = A_m @ B_m
        print(f"Reference C checksum (full sum): {C_ref.float().sum().item():.4f}")
        print(f"Kernel checksum (sum of block sums): {cksum_m.sum().item():.4f}")
        print("Note: values may differ slightly due to fp16→fp32 accumulation order")

    # --- Test Case 2: Swizzle N ---
    print(f"\n--- Test Case 2: No-WriteBack MatMul float16, swizzle N ---")
    A_n = torch.randn(M_dim, K_dim, dtype=torch.float16, device='cuda')
    B_n = torch.randn(K_dim, N_dim, dtype=torch.float16, device='cuda')
    print(f"Input A shape: {A_n.shape}, dtype: {A_n.dtype}")
    print(f"Input B shape: {B_n.shape}, dtype: {B_n.dtype}")

    cksum_n = cutile_matmul_no_writeback(
        A_n, B_n,
        matmul_kernel=NoWriteBackSwizzleN.matmul_kernel_n_no_writeback)
    print(f"Checksum buffer shape: {cksum_n.shape}, "
          f"sample values: {cksum_n[:min(4, len(cksum_n))]}")

    if args.correctness_check:
        C_ref = A_n @ B_n
        print(f"Reference C checksum (full sum): {C_ref.float().sum().item():.4f}")
        print(f"Kernel checksum (sum of block sums): {cksum_n.sum().item():.4f}")
        print("Note: values may differ slightly due to fp16→fp32 accumulation order")

    print("\n--- All no-writeback matmul perf tests done. ---")
