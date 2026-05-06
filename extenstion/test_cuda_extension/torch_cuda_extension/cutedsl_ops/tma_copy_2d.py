# TMA 2D copy: global memory → shared memory → global memory (SM90+)
#
# Uses TMA G2S for gmem→smem and TMA S2G for smem→gmem, avoiding any
# 2D runtime element indexing in CuteDSL kernel mode.
#
#   Host (@cute.jit __call__):
#     1. make_tiled_tma_atom (G2S) — load descriptor
#     2. make_tiled_tma_atom (S2G) — store descriptor
#     3. kernel.launch — 2D grid: (M//tile_m, N//tile_n) CTAs
#
#   Device (@cute.kernel kernel):
#     G2S load:
#       1. smem layout (tile_m, tile_n, 1); group_modes → rank-2 for tma_partition
#       2. mbarrier_init / arrive_and_expect_tx / copy (G2S) / mbarrier_wait
#     S2G store:
#       3. fence_proxy("async.shared") — make smem visible to TMA store proxy
#       4. tma_partition (S2G) + cute.copy (S2G)
#       5. cp_async_bulk_commit_group + cp_async_bulk_wait_group(0)

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda


class TmaCopy2D:
    """
    CuteDSL TMA 2D copy kernel (SM90+).

    Each CUDA block copies one (tile_m × tile_n) fp32 tile from global memory
    into shared memory via TMA G2S, then stores it back via TMA S2G.

    Constraints
    -----------
    * M must be a multiple of tile_m; N must be a multiple of tile_n.
    * tile_m * tile_n must be a multiple of 128.
    * tile_n must be a multiple of 4 (16-byte TMA alignment on the fast dim).
    """

    def __init__(self, tile_shape=(64, 64)):
        assert tile_shape[1] % 4 == 0, "tile_n must be a multiple of 4"
        assert (tile_shape[0] * tile_shape[1]) % 128 == 0, (
            "tile_m * tile_n must be a multiple of 128"
        )
        self.tile_m = tile_shape[0]
        self.tile_n = tile_shape[1]
        self.dtype = cutlass.Float32
        self.num_threads = 128

    # ──────────────────────────────────────────────────────────────────────────
    # Host JIT
    # ──────────────────────────────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        mIn:    cute.Tensor,   # (M, N) fp32 global input
        mOut:   cute.Tensor,   # (M, N) fp32 global output
        stream: cuda.CUstream,
    ):
        tile_m = self.tile_m
        tile_n = self.tile_n

        # Non-staged 2D smem layout shared by both G2S and S2G descriptors.
        # ROW-MAJOR (stride (tile_n, 1)): TMA 2D tile copies require the smem
        # inner dim to be contiguous in matching orientation with gmem. The
        # default column-major from cute.make_layout((tile_m, tile_n)) silently
        # produces a layout that TMA does not honor — only the leading column
        # ends up populated, leaving the rest of the tile zero.
        smem_layout_1stage = cute.make_layout((tile_m, tile_n), stride=(tile_n, 1))

        tma_atom_load, tma_tensor_in = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mIn,
            smem_layout_1stage,
            (tile_m, tile_n),
            num_multicast=1,
        )

        # S2G: no num_multicast parameter
        tma_atom_store, tma_tensor_out = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            mOut,
            smem_layout_1stage,
            (tile_m, tile_n),
        )

        tma_bytes   = tile_m * tile_n * (self.dtype.width // 8)
        M           = cute.size(mIn, mode=[0])
        N           = cute.size(mIn, mode=[1])
        num_m_tiles = M // tile_m
        num_n_tiles = N // tile_n

        print(
            f"[TmaCopy2D __call__] tma_tensor_in.shape={tma_tensor_in.shape}  "
            f"tma_bytes={tma_bytes}  grid=({num_m_tiles},{num_n_tiles})",
            flush=True,
        )

        self.kernel(
            tma_atom_load,  tma_tensor_in,
            tma_atom_store, tma_tensor_out,
            # mOut,  # DEBUG: pass raw mOut for manual thread-per-element S2G (see kernel)
            tma_bytes,
        ).launch(
            grid=[num_m_tiles, num_n_tiles, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # GPU kernel
    # ──────────────────────────────────────────────────────────────────────────

    @cute.kernel
    def kernel(
        self,
        tma_atom_load:  cute.CopyAtom,
        mIn:            cute.Tensor,       # TMA-indexed view of input
        tma_atom_store: cute.CopyAtom,
        mOut:           cute.Tensor,       # TMA-indexed view of output
        # mOut_raw:     cute.Tensor,       # DEBUG: raw mOut for manual S2G
        tma_bytes:      cutlass.Constexpr,
    ):
        tile_m = self.tile_m
        tile_n = self.tile_n

        # tidx, _, _ = cute.arch.thread_idx()  # DEBUG: needed for manual S2G
        bidx, bidy, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # ── Per-CTA gmem tiles ─────────────────────────────────────────────
        # Fix M-row (bidx), keep N-column free so tma_partition can encode
        # the tile coordinate dynamically (same pattern as SM90 GEMM gA tile).
        gIn_tma  = cute.local_tile(mIn,  (tile_m, tile_n), (bidx, None))
        gOut_tma = cute.local_tile(mOut, (tile_m, tile_n), (bidx, None))
        # DEBUG: per-CTA raw output tile (both M and N coords fixed) for manual S2G
        # gOut_raw = cute.local_tile(mOut_raw, (tile_m, tile_n), (bidx, bidy))

        # ── Shared memory ──────────────────────────────────────────────────
        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, 1]
            data: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, tile_m * tile_n], 1024
            ]

        smem     = cutlass.utils.SmemAllocator()
        storage  = smem.allocate(SharedStorage)
        mbar_ptr = storage.mbar.data_ptr()

        # 3D staged smem (row-major to match TMA descriptor): stride (tile_n, 1, tile_m*tile_n)
        # group_modes(sIn, 0, 2) → ((tile_m,tile_n), 1) rank-2
        smem_layout = cute.make_layout(
            (tile_m, tile_n, 1),
            stride=(tile_n, 1, tile_m * tile_n),
        )
        sIn = storage.data.get_tensor(smem_layout)

        # ── Mbarrier init ──────────────────────────────────────────────────
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # ── TMA G2S load partition ─────────────────────────────────────────
        tAsIn, tAgIn = cute.nvgpu.cpasync.tma_partition(
            tma_atom_load,
            0,
            cute.make_layout(1),
            cute.group_modes(sIn, 0, 2),
            cute.group_modes(gIn_tma, 0, 2),
        )

        # ── Issue TMA load ─────────────────────────────────────────────────
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tma_bytes)
            cute.copy(
                tma_atom_load,
                tAgIn[(None, bidy)],
                tAsIn[(None, 0)],
                tma_bar_ptr=mbar_ptr,
                mcast_mask=0,
            )

        # ── Wait for TMA load completion ───────────────────────────────────
        cute.arch.mbarrier_wait(mbar_ptr, 0)
        cute.arch.sync_threads()

        # ── TMA S2G store partition ────────────────────────────────────────
        tAsOut, tAgOut = cute.nvgpu.cpasync.tma_partition(
            tma_atom_store,
            0,
            cute.make_layout(1),
            cute.group_modes(sIn, 0, 2),
            cute.group_modes(gOut_tma, 0, 2),
        )

        # ── Issue TMA store ────────────────────────────────────────────────
        # fence_proxy makes prior smem writes (from TMA load) visible to the
        # async-shared proxy used by the TMA store engine.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        if warp_idx == 0:
            cute.copy(
                tma_atom_store,
                tAsOut[(None, 0)],    # smem source, stage 0
                tAgOut[(None, bidy)], # gmem destination, this CTA's N column
            )
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)

        cute.arch.sync_threads()

        # ── DEBUG: manual thread-per-element smem → gmem store ──────────────
        # Alternative S2G path used during debugging to isolate G2S correctness
        # from TMA store issues. To use:
        #   1. Comment out the TMA S2G block above (tma_partition + cute.copy)
        #   2. Uncomment `mOut_raw` in kernel signature + `mOut` in __call__ launch
        #   3. Uncomment `tidx` and `gOut_raw` near the top of the kernel
        #   4. Uncomment the loop below
        # Each thread writes (tile_m * tile_n) / num_threads elements directly
        # from smem to the raw row-major gmem output tile.
        #
        # elems_per_thread = (tile_m * tile_n) // self.num_threads
        # for i in cutlass.range(elems_per_thread, unroll_full=True):
        #     flat_idx = tidx + i * self.num_threads
        #     row = flat_idx // tile_n
        #     col = flat_idx - row * tile_n
        #     gOut_raw[row, col] = sIn[row, col, 0]
        # cute.arch.sync_threads()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

_tma_copy_2d_cache: dict = {}


def run_tma_copy_2d(
    x: "torch.Tensor",
    tile_shape: tuple = (64, 64),
) -> "torch.Tensor":
    """
    Copy a 2D fp32 CUDA tensor via TMA G2S + S2G (gmem → smem → gmem).

    Args:
        x:          (M, N) fp32 CUDA tensor; M % tile_m == 0, N % tile_n == 0
        tile_shape: (tile_m, tile_n) tile dimensions

    Returns:
        output: (M, N) fp32 CUDA tensor, value-identical to x
    """
    import torch

    tile_m, tile_n = tile_shape
    assert x.dtype == torch.float32 and x.is_cuda and x.dim() == 2
    assert x.size(0) % tile_m == 0, "M must be a multiple of tile_m"
    assert x.size(1) % tile_n == 0, "N must be a multiple of tile_n"
    assert tile_n % 4 == 0, "tile_n must be a multiple of 4"
    assert (tile_m * tile_n) % 128 == 0, "tile_m * tile_n must be a multiple of 128"

    out  = torch.empty_like(x)
    mIn  = from_dlpack(x,   assumed_align=16)
    mOut = from_dlpack(out, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)

    key = (x.shape, tile_shape)
    if key not in _tma_copy_2d_cache:
        print(
            f"[run_tma_copy_2d] compiling for shape={x.shape}, tile={tile_shape}...",
            flush=True,
        )
        _tma_copy_2d_cache[key] = cute.compile(
            TmaCopy2D(tile_shape), mIn, mOut, stream
        )
        print("[run_tma_copy_2d] compilation done", flush=True)

    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    print("[run_tma_copy_2d] invoking compiled kernel...", flush=True)
    _tma_copy_2d_cache[key](mIn, mOut, stream)
    print("[run_tma_copy_2d] invocation returned, synchronizing...", flush=True)
    torch.cuda.synchronize()
    print("[run_tma_copy_2d] synchronized — kernel finished", flush=True)
    return out
