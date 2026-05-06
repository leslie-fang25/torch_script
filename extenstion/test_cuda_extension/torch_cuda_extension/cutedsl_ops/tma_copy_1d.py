# TMA 1D copy: global memory → shared memory → global memory (SM90+)
#
# Demonstrates the minimal CuteDSL TMA workflow for a 1D fp32 array:
#
#   Host (@cute.jit __call__):
#     1. make_tiled_tma_atom  — build the TMA descriptor on the host
#     2. kernel.launch        — one CTA per tile
#
#   Device (@cute.kernel kernel):
#     1. Allocate smem + one mbarrier
#     2. Thread 0: mbarrier_init + mbarrier_init_fence
#     3. sync_threads          — all threads see the initialised barrier
#     4. tma_partition         — reshape smem/gmem tensors for TMA copy call
#     5. Thread 0: mbarrier_arrive_and_expect_tx(bytes) + cute.copy (TMA load)
#     6. All threads: mbarrier_wait(phase=0) — spin until TMA finishes
#     7. sync_threads
#     8. All threads: write smem → output gmem  (for correctness check)
#
# mbarrier mechanics (SM90):
#   - mbarrier_init(ptr, 1)           : expect 1 thread arrival
#   - mbarrier_arrive_and_expect_tx(bytes):
#       * records the thread's arrival (satisfying the arrival count)
#       * arms the transaction counter with `bytes`
#   - TMA hardware decrements tx_count by `bytes` when the DMA write completes
#   - mbarrier opens (phase 0→1) when arrival_count==0 AND tx_count==0
#   - mbarrier_wait(ptr, 0)           : spin until phase==0 is satisfied
#
# Why tma_partition is required:
#   cute.copy with a TMA CopyAtom only generates cp.async.bulk.tensor.* PTX
#   when the source tensor is in "TMA partition format" (produced by
#   tma_partition).  Passing a raw local_tile() result instead causes the
#   compiler to silently fall back to a non-TMA copy that never decrements
#   the mbarrier tx_count, so mbarrier_wait spins forever.

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda


class TmaCopy1D:
    """
    CuteDSL TMA 1D copy kernel (SM90+).

    Each CUDA block copies one contiguous tile of ``tile_size`` fp32 elements
    from global memory into shared memory via a single TMA DMA transfer, then
    writes the shared memory contents back to a global output tensor.

    Constraints
    -----------
    * N (input length) must be a multiple of ``tile_size``.
    * ``tile_size`` must be a multiple of 4 (16-byte TMA alignment requirement).
    * ``num_threads`` must divide ``tile_size`` exactly.
    """

    def __init__(self, tile_size: int = 256):
        assert tile_size % 4 == 0, "tile_size must be a multiple of 4 (16-byte aligned)"
        self.tile_size   = tile_size
        self.dtype       = cutlass.Float32
        self.num_threads = 128   # threads per CTA; must divide tile_size

    # ──────────────────────────────────────────────────────────────────────────
    # Host JIT: build TMA descriptor, launch kernel
    # ──────────────────────────────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        mIn:    cute.Tensor,    # (N,) fp32 full global input tensor
        mOut:   cute.Tensor,    # (N,) fp32 full global output tensor
        stream: cuda.CUstream,
    ):
        tile_size = self.tile_size

        # smem_layout: layout of one tile in shared memory.
        # rank=1 (1D), no stage dimension — each CTA uses a single smem slot.
        smem_layout = cute.make_layout(tile_size)

        # make_tiled_tma_atom builds a TMA descriptor on the host.
        #   op          : G2S (global → shared, non-reduce)
        #   gmem_tensor : the full input tensor, shape (N,)
        #   smem_layout : layout of one tile in smem, shape (tile_size,)
        #   cta_tiler   : tile footprint per CTA, (tile_size,)
        # Returns:
        #   tma_atom   : the compiled TMA descriptor (passed to kernel as CopyAtom)
        #   tma_tensor : TMA-indexed view of mIn; used inside the kernel for
        #                local_tile / tma_partition
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mIn,
            smem_layout,
            (tile_size,),
            num_multicast=1,
        )

        # Bytes transferred per TMA call (used to arm the mbarrier tx counter).
        tma_bytes = tile_size * (self.dtype.width // 8)

        num_tiles = cute.size(mIn) // tile_size

        print(f"[host __call__] tma_tensor.shape={tma_tensor.shape}  "
              f"tma_bytes={tma_bytes}  num_tiles={num_tiles}", flush=True)

        self.kernel(
            tma_atom, tma_tensor, mOut,
            tma_bytes,
        ).launch(
            grid=[num_tiles, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # GPU kernel
    # ──────────────────────────────────────────────────────────────────────────

    @cute.kernel
    def kernel(
        self,
        tma_atom:  cute.CopyAtom,
        mIn:       cute.Tensor,       # tma_tensor: TMA-indexed view of input
        mOut:      cute.Tensor,       # (N,) fp32 output tensor
        tma_bytes: cutlass.Constexpr, # bytes per TMA transfer (compile-time)
    ):
        tile_size   = self.tile_size
        num_threads = self.num_threads

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # ── Per-CTA gmem tiles ────────────────────────────────────────────
        # Use (None,) to keep the tile dimension intact in the TMA tensor.
        # This mirrors the SM90 GEMM pattern: local_tile(mA, (bM,bK), (bidx, None))
        # where the "free" coordinate is kept so tma_partition can encode the
        # tile offset as a TMA coordinate.  We then pass bidx at copy time.
        gIn_tma = cute.local_tile(mIn,  (tile_size,), (None,))  # (tile_size, num_tiles)
        gOut    = cute.local_tile(mOut, (tile_size,), (bidx,))   # (tile_size,)

        # ── Shared memory ─────────────────────────────────────────────────
        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, 1]
            data: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, tile_size], 1024
            ]

        smem     = cutlass.utils.SmemAllocator()
        storage  = smem.allocate(SharedStorage)
        mbar_ptr = storage.mbar.data_ptr()
        # 2D staged layout: (tile_size, 1) — mode 0 is spatial, mode 1 is a
        # size-1 "stage" dim.  tma_partition requires smem and gmem to have the
        # same rank; gIn_tma is already 2D (tile_size, num_tiles), so smem must
        # also be 2D.
        smem_layout = cute.make_layout((tile_size, 1))
        sIn      = storage.data.get_tensor(smem_layout)

        # ── Mbarrier init (CUTLASS pipeline pattern) ──────────────────────
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # ── Partition smem/gmem for TMA copy ─────────────────────────────
        # group_modes(t, 0, 1) wraps mode 0 into a "tuple mode" so that
        # tma_partition sees the same ((spatial,), iteration) structure that
        # SM90 GEMM uses after group_modes(sA, 0, 2).  Both tensors must be
        # rank-2 with matching spatial mode structure before partitioning.
        #
        # After partition:
        #   tAsIn: smem partition, shape ((atom_v, rest_v), 1)
        #   tAgIn: gmem partition, shape ((atom_v, rest_v), num_tiles)
        tAsIn, tAgIn = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sIn, 0, 1),
            cute.group_modes(gIn_tma, 0, 1),
        )

        # ── Issue TMA load ────────────────────────────────────────────────
        # tAgIn[(None, bidx)] selects this CTA's tile — exactly like
        # tAgA[(None, producer_state.count)] in the SM90 GEMM pipeline.
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tma_bytes)
            cute.copy(
                tma_atom,
                tAgIn[(None, bidx)],   # select this CTA's tile from gmem
                tAsIn[(None, 0)],       # select stage 0 from smem
                tma_bar_ptr=mbar_ptr,
                mcast_mask=0,
            )

        # ── Wait for TMA completion ───────────────────────────────────────
        # All threads spin until the mbarrier's phase-0 condition is met
        # (arrival_count == 0 AND tx_count == 0).  Then sync so all threads
        # see the data written by TMA before reading from smem.
        cute.arch.mbarrier_wait(mbar_ptr, 0)
        cute.arch.sync_threads()

        # ── Write smem → output gmem ──────────────────────────────────────
        # sIn is 2D (tile_size, 1); take a 1D view of stage 0 for element access.
        sIn_1d = cute.slice_(sIn, (None, 0))
        for i in cutlass.range(tile_size // num_threads, unroll_full=True):
            idx = tidx + i * num_threads
            gOut[idx] = sIn_1d[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

_tma_copy_cache: dict = {}


def run_tma_copy_1d(x: "torch.Tensor", tile_size: int = 256) -> "torch.Tensor":
    """
    Copy a 1D fp32 CUDA tensor via TMA (gmem → smem → gmem).

    Args:
        x:         (N,) fp32 CUDA tensor; N must be a multiple of tile_size
        tile_size: elements per TMA tile; must be a multiple of 4

    Returns:
        output: (N,) fp32 CUDA tensor, value-identical to x
    """
    import torch

    assert x.dtype == torch.float32 and x.is_cuda and x.dim() == 1
    assert x.size(0) % tile_size == 0, "N must be a multiple of tile_size"
    assert tile_size % 4 == 0, "tile_size must be a multiple of 4 (16-byte aligned)"

    out  = torch.empty_like(x)
    mIn  = from_dlpack(x,   assumed_align=16)
    mOut = from_dlpack(out, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)

    key = (x.size(0), tile_size)
    if key not in _tma_copy_cache:
        print(f"[run_tma_copy_1d] compiling for N={x.size(0)}, tile_size={tile_size}...", flush=True)
        _tma_copy_cache[key] = cute.compile(TmaCopy1D(tile_size), mIn, mOut, stream)
        print("[run_tma_copy_1d] compilation done", flush=True)

    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    print("[run_tma_copy_1d] invoking compiled kernel...", flush=True)
    _tma_copy_cache[key](mIn, mOut, stream)
    print("[run_tma_copy_1d] invocation returned, synchronizing...", flush=True)
    torch.cuda.synchronize()
    print("[run_tma_copy_1d] synchronized — kernel finished", flush=True)
    return out
