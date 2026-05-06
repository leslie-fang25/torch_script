# SM80 3-stage K-pipeline GEMM using CuteDSL (CUTLASS Python DSL)
#
# A: (M, K) fp16 row-major
# B: (N, K) fp16 row-major  [caller passes B in N×K layout, i.e. transposed]
# C: (M, N) fp32 row-major
#
# Pipeline (3-stage smem):
#   Prologue: async-copy k=0 and k=1 into stages 0, 1
#   Loop k:
#     cp_async_wait_group(1)  → stage k ready; stage k+1 copy still in flight
#     ldmatrix from stage k → registers
#     MMA
#     async-copy k+2 into the just-consumed stage
#   cp_async_wait_group(0), sync
#   Epilogue: reg → gmem (optionally with relu)
#
# 3 stages are the minimum for true overlap: wait_group(stages-2)=wait_group(1)
# keeps 1 copy in flight during MMA, unlike 2-stage which needs wait_group(0).

import math
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack


class GemmPipelineSM80:
    """
    SM80 GEMM with 3-stage K-dimension smem pipeline, implemented in CuteDSL.

    Usage::

        gemm = GemmPipelineSM80()
        compiled = cute.compile(gemm, mA, mB, mC)
        compiled(mA, mB, mC)
    """

    # relu is fused into the epilogue via a Constexpr lambda; run_gemm_pipeline
    # passes ep_op to cute.compile() and does NOT call relu_() after the kernel.
    fused_epilogue = True

    @classmethod
    def build(cls, mA, mB, mC, ep_op, device):
        """Compile the kernel for a given problem shape and return the compiled fn."""
        return cute.compile(cls(), mA, mB, mC, ep_op)

    @classmethod
    def invoke(cls, compiled, mA, mB, mC, device):
        """Execute a previously compiled kernel on the current CUDA stream."""
        compiled(mA, mB, mC)

    def __init__(self, atom_layout_mnk=(2, 2, 1), num_stages=3):
        assert num_stages >= 3, "num_stages >= 3 required for wait_group(stages-2)=1"

        self.ab_dtype   = cutlass.Float16
        self.acc_dtype  = cutlass.Float32
        self.c_dtype    = cutlass.Float32
        self.bM, self.bN, self.bK = 128, 128, 32
        self.num_stages = num_stages
        self.atom_layout_mnk = atom_layout_mnk
        self.mma_inst_shape  = (16, 8, 16)   # SM80 16×8×16 fp16 tensor-core

        mM, mN, mK = atom_layout_mnk
        self.num_threads = mM * mN * mK * 32   # 128 for (2,2,1)

    # ──────────────────────────────────────────────────────────────────────────
    # Smem layout helpers (swizzled to avoid bank conflicts on ldmatrix)
    # ──────────────────────────────────────────────────────────────────────────

    def _smem_layout_ab(self, dtype, copy_bits, smem_tiler):
        """
        Swizzled composed layout for A/B smem:
          inner atom: (8, K_minor) row-major
          outer: tiled to (bM/N, bK, num_stages)
        """
        bK = smem_tiler[1]
        minor_K = min(bK, 64)
        swizzle_bits = min(int(math.log2(minor_K * dtype.width // copy_bits)), 3)

        layout_atom_outer = cute.make_layout(
            (8, minor_K), stride=(minor_K, 1)
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            layout_atom_outer,
        )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    # ──────────────────────────────────────────────────────────────────────────
    # Host-side JIT entry point
    # ──────────────────────────────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        bM, bN, bK = self.bM, self.bN, self.bK
        ab_bits   = 128
        copy_elems = ab_bits // mA.element_type.width   # 8 fp16 per 128-bit load

        # Smem layouts: (bM/N, bK, num_stages) with swizzle
        sA_layout = self._smem_layout_ab(
            mA.element_type, ab_bits, (bM, bK, self.num_stages))
        sB_layout = self._smem_layout_ab(
            mB.element_type, ab_bits, (bN, bK, self.num_stages))

        # Gmem → smem: cp.async 128-bit
        gcp_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            mA.element_type,
            num_bits_per_copy=ab_bits,
        )
        th_K = bK // copy_elems                 # threads along K: 4
        th_M = self.num_threads // th_K         # threads along M/N: 32
        row_thread_layout = cute.make_layout((th_M, th_K), stride=(th_K, 1))
        row_val_layout    = cute.make_layout((1, copy_elems))
        tiled_copy_A = cute.make_tiled_copy_tv(gcp_atom, row_thread_layout, row_val_layout)
        tiled_copy_B = cute.make_tiled_copy_tv(gcp_atom, row_thread_layout, row_val_layout)

        # TiledMMA (SM80_16x8x16_F32F16F16F32_TN with 2×2×1 warp layout)
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape)
        mM, mN, mK = self.atom_layout_mnk
        iM, iN, iK = self.mma_inst_shape
        # permutation_mnk: sets MNK coverage per TiledMMA step.
        # N is doubled (iN*2=16) to leverage wider smem→reg copy per step.
        permutation_mnk = (mM * iM, mN * iN * 2, mK * iK)
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )

        # Launch grid: ceil(M/bM) × ceil(N/bN)
        grid = cute.ceil_div(mC.shape, (bM, bN))
        self.kernel(
            mA, mB, mC,
            sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B,
            tiled_mma,
            epilogue_op,
        ).launch(
            grid=[cute.size(grid[0]), cute.size(grid[1]), 1],
            block=[self.num_threads, 1, 1],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # GPU kernel
    # ──────────────────────────────────────────────────────────────────────────

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        bM, bN, bK = self.bM, self.bN, self.bK
        NS = self.num_stages

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # ── Block-local gmem tiles ────────────────────────────────────────────
        # gA: (bM, bK, k_tiles)  — M-tile selected by bidx, K iterated
        # gB: (bN, bK, k_tiles)  — N-tile selected by bidy, K iterated
        # gC: (bM, bN)           — output tile for this block
        gA = cute.local_tile(mA, (bM, bK), (bidx, None))
        gB = cute.local_tile(mB, (bN, bK), (bidy, None))
        gC = cute.local_tile(mC, (bM, bN), (bidx, bidy))

        # ── Shared memory allocation ──────────────────────────────────────────
        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 16]
            b: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)], 16]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)
        sA = SharedStorage(storage).a.get_tensor(sA_layout)   # (bM, bK, NS) swizzled
        sB = SharedStorage(storage).b.get_tensor(sB_layout)   # (bN, bK, NS) swizzled

        # ── Gmem → smem copy partitions ──────────────────────────────────────
        g2s_A = tiled_copy_A.get_slice(tidx)
        g2s_B = tiled_copy_B.get_slice(tidx)
        tAgA = g2s_A.partition_S(gA)   # (CPY, CPY_M, CPY_K, k_tiles)
        tAsA = g2s_A.partition_D(sA)   # (CPY, CPY_M, CPY_K, NS)
        tBgB = g2s_B.partition_S(gB)
        tBsB = g2s_B.partition_D(sB)

        # ── MMA thread partition + register fragments ─────────────────────────
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA    = thr_mma.partition_A(sA)             # (MMA_A, MMA_M, MMA_K, NS)
        tCsB    = thr_mma.partition_B(sB)             # (MMA_B, MMA_N, MMA_K, NS)
        tCgC    = thr_mma.partition_C(gC)             # (MMA_C, MMA_M, MMA_N)
        tCrA    = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])  # (MMA_A, MMA_M, MMA_K)
        tCrB    = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])  # (MMA_B, MMA_N, MMA_K)
        tCrC    = tiled_mma.make_fragment_C(tCgC)     # accumulator
        tCrC.fill(0.0)

        # ── Smem → reg copy (ldmatrix) ────────────────────────────────────────
        # LdMatrix8x8x16bOp(transposed=False, num_matrices=4):
        #   loads 4 × 8×8 fp16 sub-matrices non-transposed = SM75_U32x4_LDSM_N
        s2r_A_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mA.element_type)
        s2r_B_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mB.element_type)
        tc_s2r_A = cute.make_tiled_copy_A(s2r_A_atom, tiled_mma)
        tc_s2r_B = cute.make_tiled_copy_B(s2r_B_atom, tiled_mma)
        s2r_thr_A = tc_s2r_A.get_slice(tidx)
        s2r_thr_B = tc_s2r_B.get_slice(tidx)

        # partition_S: smem view partitioned for ldmatrix  (CPY, CPY_M, CPY_K, NS)
        # retile:      register view re-interpreted for ldmatrix (= C++ retile_D)
        tCsA_copy = s2r_thr_A.partition_S(sA)
        tCrA_copy = s2r_thr_A.retile(tCrA)
        tCsB_copy = s2r_thr_B.partition_S(sB)
        tCrB_copy = s2r_thr_B.retile(tCrB)

        k_tile_count = cute.size(tAgA, mode=[3])   # K // bK
        num_k_block  = cute.size(tCrA, mode=[2])   # bK // mmaK (typically 2)

        # ── Prologue: fill first NS-1 smem stages asynchronously ─────────────
        # After prologue: NS-1 groups in flight; main loop keeps NS-2 in flight
        # (for NS=3: 2 groups after prologue, wait_group(1) keeps 1 in flight)
        smem_pipe_read  = 0
        smem_pipe_write = 0
        k_fetch         = 0

        # Pre-clear smem (guards against partial first tile)
        tAsA.fill(0)
        tBsB.fill(0)
        cute.arch.sync_threads()

        for s in range(NS - 1):
            if k_fetch < k_tile_count:
                cute.copy(tiled_copy_A,
                    tAgA[None, None, None, k_fetch],
                    tAsA[None, None, None, smem_pipe_write])
                cute.copy(tiled_copy_B,
                    tBgB[None, None, None, k_fetch],
                    tBsB[None, None, None, smem_pipe_write])
            cute.arch.cp_async_commit_group()
            smem_pipe_write = smem_pipe_write + 1
            k_fetch = k_fetch + 1

        # ── Main loop (K-tiles) ───────────────────────────────────────────────
        # wait_group(NS-2) = wait_group(1) for NS=3:
        #   drains the oldest group (current stage ready);
        #   keeps 1 group in flight (next stage copy overlaps with MMA below)
        for k_tile in range(k_tile_count):

            cute.arch.cp_async_wait_group(NS - 2)
            cute.arch.sync_threads()

            # Smem → Reg (ldmatrix) + MMA, unrolled over k-blocks within the tile
            for k_block in cutlass.range(num_k_block, unroll_full=True):
                # ldmatrix: smem[smem_pipe_read, k_block] → registers
                cute.copy(tc_s2r_A,
                    tCsA_copy[None, None, k_block, smem_pipe_read],
                    tCrA_copy[None, None, k_block])
                cute.copy(tc_s2r_B,
                    tCsB_copy[None, None, k_block, smem_pipe_read],
                    tCrB_copy[None, None, k_block])
                # MMA using the MMA-semantic view (different layout from ldmatrix view)
                cute.gemm(tiled_mma, tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC)

            # Issue async copy for k+NS into the just-consumed stage
            # (smem_pipe_write still points at the old consumed stage)
            if k_fetch < k_tile_count:
                cute.copy(tiled_copy_A,
                    tAgA[None, None, None, k_fetch],
                    tAsA[None, None, None, smem_pipe_write])
                cute.copy(tiled_copy_B,
                    tBgB[None, None, None, k_fetch],
                    tBsB[None, None, None, smem_pipe_write])
            cute.arch.cp_async_commit_group()
            k_fetch = k_fetch + 1

            # Advance ring-buffer indices:
            #   consumed stage (smem_pipe_read) becomes the next write target
            smem_pipe_write = smem_pipe_read
            smem_pipe_read  = smem_pipe_read + 1
            if smem_pipe_read == NS:
                smem_pipe_read = 0

        # Drain remaining in-flight copies before epilogue
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # ── Epilogue ──────────────────────────────────────────────────────────
        tCrC_out = cute.make_fragment_like(tCrC, self.c_dtype)
        tCrC_out[None] = epilogue_op(tCrC.load()).to(self.c_dtype)
        cute.autovec_copy(tCrC_out, tCgC)


# ─────────────────────────────────────────────────────────────────────────────
# Public helper: compile once per process, then reuse
# ─────────────────────────────────────────────────────────────────────────────

from .gemm_pipeline_sm90 import GemmPipelineSM90  # noqa: E402
from .gemm_pipeline_sm100 import GemmPipelineSM100  # noqa: E402

_gemm_cache: dict = {}

# Registry: sm_major → GemmClass (add new arch classes here)
_ARCH_REGISTRY: dict = {
    8:  GemmPipelineSM80,   # Ampere   (sm80, sm86, sm87)
    9:  GemmPipelineSM90,   # Hopper   (sm90)
    10: GemmPipelineSM100,  # Blackwell (sm100, sm103)
}


def _get_gemm_class(device: "torch.device"):
    import torch
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    cls = _ARCH_REGISTRY.get(sm_major)
    if cls is None:
        supported = ", ".join(f"sm{k}x" for k in sorted(_ARCH_REGISTRY))
        raise NotImplementedError(
            f"No CuteDSL GEMM implementation for sm{sm_major}{sm_minor}. "
            f"Supported: {supported}"
        )
    return cls


def run_gemm_pipeline(
    a: "torch.Tensor",
    b: "torch.Tensor",
    epilogue: str = "none",
) -> "torch.Tensor":
    """
    Compute C = A @ B.T using the arch-appropriate pipeline GEMM.

    Args:
        a:        (M, K) fp16 CUDA tensor, row-major
        b:        (N, K) fp16 CUDA tensor, row-major  [already transposed by caller]
        epilogue: "none" or "relu"

    Returns:
        c:  (M, N) fp32 CUDA tensor
    """
    import torch

    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2
    assert a.size(1) == b.size(1), "K dimension mismatch"

    gemm_cls = _get_gemm_class(a.device)

    use_relu = epilogue == "relu"
    M, K = a.shape
    N    = b.shape[0]

    c = torch.empty(M, N, dtype=torch.float32, device=a.device)

    # Convert to CuTe tensors. Default is a straight 2-D from_dlpack; arches
    # that need a different layout (e.g. SM100 needs batched 3-D for
    # DenseGemmKernel) override make_cute_tensors.
    if hasattr(gemm_cls, "make_cute_tensors"):
        mA, mB, mC = gemm_cls.make_cute_tensors(a, b, c)
    else:
        mA = from_dlpack(a, assumed_align=16)
        mB = from_dlpack(b, assumed_align=16)
        mC = from_dlpack(c, assumed_align=16)

    # SM80: relu fused as Constexpr epilogue_op; SM90: relu applied post-kernel
    if gemm_cls.fused_epilogue:
        ep_op = (
            (lambda x: cute.where(x > 0, x, cute.full_like(x, 0)))
            if use_relu
            else (lambda x: x)
        )
    else:
        ep_op = None

    key = (gemm_cls, M, N, K, use_relu)
    if key not in _gemm_cache:
        _gemm_cache[key] = gemm_cls.build(mA, mB, mC, ep_op, a.device)

    gemm_cls.invoke(_gemm_cache[key], mA, mB, mC, a.device)

    if not gemm_cls.fused_epilogue and use_relu:
        c.relu_()

    return c
