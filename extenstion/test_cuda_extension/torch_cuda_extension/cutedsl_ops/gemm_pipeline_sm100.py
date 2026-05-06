# SM100 (Blackwell) tcgen05 + TMA GEMM using CuteDSL.
#
# Computes C = A @ B.T
#   A: (M, K) fp16 row-major  (k-major)
#   B: (N, K) fp16 row-major  (k-major, caller pre-transposes)
#   C: (M, N) fp32 row-major  (n-major)
#
# SM90 vs SM100 architectural differences this kernel exercises:
#   MMA:    WGMMA (warpgroup-level, 128 threads, smem→smem→register acc)
#         → tcgen05.mma (single warp issues, smem→smem→TMEM acc)
#   Acc:    fp32 accumulator lives in registers
#         → fp32 accumulator lives in TMEM (load to register in epilogue)
#   Sync:   PipelineTmaAsync (TMA → consumer mbarrier)
#         → PipelineTmaUmma   (TMA → UMMA mbarrier, MMA-aware tx_count)
#         + PipelineUmmaAsync (UMMA → AsyncThread mbarrier, acc-ready signal)
#
# This implementation deliberately covers only the simplest Blackwell config
# (1 CTA, no cluster, no 2cta MMA, no TMA multicast, fp16→fp32, fixed
# 128x128 tile). The structure mirrors the upstream reference
#   csrc/include/cutlass/examples/python/CuTeDSL/blackwell/dense_gemm.py
# but is rewritten against the installed cutlass-dsl 4.4.x APIs and stripped
# of cluster/2cta/persistent paths so the SM100 mainloop and acc-pipeline
# structure stays readable.

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack


class GemmPipelineSM100:
    """
    SM100 (Blackwell) fp16 GEMM with tcgen05.mma + TMA + TMEM accumulator.

    Tile constraints:
      M: multiple of 128 (cta tile M = mma_tiler_m / atom_thr_size = 128 / 1)
      N: multiple of 32, in [32, 256]   (mma_tiler N step is 32)
      K: multiple of 64                 (bK = mma_inst_k * 4 = 16 * 4)
    """

    # Like SM80, the relu lambda is fused into the epilogue (applied after the
    # TMEM→register load, before the smem store + TMA store back to gmem).
    fused_epilogue = True

    def __init__(self, mma_tiler_mn=(128, 128)):
        self.ab_dtype  = cutlass.Float16
        self.acc_dtype = cutlass.Float32
        self.c_dtype   = cutlass.Float32

        # 1cta MMA: each CTA owns its full output tile. (No 2cta cooperation.)
        self.cta_group        = tcgen05.CtaGroup.ONE
        self.mma_tiler_mn     = mma_tiler_mn
        self.cluster_shape_mn = (1, 1)
        self.threads_per_cta  = 128
        self.occupancy        = 1
        self.buffer_align_bytes = 1024  # TMA needs 128B; we over-align like SM90

    # ──────────────────────────────────────────────────────────────────────────
    # Driver-side glue used by run_gemm_pipeline
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def make_cute_tensors(a, b, c):
        """
        Promote 2-D torch tensors to 3-D CuTe tensors (M,K,1) / (N,K,1) /
        (M,N,1). The blackwell_helpers (make_smem_layout_a/b, TMA atom A/B,
        epilogue helpers) are written for batched (M,K,L) shapes; using L=1
        lets us reuse them without modification.
        """
        return (
            from_dlpack(a.unsqueeze(-1), assumed_align=16),
            from_dlpack(b.unsqueeze(-1), assumed_align=16),
            from_dlpack(c.unsqueeze(-1), assumed_align=16),
        )

    @classmethod
    def build(cls, mA, mB, mC, ep_op, device):
        import torch
        ep_op = ep_op if ep_op is not None else (lambda x: x)
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        return cute.compile(cls(), mA, mB, mC, stream, ep_op)

    @classmethod
    def invoke(cls, compiled, mA, mB, mC, device):
        import torch
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        compiled(mA, mB, mC, stream)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage / smem-budget heuristic
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_stages(self, tiled_mma, mma_tiler, epi_tile, c_layout):
        """
        Default to 1 acc stage (a single in-flight accumulator in TMEM is
        sufficient for a non-persistent kernel) and 2 epilogue stages, then
        fill the remaining smem budget with mainloop A/B stages.
        """
        num_acc_stage = 1
        num_c_stage   = 2

        a_smem_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler, self.ab_dtype, 1,
        )
        b_smem_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler, self.ab_dtype, 1,
        )
        c_smem_one = sm100_utils.make_smem_layout_epi(
            self.c_dtype, c_layout, epi_tile, 1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(self.ab_dtype, a_smem_one)
            + cute.size_in_bytes(self.ab_dtype, b_smem_one)
        )
        c_bytes_per_stage = cute.size_in_bytes(self.c_dtype, c_smem_one)
        mbar_helpers_bytes = 1024
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            self.smem_capacity
            - (self.occupancy + 1) * (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Repurpose any leftover smem for additional epilogue stages
        leftover = (
            self.smem_capacity
            - ab_bytes_per_stage * num_ab_stage
            - (self.occupancy + 1) * (mbar_helpers_bytes + c_bytes)
        )
        num_c_stage += leftover // ((self.occupancy + 1) * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    # ──────────────────────────────────────────────────────────────────────────
    # Host JIT entry: build TMA descriptors, smem layouts, launch kernel
    # ──────────────────────────────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        a_layout = utils.LayoutEnum.from_tensor(a)
        b_layout = utils.LayoutEnum.from_tensor(b)
        c_layout = utils.LayoutEnum.from_tensor(c)

        # tcgen05 trivial tiled MMA: fp16 × fp16 → fp32, 1cta variant.
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype,
            a_layout.mma_major_mode(),
            b_layout.mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        # K dim of one MMA tile. For fp16 tcgen05 MMA the instruction K-depth
        # is 16; we issue 4 MMA k-blocks per smem stage, so bK = 64.
        # Hard-coded (rather than via cute.size on the static layout) so the
        # value stays a Python int and can cross @cute.jit region boundaries
        # as a Constexpr — cute.size returns an MLIR Int even for static
        # shapes, which fails MLIR isolation when reused in another region.
        mma_inst_shape_k = 16
        mma_inst_tile_k  = 4
        bK = mma_inst_shape_k * mma_inst_tile_k          # 64
        mma_tiler = (*self.mma_tiler_mn, bK)

        # 1cta MMA: atom_thr_size == 1, so each CTA owns the full M tile.
        atom_thr_size = 1
        cta_tile_shape_mnk = (
            self.mma_tiler_mn[0] // atom_thr_size,
            self.mma_tiler_mn[1],
            bK,
        )

        # Cluster layout: (V, M, N, L). For (1,1) cluster + 1cta MMA all sizes are 1.
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Epilogue tile (sub-tile of cta_tile_mn used for one TMA store unit)
        epi_tile = sm100_utils.compute_epilogue_tile_shape(
            cta_tile_shape_mnk, False, c_layout, self.c_dtype,
        )

        # Stage counts based on smem capacity
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        num_acc_stage, num_ab_stage, num_c_stage = self._compute_stages(
            tiled_mma, mma_tiler, epi_tile, c_layout,
        )

        # ── Smem layouts ──────────────────────────────────────────────────────
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler, self.ab_dtype, num_ab_stage,
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler, self.ab_dtype, num_ab_stage,
        )
        c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, c_layout, epi_tile, num_c_stage,
        )

        # ── TMA atoms ─────────────────────────────────────────────────────────
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id),
            a, a_smem_layout, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        )
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, tiled_mma.thr_id),
            b, b_smem_layout, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        )
        epi_smem_layout = cute.slice_(c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c, epi_smem_layout, epi_tile,
        )

        a_copy_size = cute.size_in_bytes(self.ab_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.ab_dtype, b_smem_layout)
        num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Number of TMEM columns for this MMA tile (drives TmemAllocator size)
        acc_shape   = tiled_mma.partition_shape_C(self.mma_tiler_mn)
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        # Grid: (ceil(M / cta_tile_M), ceil(N / cta_tile_N), L)
        grid = (
            cute.ceil_div(c.shape[0], cta_tile_shape_mnk[0]),
            cute.ceil_div(c.shape[1], cta_tile_shape_mnk[1]),
            c.shape[2],
        )

        # Pass everything the kernel needs as explicit parameters. Storing
        # MLIR-derived values on self crosses cute.jit region boundaries and
        # fails IR isolation; only static (Python-level) config goes via self.
        self.kernel(
            tiled_mma,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c,
            cluster_layout_vmnk,
            a_smem_layout_staged, b_smem_layout_staged, c_smem_layout_staged,
            epi_tile,
            mma_tiler,
            cta_tile_shape_mnk,
            c_layout,
            num_acc_stage,
            num_ab_stage,
            num_c_stage,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Device kernel
    # ──────────────────────────────────────────────────────────────────────────

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        # cute.Tile (not Constexpr): epi_tile contains MLIR Layouts created in
        # __call__, and Constexpr captures break MLIR isolation. Regular cute.*
        # parameter annotations route the SSA values through kernel codegen.
        epi_tile: cute.Tile,
        mma_tiler: cutlass.Constexpr,
        cta_tile_shape_mnk: cutlass.Constexpr,
        c_layout: cutlass.Constexpr,
        num_acc_stage: cutlass.Constexpr,
        num_ab_stage: cutlass.Constexpr,
        num_c_stage: cutlass.Constexpr,
        num_tmem_alloc_cols: cutlass.Constexpr,
        num_tma_load_bytes: cutlass.Constexpr,
        epilogue_op: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # ── Step 1: prefetch TMA descriptors ─────────────────────────────────
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        # ── Step 2: per-CTA tile coordinates ─────────────────────────────────
        # 1cta MMA: mma_tile_coord_v == 0 always; this CTA is always its own leader.
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_mnl = (bidx, bidy, bidz)

        # ── Step 3: shared storage layout ────────────────────────────────────
        # Mainloop ab pipeline:  num_ab_stage × 2 mbarriers (full + empty)
        # Acc pipeline:          num_acc_stage × 2 mbarriers
        # tmem_holding_buf / tmem_dealloc_mbar: addressable scalars used by
        # TmemAllocator (must be MemRange[T,1] in cutlass-dsl 4.4.x because
        # plain scalar struct fields don't expose `.ptr` in this version).
        @cute.struct
        class SharedStorage:
            ab_full_mbar:     cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
            acc_full_mbar:    cute.struct.MemRange[cutlass.Int64, num_acc_stage * 2]
            tmem_dealloc_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf:  cute.struct.MemRange[cutlass.Int32, 1]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(c_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]

        smem    = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ── Step 4: pipelines (TMA→UMMA  and  UMMA→AsyncThread) ──────────────
        # PipelineTmaUmma: producer = TMA load (warp 0 issuer);
        #                  consumer = tcgen05.mma (warp 0).
        # tx_count tells the mbarrier how many bytes the TMA engine writes
        # before signaling "full" — the consumer (MMA) can only start once
        # this byte-count is reached.
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar.data_ptr(),
            num_stages=num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            # 1cta: the consumer here is just the MMA-issuing thread.
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # PipelineUmmaAsync: producer = tcgen05.mma (warp 0);
        #                    consumer = all CTA threads doing the epilogue.
        # No tx_count: the MMA "full" signal is purely a logical handshake —
        # the TMEM data doesn't go through smem so there are no TMA bytes.
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar.data_ptr(),
            num_stages=num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_cta
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline state cursors
        ab_producer_state    = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_ab_stage)
        ab_consumer_state    = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_ab_stage)
        acc_producer_state   = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_acc_stage)
        acc_consumer_state   = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_acc_stage)

        # ── Step 5: TMEM allocator ───────────────────────────────────────────
        # TmemAllocator stores the dynamically-allocated TMEM base pointer in
        # smem (`tmem_holding_buf`); after `wait_for_alloc`, all threads can
        # read the pointer back via `retrieve_ptr`.
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.data_ptr(),
            barrier_for_retrieve=tmem_alloc_barrier,
            is_two_cta=False,
        )

        # Cluster-arrive must come after barrier init and before any cluster wait.
        pipeline_init_arrive(
            cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True,
        )

        # ── Step 6: smem tensors ─────────────────────────────────────────────
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)

        # ── Step 7: gmem tile partitioning ───────────────────────────────────
        bM, bN, bK = mma_tiler

        # gA_mkl: (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        # gB_nkl: (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(mma_tiler, (0, None, None)),
            (None, None, None),
        )
        # gC_mnl: (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # ── Step 8: tiled-mma partitions ─────────────────────────────────────
        # mma_tile_coord_v: V index inside tiled_mma. With 1cta MMA this is 0.
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)  # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)  # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)  # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)

        # ── Step 9: TMA partitions for A/B (per-stage smem source/dest views) ──
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # Slice down to this CTA's M/N/L tile
        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

        # ── Step 10: MMA fragments ───────────────────────────────────────────
        tCrA = tiled_mma.make_fragment_A(sA)  # (MMA, MMA_M, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)  # (MMA, MMA_N, MMA_K, STAGE)
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        # Cluster-wait must come BEFORE TMEM allocation so all CTAs in the
        # cluster are past barrier init.
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # Allocate TMEM (warp 0 issues; all threads wait for the pointer to
        # land in tmem_holding_buf, then re-read it as fp32).
        tmem.allocate(num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # ── Step 11: prologue — prefetch (num_ab_stage - 1) k-tiles ──────────
        # Leave one stage free so the consumer (MMA) can start while the
        # producer fills the remaining stages.
        prefetch_cnt = cutlass.min(num_ab_stage - 1, k_tile_cnt)
        if warp_idx == 0:
            for _i in cutlass.range(prefetch_cnt, unroll=1):
                ab_pipeline.producer_acquire(ab_producer_state)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, ab_producer_state.count)],
                    tAsA[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, ab_producer_state.count)],
                    tBsB[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
                # producer_commit is a no-op for PipelineTmaUmma — TMA itself
                # decrements the transaction barrier.
                ab_pipeline.producer_commit(ab_producer_state)
                ab_producer_state.advance()

        # ── Step 12: mainloop — interleave TMA load and tcgen05.mma ──────────
        # Single warp drives the MMA. Disable initial accumulate (acc=0 first),
        # then enable it for subsequent k-blocks within a tile.
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

        if warp_idx == 0:
            num_kblks = cute.size(tCrA, mode=[2])

            for k_tile_idx in cutlass.range(k_tile_cnt, unroll=1):
                # Issue the next TMA load if there is still work to feed.
                if ab_producer_state.count < k_tile_cnt:
                    ab_pipeline.producer_acquire(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(
                            ab_producer_state),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(
                            ab_producer_state),
                    )
                    ab_pipeline.producer_commit(ab_producer_state)
                    ab_producer_state.advance()

                # Wait for THIS tile's TMA load to finish, then issue MMA.
                ab_pipeline.consumer_wait(ab_consumer_state)
                for kblk in cutlass.range(num_kblks, unroll_full=True):
                    coord = (None, None, kblk, ab_consumer_state.index)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[coord],
                        tCrB[coord],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                # Release the smem stage so the producer can refill it.
                ab_pipeline.consumer_release(ab_consumer_state)
                ab_consumer_state.advance()

            # Tell the epilogue: the (only) accumulator slot is full.
            acc_pipeline.producer_commit(acc_producer_state)

        # ── Step 13: epilogue ────────────────────────────────────────────────
        # Release the TMEM-allocation lock (held since `tmem.allocate`) so the
        # SM can reclaim TMEM after we are done in the epilogue.
        tmem.relinquish_alloc_permit()

        # All threads wait for "acc full".
        acc_pipeline.consumer_wait(acc_consumer_state)

        # ── Step 14: epilogue inlined here ────────────────────────────────────
        # The epilogue must live inside `kernel` rather than a helper @cute.jit
        # method: epi_tile is a tuple of MLIR Layouts created in __call__'s
        # region, and even passing it as Constexpr to a separate @cute.jit
        # function fails MLIR isolation ("value defined outside the region").
        # Keeping everything in one @cute.kernel region sidesteps the issue.

        # TMEM → register tiled copy
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            cta_tile_shape_mnk,
            c_layout, self.c_dtype, self.acc_dtype, epi_tile,
            False,  # use_2cta_instrs
        )
        # tAcc_epi: (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0)],
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # Partition gC over epilogue tiles
        gC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile,
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype,
        )
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

        # register → smem
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s   = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)

        # TMA-store partition (smem → gmem)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(gC_epi, 0, 2),
        )
        bSG_gC = bSG_gC[(None, None, None, *mma_tile_coord_mnl)]
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

        # Pipeline the TMA stores so the smem buffer can ping-pong.
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=num_c_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_cta),
        )

        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in cutlass.range(subtile_cnt):
            # TMEM → register
            cute.copy(
                tiled_copy_t2r,
                tTR_tAcc[(None, None, None, subtile_idx)],
                tTR_rAcc,
            )
            # Apply (relu) epilogue, convert acc → c_dtype
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
            tRS_rC.store(acc_vec)
            # register → smem (ping-pong stage)
            c_buffer = subtile_idx % num_c_stage
            cute.copy(
                tiled_copy_r2s, tRS_rC,
                tRS_sC[(None, None, None, c_buffer)],
            )
            # Make the smem store visible to TMA
            cute.arch.fence_proxy("async.shared", space="cta")
            pipeline.sync(barrier_id=1)
            # TMA store smem → gmem (warp 0 issues)
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c,
                    bSG_sC[(None, c_buffer)],
                    bSG_gC[(None, subtile_idx)],
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            pipeline.sync(barrier_id=1)

        # Wait for the last TMA store to finish (TMA store is async)
        c_pipeline.producer_tail()

        # Free TMEM (only after the epilogue has fully consumed it).
        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)

        # Drain any in-flight TMA loads so we don't leave dangling mbarrier signals.
        if warp_idx == 0:
            ab_pipeline.producer_tail(ab_producer_state)
