# SM90 (Hopper) WGMMA + TMA GEMM using CuteDSL
#
# Computes C = A @ B.T
#   A: (M, K) fp16 row-major  (k-major)
#   B: (N, K) fp16 row-major  (k-major, caller pre-transposes)
#   C: (M, N) fp32 row-major  (n-major)
#
# SM90 vs SM80 key differences:
#   Copy:  cp.async (128-bit, per-thread) → TMA (bulk, single-thread, hardware DMA)
#   MMA:   warp-level mma.sync (16×8×16)  → warpgroup WGMMA (wgmma.mma_async, 128 threads)
#   Sync:  cp_async_wait_group            → mbarrier (PipelineTmaAsync producer/consumer)
#   B reg: smem → ldmatrix → registers    → WGMMA reads B directly from smem
#
# Pipeline structure:
#   Producer (warp 0):  TMA load A/B → smem stage[i],  signal mbarrier "full"
#   Consumer (all threads): wait mbarrier "full", WGMMA, signal mbarrier "empty"
#
# Epilogue (relu applied post-kernel by run_gemm_pipeline, not fused here):
#   acc (fp32 registers) → stmatrix → smem → TMA store → gmem C

import math
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda


class GemmPipelineSM90:
    """
    SM90 (Hopper) fp16 GEMM with TMA + WGMMA multi-stage pipeline.

    Tile constraints:
      M: multiple of 64 or 128
      N: multiple of 64, 128, or 256
      K: multiple of 64  (bK = mma_k*4 = 16*4 = 64 for fp16)
    """

    # relu is not fused into the WGMMA epilogue; run_gemm_pipeline applies it
    # as a separate torch.relu_() call after the kernel completes.
    fused_epilogue = False

    def __init__(self, tile_shape_mn=(128, 128), cluster_shape_mn=(1, 1)):
        self.tile_shape_mn   = tile_shape_mn
        self.cluster_shape_mn = cluster_shape_mn

        self.ab_dtype  = cutlass.Float16
        self.acc_dtype = cutlass.Float32
        self.c_dtype   = cutlass.Float32

        # Use 2 warpgroups for large tiles to avoid register spill.
        # (2,1,1) → 2 warpgroups × 128 threads = 256 threads per CTA.
        self.atom_layout_mnk = (
            (2, 1, 1) if tile_shape_mn[0] > 64 and tile_shape_mn[1] > 128
            else (1, 1, 1)
        )
        self.num_threads_per_warpgroup = 128
        self.mma_warpgroups  = math.prod(self.atom_layout_mnk)
        self.threads_per_cta = self.mma_warpgroups * self.num_threads_per_warpgroup

        # SM90 provides 228 KB smem per SM; use all of it for pipeline stages.
        self.smem_capacity    = utils.get_smem_capacity_in_bytes("sm_90")
        self.occupancy        = 1
        self.buffer_align_bytes = 1024   # TMA requires 1 KB alignment for smem buffers

    # ──────────────────────────────────────────────────────────────────────────
    # Dispatch helpers (called by run_gemm_pipeline)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, mA, mB, mC, ep_op, device):
        """Compile the kernel for a given problem shape and return the compiled fn."""
        import torch
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        return cute.compile(cls(), mA, mB, mC, stream)

    @classmethod
    def invoke(cls, compiled, mA, mB, mC, device):
        """Execute a previously compiled kernel on the current CUDA stream."""
        import torch
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        compiled(mA, mB, mC, stream)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage count heuristic
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_stages(tile_shape_mnk, a_dtype, b_dtype, smem_capacity, occupancy):
        """
        Fill smem with as many pipeline stages as possible.

        epi_stage=4: epilogue reuses sA smem after the mainloop, so its stages
        don't add to the smem budget (they alias the mainloop buffer).
        mbar_bytes: space for ab_stage × 2 mbarriers (full + empty per stage).
        """
        epi_stage = 4
        ab_bytes_per_stage = (
            tile_shape_mnk[0] * tile_shape_mnk[2] * a_dtype.width // 8
            + tile_shape_mnk[1] * tile_shape_mnk[2] * b_dtype.width // 8
        )
        mbar_bytes = 1024
        ab_stage = (smem_capacity // occupancy - mbar_bytes) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _compute_grid(mC, tile_shape_mnk, cluster_shape_mn):
        grid = cute.ceil_div(mC.shape, (tile_shape_mnk[0], tile_shape_mnk[1]))
        return (cute.size(grid[0]), cute.size(grid[1]), 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Host JIT: build TMA descriptors, smem layouts, launch kernel
    # ──────────────────────────────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        a_layout = utils.LayoutEnum.from_tensor(mA)   # ROW_MAJOR: K is contiguous
        b_layout = utils.LayoutEnum.from_tensor(mB)   # ROW_MAJOR: K is contiguous
        c_layout = utils.LayoutEnum.from_tensor(mC)   # ROW_MAJOR: N is contiguous

        # ── WGMMA tiled MMA ───────────────────────────────────────────────────
        # make_trivial_tiled_mma constructs an SM90 WGMMA MMA atom:
        #   fp16×fp16 → fp32, k-major A, k-major B
        #   mma_k = 16 (fp16 K-depth per WGMMA instruction)
        #   bK = mma_k × 4 = 64 (4 WGMMA k-steps per smem stage)
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            mA.element_type,
            mB.element_type,
            a_layout.sm90_mma_major_mode(),
            b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mn[1]),
        )
        mma_k        = cute.size(tiled_mma.shape_mnk, mode=[2])
        bK           = mma_k * 4
        tile_shape_mnk = (*self.tile_shape_mn, bK)

        # ── Stage count ───────────────────────────────────────────────────────
        ab_stage, epi_stage = self._compute_stages(
            tile_shape_mnk, mA.element_type, mB.element_type,
            self.smem_capacity, self.occupancy,
        )

        # ── Epilogue tile shape ───────────────────────────────────────────────
        is_cooperative = (self.atom_layout_mnk == (2, 1, 1))
        epi_tile = sm90_utils.compute_tile_shape_or_override(
            tile_shape_mnk, mC.element_type, is_cooperative=is_cooperative
        )

        # ── Smem layouts (SM90-specific swizzle for TMA alignment) ────────────
        # The TMA engine requires specific smem address alignment (128-byte).
        # sm90_utils helpers choose the right swizzle pattern for each operand.
        sA_layout = sm90_utils.make_smem_layout_a(
            a_layout, tile_shape_mnk, mA.element_type, ab_stage)
        sB_layout = sm90_utils.make_smem_layout_b(
            b_layout, tile_shape_mnk, mB.element_type, ab_stage)
        sC_layout = sm90_utils.make_smem_layout_epi(
            mC.element_type, c_layout, epi_tile, epi_stage)

        # ── TMA atoms and descriptors ─────────────────────────────────────────
        # make_tiled_tma_atom creates a TMA descriptor on the host.
        # The descriptor encodes: src pointer, shape, stride, swizzle mode.
        # cute.slice_(layout, (None, None, 0)) extracts one stage's layout to
        # tell the TMA engine the per-transfer smem footprint.
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mA,
            cute.slice_(sA_layout, (None, None, 0)),
            (tile_shape_mnk[0], bK),
            num_multicast=1,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mB,
            cute.slice_(sB_layout, (None, None, 0)),
            (tile_shape_mnk[1], bK),
            num_multicast=1,
        )
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            mC,
            cute.slice_(sC_layout, (None, None, 0)),
            epi_tile,
        )

        # ── Shared memory layout ──────────────────────────────────────────────
        # mainloop_pipeline_array_ptr: storage for ab_stage × 2 mbarriers
        #   (each stage needs one "full" and one "empty" barrier, each 8 bytes)
        # sA, sB: pipeline-staged buffers (ab_stage slots × one bM×bK tile each)
        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)],
                self.buffer_align_bytes,
            ]

        self._shared_storage = SharedStorage

        grid = self._compute_grid(mC, tile_shape_mnk, self.cluster_shape_mn)
        cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))

        self.kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c,
            tiled_mma,
            cta_layout_mnk,
            sA_layout, sB_layout, sC_layout,
            ab_stage, epi_stage, epi_tile,
            c_layout,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # GPU kernel
    # ──────────────────────────────────────────────────────────────────────────

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        ab_stage:  cutlass.Constexpr,
        epi_stage: cutlass.Constexpr,
        epi_tile:  cutlass.Constexpr,
        c_layout:  cutlass.Constexpr,
    ):
        # make_warp_uniform broadcasts the value across all threads in a warp,
        # ensuring divergence-free conditionals keyed on warp_idx.
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        bM  = self.tile_shape_mn[0]
        bN  = self.tile_shape_mn[1]
        bK  = cute.size(tiled_mma.shape_mnk, mode=[2]) * 4

        # ── Step 1: prefetch TMA descriptors ─────────────────────────────────
        # Warp 0 prefetches both TMA descriptors into L2 to hide descriptor
        # fetch latency before the first TMA load issues.
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ── Step 2: select this CTA's gmem tiles ─────────────────────────────
        # gA: (bM, bK, k_tile_count)  — M-tile for this CTA, K iterated
        # gB: (bN, bK, k_tile_count)  — N-tile for this CTA, K iterated
        # gC: (bM, bN)                — output tile for this CTA
        gA = cute.local_tile(mA, (bM, bK), (bidx, None))
        gB = cute.local_tile(mB, (bN, bK), (bidy, None))
        gC = cute.local_tile(mC, (bM, bN), (bidx, bidy))

        # ── Step 3: smem allocation + mbarrier pipeline init ─────────────────
        smem    = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._shared_storage)

        mbar_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # TMA transaction count: how many bytes a single TMA load writes.
        # The mbarrier waits for exactly this many bytes before signaling "full".
        sA_one = cute.slice_(sA_layout, (None, None, 0))
        sB_one = cute.slice_(sB_layout, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.ab_dtype, sA_one)
            + cute.size_in_bytes(self.ab_dtype, sB_one)
        )

        # PipelineTmaAsync: producer = warp 0 (issues TMA loads),
        #                   consumer = all threads (run WGMMA).
        # Each stage has an mbarrier pair (full/empty).  The TMA engine
        # decrements the mbarrier by tma_copy_bytes when the DMA completes;
        # consumer_wait spins until the count hits zero.
        num_warps = self.threads_per_cta // 32
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_warps
            ),
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )
        # Cluster arrive: for cluster=(1,1) this is a no-op but required by API.
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ── Step 4: smem tensors ──────────────────────────────────────────────
        # get_tensor(outer, swizzle=inner) reassembles the ComposedLayout:
        #   outer holds the stage dimension; inner holds the per-stage swizzle.
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
        # sC reuses sA's smem (epilogue starts only after mainloop completes,
        # so there's no overlap between mainloop smem and epilogue smem use).
        sC_ptr = cute.recast_ptr(sA.iterator, sC_layout.inner, dtype=self.c_dtype)
        sC = cute.make_tensor(sC_ptr, sC_layout.outer)

        # ── Step 5: TMA partitions ────────────────────────────────────────────
        # tma_partition creates the smem/gmem views for the TMA copy.
        # group_modes(t, 0, 2) collapses the first two modes (tile dims) into
        # one, giving ((bM×bK), stage) for smem and ((bM×bK), k_tiles) for gmem.
        # For cluster=(1,1) there is no multicast: cta_coord=0, cta_layout=size1.
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,                          # CTA coordinate in cluster (no multicast)
            cute.make_layout(1),        # single-CTA cluster layout
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        # ── Step 6: WGMMA fragments ───────────────────────────────────────────
        # Each warpgroup uses a slice of tiled_mma.
        # WGMMA reads A from smem directly; B is also read from smem.
        wg_layout = cute.make_layout(
            self.mma_warpgroups, stride=self.num_threads_per_warpgroup
        )
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warpgroup
        )
        thr_mma = tiled_mma.get_slice(wg_layout(warp_group_idx))

        tCsA = thr_mma.partition_A(sA)   # (MMA_A, MMA_M, MMA_K, ab_stage)
        tCsB = thr_mma.partition_B(sB)   # (MMA_B, MMA_N, MMA_K, ab_stage)
        tCgC = thr_mma.partition_C(gC)   # (MMA_C, MMA_M, MMA_N)

        # make_fragment_A/B allocate register fragments matching WGMMA layout.
        # Unlike SM80, WGMMA reads operands from smem directly in hardware;
        # the fragment here is only for the register-level A pointer (sm90a) or
        # is a smem-backed descriptor reference.
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        acc  = cute.make_rmem_tensor(tCgC.shape, self.acc_dtype)

        # Cluster barrier init done; now wait for peer CTAs in the cluster.
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ── Step 7: prologue — TMA prefetch ab_stage tiles ───────────────────
        # Fill all pipeline stages with async TMA copies before the mainloop.
        # Only warp 0 issues TMA loads; all threads participate in WGMMA.
        k_tile_cnt   = cute.size(gA, mode=[2])    # K // bK
        prefetch_cnt = cutlass.max(cutlass.min(ab_stage, k_tile_cnt), 0)

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, ab_stage
        )
        if warp_idx == 0:
            for _prefetch in cutlass.range(prefetch_cnt, unroll=1):
                # producer_acquire waits for the "empty" mbarrier (stage is free)
                # and arms the "full" transaction barrier with tma_copy_bytes.
                mainloop_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, producer_state.count)],
                    tAsA[(None, producer_state.index)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        producer_state
                    ),
                    mcast_mask=0,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, producer_state.count)],
                    tBsB[(None, producer_state.index)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        producer_state
                    ),
                    mcast_mask=0,
                )
                # producer_commit is a NOP for PipelineTmaAsync (the mbarrier
                # is signaled automatically by the TMA engine on completion).
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

        # ── Step 8: mainloop — WGMMA + concurrent TMA load ───────────────────
        # k_pipe_mmas=1: run one prologue MMA before interleaving with TMA loads.
        # This ensures the first WGMMA has data ready while TMA fetches k+1.
        k_pipe_mmas = 1
        consumer_read_state    = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, ab_stage
        )
        consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, ab_stage
        )

        # consumer_try_wait: non-blocking poll of the mbarrier.  Returns True
        # if the stage is already full (copy done); otherwise returns False and
        # consumer_wait will spin until it completes.
        peek_status = cutlass.Boolean(1)
        if consumer_read_state.count < k_tile_cnt:
            peek_status = mainloop_pipeline.consumer_try_wait(consumer_read_state)

        num_k_blocks = cute.size(tCrA, mode=[2])  # bK // mma_k (= 4)

        # Prologue MMA: consume the first k_pipe_mmas stages without issuing
        # new TMA loads (new loads come in the main loop below).
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        for _prologue in cutlass.range_constexpr(k_pipe_mmas):
            mainloop_pipeline.consumer_wait(consumer_read_state, peek_status)
            # warpgroup.fence: ensures prior smem stores are visible before WGMMA.
            cute.nvgpu.warpgroup.fence()
            for k_blk in cutlass.range(num_k_blocks, unroll_full=True):
                coord = (None, None, k_blk, consumer_read_state.index)
                # WGMMA: async warpgroup MMA.  acc += A[k_blk] × B[k_blk].
                # The instruction is asynchronous: must commit+wait before reading acc.
                cute.gemm(tiled_mma, acc, tCrA[coord], tCrB[coord], acc)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            # commit_group: seals the outstanding WGMMA instructions into a group.
            cute.nvgpu.warpgroup.commit_group()
            consumer_read_state.advance()
            peek_status = cutlass.Boolean(1)
            if consumer_read_state.count < k_tile_cnt:
                peek_status = mainloop_pipeline.consumer_try_wait(consumer_read_state)

        # Main K-tile loop: interleave WGMMA (consumer) and TMA load (producer).
        for _k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
            # Wait for the current stage's TMA copy to complete.
            mainloop_pipeline.consumer_wait(consumer_read_state, peek_status)

            cute.nvgpu.warpgroup.fence()
            for k_blk in cutlass.range(num_k_blocks, unroll_full=True):
                coord = (None, None, k_blk, consumer_read_state.index)
                cute.gemm(tiled_mma, acc, tCrA[coord], tCrB[coord], acc)
            cute.nvgpu.warpgroup.commit_group()
            # wait_group(k_pipe_mmas): wait for all but the last k_pipe_mmas WGMMA
            # groups to complete.  This allows the current WGMMA to run while
            # TMA fetch below overlaps with it.
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

            # Release the consumed stage so the producer can refill it.
            mainloop_pipeline.consumer_release(consumer_release_state)
            consumer_read_state.advance()
            consumer_release_state.advance()

            peek_status = cutlass.Boolean(1)
            if consumer_read_state.count < k_tile_cnt:
                peek_status = mainloop_pipeline.consumer_try_wait(consumer_read_state)

            # Warp 0 issues the next TMA load while WGMMA runs above (overlap).
            if warp_idx == 0 and producer_state.count < k_tile_cnt:
                mainloop_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, producer_state.count)],
                    tAsA[(None, producer_state.index)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        producer_state
                    ),
                    mcast_mask=0,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, producer_state.count)],
                    tBsB[(None, producer_state.index)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        producer_state
                    ),
                    mcast_mask=0,
                )
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

        # Drain all remaining in-flight WGMMA instructions.
        cute.nvgpu.warpgroup.wait_group(0)

        # Sync all threads before epilogue (smem sA will be reused as sC).
        cute.arch.sync_threads()

        # ── Step 9: epilogue — acc(reg) → sC(smem) → gmem via TMA store ──────
        # copy_atom_r2s: stmatrix instruction to move fp16/fp32 from registers
        # to smem in the layout expected by TMA store.
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                c_layout.is_m_major_c(), 4
            ),
            self.c_dtype,
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s    = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)

        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD       = thr_copy_r2s.partition_D(sC)
        tRS_rAcc     = tiled_copy_r2s.retile(acc)

        rD_layout    = cute.make_layout(cute.shape(thr_copy_r2s.partition_S(sC))[:3])
        tRS_rD       = cute.make_rmem_tensor_like(rD_layout, self.acc_dtype)
        tRS_rD_out   = cute.make_rmem_tensor_like(rD_layout, self.c_dtype)
        size_rD      = cute.size(tRS_rD)

        # TMA store partition: maps smem epilogue tile to gmem.
        sepi_for_tma  = cute.group_modes(sC, 0, 2)
        tCgC_for_tma  = cute.zipped_divide(gC, epi_tile)
        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sepi_for_tma,
            tCgC_for_tma,
        )
        epi_tile_num    = cute.size(tCgC_for_tma, mode=[1])
        epi_tile_shape  = tCgC_for_tma.shape[1]
        epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))

        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=epi_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_cta
            ),
        )

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # Scatter accumulator slice into D registers.
            for v in cutlass.range_constexpr(size_rD):
                tRS_rD[v] = tRS_rAcc[epi_idx * size_rD + v]

            # Type conversion: fp32 accumulator → fp32 output (identity here).
            tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))

            # stmatrix: registers → smem (ping-pong across epi_stage slots).
            epi_buf = epi_idx % cute.size(tRS_sD, mode=[3])
            cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buf)])

            # fence_proxy: ensure stmatrix writes are visible to TMA store.
            cute.arch.fence_proxy("async.shared", space="cta")
            pipeline.sync(barrier_id=1)

            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            if warp_idx == 0:
                # TMA store: one warp issues the bulk smem→gmem DMA.
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buf)],
                    bSG_gD[(None, gmem_coord)],
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            pipeline.sync(barrier_id=1)

        if warp_idx == 0:
            c_pipeline.producer_tail()
