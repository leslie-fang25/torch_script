#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda/pipeline>
#include <cudaTypedefs.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include <ATen/AccumulateType.h>
#include <torch/library.h>
#include <ATen/functorch/BatchRulesHelper.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/Dispatch.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
#include <cutlass/gemm/threadblock/default_mma_core.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>

#include "extended_tma.h"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace ptx = cuda::ptx;

static constexpr size_t buf_len = 1024;
__global__ void _extended_add_one_tma_kernel(int * in_ptr, int * out_ptr) {

#if __CUDA_ARCH__ >= 900
  int offset = blockIdx.x * buf_len;

  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
    init(&bar, blockDim.x);                      // a)
    ptx::fence_proxy_async(ptx::space_shared);   // b)
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory.
  if (threadIdx.x == 0) {
    // 3a. cuda::memcpy_async arrives on the barrier and communicates
    //     how many bytes are expected to come in (the transaction count)
    memcpy_async(
        smem_data, 
        in_ptr + offset,
        sizeof(smem_data),
        bar
    );
  }
  // 3b. All threads arrive on the barrier
  barrier::arrival_token token = bar.arrive();
  
  // 3c. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);   // b)
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk(
        ptx::space_global,
        ptx::space_shared,
        out_ptr + offset, smem_data, sizeof(smem_data));
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
#endif
}

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  // CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
  cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status);
  assert(driver_status == cudaDriverEntryPointSuccess);

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

static constexpr int SMEM_WIDTH = 32;
static constexpr int SMEM_HEIGHT = 32;
__global__ void _extended_add_one_tma_2d_kernel(
    const __grid_constant__ CUtensorMap in_tensor_map,
    const __grid_constant__ CUtensorMap out_tensor_map,
    int g_w,
    int g_h) {

#if __CUDA_ARCH__ >= 900

  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);
    // Make initialized barrier visible in async proxy.
    cde::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int block_id = blockIdx.x;
  int n_block_w = g_w / SMEM_WIDTH;
  int n_block_h = g_h / SMEM_HEIGHT;
  // The top-left corner of the tile is indicated by the indices x and y.
  int x = block_id % n_block_w * SMEM_WIDTH;
  int y = block_id / n_block_w * SMEM_HEIGHT;

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    // Initiate bulk tensor copy.
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &in_tensor_map, x, y, bar);
    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }
  // Wait for the data to have arrived.
  bar.wait(std::move(token));

  // Symbolically modify a value in shared memory.
  for (int i=0; i<SMEM_HEIGHT; i++) {
    smem_buffer[i][threadIdx.x] += 1;
  }

  // Wait for shared memory writes to be visible to TMA engine.
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&out_tensor_map, x, y, &smem_buffer);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cde::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cde::cp_async_bulk_wait_group_read<0>();
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }

#endif
}

template <int SMEM_HEIGHT, int SMEM_WIDTH, class TmaLoad, class GmemInTensor, class TmaStore, class GmemOutTensor>
__global__ void _extended_extended_add_tma_cute_kernel(
  __grid_constant__ const TmaLoad tma_load,
  GmemInTensor gmem_in_tensor,
 __grid_constant__ const TmaStore tma_store,
  GmemOutTensor gmem_out_tensor) {

#if __CUDA_ARCH__ >= 900
  constexpr int tma_transaction_bytes = SMEM_HEIGHT * SMEM_WIDTH * sizeof(int);
 
  __shared__ int smem_data[SMEM_HEIGHT * SMEM_WIDTH];
  __shared__ uint64_t tma_load_mbar;
 
  auto smem_layout = cute::make_layout(cute::make_shape(SMEM_HEIGHT, SMEM_WIDTH), cute::LayoutRight{});
  auto smem_tensor = cute::make_tensor(cute::make_smem_ptr(smem_data), smem_layout);
 
  if (threadIdx.x == 0) {
    auto gmem_tensor_coord = tma_load.get_tma_tensor(cute::shape(gmem_in_tensor));
 
    auto gmem_tensor_coord_cta = cute::local_tile(
        gmem_tensor_coord,
        cute::Tile<cute::Int<SMEM_HEIGHT>, cute::Int<SMEM_WIDTH>>{},
        cute::make_coord(blockIdx.x, blockIdx.y)); // 这里指定了 这个要copy的全局小块，在全局大块中的位置
 
    cute::initialize_barrier(tma_load_mbar, /* arrival count */ 1);
 
    cute::set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);
 
    auto tma_load_per_cta = tma_load.get_slice(0);
    cute::copy(tma_load.with(tma_load_mbar),
         tma_load_per_cta.partition_S(gmem_tensor_coord_cta),
         tma_load_per_cta.partition_D(smem_tensor));
  }
  __syncthreads();
  cute::wait_barrier(tma_load_mbar, /* phase */ 0);
 
  // after this line, the TMA load is finished
  for (int i=0; i<SMEM_HEIGHT; i++) {
    smem_data[threadIdx.x + i * SMEM_WIDTH] += 1;
  }

  __syncthreads();
  cute::tma_store_fence();
  if (threadIdx.x == 0) {
    auto gmem_tensor_coord = tma_store.get_tma_tensor(cute::shape(gmem_out_tensor));
 
    auto gmem_tensor_coord_cta = cute::local_tile(
      gmem_tensor_coord,
      cute::Tile<cute::Int<SMEM_HEIGHT>, cute::Int<SMEM_WIDTH>>{},
      cute::make_coord(blockIdx.x, blockIdx.y));
 
    auto tma_store_per_cta = tma_store.get_slice(0);
    copy(tma_store,
         tma_store_per_cta.partition_S(smem_tensor),
         tma_store_per_cta.partition_D(gmem_tensor_coord_cta));
    cute::tma_store_arrive();
  }
  cute::tma_store_wait<0>();
#endif
}

namespace at {
namespace native {

Tensor extended_add_one_tma_kernel(
    Tensor input,
    Tensor output) {

    auto in_ptr = input.data_ptr();
    auto out_ptr = output.data_ptr();
    int num_elements = input.size(0) * input.size(1);
    auto N1 = num_elements / buf_len;
    auto N2 = buf_len;
    // size_t sharedMemSize = N2 * sizeof(int);

    std::cout<<"---- before submit kernel ----"<<std::endl;

    _extended_add_one_tma_kernel<<<N1, N2>>>((int*)in_ptr, (int*)out_ptr);
    return output;
}

Tensor extended_add_one_tma_2d_kernel(
    Tensor input,
    Tensor output) {

    auto in_ptr = input.data_ptr();
    auto out_ptr = output.data_ptr();
    int num_elements = input.size(0) * input.size(1);
    int GMEM_WIDTH = input.size(0);
    int GMEM_HEIGHT = input.size(1);

    std::cout<<"---- before submit kernel ----"<<std::endl;


    CUtensorMap in_tensor_map{};
    CUtensorMap out_tensor_map{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride[rank] = {1, 1};

    // Get a function pointer to the cuTensorMapEncodeTiled driver API.
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

    // Create the tensor descriptor.
    CUresult res = cuTensorMapEncodeTiled(
        &in_tensor_map,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank,                       // cuuint32_t tensorRank,
        in_ptr,                     // void *globalAddress,
        size,                       // const cuuint64_t *globalDim,
        stride,                     // const cuuint64_t *globalStrides,
        box_size,                   // const cuuint32_t *boxDim,
        elem_stride,                // const cuuint32_t *elementStrides,
        // Interleave patterns can be used to accelerate loading of values that
        // are less than 4 bytes long.
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        // Swizzling can be used to avoid shared memory bank conflicts.
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        // L2 Promotion can be used to widen the effect of a cache-policy to a wider
        // set of L2 cache lines.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        // Any element that is outside of bounds will be set to zero by the TMA transfer.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

        // Create the tensor descriptor.
    CUresult res_1 = cuTensorMapEncodeTiled(
        &out_tensor_map,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank,                       // cuuint32_t tensorRank,
        out_ptr,                     // void *globalAddress,
        size,                       // const cuuint64_t *globalDim,
        stride,                     // const cuuint64_t *globalStrides,
        box_size,                   // const cuuint32_t *boxDim,
        elem_stride,                // const cuuint32_t *elementStrides,
        // Interleave patterns can be used to accelerate loading of values that
        // are less than 4 bytes long.
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        // Swizzling can be used to avoid shared memory bank conflicts.
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        // L2 Promotion can be used to widen the effect of a cache-policy to a wider
        // set of L2 cache lines.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        // Any element that is outside of bounds will be set to zero by the TMA transfer.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    int block_num = num_elements / (SMEM_HEIGHT * SMEM_WIDTH);
    _extended_add_one_tma_2d_kernel<<<block_num, SMEM_WIDTH>>>(in_tensor_map, out_tensor_map, GMEM_WIDTH, GMEM_HEIGHT);
    return output;
}


Tensor extended_add_tma_cute_kernel(
    Tensor input,
    Tensor output) {

    // using namespace cute;

    const int* in_ptr = (int*)(input.data_ptr());
    int* out_ptr = (int*)(output.data_ptr());
    int num_elements = input.size(0) * input.size(1);
    int GMEM_WIDTH = input.size(0);
    int GMEM_HEIGHT = input.size(1);

    std::cout<<"---- before the start of extended_add_tma_cute_kernel ----"<<std::endl;

    auto M = int(GMEM_HEIGHT);
    auto K = int(GMEM_WIDTH);

    int ldA = GMEM_WIDTH;
    auto dA = cute::make_stride(ldA, cute::Int<1>{});
    auto in_shape = cute::make_shape(M, K);


    // create the GMEM tensor
    // auto gmem_layout = cute::make_layout(cute::make_shape(M, K), cute::LayoutRight{});
    // auto gmem_tensor = cute::make_tensor(cute::make_gmem_ptr(T), gmem_layout);
    cute::Tensor mA = cute::make_tensor(in_ptr, in_shape, dA);
    cute::Tensor mB = cute::make_tensor(out_ptr, in_shape, dA);
    
    // cute::Tensor mB = cute::make_tensor(cute::make_gmem_ptr(out_ptr), in_shape, dA);
    auto bM = cute::Int<SMEM_HEIGHT>{};
    auto bK = cute::Int<SMEM_WIDTH>{};
    auto smem_layout = cute::make_layout(cute::make_shape(bM, bK), cute::LayoutRight{});
    auto tma_load = cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        mA,
        smem_layout);
    
    auto tma_store = cute::make_tma_copy(
        cute::SM90_TMA_STORE{},
        mB,
        smem_layout);

    int block_num = num_elements / (SMEM_HEIGHT * SMEM_WIDTH);
    int n_block_w = GMEM_WIDTH / SMEM_WIDTH;
    int n_block_h = GMEM_HEIGHT / SMEM_HEIGHT;
    dim3 numBlocks(n_block_h, n_block_w);
    _extended_extended_add_tma_cute_kernel<SMEM_HEIGHT, SMEM_WIDTH><<<numBlocks, SMEM_WIDTH>>>(
        tma_load, mA, tma_store, mB);
    return output;
}
}
}
