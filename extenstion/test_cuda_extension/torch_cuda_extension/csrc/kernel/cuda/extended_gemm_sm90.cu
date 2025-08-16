#include <ATen/AccumulateType.h>
#include <torch/library.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/Dispatch.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
#include <cutlass/gemm/threadblock/default_mma_core.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include "extended_gemm.h"
#include "extended_gemm_collective_api.h"

namespace at {
namespace native {

__global__ void _extended_gemm_block_naive_kernel(Half * a_ptr, Half * b_ptr, float * out_ptr, int M, int N, int K, int lda, int ldb, int ldc) {
    // naive cuda implementation, each block calculate a output element
    auto m = blockIdx.x;
    auto n = threadIdx.x;
    if (m < M && n < N) {
      out_ptr[m * ldc + n] = 0.0;
      for (int k=0; k<K; k++) {
        out_ptr[m * ldc + n] += (float)(a_ptr[m * lda + k]) * (float)(b_ptr[n * ldb + k]);
      }
    }
}

template <typename input_dtype, typename output_dtype, bool use_relu, std::enable_if_t<!std::is_same_v<input_dtype, Half>, int> =0>
void _extended_gemm_kernel_low_level_api(
  input_dtype * a_ptr,
  input_dtype * b_ptr,
  output_dtype * out_ptr,
  int M,
  int N,
  int K,
  int lda,
  int ldb,
  int ldc) {
  TORCH_CHECK(false, "None Half input not support yet");
}


template <typename input_dtype, typename output_dtype, bool use_relu, std::enable_if_t<std::is_same_v<input_dtype, Half>, int> =0> // std::enable_if_t<std::is_same_v<input_dtype, Half>, int> =0
void _extended_gemm_kernel_low_level_api(
  input_dtype * a_ptr,
  input_dtype * b_ptr,
  output_dtype * out_ptr,
  int M,
  int N,
  int K,
  int lda,
  int ldb,
  int ldc) {

}

Tensor extended_gemm_sm90_kernel(
    Tensor a,
    Tensor b,
    Tensor out,
    std::string_view epilogue,
    bool transpose_B,
    int64_t api_level) {
    TORCH_CHECK(transpose_B, "for extended_gemm_sm90_kernel, transpose_B must be true");
    auto a_ptr = a.data_ptr();
    auto b_ptr = b.data_ptr();
    auto out_ptr = out.data_ptr();

    // A is: M x K
    // B is: N x K
    // C is: M x N
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(0);

    int lda = a.size(1);
    int ldb = b.size(1);
    int ldc = out.size(1);

    int runtime_cc = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    runtime_cc = prop.major * 100 + prop.minor * 10;
    std::cout<<"runtime_cc is: "<<runtime_cc<<std::endl;
    if(runtime_cc >= 900) {
        std::cout<<"--- hit sm90 above path ----"<<std::endl;
        // std::cout<<std::is_same_v<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Half>, Half><<std::endl;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::BFloat16, at::ScalarType::Half, out.scalar_type(),
            "_extended_gemm_kernel_low_level_api_kernel_impl",
            [&] { 
            // std::cout<<std::is_same_v<scalar_t, Half><<std::endl;
            at::Half* a_ptr = a.data_ptr<at::Half>();
            at::Half* b_ptr = b.data_ptr<at::Half>();
            scalar_t* out_ptr = out.data_ptr<scalar_t>();
            if (epilogue == "relu") {
                // _extended_gemm_kernel_low_level_api<at::Half, scalar_t, true>(a_ptr, b_ptr, out_ptr, M, N, K, lda, ldb, ldc);
            } else {
                _extended_gemm_kernel_low_level_api<at::Half, scalar_t, false>(a_ptr, b_ptr, out_ptr, M, N, K, lda, ldb, ldc);
            }
            });
    } else {
        std::cout<<"--- hit sm90 below path ----"<<std::endl;
        _extended_gemm_block_naive_kernel<<<M,N>>>((Half*)a_ptr, (Half*)b_ptr, (float*)out_ptr, M, N, K, lda, ldb, ldc);
    }

    return out;
}

}
}