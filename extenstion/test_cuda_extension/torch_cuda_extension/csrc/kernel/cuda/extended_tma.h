#pragma once

namespace at {
namespace native {

TORCH_API Tensor extended_add_one_tma_kernel(Tensor input, Tensor output);
TORCH_API Tensor extended_add_one_tma_2d_kernel(Tensor input, Tensor output);
TORCH_API Tensor extended_add_tma_cute_kernel(Tensor input, Tensor output);

} // namespace native
} // namespace at
