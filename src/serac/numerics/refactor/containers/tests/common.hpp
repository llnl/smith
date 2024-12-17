#pragma once

#include "serac/numerics/refactor/containers/ndarray.hpp"

#if __CUDACC__
// dimension should each be less than 10
template < uint32_t rank >
__global__ void patterned_ndarray_kernel(nd::view<double, rank> output, stack::array<uint32_t, rank> dimensions) {
  uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < dimensions[0]) {
    if constexpr (rank > 1) {
      for (int j = 0; j < dimensions[1]; j++) {
        if constexpr (rank > 2) {
          for (int k = 0; k < dimensions[2]; k++) {
            output(i, j, k) = 100 * i + 10 * j + k;
          }
        } else {
          output(i, j) = 10 * i + j;
        }
      }
    } else {
      output(i) = i;
    }
  }
}

template < uint32_t rank >
nd::array<double, rank> make_patterned_ndarray_on_GPU(stack::array<uint32_t, rank> dimensions) {
  static_assert(rank <= 3);

  nd::array<double, rank> output({dimensions}); 
  nd::view<double, rank> v = output;

  uint32_t blocksize = 64;
  uint32_t gridsize = (dimensions[0] + blocksize - 1) / blocksize;
  patterned_ndarray_kernel<<<blocksize, gridsize>>>(v, dimensions);
  cudaDeviceSynchronize();

  return output;
}
#endif

// dimension should each be less than 10
template < uint32_t rank >
nd::array<double, rank> make_patterned_ndarray(stack::array<uint32_t, rank> dimensions) {

  static_assert(rank <= 3);

  nd::array<double, rank> output({dimensions}); 

  for (int i = 0; i < dimensions[0]; i++) {
    if constexpr (rank > 1) {
      for (int j = 0; j < dimensions[1]; j++) {
        if constexpr (rank > 2) {
          for (int k = 0; k < dimensions[2]; k++) {
            output(i, j, k) = 100 * i + 10 * j + k;
          }
        } else {
          output(i, j) = 10 * i + j;
        }
      }
    } else {
      output(i) = i;
    }
  }

  return output;

}
