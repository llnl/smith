// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(float* out, float* a, float* b, int n)
{
  for (int i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

void checked_op(int err, const char* msg)
{
  if (err) {
    printf("CUDA %s error: %d\n", msg, err);
  }
}

void vector_add(float* out, float* a, float* b, int n)
{
  float* device_a;
  float* device_b;
  checked_op(cudaMalloc((void**)(&device_a), sizeof(float) * n), "malloc a");
  checked_op(cudaMalloc((void**)(&device_b), sizeof(float) * n), "malloc b");
  checked_op(cudaMemcpy(device_a, a, sizeof(float) * n, cudaMemcpyHostToDevice), "copy a");
  checked_op(cudaMemcpy(device_b, b, sizeof(float) * n, cudaMemcpyHostToDevice), "copy b");

  float* device_out;
  checked_op(cudaMalloc((void**)(&device_out), sizeof(float) * n), "malloc out");

  vector_add_kernel<<<1, 1>>>(device_out, device_a, device_b, n);
  cudaMemcpy(out, device_out, sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_out);
}
