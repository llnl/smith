// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "smith/numerics/functional/tensor.hpp"

using namespace smith;

static constexpr double tolerance = 4.0e-16;

// this is intended to mimic GTEST's EXPECT_LT macro, except
// that it works inside a cuda kernel. This macro prints the error message
// on a failed test, and sets an error flag (requires an `int * error` to be
// in the scope where the macro is used)
#define CUDA_EXPECT_LT(value, threshold)                             \
  if (value >= threshold) {                                          \
    printf("%s:%d: Failure\n", __FILE__, __LINE__);                  \
    printf("Expected less than %f, actual: %f\n", threshold, value); \
    *error = 1;                                                      \
  }

__global__ void basic_operations(int* error)
{
  auto I = Identity<3>();
  auto abs = [](auto x) { return (x < 0) ? -x : x; };

  tensor<double, 3> u = {1, 2, 3};
  tensor<double, 4> v = {4, 5, 6, 7};

  // for some reason make_tensor(...) is producing a compiler error about
  // "calling constexpr __device__ function from __host__ __device__".
  // I have a minimal reproducer for NVIDIA to investigate
  tensor<double, 3, 3> A = {{
      {0.0, 2.0, 4.0},
      {1.0, 3.0, 5.0},
      {2.0, 4.0, 6.0},
  }};

  double squared_normA = 111.0;
  CUDA_EXPECT_LT(abs(squared_norm(A) - squared_normA), tolerance);

  tensor<double, 3, 3> symA = {{{0, 1.5, 3}, {1.5, 3, 4.5}, {3, 4.5, 6}}};
  CUDA_EXPECT_LT(abs(squared_norm(sym(A) - symA)), tolerance);

  tensor<double, 3, 3> devA = {{{-3, 2, 4}, {1, 0, 5}, {2, 4, 3}}};
  CUDA_EXPECT_LT(abs(squared_norm(dev(A) - devA)), tolerance);

  tensor<double, 3, 3> invAp1 = {{{-4, -1, 3}, {-1.5, 0.5, 0.5}, {2, 0, -1}}};
  CUDA_EXPECT_LT(abs(squared_norm(inv(A + I) - invAp1)), tolerance);

  tensor<double, 3> Au = {16, 22, 28};
  CUDA_EXPECT_LT(abs(squared_norm(dot(A, u) - Au)), tolerance);

  tensor<double, 3> uA = {8, 20, 32};
  CUDA_EXPECT_LT(abs(squared_norm(dot(u, A) - uA)), tolerance);

  double uAu = 144;
  CUDA_EXPECT_LT(abs(dot(u, A, u) - uAu), tolerance);

  tensor<double, 3, 4> B = {{{0.0, -1.0, -2.0, -3.0}, {3.0, 2.0, 1.0, 0.0}, {6.0, 5.0, 4.0, 3.0}}};

  double uBv = 300;
  CUDA_EXPECT_LT(abs(dot(u, B, v) - uBv), tolerance);
}

TEST(Tensor, BasicOperations)
{
  int* error;
  cudaMallocManaged(&error, sizeof(int));
  *error = 0;

  basic_operations<<<1, 1>>>(error);
  cudaDeviceSynchronize();

  EXPECT_EQ(*error, 0);

  cudaFree(error);
}

__global__ void elasticity(int* error)
{
  auto I = Identity<3>();
  static auto abs = [](auto x) { return (x < 0) ? -x : x; };

  double lambda = 5.0;
  double mu = 3.0;
  tensor<double, 3, 3, 3, 3> C;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 3; l++) {
          C(i, j, k, l) = lambda * (i == j) * (k == l) + mu * ((i == k) * (j == l) + (i == l) * (j == k));
        }
      }
    }
  }

  auto sigma = [=](auto epsilon) { return lambda * tr(epsilon) * I + 2.0 * mu * epsilon; };

  tensor<double, 3, 3> grad_u = {{
      {0.0, 2.0, 4.0},
      {1.0, 3.0, 5.0},
      {2.0, 4.0, 6.0},
  }};

  CUDA_EXPECT_LT(abs(squared_norm(double_dot(C, sym(grad_u)) - sigma(sym(grad_u)))), tolerance);

  auto epsilon = sym(make_dual(grad_u));

  tensor dsigma_depsilon = get_gradient(sigma(epsilon));

  CUDA_EXPECT_LT(abs(squared_norm(dsigma_depsilon - C)), tolerance);
}

TEST(Tensor, Elasticity)
{
  int* error;
  cudaMallocManaged(&error, sizeof(int));
  *error = 0;

  elasticity<<<1, 1>>>(error);
  cudaDeviceSynchronize();

  EXPECT_EQ(*error, 0);

  cudaFree(error);
}

__global__ void navier_stokes(int* error)
{
  auto I = Identity<3>();
  static auto abs = [](auto x) { return (x < 0) ? -x : x; };

  static constexpr double rho = 3.0;
  static constexpr double mu = 2.0;
  auto sigma = [&](auto p, auto v, auto L) { return rho * outer(v, v) + 2.0 * mu * sym(L) - p * I; };

  auto dsigma_dp = [&](auto /*p*/, auto /*v*/, auto /*L*/) { return -1.0 * I; };

  auto dsigma_dv = [&](auto /*p*/, auto v, auto /*L*/) {
    tensor<double, 3, 3, 3> A{};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          A(i, j, k) = rho * ((i == k) * v[j] + (j == k) * v[i]);
        }
      }
    }
    return A;
  };

  auto dsigma_dL = [&](auto /*p*/, auto /*v*/, auto /*L*/) {
    tensor<double, 3, 3, 3, 3> A{};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            A(i, j, k, l) = mu * ((i == k) * (j == l) + (i == l) * (j == k));
          }
        }
      }
    }
    return A;
  };

  double p = 3.14;
  tensor v = {{1.0, 2.0, 3.0}};
  tensor<double, 3, 3> L = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  {
    auto exact = dsigma_dp(p, v, L);
    auto ad = get_gradient(sigma(make_dual(p), v, L));
    CUDA_EXPECT_LT(abs(squared_norm(exact - ad)), tolerance);
  }

  {
    auto exact = dsigma_dv(p, v, L);
    auto ad = get_gradient(sigma(p, make_dual(v), L));
    CUDA_EXPECT_LT(abs(squared_norm(exact - ad)), tolerance);
  }

  {
    auto exact = dsigma_dL(p, v, L);
    auto ad = get_gradient(sigma(p, v, make_dual(L)));
    CUDA_EXPECT_LT(abs(squared_norm(exact - ad)), tolerance);
  }
}

TEST(Tensor, NavierStokes)
{
  int* error;
  cudaMallocManaged(&error, sizeof(int));
  *error = 0;

  navier_stokes<<<1, 1>>>(error);
  cudaDeviceSynchronize();

  EXPECT_EQ(*error, 0);

  cudaFree(error);
}

__global__ void isotropic_operations(int* error)
{
  auto I = Identity<3>();

  double lambda = 5.0;
  double mu = 3.0;

  tensor<double, 3> u = {1, 2, 3};

  tensor<double, 3, 3> A = {{
      {0.0, 2.0, 4.0},
      {1.0, 3.0, 5.0},
      {2.0, 4.0, 6.0},
  }};

  CUDA_EXPECT_LT(abs(squared_norm(dot(I, u) - u)), tolerance);
  CUDA_EXPECT_LT(abs(squared_norm(dot(u, I) - u)), tolerance);

  CUDA_EXPECT_LT(abs(squared_norm(dot(I, A) - A)), tolerance);
  CUDA_EXPECT_LT(abs(squared_norm(dot(A, I) - A)), tolerance);

  CUDA_EXPECT_LT(double_dot(I, A) - tr(A), tolerance);

  auto sigma = [=](auto epsilon) { return lambda * tr(epsilon) * I + 2.0 * mu * epsilon; };

  isotropic_tensor<double, 3, 3, 3, 3> C{lambda, 2 * mu, 0.0};

  auto strain = sym(A);

  CUDA_EXPECT_LT(squared_norm(double_dot(C, strain) - sigma(strain)), tolerance);

  CUDA_EXPECT_LT(det(I) - 1, tolerance);
  CUDA_EXPECT_LT(tr(I) - 3, tolerance);
  CUDA_EXPECT_LT(squared_norm(sym(I) - I), tolerance);
}

TEST(Tensor, IsotropicOperations)
{
  int* error;
  cudaMallocManaged(&error, sizeof(int));
  *error = 0;

  isotropic_operations<<<1, 1>>>(error);
  cudaDeviceSynchronize();

  EXPECT_EQ(*error, 0);

  cudaFree(error);
}
