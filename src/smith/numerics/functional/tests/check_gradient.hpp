// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include <iostream>

#include "mfem.hpp"

#include "smith/infrastructure/accelerator.hpp"
#include "smith/smith_config.hpp"
#include "smith/numerics/functional/functional.hpp"

template <typename T>
void debug_sparse_matrix(smith::Functional<T>& f, double t, const mfem::Vector& U,
                         [[maybe_unused]] double epsilon = 1.0e-4)
{
  mfem::Vector dU(U.Size());
  dU = 0.0;

  auto [value, dfdU] = f(t, smith::differentiate_wrt(U));
  std::unique_ptr<mfem::HypreParMatrix> dfdU_matrix = assemble(dfdU);

  std::cout << "{";
  for (int i = 0; i < U.Size(); i++) {
    dU[i] = 1;
    mfem::Vector df_jvp = dfdU(dU);  // matrix-free

    std::cout << "{";
    for (int j = 0; j < df_jvp.Size(); j++) {
      std::cout << df_jvp[j];
      if (j != df_jvp.Size() - 1) {
        std::cout << ",";
      } else {
        std::cout << " ";
      }
    }
    std::cout << "}";
    if (i != U.Size() - 1) {
      std::cout << ",\n";
    } else {
      std::cout << "\n";
    }

    dU[i] = 0;
  }
  std::cout << "}" << std::endl;

  dfdU_matrix->Print("K.mtx");
}

template <typename T, smith::ExecutionSpace exec = smith::ExecutionSpace::CPU>
void check_gradient(smith::Functional<T, exec>& f, double t, mfem::Vector& U, double epsilon = 1.0e-4)
{
  int seed = 42;

  mfem::Vector dU(U.Size());
  // Set memory backend to device if using GPU execution.
  if constexpr (exec == smith::ExecutionSpace::GPU) {
    dU.UseDevice(true);
  }
  dU.Randomize(seed);

  auto [value, dfdU] = f(t, smith::differentiate_wrt(U));
  std::unique_ptr<mfem::HypreParMatrix> dfdU_matrix = assemble(dfdU);

  // jacobian vector products
  mfem::Vector df_jvp1;
  df_jvp1.UseDevice(true);
  df_jvp1 = dfdU(dU);  // matrix-free

  mfem::Vector df_jvp2(df_jvp1.Size());
  if constexpr (exec == smith::ExecutionSpace::GPU) {
    df_jvp2.UseDevice(true);
  }
  dfdU_matrix->Mult(dU, df_jvp2);  // sparse matvec

  if (df_jvp1.Norml2() != 0) {
    double relative_error = df_jvp1.DistanceTo(df_jvp2.GetData()) / df_jvp1.Norml2();
    EXPECT_NEAR(0., relative_error, 5.e-6);
  }

  // {f(x - 2 * h), f(x - h), f(x), f(x + h), f(x + 2 * h)}
  mfem::Vector f_values[5];
  for (int i = 0; i < 5; i++) {
    auto U_plus_small = U;
    U_plus_small.Add((i - 2) * epsilon, dU);
    f_values[i] = f(t, U_plus_small);
  }

  // forward-difference approximations
  mfem::Vector df1_fd[2];
  df1_fd[0] = f_values[4];
  df1_fd[0] -= f_values[2];
  df1_fd[0] /= (2.0 * epsilon);

  df1_fd[1] = f_values[3];
  df1_fd[1] -= f_values[2];
  df1_fd[1] /= epsilon;

  // center-difference approximations
  mfem::Vector df1_cd[2];
  df1_cd[0] = f_values[4];
  df1_cd[0] -= f_values[0];
  df1_cd[0] /= (4.0 * epsilon);

  df1_cd[1] = f_values[3];
  df1_cd[1] -= f_values[1];
  df1_cd[1] /= (2.0 * epsilon);

  // this makes e1-e4 relative errors when |df| > 0
  //              and absolute errors when |df| ~ 0
  double denominator = (df_jvp1.Norml2() < 1.0e-10) ? 1.0 : df_jvp1.Norml2();

  // halving epsilon should make the error decrease
  // by about a factor of two for the forward-difference stencil
  double e1 = df1_fd[0].DistanceTo(df_jvp1.GetData()) / denominator;
  double e2 = df1_fd[1].DistanceTo(df_jvp1.GetData()) / denominator;
  EXPECT_TRUE(fabs(e1 / e2 - 2.0) < 0.1 || fmin(e1, e2) < 1.0e-9);

  // halving epsilon should make the error decrease
  // by about a factor of four for the center-difference stencil
  double e3 = df1_cd[0].DistanceTo(df_jvp1.GetData()) / denominator;
  double e4 = df1_cd[1].DistanceTo(df_jvp1.GetData()) / denominator;
  EXPECT_TRUE((fabs(e3 / e4 - 4.0) < 0.1) || fmin(e3, e4) < 1.0e-9);
}

template <typename T>
void check_gradient(smith::Functional<T>& f, double t, const mfem::Vector& U, const mfem::Vector& dU_dt,
                    double epsilon = 1.0e-4)
{
  int seed = 42;

  mfem::Vector dU(U.Size());
  dU.Randomize(seed);

  mfem::Vector ddU_dt(dU_dt.Size());
  ddU_dt.Randomize(seed + 1);

  {
    auto [value, dfdU] = f(t, smith::differentiate_wrt(U), dU_dt);
    std::unique_ptr<mfem::HypreParMatrix> dfdU_matrix = assemble(dfdU);

    // jacobian vector products
    mfem::Vector df_jvp1 = dfdU(dU);  // matrix-free

    mfem::Vector df_jvp2(df_jvp1.Size());
    dfdU_matrix->Mult(dU, df_jvp2);  // sparse matvec

    if (df_jvp1.Norml2() < 1.0e-13) {
      double absolute_error = df_jvp1.DistanceTo(df_jvp2.GetData());
      EXPECT_NEAR(0., absolute_error, 5.e-14);
    } else {
      double relative_error = df_jvp1.DistanceTo(df_jvp2.GetData()) / df_jvp1.Norml2();
      EXPECT_NEAR(0., relative_error, 5.e-14);
    }

    // {f(x - 2 * h), f(x - h), f(x), f(x + h), f(x + 2 * h)}
    mfem::Vector f_values[5];
    for (int i = 0; i < 5; i++) {
      auto U_plus_small = U;
      U_plus_small.Add((i - 2) * epsilon, dU);
      f_values[i] = f(t, U_plus_small, dU_dt);
    }

    // forward-difference approximations
    mfem::Vector df1_fd[2];
    df1_fd[0] = f_values[4];
    df1_fd[0] -= f_values[2];
    df1_fd[0] /= (2.0 * epsilon);

    df1_fd[1] = f_values[3];
    df1_fd[1] -= f_values[2];
    df1_fd[1] /= epsilon;

    // center-difference approximations
    mfem::Vector df1_cd[2];
    df1_cd[0] = f_values[4];
    df1_cd[0] -= f_values[0];
    df1_cd[0] /= (4.0 * epsilon);

    df1_cd[1] = f_values[3];
    df1_cd[1] -= f_values[1];
    df1_cd[1] /= (2.0 * epsilon);

    // this makes e1-e4 relative errors when |df| > 0
    //              and absolute errors when |df| ~ 0
    double denominator = (df_jvp1.Norml2() < 1.0e-10) ? 1.0 : df_jvp1.Norml2();

    // halving epsilon should make the error decrease
    // by about a factor of two for the forward-difference stencil
    double e1 = df1_fd[0].DistanceTo(df_jvp1.GetData()) / denominator;
    double e2 = df1_fd[1].DistanceTo(df_jvp1.GetData()) / denominator;
    EXPECT_TRUE(fabs(e1 / e2 - 2.0) < 0.1 || fmin(e1, e2) < 1.0e-9);

    // halving epsilon should make the error decrease
    // by about a factor of four for the center-difference stencil
    double e3 = df1_cd[0].DistanceTo(df_jvp1.GetData()) / denominator;
    double e4 = df1_cd[1].DistanceTo(df_jvp1.GetData()) / denominator;
    EXPECT_TRUE((fabs(e3 / e4 - 4.0) < 0.1) || fmin(e3, e4) < 1.0e-9);
  }

  {
    auto [value, df_ddU_dt] = f(t, U, smith::differentiate_wrt(dU_dt));
    std::unique_ptr<mfem::HypreParMatrix> df_ddU_dt_matrix = assemble(df_ddU_dt);

    // jacobian vector products
    mfem::Vector df_jvp1 = df_ddU_dt(ddU_dt);  // matrix-free

    mfem::Vector df_jvp2(df_jvp1.Size());
    df_ddU_dt_matrix->Mult(ddU_dt, df_jvp2);  // sparse matvec

    if (df_jvp1.Norml2() < 1.0e-13) {
      double absolute_error = df_jvp1.DistanceTo(df_jvp2.GetData());
      EXPECT_NEAR(0., absolute_error, 5.e-14);
    } else {
      double relative_error = df_jvp1.DistanceTo(df_jvp2.GetData()) / df_jvp1.Norml2();
      EXPECT_NEAR(0., relative_error, 5.e-14);
    }

    // {f(x - 2 * h), f(x - h), f(x), f(x + h), f(x + 2 * h)}
    mfem::Vector f_values[5];
    for (int i = 0; i < 5; i++) {
      auto dU_dt_plus_small = dU_dt;
      dU_dt_plus_small.Add((i - 2) * epsilon, ddU_dt);
      f_values[i] = f(t, U, dU_dt_plus_small);
    }

    // forward-difference approximations
    mfem::Vector df1_fd[2];
    df1_fd[0] = f_values[4];
    df1_fd[0] -= f_values[2];
    df1_fd[0] /= (2.0 * epsilon);

    df1_fd[1] = f_values[3];
    df1_fd[1] -= f_values[2];
    df1_fd[1] /= epsilon;

    // center-difference approximations
    mfem::Vector df1_cd[2];
    df1_cd[0] = f_values[4];
    df1_cd[0] -= f_values[0];
    df1_cd[0] /= (4.0 * epsilon);

    df1_cd[1] = f_values[3];
    df1_cd[1] -= f_values[1];
    df1_cd[1] /= (2.0 * epsilon);

    // this makes e1-e4 relative errors when |df| > 0
    //              and absolute errors when |df| ~ 0
    double denominator = (df_jvp1.Norml2() < 1.0e-10) ? 1.0 : df_jvp1.Norml2();

    // halving epsilon should make the error decrease
    // by about a factor of two for the forward-difference stencil
    double e1 = df1_fd[0].DistanceTo(df_jvp1.GetData()) / denominator;
    double e2 = df1_fd[1].DistanceTo(df_jvp1.GetData()) / denominator;
    EXPECT_TRUE(fabs(e1 / e2 - 2.0) < 0.1 || fmin(e1, e2) < 1.0e-9);

    // halving epsilon should make the error decrease
    // by about a factor of four for the center-difference stencil
    double e3 = df1_cd[0].DistanceTo(df_jvp1.GetData()) / denominator;
    double e4 = df1_cd[1].DistanceTo(df_jvp1.GetData()) / denominator;
    EXPECT_TRUE((fabs(e3 / e4 - 4.0) < 0.1) || fmin(e3, e4) < 1.0e-9);
  }
}

///////////////////
// qoi overloads //
///////////////////

template <typename T, smith::ExecutionSpace execution_space = smith::ExecutionSpace::CPU>
void check_gradient(smith::Functional<double(T), execution_space>& f, double t, const mfem::HypreParVector& U)
{
  int seed = 42;

  mfem::HypreParVector dU = U;
  dU = U;
  dU.Randomize(seed);

  double epsilon = 1.0e-8;

  auto [unused, dfdU] = f(t, smith::differentiate_wrt(U));

  std::unique_ptr<mfem::HypreParVector> dfdU_vec = assemble(dfdU);

  // TODO: fix this weird copy ctor behavior in mfem::HypreParVector
  auto U_plus = U;
  U_plus = U;
  U_plus.Add(epsilon, dU);

  auto U_minus = U;
  U_minus = U;
  U_minus.Add(-epsilon, dU);

  double df1 = (f(t, U_plus) - f(t, U_minus)) / (2 * epsilon);
  double df2 = InnerProduct(*dfdU_vec, dU);
  double df3 = dfdU(dU);

  double relative_error1 = (df1 - df2) / df1;
  double relative_error2 = (df1 - df3) / df1;

  EXPECT_NEAR(0., relative_error1, 2.e-5);
  EXPECT_NEAR(0., relative_error2, 2.e-5);

  // if (verbose) {
  //  std::cout << "errors: " << df1 << " " << df2 << " " << df3 << std::endl;
  //}
}

template <typename T1, typename T2>
void check_gradient(smith::Functional<double(T1, T2)>& f, double t, const mfem::HypreParVector& U,
                    const mfem::HypreParVector& dU_dt)
{
  int seed = 42;
  double epsilon = 1.0e-8;

  mfem::HypreParVector dU(U);
  dU.Randomize(seed);

  mfem::HypreParVector ddU_dt(dU_dt);
  ddU_dt.Randomize(seed + 1);

  // TODO: fix this weird copy ctor behavior in mfem::HypreParVector
  auto U_plus = U;
  U_plus = U;
  U_plus.Add(epsilon, dU);

  auto U_minus = U;
  U_minus = U;
  U_minus.Add(-epsilon, dU);

  {
    double df1 = (f(t, U_plus, dU_dt) - f(t, U_minus, dU_dt)) / (2 * epsilon);

    auto [value, dfdU] = f(t, smith::differentiate_wrt(U), dU_dt);
    double df2 = dfdU(dU);

    std::unique_ptr<mfem::HypreParVector> dfdU_vector = assemble(dfdU);

    double df3 = mfem::InnerProduct(*dfdU_vector, dU);

    double relative_error1 = fabs(df1 - df2) / std::max(fabs(df1), 1.0e-8);
    double relative_error2 = fabs(df1 - df3) / std::max(fabs(df1), 1.0e-8);

    EXPECT_NEAR(0., relative_error1, 5.e-5);
    EXPECT_NEAR(0., relative_error2, 5.e-5);
  }

  auto dU_dt_plus = dU_dt;
  dU_dt_plus = dU_dt;
  dU_dt_plus.Add(epsilon, ddU_dt);

  auto dU_dt_minus = dU_dt;
  dU_dt_minus = dU_dt;
  dU_dt_minus.Add(-epsilon, ddU_dt);

  {
    double df1 = (f(t, U, dU_dt_plus) - f(t, U, dU_dt_minus)) / (2 * epsilon);

    auto [value, df_ddU_dt] = f(t, U, smith::differentiate_wrt(dU_dt));
    double df2 = df_ddU_dt(ddU_dt);

    std::unique_ptr<mfem::HypreParVector> df_ddU_dt_vector = assemble(df_ddU_dt);

    double df3 = mfem::InnerProduct(*df_ddU_dt_vector, ddU_dt);

    double relative_error1 = fabs(df1 - df2) / std::max(fabs(df1), 1.0e-8);
    double relative_error2 = fabs(df1 - df3) / std::max(fabs(df1), 1.0e-8);
    double relative_error3 = fabs(df2 - df3) / std::max(fabs(df2), 1.0e-8);

    // note: these first two relative tolerances are really coarse,
    // since it seems the finite-difference approximation of the derivative
    // of this function is not very accurate (?)
    //
    // the action-of-gradient and gradient vector versions seem to agree to
    // machine precision
    EXPECT_NEAR(0., relative_error1, 1.e-2);
    EXPECT_NEAR(0., relative_error2, 1.e-2);
    EXPECT_NEAR(0., relative_error3, 5.e-14);
  }
}
