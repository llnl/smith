// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/serac_config.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils_base.hpp"

using namespace serac;
using namespace serac::profiling;
int serial_refinement = 1;
int parallel_refinement = 0;

int num_procs, myid;
int nsamples = 1;  // because mfem doesn't take in unsigned int

constexpr bool verbose = true;
std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

const std::map<int, std::string> meshfiles = {{2, SERAC_REPO_DIR "/data/meshes/star.mesh"},
                                              {3, SERAC_REPO_DIR "/data/meshes/beam-hex.mesh"}};

static constexpr double a = 1.7;
static constexpr double b = 2.1;

template <int dim>
struct thermal_qfunction {
  template <typename x_t, typename temperature_t>
  __host__ __device__ auto operator()(double /*t*/, x_t x, temperature_t temperature) const
  {
    // get the value and the gradient from the input tuple
    auto [u, du_dx] = temperature;
    auto source = a * u - (100 * x[0] * x[1]);
    auto flux = b * du_dx;
    return serac::tuple{source, flux};
  }
};

template <int dim>
struct elastic_qfunction {
  template <typename x_t, typename displacement_t>
  __host__ __device__ auto operator()(double /*t*/, x_t x, displacement_t displacement) const
  {
    // get the value and the gradient from the input tuple
    auto [u, du_dx] = displacement;
    constexpr auto I = Identity<dim>();
    auto body_force = a * u + I[0];
    auto strain = 0.5 * (du_dx + transpose(du_dx));
    auto stress = b * tr(strain) * I + 2.0 * b * strain;
    return serac::tuple{body_force, stress};
  }
};

template <int dim>
struct hcurl_qfunction {
  template <typename x_t, typename vector_potential_t>
  __host__ __device__ auto operator()(double /*t*/, x_t x, vector_potential_t vector_potential) const
  {
    auto [A, curl_A] = vector_potential;
    auto J_term = a * A - tensor<double, dim>{10 * x[0] * x[1], -5 * (x[0] - x[1]) * x[1]};
    auto H_term = b * curl_A;
    return serac::tuple{J_term, H_term};
  }
};

// this test sets up a toy "thermal" problem where the residual includes contributions
// from a temperature-dependent source term and a temperature-gradient-dependent flux
//
// the same problem is expressed with mfem and functional, and their residuals and gradient action
// are compared to ensure the implementations are in agreement.
template <int p, int dim>
void functional_test(H1<p> test, H1<p> trial, Dimension<dim>)
{
  auto pmesh = mesh::refineAndDistribute(buildMeshFromFile(meshfiles.at(dim)), serial_refinement, parallel_refinement);
  mfem::ParMesh& mesh = *pmesh;

  std::string postfix = concat("_H1<", p, ">");

  std::cout << serac::accelerator::getCUDAMemInfoString() << std::endl;

  serac::profiling::initialize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space = decltype(test);
  using trial_space = decltype(trial);

  // Create standard MFEM bilinear and linear forms on H1
  auto [fespace, fec] = serac::generateParFiniteElementSpace<test_space>(&mesh);

  mfem::ParBilinearForm A(fespace.get());

  // Add the mass term using the standard MFEM method
  mfem::ConstantCoefficient a_coef(a);
  A.AddDomainIntegrator(new mfem::MassIntegrator(a_coef));

  // Add the diffusion term using the standard MFEM method
  mfem::ConstantCoefficient b_coef(b);
  A.AddDomainIntegrator(new mfem::DiffusionIntegrator(b_coef));

  // Assemble the bilinear form into a matrix
  {
    SERAC_MARK_SCOPE(concat("mfem_localAssemble", postfix));
    A.Assemble(0);
  }

  A.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(
      SERAC_PROFILE_EXPR(concat("mfem_parallelAssemble", postfix), A.ParallelAssemble()));

  // Create a linear form for the load term using the standard MFEM method
  mfem::ParLinearForm f(fespace.get());
  mfem::FunctionCoefficient load_func([&](const mfem::Vector& coords) { return 100 * coords(0) * coords(1); });

  // Create and assemble the linear load term into a vector
  f.AddDomainIntegrator(new mfem::DomainLFIntegrator(load_func));
  SERAC_PROFILE_EXPR(serac::profiling::concat("mfem_fAssemble", postfix), f.Assemble());
  std::unique_ptr<mfem::HypreParVector> F(
      SERAC_PROFILE_EXPR(concat("mfem_fParallelAssemble", postfix), f.ParallelAssemble()));
  F->UseDevice(true);

  // Set a random state to evaluate the residual
  mfem::ParGridFunction u_global(fespace.get());
  u_global.Randomize();

  mfem::Vector U(fespace->TrueVSize());
  U.UseDevice(true);
  u_global.GetTrueDofs(U);

  cudaDeviceSynchronize();
  std::cout << "MFEM " << serac::accelerator::getCUDAMemInfoString() << std::endl;

  // Set up the same problem using functional

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space), ExecutionSpace::GPU> residual(fespace.get(), fespace.get());

  // Add the total domain residual term to the functional
  // note: NVCC's limited support for __device__ lambda functions
  // means we have to use a functor class to describe the qfunction instead of a lambda
  residual.AddDomainIntegral(Dimension<dim>{}, thermal_qfunction<dim>{}, mesh);

  // Compute the residual using standard MFEM methods
  mfem::Vector r1(U.Size());
  J->Mult(U, r1);
  r1 -= *F;

  std::cout << "using device: " << r1.UseDevice() << std::endl;

  // Compute the residual using functional
  mfem::Vector r2 = residual(U);

  if (verbose) {
    std::cout << "||r1||: " << r1.Norml2() << std::endl;
    std::cout << "||r2||: " << r2.Norml2() << std::endl;
    std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  }

  // Test that the two residuals are equivalent
  mfem::Vector error = r1;
  error -= r2;

  EXPECT_NEAR(0., error.Norml2() / r1.Norml2(), 1.e-14);

  serac::accelerator::displayLastCUDAMessage();

  // Compute the gradient using functional
  mfem::Operator& grad2 = SERAC_PROFILE_EXPR(concat("functional_GetGradient", postfix), residual.GetGradient(U));

  // Compute the gradient action using standard MFEM and functional
  mfem::Vector g1 = SERAC_PROFILE_EXPR_LOOP(concat("mfem_ApplyGradient", postfix), (*J) * U, nsamples);
  mfem::Vector g2 = SERAC_PROFILE_EXPR_LOOP(concat("functional_ApplyGradient", postfix), grad2 * U, nsamples);

  if (verbose) {
    std::cout << "||g1||: " << g1.Norml2() << std::endl;
    std::cout << "||g2||: " << g2.Norml2() << std::endl;
    std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;
  }

  // Ensure the two methods generate the same result
  EXPECT_NEAR(0., mfem::Vector(g1 - g2).Norml2() / g1.Norml2(), 1.e-14);

  std::cout << "Functional:" << serac::accelerator::getCUDAMemInfoString() << std::endl;

  serac::profiling::finalize();
}

// this test sets up a toy "elasticity" problem where the residual includes contributions
// from a displacement-dependent body force term and an isotropically linear elastic stress response
//
// the same problem is expressed with mfem and functional, and their residuals and gradient action
// are compared to ensure the implementations are in agreement.
template <int p, int dim>
void functional_test(H1<p, dim> test, H1<p, dim> trial, Dimension<dim>)
{
  auto pmesh = mesh::refineAndDistribute(buildMeshFromFile(meshfiles.at(dim)), serial_refinement, parallel_refinement);
  mfem::ParMesh& mesh = *pmesh;

  std::string postfix = concat("_H1<", p, ",", dim, ">");

  serac::profiling::initialize();

  using test_space = decltype(test);
  using trial_space = decltype(trial);

  auto [fespace, fec] = serac::generateParFiniteElementSpace<test_space>(&mesh);

  mfem::ParBilinearForm A(fespace.get());

  mfem::ConstantCoefficient a_coef(a);
  A.AddDomainIntegrator(new mfem::VectorMassIntegrator(a_coef));

  mfem::ConstantCoefficient lambda_coef(b);
  mfem::ConstantCoefficient mu_coef(b);
  A.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda_coef, mu_coef));
  {
    SERAC_MARK_SCOPE(concat("mfem_localAssemble", postfix));
    A.Assemble(0);
  }
  A.Finalize();

  std::unique_ptr<mfem::HypreParMatrix> J(
      SERAC_PROFILE_EXPR(concat("mfem_parallelAssemble", postfix), A.ParallelAssemble()));

  mfem::ParLinearForm f(fespace.get());
  mfem::VectorFunctionCoefficient load_func(dim, [&](const mfem::Vector& /*coords*/, mfem::Vector& force) {
    force = 0.0;
    force(0) = -1.;
  });

  f.AddDomainIntegrator(new mfem::VectorDomainLFIntegrator(load_func));
  {
    SERAC_MARK_SCOPE(concat("mfem_fAssemble", postfix));
    f.Assemble();
  }

  std::unique_ptr<mfem::HypreParVector> F(
      SERAC_PROFILE_EXPR(concat("mfem_fParallelAssemble", postfix), f.ParallelAssemble()));
  F->UseDevice(true);

  F->HostRead();

  mfem::ParGridFunction u_global(fespace.get());
  u_global.Randomize();

  mfem::Vector U(fespace->TrueVSize());
  U.UseDevice(true);
  u_global.GetTrueDofs(U);

  [[maybe_unused]] static constexpr auto I = Identity<dim>();

  Functional<test_space(trial_space), ExecutionSpace::GPU> residual(fespace.get(), fespace.get());
  residual.AddDomainIntegral(Dimension<dim>{}, elastic_qfunction<dim>{}, mesh);

  mfem::Vector r1 = SERAC_PROFILE_EXPR(concat("mfem_Apply", postfix), (*J) * U - (*F));
  mfem::Vector r2 = SERAC_PROFILE_EXPR(concat("functional_Apply", postfix), residual(U));

  if (verbose) {
    std::cout << "||r1||: " << r1.Norml2() << std::endl;
    std::cout << "||r2||: " << r2.Norml2() << std::endl;
    std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  }
  EXPECT_NEAR(0., mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-14);

  mfem::Operator& grad = SERAC_PROFILE_EXPR(concat("functional_GetGradient", postfix), residual.GetGradient(U));

  mfem::Vector g1 = SERAC_PROFILE_EXPR(concat("mfem_ApplyGradient", postfix), (*J) * U);
  mfem::Vector g2 = SERAC_PROFILE_EXPR(concat("functional_ApplyGradient", postfix), grad * U);

  if (verbose) {
    std::cout << "||g1||: " << g1.Norml2() << std::endl;
    std::cout << "||g2||: " << g2.Norml2() << std::endl;
    std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;
  }
  EXPECT_NEAR(0., mfem::Vector(g1 - g2).Norml2() / g1.Norml2(), 1.e-14);

  serac::profiling::finalize();
}

// this test sets up part of a toy "magnetic diffusion" problem where the residual includes contributions
// from a vector-potential-proportional J and an isotropically linear H
//
// the same problem is expressed with mfem and functional, and their residuals and gradient action
// are compared to ensure the implementations are in agreement.
template <int p, int dim>
void functional_test(Hcurl<p> test, Hcurl<p> trial, Dimension<dim>)
{
  auto pmesh = mesh::refineAndDistribute(buildMeshFromFile(meshfiles.at(dim)), serial_refinement, parallel_refinement);
  mfem::ParMesh& mesh = *pmesh;

  std::string postfix = concat("_Hcurl<", p, ">");

  serac::profiling::initialize();

  using test_space = decltype(test);
  using trial_space = decltype(trial);

  auto [fespace, fec] = serac::generateParFiniteElementSpace<test_space>(&mesh);

  mfem::ParBilinearForm B(fespace.get());

  mfem::ConstantCoefficient a_coef(a);
  B.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(a_coef));

  mfem::ConstantCoefficient b_coef(b);
  B.AddDomainIntegrator(new mfem::CurlCurlIntegrator(b_coef));
  {
    SERAC_MARK_SCOPE(concat("mfem_localAssemble", postfix));
    B.Assemble(0);
  }
  B.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(
      SERAC_PROFILE_EXPR(concat("mfem_parallelAssemble", postfix), B.ParallelAssemble()));

  mfem::ParLinearForm f(fespace.get());
  mfem::VectorFunctionCoefficient load_func(dim, [&](const mfem::Vector& coords, mfem::Vector& output) {
    double x = coords(0);
    double y = coords(1);
    output = 0.0;
    output(0) = 10 * x * y;
    output(1) = -5 * (x - y) * y;
  });

  f.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(load_func));
  {
    SERAC_MARK_SCOPE(concat("mfem_fAssemble", postfix));
    f.Assemble();
  }
  std::unique_ptr<mfem::HypreParVector> F(
      SERAC_PROFILE_EXPR(concat("mfem_fParallelAssemble", postfix), f.ParallelAssemble()));
  F->UseDevice(true);

  mfem::ParGridFunction u_global(fespace.get());
  u_global.Randomize();

  mfem::Vector U(fespace->TrueVSize());
  U.UseDevice(true);
  u_global.GetTrueDofs(U);

  using test_space = decltype(test);
  using trial_space = decltype(trial);

  Functional<test_space(trial_space), ExecutionSpace::GPU> residual(fespace.get(), fespace.get());

  residual.AddDomainIntegral(Dimension<dim>{}, hcurl_qfunction<dim>{}, mesh);

  // Compute the residual using standard MFEM methods
  mfem::Vector r1(U.Size());
  {
    SERAC_MARK_SCOPE(concat("mfem_Apply", postfix));
    J->Mult(U, r1);
    r1 -= *F;
  }

  mfem::Vector r2 = SERAC_PROFILE_EXPR(concat("functional_Apply", postfix), residual(U));

  if (verbose) {
    std::cout << "||r1||: " << r1.Norml2() << std::endl;
    std::cout << "||r2||: " << r2.Norml2() << std::endl;
    std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  }
  EXPECT_NEAR(0., mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-13);

  mfem::Operator& grad = SERAC_PROFILE_EXPR(concat("functional_GetGradient", postfix), residual.GetGradient(U));

  mfem::Vector g1 = SERAC_PROFILE_EXPR(concat("mfem_ApplyGradient", postfix), (*J) * U);
  mfem::Vector g2 = SERAC_PROFILE_EXPR(concat("functional_ApplyGradient", postfix), grad * U);

  if (verbose) {
    std::cout << "||g1||: " << g1.Norml2() << std::endl;
    std::cout << "||g2||: " << g2.Norml2() << std::endl;
    std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;
  }
  EXPECT_NEAR(0., mfem::Vector(g1 - g2).Norml2() / g1.Norml2(), 1.e-13);

  serac::profiling::finalize();
}

TEST(Thermal, 2DLinear) { functional_test(H1<1>{}, H1<1>{}, Dimension<2>{}); };
TEST(Thermal, 2DQuadratic) { functional_test(H1<2>{}, H1<2>{}, Dimension<2>{}); }
TEST(Thermal, 2DCubic) { functional_test(H1<3>{}, H1<3>{}, Dimension<2>{}); }

TEST(Thermal, 3DLinear) { functional_test(H1<1>{}, H1<1>{}, Dimension<3>{}); }
TEST(Thermal, 3DQuadratic) { functional_test(H1<2>{}, H1<2>{}, Dimension<3>{}); }
TEST(Thermal, 3DCubic) { functional_test(H1<3>{}, H1<3>{}, Dimension<3>{}); }

TEST(Hcurl, 2DLinear) { functional_test(Hcurl<1>{}, Hcurl<1>{}, Dimension<2>{}); }
TEST(Hcurl, 2DQuadratic) { functional_test(Hcurl<2>{}, Hcurl<2>{}, Dimension<2>{}); }
TEST(Hcurl, 2DCubic) { functional_test(Hcurl<3>{}, Hcurl<3>{}, Dimension<2>{}); }

TEST(Hcurl, 3DLinear) { functional_test(Hcurl<1>{}, Hcurl<1>{}, Dimension<3>{}); }
TEST(Hcurl, 3DQuadratic) { functional_test(Hcurl<2>{}, Hcurl<2>{}, Dimension<3>{}); }
TEST(Hcurl, 3DCubic) { functional_test(Hcurl<3>{}, Hcurl<3>{}, Dimension<3>{}); }

TEST(elasticity, 2DLinear) { functional_test(H1<1, 2>{}, H1<1, 2>{}, Dimension<2>{}); }
TEST(elasticity, 2DQuadratic) { functional_test(H1<2, 2>{}, H1<2, 2>{}, Dimension<2>{}); }
TEST(elasticity, 2DCubic) { functional_test(H1<3, 2>{}, H1<3, 2>{}, Dimension<2>{}); }

TEST(elasticity, 3DLinear) { functional_test(H1<1, 3>{}, H1<1, 3>{}, Dimension<3>{}); }
TEST(elasticity, 3DQuadratic) { functional_test(H1<2, 3>{}, H1<2, 3>{}, Dimension<3>{}); }
TEST(elasticity, 3DCubic) { functional_test(H1<3, 3>{}, H1<3, 3>{}, Dimension<3>{}); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  cudaDeviceSynchronize();
  std::cout << "Initial:" << serac::accelerator::getCUDAMemInfoString() << std::endl;

  auto [num_procs, myid] = serac::initialize(argc, argv);

  serac::accelerator::initializeDevice();

  mfem::OptionsParser args(argc, argv);
  args.AddOption(&serial_refinement, "-r", "--ref", "");
  args.AddOption(&parallel_refinement, "-pr", "--pref", "");
  args.AddOption(&nsamples, "-n", "--n-samples", "Samples per test");

  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(std::cout);
    }
    serac::exitGracefully(true);
  }
  if (myid == 0) {
    args.PrintOptions(std::cout);
  }

  int result = RUN_ALL_TESTS();

  serac::exitGracefully(result);
}
