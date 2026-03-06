// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/smith_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tests/check_gradient.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/tuple.hpp"

using namespace smith;

#ifdef SMITH_USE_CUDA_KERNEL_EVALUATION
constexpr auto exec_space = smith::ExecutionSpace::GPU;
#else
constexpr auto exec_space = smith::ExecutionSpace::CPU;
#endif

template <int dim>
struct MixedModelOne {
  template <typename unused_type, typename displacement_type>
  SMITH_HOST_DEVICE auto operator()(double, unused_type, displacement_type displacement) const
  {
    constexpr static auto d11 =
        1.0 * make_tensor<dim + 1, dim, dim + 2, dim>([](int i, int j, int k, int l) { return i - j + 2 * k - 3 * l; });
    auto [u, du_dx] = displacement;
    auto source = zero{};
    auto flux = double_dot(d11, du_dx);
    return smith::tuple{source, flux};
  }
};

template <int dim>
struct MixedModelTwo {
  template <typename position_type, typename displacement_type>
  SMITH_HOST_DEVICE auto operator()(double, position_type position, displacement_type displacement) const
  {
    constexpr static auto s11 = 1.0 * make_tensor<dim + 1, dim + 2>([](int i, int j) { return i * i - j; });
    auto [X, dX_dxi] = position;
    auto [u, du_dxi] = displacement;
    return dot(s11, u) * X[0];
  }
};

template <int dim>
struct ElasticityTestModelOne {
  template <typename position_type, typename displacement_type>
  SMITH_HOST_DEVICE auto operator()(double, position_type, displacement_type displacement) const
  {
    constexpr static auto d00 = make_tensor<dim, dim>([](int i, int j) { return i + 2 * j + 1; });
    constexpr static auto d01 = make_tensor<dim, dim, dim>([](int i, int j, int k) { return i + 2 * j - k + 1; });
    constexpr static auto d10 = make_tensor<dim, dim, dim>([](int i, int j, int k) { return i + 3 * j - 2 * k; });
    constexpr static auto d11 =
        make_tensor<dim, dim, dim, dim>([](int i, int j, int k, int l) { return i - j + 2 * k - 3 * l + 1; });
    auto [u, du_dx] = displacement;
    auto source = dot(d00, u) + double_dot(d01, du_dx);
    auto flux = dot(d10, u) + double_dot(d11, du_dx);
    return smith::tuple{source, flux};
  }
};

template <int dim>
struct ElasticityTestModelTwo {
  template <typename position_type, typename displacement_type>
  SMITH_HOST_DEVICE auto operator()(double, position_type position, displacement_type displacement) const
  {
    auto [X, dX_dxi] = position;
    auto [u, du_dxi] = displacement;
    return u * X[0];
  }
};

template <int p, int dim>
void weird_mixed_test(std::unique_ptr<mfem::ParMesh>& mesh)
{
  // Define vector-valued test and trial spaces of different sizes
  using test_space = H1<p, dim + 1>;
  using trial_space = H1<p, dim + 2>;

  auto [trial_fes, trial_col] = generateParFiniteElementSpace<trial_space>(mesh.get());
  auto [test_fes, test_col] = generateParFiniteElementSpace<test_space>(mesh.get());

  mfem::Vector U(trial_fes->TrueVSize());

  Functional<test_space(trial_space), exec_space> residual(test_fes.get(), {trial_fes.get()});
  int seed = 5;
  U.Randomize(seed);

  // note: this is not really an elasticity problem, it's testing source and flux
  // terms that have the appropriate shapes to ensure that all the differentiation
  // code works as intended
  Domain dom = EntireDomain(*mesh);
  Domain bdr = EntireBoundary(*mesh);
  residual.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, MixedModelOne<dim>{}, dom);
  residual.AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<0>{}, MixedModelTwo<dim>{}, bdr);

  double t = 0.0;
  check_gradient(residual, t, U);
}

template <int p, int dim>
void elasticity_test(std::unique_ptr<mfem::ParMesh>& mesh)
{
  // Define the test and trial spaces for an elasticity-like problem
  using test_space = H1<p, dim>;
  using trial_space = H1<p, dim>;

  auto [trial_fes, trial_col] = generateParFiniteElementSpace<trial_space>(mesh.get());
  auto [test_fes, test_col] = generateParFiniteElementSpace<test_space>(mesh.get());

  mfem::Vector U(trial_fes->TrueVSize());
  int seed = 6;
  U.Randomize(seed);

  Functional<test_space(trial_space), exec_space> residual(test_fes.get(), {trial_fes.get()});

  // note: this is not really an elasticity problem, it's testing source and flux
  // terms that have the appropriate shapes to ensure that all the differentiation
  // code works as intended
  Domain dom = EntireDomain(*mesh);
  Domain bdr = EntireBoundary(*mesh);

  residual.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, ElasticityTestModelOne<dim>{}, dom);
  residual.AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<0>{}, ElasticityTestModelTwo<dim>{}, bdr);

  double t = 0.0;
  check_gradient(residual, t, U);
}

void test_suite(std::string meshfile)
{
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SMITH_REPO_DIR + meshfile), 1);

  if (mesh->Dimension() == 2) {
    constexpr int dim = 2;
    elasticity_test<1, dim>(mesh);
    elasticity_test<2, dim>(mesh);
    weird_mixed_test<1, dim>(mesh);
    weird_mixed_test<2, dim>(mesh);
  }

  if (mesh->Dimension() == 3) {
    constexpr int dim = 3;
    elasticity_test<1, dim>(mesh);
    elasticity_test<2, dim>(mesh);
    weird_mixed_test<1, dim>(mesh);
    weird_mixed_test<2, dim>(mesh);
  }
}

TEST(VectorValuedH1, test_suite_hexes) { test_suite("/data/meshes/patch3D_hexes.mesh"); }

#ifndef SMITH_USE_CUDA_KERNEL_EVALUATION
TEST(VectorValuedH1, test_suite_tris) { test_suite("/data/meshes/patch2D_tris.mesh"); }
TEST(VectorValuedH1, test_suite_quads) { test_suite("/data/meshes/patch2D_quads.mesh"); }
TEST(VectorValuedH1, test_suite_tris_and_quads) { test_suite("/data/meshes/patch2D_tris_and_quads.mesh"); }
TEST(VectorValuedH1, test_suite_tets) { test_suite("/data/meshes/patch3D_tets.mesh"); }
TEST(VectorValuedH1, test_suite_tets_and_hexes) { test_suite("/data/meshes/patch3D_tets_and_hexes.mesh"); }
#endif

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
