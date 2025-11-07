// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <set>
#include <string>
#include <memory>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/physics/heat_transfer.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/thermal_material.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/tensor.hpp"

namespace smith {

/**
 * @brief Exact temperature solution that is an affine function
 *
 * @tparam dim number of spatial dimensions
 */
template <int dim>
class AffineSolution {
 public:
  AffineSolution() : A(dim)
  {
    // clang-format off
    A(0) = 0.2; A(1) = 0.3;
    if constexpr (dim == 3) { A(2) = 0.4; }

    b = 1.0;
    //clang-format on
  };

  /**
   * @brief MFEM-style coefficient function corresponding to this solution
   *
   * @param X Coordinates of point in reference configuration at which solution is sought
   */
  double operator()(const mfem::Vector& X) const
  {
    return A * X + b;
  }

  /**
   * @brief Apply forcing that should produce this exact temperature
   *
   * Given the physics module, apply boundary conditions and a source
   * term that are consistent with the exact solution. This is
   * independent of the domain. The solution is imposed as an essential
   * boundary condition on the parts of the boundary identified by \p
   * essential_boundaries. On the complement of
   * \p essential_boundaries, the traction corresponding to the exact
   * solution is applied.
   *
   * @tparam p Polynomial degree of the finite element approximation
   * @tparam Material Type of the material model used in the problem
   *
   * @param material Material model used in the problem
   * @param physics The HeatTransfer module for the problem
   * @param essential_boundaries Boundary attributes on which essential boundary conditions are desired
   */
  template <int p, typename Material>
  void applyLoads(const Material& material, HeatTransfer<p, dim>& physics, std::set<int> essential_boundaries, Domain & boundary) const
  {
    // essential BCs
    auto ebc_func = [*this](const auto& X, auto){ return this->operator()(X); };
    physics.setTemperatureBCs(essential_boundaries, ebc_func);

    // natural BCs
    auto temp_grad = make_tensor<dim>([&](int i) { return A(i); });
    tensor<double, dim> dummy_x;
    auto flux = smith::get<1>(material(dummy_x, 1.0, temp_grad));

    auto surface_flux = [flux](auto, auto n0, auto, auto) { return dot(flux, n0); };
    physics.setFluxBCs(surface_flux, boundary);
  }

 private:
  mfem::Vector A; /// Linear part of solution. Equivalently, the temperature gradient.
  double b;      /// Constant part of solution. Temperature offset.
};

/**
 * @brief Specify the kinds of boundary condition to apply
 */
enum class PatchBoundaryCondition { Essential, EssentialAndNatural };

/**
 * @brief Get boundary attributes for patch meshes on which to apply essential boundary conditions
 *
 * Parameterizes patch tests boundary conditions, as either essential
 * boundary conditions or partly essential boundary conditions and
 * partly natural boundary conditions. The return values are specific
 * to the meshes "patch2d.mesh" and "patch3d.mesh". The particular
 * portions of the boundary that get essential boundary conditions
 * are arbitrarily chosen.
 *
 * @tparam dim Spatial dimension
 *
 * @param b Kind of boundary conditions to apply in the problem
 * @return std::set<int> Boundary attributes for the essential boundary condition
 */
template <int dim>
std::set<int> essentialBoundaryAttributes(PatchBoundaryCondition bc)
{
  std::set<int> essential_boundaries;
  if constexpr (dim == 2) {
    switch (bc) {
      case PatchBoundaryCondition::Essential:
        essential_boundaries = {1, 2, 3, 4};
        break;
      case PatchBoundaryCondition::EssentialAndNatural:
        essential_boundaries = {1, 4};
        break;
    }
  } else {
    switch (bc) {
      case PatchBoundaryCondition::Essential:
        essential_boundaries = {1, 2, 3, 4, 5, 6};
        break;
      case PatchBoundaryCondition::EssentialAndNatural:
        essential_boundaries = {1, 2};
        break;
    }
  }
  return essential_boundaries;
}

/**
 * @brief Solve problem and compare numerical solution to exact answer
 *
 * @tparam p Polynomial degree of finite element approximation
 * @tparam dim Number of spatial dimensions
 * @tparam ExactSolution A class that satisfies the exact solution concept
 *
 * @param exact_temperature Exact solution of problem
 * @param bc Specifier for boundary condition type to test
 * @return double L2 norm (continuous) of error in computed solution
 * *
 * @pre ExactSolution must implement operator() that is an MFEM
 * coefficient-generating function
 * @pre ExactSolution must have a method applyLoads that applies forcing terms to the
 * heat transfer that should lead to the exact solution
 * See AffineSolution for an example
 */
template <int p, int dim, typename ExactSolution>
double solution_error(const ExactSolution& exact_temperature, PatchBoundaryCondition bc)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermal_static_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for heat transfer test");

  std::string filename = std::string(SMITH_REPO_DIR) +  "/data/meshes/patch" + std::to_string(dim) + "D.mesh";

  std::string mesh_tag{"mesh"};
  auto mesh = std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag);

  // Construct a heat transfer mechanics solver
  auto nonlinear_opts = heat_transfer::default_nonlinear_options;
  nonlinear_opts.absolute_tol = 1e-14;
  nonlinear_opts.relative_tol = 1e-14;
  HeatTransfer<p, dim> thermal(nonlinear_opts, heat_transfer::direct_linear_options, heat_transfer::default_static_options, "thermal", mesh);

  heat_transfer::LinearIsotropicConductor mat(1.0,1.0,1.0);
  thermal.setMaterial(mat, mesh->entireBody());

  exact_temperature.applyLoads(mat, thermal, essentialBoundaryAttributes<dim>(bc), mesh->entireBoundary());

  // Finalize the data structures
  thermal.completeSetup();

  // Perform the quasi-static solve
  thermal.advanceTimestep(1.0);

  // Compute norm of error
  mfem::FunctionCoefficient exact_solution_coef(exact_temperature);
  return computeL2Error(thermal.temperature(), exact_solution_coef);
}

const double tol = 1e-13;

TEST(HeatTransfer, PatchTest2dQ1EssentialBcs)
{
  constexpr int p = 1;
  constexpr int dim = 2;
  double error = solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(HeatTransfer, PatchTest3dQ1EssentialBcs)
{
  constexpr int p = 1;
  constexpr int dim   = 3;
  double error = solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(HeatTransfer, PatchTest2dQ2EssentialBcs)
{
  constexpr int p = 2;
  constexpr int dim   = 2;
  double error = solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(HeatTransfer, PatchTest3dQ2EssentialBcs)
{
  constexpr int p = 2;
  constexpr int dim   = 3;
  double error = solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(HeatTransfer, PatchTest2dQ1FluxBcs)
{
  constexpr int p = 1;
  constexpr int dim   = 2;
  double error = solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(HeatTransfer, PatchTest3dQ1FluxBcs)
{
  constexpr int p = 1;
  constexpr int dim   = 3;
  double error = solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(HeatTransfer, PatchTest2dQ2FluxBcs)
{
  constexpr int p = 2;
  constexpr int dim   = 2;
  double error = solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(HeatTransfer, PatchTest3dQ2FluxBcs)
{
  constexpr int p = 2;
  constexpr int dim   = 3;
  double error = solution_error<p, dim>(AffineSolution<dim>(), PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
