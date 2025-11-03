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
#include "smith/numerics/solver_config.hpp"

namespace smith {

/**
 * @brief Specify the kinds of boundary condition to apply
 */
enum class PatchBoundaryCondition
{
  Essential,
  MixedEssentialAndNatural
};

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
      case PatchBoundaryCondition::MixedEssentialAndNatural:
        essential_boundaries = {1, 4};
        break;
    }
  } else {
    switch (bc) {
      case PatchBoundaryCondition::Essential:
        essential_boundaries = {1, 2, 3, 4, 5, 6};
        break;
      case PatchBoundaryCondition::MixedEssentialAndNatural:
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
 * @param exact_solution Exact solution of problem
 * @param bc Specifier for boundary condition type to test
 * @return double L2 norm (continuous) of error in computed solution
 * *
 * @pre ExactSolution must implement operator() that is an MFEM
 * coefficient-generating function for the exact solution of the temperature
 * as a function of space and time.
 * @pre ExactSolution must have a method applyLoads that applies forcing terms to the
 * thermal functional that should lead to the exact solution
 */
template <int p, int dim, typename ExactSolution>
double dynamic_solution_error(const ExactSolution& exact_solution, PatchBoundaryCondition bc)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermal_dynamic_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for heat transfer test");

  std::string mesh_tag{"mesh"};
  std::string filename = std::string(SMITH_REPO_DIR) + "/data/meshes/patch" + std::to_string(dim) + "D.mesh";
  auto mesh = std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag);

  // Construct a heat transfer solver
  NonlinearSolverOptions nonlinear_opts{.relative_tol = 5.0e-13, .absolute_tol = 5.0e-13};

  TimesteppingOptions dyn_opts{.timestepper = TimestepMethod::BackwardEuler,
                               .enforcement_method = DirichletEnforcementMethod::DirectControl};

  HeatTransfer<p, dim> thermal(nonlinear_opts, heat_transfer::direct_linear_options, dyn_opts, "thermal", mesh);

  heat_transfer::LinearIsotropicConductor mat(1.0, 1.0, 1.0);
  thermal.setMaterial(mat, mesh->entireBody());

  // initial conditions
  thermal.setTemperature([exact_solution](const mfem::Vector& x, double) { return exact_solution(x, 0.0); });

  // forcing terms
  exact_solution.applyLoads(mat, thermal, mesh->entireBody(), mesh->entireBoundary(),
                            essentialBoundaryAttributes<dim>(bc));

  // Finalize the data structures
  thermal.completeSetup();

  // Integrate in time
  thermal.outputStateToDisk();
  for (int i = 0; i < 3; i++) {
    thermal.advanceTimestep(1.0);
    thermal.outputStateToDisk();
  }

  // Compute norm of error
  mfem::FunctionCoefficient exact_solution_coef(exact_solution);
  exact_solution_coef.SetTime(thermal.time());
  return computeL2Error(thermal.temperature(), exact_solution_coef);
}

// clang-format off
const tensor<double, 3> A{{10.0, 8.0, 4.0}};

const double b{3.0};
// clang-format on

/**
 * @brief Exact solution that is linear in space and time
 *
 * @tparam dim number of spatial dimensions
 */
template <int dim>
class LinearSolution {
 public:
  LinearSolution() : temp_grad_rate(dim)
  {
    initial_temperature = b;

    for (int i = 0; i < dim; i++) {
      temp_grad_rate(i) = A[i];
    }
  };

  /**
   * @brief MFEM-style coefficient function corresponding to this solution
   *
   * @param X Coordinates of point in reference configuration at which solution is sought
   * @param t Time of the evaluation
   * @return Exact solution evaluated at \p X
   */
  double operator()(const mfem::Vector& X, double t) const { return (temp_grad_rate * X) * t + initial_temperature; }

  /**
   * @brief Apply forcing that should produce this exact temperature
   *
   * Given the physics module, apply boundary conditions and a source
   * term that are consistent with the exact solution. This is
   * independent of the domain. The solution is imposed as an essential
   * boundary condition on the parts of the boundary identified by \p
   * essential_boundaries. On the complement of
   * \p essential_boundaries, the flux corresponding to the exact
   * solution is applied.
   *
   * @tparam p Polynomial degree of the finite element approximation
   * @tparam Material Type of the material model used in the problem
   *
   * @param material Material model used in the problem
   * @param thermal The HeatTransfer module for the problem
   * @param essential_boundaries Boundary attributes on which essential boundary conditions are desired
   */
  template <int p, typename Material>
  void applyLoads(const Material& material, HeatTransfer<p, dim>& thermal, Domain& dom, Domain& bdr,
                  std::set<int> essential_boundaries) const
  {
    // essential BCs
    auto ebc_func = [*this](const auto& X, double t) { return this->operator()(X, t); };
    thermal.setTemperatureBCs(essential_boundaries, ebc_func);

    // natural BCs
    auto temp_rate_grad = make_tensor<dim>([&](int i) { return temp_grad_rate(i); });
    auto flux_function = [material, temp_rate_grad](auto X, auto n0, auto t, auto temp) {
      auto temp_grad = temp_rate_grad * t;

      auto flux = smith::get<1>(material(X, temp, temp_grad));

      return dot(flux, n0);
    };
    thermal.setFluxBCs(flux_function, bdr);

    // volumetric source
    auto source_function = [temp_rate_grad](auto position, auto /* time */, auto /* u */, auto /* du_dx */) {
      return dot(get<VALUE>(position), temp_rate_grad);
    };
    thermal.setSource(source_function, dom);
  }

 private:
  mfem::Vector temp_grad_rate;  /// Linear part of solution. Equivalently, the temperature gradient rate
  double initial_temperature;   /// Constant part of temperature
};

const double tol = 1e-12;

TEST(HeatTransferDynamic, PatchTest2dQ1EssentialBcs)
{
  constexpr int p = 1;
  constexpr int dim = 2;
  double error = dynamic_solution_error<p, dim>(LinearSolution<dim>(), PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(HeatTransferDynamic, PatchTest3dQ1EssentialBcs)
{
  constexpr int p = 1;
  constexpr int dim = 3;
  double error = dynamic_solution_error<p, dim>(LinearSolution<dim>(), PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(HeatTransferDynamic, PatchTest2dQ2EssentialBcs)
{
  constexpr int p = 2;
  constexpr int dim = 2;
  double error = dynamic_solution_error<p, dim>(LinearSolution<dim>(), PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(HeatTransferDynamic, PatchTest3dQ2EssentialBcs)
{
  constexpr int p = 2;
  constexpr int dim = 3;
  double error = dynamic_solution_error<p, dim>(LinearSolution<dim>(), PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(HeatTransferDynamic, PatchTest2dQ1FluxBcs)
{
  constexpr int p = 1;
  constexpr int dim = 2;
  double error =
      dynamic_solution_error<p, dim>(LinearSolution<dim>(), PatchBoundaryCondition::MixedEssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(HeatTransferDynamic, PatchTest3dQ1FluxBcs)
{
  constexpr int p = 1;
  constexpr int dim = 3;
  double error =
      dynamic_solution_error<p, dim>(LinearSolution<dim>(), PatchBoundaryCondition::MixedEssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(HeatTransferDynamic, PatchTest2dQ2FluxBcs)
{
  constexpr int p = 2;
  constexpr int dim = 2;
  double error =
      dynamic_solution_error<p, dim>(LinearSolution<dim>(), PatchBoundaryCondition::MixedEssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(HeatTransferDynamic, PatchTest3dQ2FluxBcs)
{
  constexpr int p = 2;
  constexpr int dim = 3;
  double error =
      dynamic_solution_error<p, dim>(LinearSolution<dim>(), PatchBoundaryCondition::MixedEssentialAndNatural);
  EXPECT_LT(error, tol);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
