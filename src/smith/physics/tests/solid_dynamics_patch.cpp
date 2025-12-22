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

#include "smith/physics/solid_mechanics.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/finite_element.hpp"  // for H1
#include "smith/numerics/functional/geometry.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/numerics/solver_config.hpp"

namespace smith {

/**
 * @brief Specify the kinds of boundary condition to apply
 */
enum class PatchBoundaryCondition
{
  Essential,
  EssentialAndNatural
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

// clang-format off
constexpr tensor<double, 3, 3> A_3d{{{0.110791568544027, 0.230421268325901, 0.15167673653354},
                                     {0.198344644470483, 0.060514559793513, 0.084137393813728},
                                     {0.011544253485023, 0.060942846497753, 0.186383473579596}}};

constexpr tensor<double, 3> b_3d{{0.765645367640828, 0.992487355850465, 0.162199373722092}};
// clang-format on

/**
 * @brief Exact solution that is affine in space and time
 *
 * @tparam dim number of spatial dimensions
 */
template <int dim>
class AffineSolution {
 public:
  AffineSolution()
      : disp_grad_rate(make_tensor<dim, dim>([](int i, int j) { return A_3d[i][j]; })),
        initial_displacement(make_tensor<dim>([](int i) { return b_3d[i]; })) {};

  /// exact solution for displacement field
  tensor<double, dim> displacement(tensor<double, dim> X, double t) const
  {
    return t * dot(disp_grad_rate, X) + initial_displacement;
  }

  /**
   * @brief MFEM-style coefficient function corresponding to this solution
   *
   * @param X Coordinates of point in reference configuration at which solution is sought
   * @param u Exact solution evaluated at \p X
   */
  void operator()(const mfem::Vector& X, double t, mfem::Vector& u) const
  {
    auto X_tensor = make_tensor<dim>([&X](int i) { return X[i]; });
    auto u_tensor = this->displacement(X_tensor, t);
    for (int i = 0; i < dim; ++i) u[i] = u_tensor[i];
  }

  tensor<double, dim> velocity(tensor<double, dim> X, double /* t */) const { return dot(disp_grad_rate, X); }

  /**
   * @brief Apply forcing that should produce this exact displacement
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
   * @param solid The SolidMechanics module for the problem
   * @param essential_boundaries Boundary attributes on which essential boundary conditions are desired
   */
  template <int p, typename Material>
  void applyLoads(const Material& material, SolidMechanics<p, dim>& solid, Domain essential_boundary,
                  Domain /* whole_domain */) const
  {
    // essential BCs
    auto ebc_func = [*this](tensor<double, dim> X, double t) { return this->displacement(X, t); };
    solid.setDisplacementBCs(ebc_func, essential_boundary);

    // It's tempting to restrict the traction to the appropriate sub-boundary by defining
    // Domain natural_boundary = EntireBoundary(solid.mesh()) - essential_boundary;
    // ... and then passing natural_boundary to setTraction(). However, in some cases,
    // natural_boundary will be empty. It's easier to assign the traction to the
    // entire boundary, and it will automatically be ignored on the essential boundary.

    // natural BCs
    auto traction = [material, *this](auto, auto n0, auto t) {
      auto H = this->disp_grad_rate * t;
      typename Material::State state;  // needs to be reconfigured for mats with state
      tensor<double, dim, dim> P = material(state, H);
      return dot(P, n0);
    };
    solid.setTraction(traction, solid.mesh().entireBoundary());
  }

 private:
  tensor<double, dim, dim> disp_grad_rate;   /// Linear part of solution. Equivalently, the displacement gradient rate
  tensor<double, dim> initial_displacement;  /// Constant part of solution. Rigid body displacement.
};

constexpr tensor<double, 3> a_3d{{0.1, -0.2, 0.25}};

/**
 * @brief Constant acceleration exact solution
 *
 * This test can only be passed by second-order accurate time integrators.
 *
 * @tparam dim number of spatial dimensions
 */
template <int dim>
class ConstantAccelerationSolution {
 public:
  ConstantAccelerationSolution() : acceleration(make_tensor<dim>([](int i) { return a_3d[i]; })) {};

  tensor<double, dim> displacement(tensor<double, dim>, double t) const { return 0.5 * t * t * acceleration; };

  /**
   * @brief MFEM-style coefficient function corresponding to this solution
   *
   * @param X Coordinates of point in reference configuration at which solution is sought
   * @param u Exact solution evaluated at \p X
   */
  void operator()(const mfem::Vector& X, double t, mfem::Vector& u) const
  {
    tensor<double, dim> X_tensor{make_tensor<dim>([&X](int i) { return X[i]; })};
    tensor<double, dim> u_tensor = displacement(X_tensor, t);
    for (int i = 0; i < dim; ++i) u[i] = u_tensor[i];
  }

  tensor<double, dim> velocity(tensor<double, dim>, double t) const { return t * acceleration; }

  /**
   * @brief Apply forcing that should produce this exact displacement
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
   * @param solid The SolidMechanics module for the problem
   * @param essential_boundaries Boundary attributes on which essential boundary conditions are desired
   */
  template <int p, typename Material>
  void applyLoads(const Material& material, SolidMechanics<p, dim>& solid, Domain essential_boundary,
                  Domain whole_domain) const
  {
    // essential BCs
    auto ebc_func = [*this](tensor<double, dim> X, double t) { return this->displacement(X, t); };
    solid.setDisplacementBCs(ebc_func, essential_boundary);

    // no natural BCs

    // body force
    solid.addBodyForce([&material, *this](auto /* X */, auto /* t */) { return material.density * this->acceleration; },
                       whole_domain);
  }

 private:
  tensor<double, dim> acceleration;  /// Constant acceleration vector of solution
};

/**
 * @brief compute the error between the approximate and exact solution
 *
 * @tparam element_type the kind of element to use in the mesh
 * @tparam solution_type A class that satisfies the exact solution concept
 *
 * @param exact_solution Exact solution of problem
 * @param bc Specifier for boundary condition type to test
 * @return double L2 norm (continuous) of error in computed solution
 * *
 * @pre exact_solution must implement operator() that is an MFEM
 * coefficient-generating function for the exact solution of the displacement
 * as a function of space and time.
 * @pre exact_solution must implement velocity() that is an MFEM
 * coefficient-generating function for the exact solution of the velocity
 * as a function of space and time.
 * @pre exact_solution must have a method applyLoads that applies forcing terms to the
 * solid functional that should lead to the exact solution
 */
template <typename element_type, typename solution_type>
double solution_error(solution_type exact_solution, PatchBoundaryCondition bc)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = element_type::order;
  constexpr int dim = dimension_of(element_type::geometry);

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid_functional_dynamic_solve");

  // BT: shouldn't this assertion be in the physics module?
  // Putting it here prevents tests from having a nonsensical spatial dimension value,
  // but the physics module should be catching this error to protect users.
  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid functional test");

  std::string meshdir = std::string(SMITH_REPO_DIR) + "/data/meshes/";
  std::string filename;
  switch (element_type::geometry) {
    case mfem::Geometry::TRIANGLE:
      filename = meshdir + "patch2D_tris.mesh";
      break;
    case mfem::Geometry::SQUARE:
      filename = meshdir + "patch2D_quads.mesh";
      break;
    case mfem::Geometry::TETRAHEDRON:
      filename = meshdir + "patch3D_tets.mesh";
      break;
    case mfem::Geometry::CUBE:
      filename = meshdir + "patch3D_hexes.mesh";
      break;
    default:
      SLIC_ERROR_ROOT("unsupported element type for patch test");
      break;
  }

  std::string mesh_tag{"mesh"};
  auto mesh = std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag);

  // Construct a functional-based solid mechanics solver
  smith::NonlinearSolverOptions nonlin_opts{.relative_tol = 1.0e-13, .absolute_tol = 1.0e-13};

  SolidMechanics<p, dim> solid(nonlin_opts, smith::solid_mechanics::default_linear_options,
                               TimesteppingOptions{TimestepMethod::Newmark, DirichletEnforcementMethod::DirectControl},
                               "solid_dynamics", mesh);

  solid_mechanics::NeoHookean mat{.density = 1.0, .K = 1.0, .G = 1.0};
  solid.setMaterial(mat, mesh->entireBody());

  // initial conditions
  solid.setVelocity([&exact_solution](tensor<double, dim> X) { return exact_solution.velocity(X, 0.0); });
  solid.setDisplacement([&exact_solution](tensor<double, dim> X) { return exact_solution.displacement(X, 0.0); });

  // forcing terms
  auto essBC = Domain::ofBoundaryElements(mesh->mfemParMesh(), by_attr<dim>(essentialBoundaryAttributes<dim>(bc)));
  mesh->insertDomain("essential_boundary", essBC);
  exact_solution.applyLoads(mat, solid, mesh->domain("essential_boundary"), mesh->entireBody());

  // Finalize the data structures
  solid.completeSetup();

  // Integrate in time
  for (int i = 0; i < 3; i++) {
    solid.advanceTimestep(1.0);
  }

  // Compute norm of error
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_solution);
  exact_solution_coef.SetTime(solid.time());
  return computeL2Error(solid.displacement(), exact_solution_coef);
}

template <typename element_type>
double affine_velocity_test(PatchBoundaryCondition bc)
{
  constexpr int dim = dimension_of(element_type::geometry);
  return solution_error<element_type>(AffineSolution<dim>(), bc);
}

template <typename element_type>
double constant_acceleration_test(PatchBoundaryCondition bc)
{
  constexpr int dim = dimension_of(element_type::geometry);
  return solution_error<element_type>(ConstantAccelerationSolution<dim>(), bc);
}

const double tol = 1e-12;

constexpr int LINEAR = 1;
constexpr int QUADRATIC = 2;
constexpr int CUBIC = 3;

//
// 2D, Essential
//
TEST(SolidMechanicsDynamic, PatchTestTriQ1EssentialBcs)
{
  using element_type = finite_element<mfem::Geometry::TRIANGLE, H1<LINEAR> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestQuadQ1EssentialBcs)
{
  using element_type = finite_element<mfem::Geometry::SQUARE, H1<LINEAR> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestTriQ2EssentialBcs)
{
  using element_type = finite_element<mfem::Geometry::TRIANGLE, H1<QUADRATIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestQuadQ2EssentialBcs)
{
  using element_type = finite_element<mfem::Geometry::SQUARE, H1<QUADRATIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestTriQ3EssentialBcs)
{
  using element_type = finite_element<mfem::Geometry::TRIANGLE, H1<CUBIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestQuadQ3EssentialBcs)
{
  using element_type = finite_element<mfem::Geometry::SQUARE, H1<CUBIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

//
// 3D, Essential
//
TEST(SolidMechanicsDynamic, PatchTestTetQ1EssentialBcs)
{
  using element_type = finite_element<mfem::Geometry::TETRAHEDRON, H1<LINEAR> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestHexQ1EssentialBcs)
{
  using element_type = finite_element<mfem::Geometry::CUBE, H1<LINEAR> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestTetQ2EssentialBcs)
{
  using element_type = finite_element<mfem::Geometry::TETRAHEDRON, H1<QUADRATIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestHexQ2EssentialBcs)
{
  using element_type = finite_element<mfem::Geometry::CUBE, H1<QUADRATIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestTetQ3EssentialBcs)
{
  using element_type = finite_element<mfem::Geometry::TETRAHEDRON, H1<CUBIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestHexQ3EssentialBcs)
{
  using element_type = finite_element<mfem::Geometry::CUBE, H1<CUBIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::Essential);
  EXPECT_LT(error, tol);
}

//
// 2D, EssentialAndNatural
//
TEST(SolidMechanicsDynamic, PatchTestTriQ1EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::TRIANGLE, H1<LINEAR> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestQuadQ1EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::SQUARE, H1<LINEAR> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestTriQ2EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::TRIANGLE, H1<QUADRATIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestQuadQ2EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::SQUARE, H1<QUADRATIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestTriQ3EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::TRIANGLE, H1<CUBIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestQuadQ3EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::SQUARE, H1<CUBIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

//
// 3D, EssentialAndNatural
//
TEST(SolidMechanicsDynamic, PatchTestTetQ1EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::TETRAHEDRON, H1<LINEAR> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestHexQ1EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::CUBE, H1<LINEAR> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestTetQ2EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::TETRAHEDRON, H1<QUADRATIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestHexQ2EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::CUBE, H1<QUADRATIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestTetQ3EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::TETRAHEDRON, H1<CUBIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, PatchTestHexQ3EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::CUBE, H1<CUBIC> >;
  double error = affine_velocity_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

//
// Some misc. constant acceleration tests
//
TEST(SolidMechanicsDynamic, ConstantAccelerationTriQ2EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::TRIANGLE, H1<QUADRATIC> >;
  double error = constant_acceleration_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, ConstantAccelerationQuadQ2EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::SQUARE, H1<QUADRATIC> >;
  double error = constant_acceleration_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, ConstantAccelerationTetQ2EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::TETRAHEDRON, H1<QUADRATIC> >;
  double error = constant_acceleration_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

TEST(SolidMechanicsDynamic, ConstantAccelerationHexQ2EssentialAndNaturalBcs)
{
  using element_type = finite_element<mfem::Geometry::CUBE, H1<QUADRATIC> >;
  double error = constant_acceleration_test<element_type>(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(error, tol);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
