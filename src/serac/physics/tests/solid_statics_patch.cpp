// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include <functional>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"

namespace serac {

/**
 * @brief Exact displacement solution that is an affine function
 *
 * @tparam dim number of spatial dimensions
 */
template <int dim>
class AffineSolution {
public:
  AffineSolution() : A(dim), b(dim)
  {
    // clang-format off
    A(0, 0) = 0.110791568544027; A(0, 1) = 0.230421268325901;
    A(1, 0) = 0.198344644470483; A(1, 1) = 0.060514559793513;
    if constexpr (dim == 3) {
                                                                A(0, 2) = 0.15167673653354;
                                                                A(1, 2) = 0.084137393813728;
      A(2, 0) = 0.011544253485023; A(2, 1) = 0.060942846497753; A(2, 2) = 0.186383473579596;
    }

    b(0) = 0.765645367640828;
    b(1) = 0.992487355850465;
    if constexpr (dim == 3) {
      b(2) = 0.162199373722092;
    }
    //clang-format on
  };

  /**
   * @brief MFEM-style coefficient function corresponding to this solution
   *
   * @param X Coordinates of point in reference configuration at which solution is sought
   * @param u Exact solution evaluated at \p X
   */
  void operator()(const mfem::Vector& X, mfem::Vector& u) const
  {
    A.Mult(X, u);
    u += b;
  }

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
   * @param sf The SolidMechanics module for the problem
   * @param essential_boundaries Boundary attributes on which essential boundary conditions are desired
   */
  template <int p, typename Material>
  void applyLoads(const Material& material, SolidMechanics<p, dim>& sf, std::set<int> essential_boundaries) const
  {
    // essential BCs
    auto ebc_func = [*this](const auto& X, auto& u){ this->operator()(X, u); };
    sf.setDisplacementBCs(essential_boundaries, ebc_func);

    // natural BCs
    typename Material::State state;
    auto H = make_tensor<dim, dim>([&](int i, int j) { return A(i,j); });
    tensor<double, dim, dim> P = material(state, H);
    auto traction = [P](auto, auto n0, auto) { return dot(P, n0); };
    sf.setTraction(traction);
  }

 private:
  mfem::DenseMatrix A; /// Linear part of solution. Equivalently, the displacement gradient
  mfem::Vector b;      /// Constant part of solution. Rigid mody displacement.
};

/**
 * @brief Exact displacement solution of the form:
 *
 *   p == 1: u(X) := A.X + b (affine)
 *   p == 2: u(X) := A.diag(X).X + b
 *   p == 3: u(X) := A.diag(X).diag(X).X + b
 *
 * @tparam dim number of spatial dimensions
 */
template <int dim>
class ManufacturedSolution {
public:

  ManufacturedSolution(int order) : p(order), A(dim), b(dim)
  {
    // clang-format off
    if constexpr (dim == 2) {
      A(0, 0) = 0.110791568544027; A(0, 1) = 0.230421268325901;
      A(1, 0) = 0.198344644470483; A(1, 1) = 0.060514559793513;

      b(0) = 0.765645367640828;
      b(1) = 0.992487355850465;
    }
    if constexpr (dim == 3) {
      A(0, 0) = 0.110791568544027; A(0, 1) = 0.230421268325901; A(0, 2) = 0.15167673653354;
      A(1, 0) = 0.198344644470483; A(1, 1) = 0.060514559793513; A(1, 2) = 0.084137393813728;
      A(2, 0) = 0.011544253485023; A(2, 1) = 0.060942846497753; A(2, 2) = 0.186383473579596;

      b(0) = 0.765645367640828;
      b(1) = 0.992487355850465;
      b(2) = 0.162199373722092;
    }

    A *= 0.01;

    //clang-format on
  };

  /**
   * @brief MFEM-style coefficient function corresponding to this solution
   *
   * @param X Coordinates of point in reference configuration at which solution is sought
   * @param u Exact solution evaluated at \p X
   */
  void operator()(const mfem::Vector& X, mfem::Vector& u) const
  {
    A.Mult(X, u);
    u += b;
  }

  /// @brief computes du/dX
  template < typename T >
  auto gradient(const tensor<T, dim> & X) const
  {
    return make_tensor<dim, dim>([&](int i, int j){
      using std::pow;
      return A(i, j) * p * pow(X[j], p-1);
    });
  }

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
   * @tparam material_type Type of the material model used in the problem
   * @tparam p Polynomial degree of the finite element approximation
   *
   * @param material Material model used in the problem
   * @param sf The SolidMechanics module for the problem
   * @param essential_boundaries Boundary attributes on which essential boundary conditions are desired
   */
  template <typename material_type, int p>
  void applyLoads(const material_type & material, SolidMechanics<p, dim>& sf, std::set<int> essential_boundaries) const
  {
    // essential BCs
    auto ebc_func = [*this](const auto& X, auto& u){ this->operator()(X, u); };
    sf.setDisplacementBCs(essential_boundaries, ebc_func);

    // natural BCs
    auto traction = [=](auto X, auto n0, auto) {
      auto H = gradient(get_value(X));
      typename material_type::State state{};
      auto P = material(state, H);
      return dot(P, n0);
    };

    sf.setTraction(traction, EntireBoundary<dim>(sf.mesh()));

    auto bf = [=](auto X, auto) {
      auto X_val = get_value(X);
      auto H = gradient(make_dual(X_val));
      solid_mechanics::LinearIsotropic::State state{};
      auto P = material(state, H);
      auto dPdX = get_gradient(P);
      tensor<double,dim> divP{};
      for (int i = 0; i < dim; i++) {
        divP[i] = tr(dPdX[i]);
      }
      return divP;
    };

    sf.addBodyForce(DependsOn<>{}, bf, EntireDomain<dim>(sf.mesh()));

  }

 private:
  int p;               /// polynomial order
  mfem::DenseMatrix A; /// Linear part of solution. Equivalently, the displacement gradient
  mfem::Vector b;      /// Rigid body displacement.
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
 * @tparam element_type type describing element geometry and polynomial order to use for this test
 *
 * @param bc Specifier for boundary condition type to test
 * @return double L2 norm (continuous) of error in computed solution
 */
template < typename element_type>
double solution_error(PatchBoundaryCondition bc)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_static_solve");

  constexpr int p = element_type::order;
  constexpr int dim = dimension_of(element_type::geometry);

  // BT: shouldn't this assertion be in the physics module?
  // Putting it here prevents tests from having a nonsensical spatial dimension value,
  // but the physics module should be catching this error to protect users.
  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid test");

  // Sam: for now, this polynomial order is set to 1.
  //      In the future, it would be better to generalize it
  //      so that solution_polynomial_order = p, but my initial
  //      attempt had some issues. I'll revisit it at a later date
  //
  //      relevant issue: https://github.com/LLNL/serac/issues/926
  constexpr int solution_polynomial_order = 1;
  auto exact_displacement = ManufacturedSolution<dim>(solution_polynomial_order);

  std::string meshdir = std::string(SERAC_REPO_DIR) + "/data/meshes/";
  std::string filename;
  switch (element_type::geometry) {
    case mfem::Geometry::TRIANGLE:    filename = meshdir + "patch2D_tris.mesh"; break;
    case mfem::Geometry::SQUARE:      filename = meshdir + "patch2D_quads.mesh"; break;
    case mfem::Geometry::TETRAHEDRON: filename = meshdir + "patch3D_tets.mesh"; break;
    case mfem::Geometry::CUBE:        filename = meshdir + "patch3D_hexes.mesh"; break;
    default: SLIC_ERROR_ROOT("unsupported element type for patch test"); break;
  }
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename));

  std::string mesh_tag{"mesh_tag"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Construct a solid mechanics solver
  #ifdef SERAC_USE_SUNDIALS
  serac::NonlinearSolverOptions nonlin_solver_options{.nonlin_solver = NonlinearSolver::KINBacktrackingLineSearch, .relative_tol = 0.0, .absolute_tol = 1.0e-14, .max_iterations = 30};
  #else
  serac::NonlinearSolverOptions nonlin_solver_options{.nonlin_solver = NonlinearSolver::Newton, .relative_tol = 0.0, .absolute_tol = 1.0e-14, .max_iterations = 30};
  #endif

  auto equation_solver = std::make_unique<EquationSolver>(nonlin_solver_options, serac::solid_mechanics::default_linear_options, pmesh.GetComm());

  SolidMechanics<p, dim> solid(std::move(equation_solver), solid_mechanics::default_quasistatic_options, "solid", mesh_tag);

  solid_mechanics::NeoHookean mat{.density=1.0, .K=1.0, .G=1.0};
  solid.setMaterial(mat);

  exact_displacement.applyLoads(mat, solid, essentialBoundaryAttributes<dim>(bc));

  // Finalize the data structures
  solid.completeSetup();

  // Perform the quasi-static solve
  solid.advanceTimestep(1.0);

  // Output solution for debugging
  // solid.outputStateToDisk("paraview_output");
  // std::cout << "displacement =\n";
  // solid.displacement().Print(std::cout);
  // std::cout << "forces =\n";
  // solid_functional.reactions().Print();

  // Compute norm of error
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_displacement);
  return computeL2Error(solid.displacement(), exact_solution_coef);
}

/**
 * @brief Solve pressure-driven problem with 10% uniaxial strain and compare numerical solution to exact answer
 *
 * @tparam element_type type describing element geometry and polynomial order to use for this test
 *
 * @return double L2 norm (continuous) of error in computed solution
 */
template < typename element_type>
double pressure_error()
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_static_solve");

  constexpr int p = element_type::order;
  constexpr int dim = dimension_of(element_type::geometry);

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid test");

  auto exact_uniaxial_strain = [](const mfem::Vector& X, mfem::Vector& u) {
    u = 0.0;
    u(0) = -0.1 * X(0);
  };

  std::string meshdir = std::string(SERAC_REPO_DIR) + "/data/meshes/";
  std::string filename;
  switch (element_type::geometry) {
    case mfem::Geometry::TRIANGLE:    filename = meshdir + "patch2D_tris.mesh"; break;
    case mfem::Geometry::SQUARE:      filename = meshdir + "patch2D_quads.mesh"; break;
    case mfem::Geometry::TETRAHEDRON: filename = meshdir + "patch3D_tets.mesh"; break;
    case mfem::Geometry::CUBE:        filename = meshdir + "patch3D_hexes.mesh"; break;
    default: SLIC_ERROR_ROOT("unsupported element type for patch test"); break;
  }
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename));

  std::string mesh_tag{"mesh"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Construct a solid mechanics solver
  #ifdef SERAC_USE_SUNDIALS
  serac::NonlinearSolverOptions nonlin_solver_options{.nonlin_solver = NonlinearSolver::KINBacktrackingLineSearch, .relative_tol = 0.0, .absolute_tol = 1.0e-14, .max_iterations = 30};
  #else
  serac::NonlinearSolverOptions nonlin_solver_options{.nonlin_solver = NonlinearSolver::Newton, .relative_tol = 0.0, .absolute_tol = 1.0e-14, .max_iterations = 30};
  #endif

  auto equation_solver = std::make_unique<EquationSolver>(nonlin_solver_options, serac::solid_mechanics::default_linear_options, pmesh.GetComm());

  SolidMechanics<p, dim> solid(std::move(equation_solver), solid_mechanics::default_quasistatic_options, "solid", mesh_tag);

  solid_mechanics::NeoHookean mat{.density=1.0, .K=1.0, .G=1.0};
  solid.setMaterial(mat);

  typename solid_mechanics::NeoHookean::State state;
  auto H = make_tensor<dim, dim>([](int i, int j) {
      if ( i == 0 && j == 0) {
        return -0.1;
      }
      return 0.0;
  });

  tensor<double, dim, dim> sigma = mat(state, H);
  auto P = solid_mechanics::CauchyToPiola(sigma, H);
  double pressure = -1.0 * P(0,0);

  // Set the pressure corresponding to 10% uniaxial strain
  solid.setPressure([pressure](auto&, double) {
    return pressure;
  });

  // Define the essential boundary conditions corresponding to 10% uniaxial strain everywhere
  // except the pressure loaded surface
  if (dim == 2) {
    if (element_type::geometry == mfem::Geometry::TRIANGLE) {
      solid.setDisplacementBCs({4}, exact_uniaxial_strain);
      solid.setDisplacementBCs({1,3}, [](auto&){return 0.0; }, 1);
    } else if (element_type::geometry == mfem::Geometry::SQUARE) {
      solid.setDisplacementBCs({1}, exact_uniaxial_strain);
      solid.setDisplacementBCs({2,4}, [](auto&){return 0.0; }, 1);
    }
  } else if (dim == 3) {
    solid.setDisplacementBCs({1}, exact_uniaxial_strain);
    solid.setDisplacementBCs({2,5}, [](auto&){return 0.0; }, 1);
    solid.setDisplacementBCs({3,6}, [](auto&){return 0.0; }, 2);
  }

  // Finalize the data structures
  solid.completeSetup();

  // Perform the quasi-static solve
  solid.advanceTimestep(1.0);

  solid.outputStateToDisk();

  // Output solution for debugging
  // solid.outputStateToDisk("paraview_output");
  // std::cout << "displacement =\n";
  // solid.displacement().Print(std::cout);
  // std::cout << "forces =\n";
  // solid_functional.reactions().Print();

  // Compute norm of error
  mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_uniaxial_strain);
  return computeL2Error(solid.displacement(), exact_solution_coef);
}


const double tol = 1e-13;

constexpr int LINEAR = 1;
constexpr int QUADRATIC = 2;
constexpr int CUBIC = 3;

TEST(SolidMechanics, PatchTest2dQ1EssentialBcs)
{
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< LINEAR > >;
  double tri_error = solution_error< triangle >(PatchBoundaryCondition::Essential);
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< LINEAR > >;
  double quad_error = solution_error< quadrilateral >(PatchBoundaryCondition::Essential);
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ1EssentialBcs)
{
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< LINEAR > >;
  double tet_error = solution_error< tetrahedron >(PatchBoundaryCondition::Essential);
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< LINEAR > >;
  double hex_error = solution_error< hexahedron >(PatchBoundaryCondition::Essential);
  EXPECT_LT(hex_error, tol);
}

TEST(SolidMechanics, PatchTest2dQ2EssentialBcs)
{
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< QUADRATIC > >;
  double tri_error = solution_error< triangle >(PatchBoundaryCondition::Essential);
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< QUADRATIC > >;
  double quad_error = solution_error< quadrilateral >(PatchBoundaryCondition::Essential);
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ2EssentialBcs)
{
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< QUADRATIC > >;
  double tet_error = solution_error< tetrahedron >(PatchBoundaryCondition::Essential);
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< QUADRATIC > >;
  double hex_error = solution_error< hexahedron >(PatchBoundaryCondition::Essential);
  EXPECT_LT(hex_error, tol);
}

TEST(SolidMechanics, PatchTest2dQ3EssentialBcs)
{
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< CUBIC > >;
  double tri_error = solution_error< triangle >(PatchBoundaryCondition::Essential);
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< CUBIC > >;
  double quad_error = solution_error< quadrilateral >(PatchBoundaryCondition::Essential);
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ3EssentialBcs)
{
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< CUBIC > >;
  double tet_error = solution_error< tetrahedron >(PatchBoundaryCondition::Essential);
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< CUBIC > >;
  double hex_error = solution_error< hexahedron >(PatchBoundaryCondition::Essential);
  EXPECT_LT(hex_error, tol);
}

///////////////////////////////////////////////////////////////////////////////

TEST(SolidMechanics, PatchTest2dQ1EssentialAndNaturalBcs)
{
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< LINEAR > >;
  double tri_error = solution_error< triangle >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< LINEAR > >;
  double quad_error = solution_error< quadrilateral >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ1EssentialAndNaturalBcs)
{
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< LINEAR > >;
  double tet_error = solution_error< tetrahedron >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< LINEAR > >;
  double hex_error = solution_error< hexahedron >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(hex_error, tol);
}

TEST(SolidMechanics, PatchTest2dQ2EssentialAndNaturalBcs)
{
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< QUADRATIC > >;
  double tri_error = solution_error< triangle >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< QUADRATIC > >;
  double quad_error = solution_error< quadrilateral >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ2EssentialAndNaturalBcs)
{
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< QUADRATIC > >;
  double tet_error = solution_error< tetrahedron >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< QUADRATIC > >;
  double hex_error = solution_error< hexahedron >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(hex_error, tol);
}

TEST(SolidMechanics, PatchTest2dQ3EssentialAndNaturalBcs)
{
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< CUBIC > >;
  double tri_error = solution_error< triangle >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< CUBIC > >;
  double quad_error = solution_error< quadrilateral >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ3EssentialAndNaturalBcs)
{
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< CUBIC > >;
  double tet_error = solution_error< tetrahedron >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< CUBIC > >;
  double hex_error = solution_error< hexahedron >(PatchBoundaryCondition::EssentialAndNatural);
  EXPECT_LT(hex_error, tol);
}

TEST(SolidMechanics, PatchTest2dQ1Pressure){
  using triangle = finite_element< mfem::Geometry::TRIANGLE, H1< LINEAR > >;
  double tri_error = pressure_error< triangle >();
  EXPECT_LT(tri_error, tol);

  using quadrilateral = finite_element< mfem::Geometry::SQUARE, H1< LINEAR > >;
  double quad_error = pressure_error< quadrilateral >();
  EXPECT_LT(quad_error, tol);
}

TEST(SolidMechanics, PatchTest3dQ1Pressure){
  using tetrahedron = finite_element< mfem::Geometry::TETRAHEDRON, H1< LINEAR > >;
  double tet_error = pressure_error< tetrahedron >();
  EXPECT_LT(tet_error, tol);

  using hexahedron = finite_element< mfem::Geometry::CUBE, H1< LINEAR > >;
  double hex_error = pressure_error< hexahedron >();
  EXPECT_LT(hex_error, tol);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
