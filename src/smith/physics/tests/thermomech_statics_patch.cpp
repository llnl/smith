// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <array>
#include <complex>
#include <memory>
#include <set>
#include <string>
#include <tuple>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/physics/thermomechanics_monolithic.hpp"
#include "smith/physics/materials/green_saint_venant_thermoelastic.hpp"
#include "smith/smith_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/dual.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/geometry.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/physics/boundary_conditions/components.hpp"

namespace smith {

// Heat driven solid deformation test
// Fixed bcs for displacement: u(0, y) = 0, u(x, 0) = 0
template <int p, int dim, typename TempBC, typename FluxBC, typename ThermalSource>
void ThermomechHeatedDeform(const std::set<int>& temp_ess_bcs, const TempBC& temp_bc_function,
                            const FluxBC& flux_bc_function, const ThermalSource& source_function)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 4;
  int parallel_refinement = 2;

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermomechHeatedDeform");

  std::string filename = SMITH_REPO_DIR "/data/meshes/square_attribute.mesh";

  const std::string mesh_tag = "mesh";
  auto mesh =
      std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

  auto linear_opts = thermomechanics::direct_linear_options;
  auto nonlinear_opts = thermomechanics::default_nonlinear_options;
  ThermomechanicsMonolithic<p, dim> thermomech_solver(nonlinear_opts, linear_opts, "thermomechHeatedDeform", mesh);

  double rho = 1.0;
  double E = 100.0;
  double nu = 0.25;
  double c = 1.0;
  double alpha = 1.0e-3;
  double theta_ref = 0.0;
  double k = 1.0;
  thermomechanics::GreenSaintVenantThermoelasticMaterial material{rho, E, nu, c, alpha, theta_ref, k};

  thermomech_solver.setMaterial(material, mesh->entireBody());

  auto zero = [](const mfem::Vector&, double) -> double { return 0.0; };
  thermomech_solver.setTemperatureBCs(temp_ess_bcs, temp_bc_function);
  thermomech_solver.setFluxBCs(flux_bc_function, mesh->entireBoundary());

  thermomech_solver.setSource(source_function, mesh->entireBody());
  thermomech_solver.setTemperature(zero);

  std::set<int> disp_ess_bdr_y = {1};
  std::set<int> disp_ess_bdr_x = {3};
  mesh->addDomainOfBoundaryElements("ess_y_bdr", by_attr<dim>(disp_ess_bdr_y));
  mesh->addDomainOfBoundaryElements("ess_x_bdr", by_attr<dim>(disp_ess_bdr_x));

  thermomech_solver.setFixedBCs(mesh->domain("ess_y_bdr"), Component::Y);
  thermomech_solver.setFixedBCs(mesh->domain("ess_x_bdr"), Component::X);

  auto zeroVector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };
  thermomech_solver.setDisplacement(zeroVector);

  thermomech_solver.completeSetup();

  thermomech_solver.advanceTimestep(1.0);
}

/**
 * @brief Exact displacement and temperature solution of the form:
 *
 *   p == 1: u(X) := A.X + B,                 T(X) := C.X + E (affine)
 *   p == 2: u(X) := A.diag(X).X + B,         T(X) := C.X^2 + E
 *   p == 3: u(X) := A.diag(X).diag(X).X + B  T(x) := C.X^3 + E
 *
 * @tparam dim number of spatial dimensions
 */
template <int dim>
class ManufacturedSolution {
 public:
  ManufacturedSolution(int order, tensor<double, dim, dim> A, tensor<double, dim> B, tensor<double, dim> C, double D)
      : p(order), A_(A), B_(B), C_(C), D_(D)
  {
  }

  /**
   * @brief computes u_exact. Different from operator().
   * Takes a tensor as input to be compatible with applyLoad
   */
  template <typename T>
  auto evalU(const tensor<T, dim>& X) const
  {
    return make_tensor<dim>([&](int i) {
      decltype(double{} * T{}) val{};
      using std::pow;
      for (int j = 0; j < dim; ++j) {
        val += A_[i][j] * pow(X[j], p);
      }
      val += B_[i];
      return val;
    });
  }

  /**
   * @brief computes T_exact. Different from operator().
   * Takes a tensor as input to be compatible with applyLoad
   */
  template <typename T>
  auto evalT(const tensor<T, dim>& X) const
  {
    decltype(double{} * T{}) val{};
    using std::pow;
    for (int i = 0; i < dim; ++i) {
      val += C_[i] * pow(X[i], p);
    }
    val += D_;
    return val;
  }

  /**
   * @brief MFEM-style coefficient function corresponding to the displacement solution
   *
   * @param X Coordinates of point in reference configuration at which solution is sought
   * @param u Exact solution evaluated at \p X
   */
  void operator()(const mfem::Vector& X, mfem::Vector& u) const
  {
    auto Xt = make_tensor<dim>([&X](int i) { return X[i]; });
    auto ut = this->evalU(Xt);
    for (int i = 0; i < dim; ++i) {
      u[i] = ut[i];
    }
  }

  /**
   * @brief MFEM-style coefficient function corresponding to the temperature solution
   *
   * @param X Coordinates of point in reference configuration at which solution is sought
   * @param T Exact solution evaluated at \p X
   */
  double operator()(const mfem::Vector& X) const
  {
    auto Xt = make_tensor<dim>([&X](int i) { return X[i]; });
    return this->evalT(Xt);
  }

  /// @brief computes du/dX
  template <typename T>
  auto gradU(const tensor<T, dim>& X) const
  {
    return make_tensor<dim, dim>([&](int i, int j) {
      using std::pow;
      return A_[i][j] * p * pow(X[j], p - 1);
    });
  }

  /// @brief computes dT/dX
  template <typename T>
  auto gradT(const tensor<T, dim>& X) const
  {
    return make_tensor<dim>([&](int i) {
      using std::pow;
      return C_[i] * p * pow(X[i], p - 1);
    });
  }

  /**
   * @brief Apply forcing that should produce this exact temperature and displacement
   *
   * Given the physics module, apply boundary conditions and source
   * terms that are consistent with the exact solution. This is
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
   * @param tm The ThermoMechanics module for the problem
   * @param temp_ess_bcs Boundary attributes on which essential boundary conditions are desired on temperature field
   * @param disp_ess_bcs Boundary attributes on which essential boundary conditions are desired on displacement field
   */
  template <typename MaterialType, int p>
  void applyLoads(const MaterialType& material, ThermomechanicsMonolithic<p, dim>& tm, Domain& temp_ess_bdr,
                  Domain& disp_ess_bdr) const
  {
    // essential BCs
    auto temp_ebc_func = [*this](const auto& X, auto) { return this->evalT(X); };
    tm.setTemperatureBCs(temp_ebc_func, temp_ess_bdr);
    auto disp_ebc_func = [*this](const auto& X, auto) { return this->evalU(X); };
    tm.setDisplacementBCs(disp_ebc_func, disp_ess_bdr);

    // natural BCs
    auto flux = [=](auto X, auto n0, auto /* time */, auto /* T */) {
      typename MaterialType::State state{};

      auto theta = evalT(get<0>(X));
      auto dtheta_dX = gradT(get<0>(X));
      auto du_dX = gradU(get<0>(X));

      auto [stress, heat_accumulation, internal_heat_source, heat_flux] = material(state, du_dX, theta, dtheta_dX);
      return dot(heat_flux, n0);
    };
    tm.setFluxBCs(flux, tm.mesh().entireBoundary());

    auto traction = [=](auto X, auto n0, auto /* time */) {
      typename MaterialType::State state{};

      auto theta = evalT(get_value(X));
      auto dtheta_dX = gradT(get_value(X));
      auto du_dX = gradU(get_value(X));

      auto [stress, heat_accumulation, internal_heat_source, heat_flux] = material(state, du_dX, theta, dtheta_dX);
      return dot(stress, n0);
    };
    tm.setTraction(traction, tm.mesh().entireBoundary());

    // Forcing functions
    auto heat_source = [=](auto X, auto /* time */, auto /* T */, auto /* dTdX*/) {
      typename MaterialType::State state{};

      auto X_val_temp = get<0>(X);
      auto X_val = get_value(X_val_temp);
      auto theta = evalT(X_val);
      auto dtheta_dX = gradT(make_dual(X_val));
      auto du_dX = gradU(X_val);

      auto [stress, heat_accumulation, internal_heat_source, heat_flux] = material(state, du_dX, theta, dtheta_dX);
      auto dFluxdX = get_gradient(heat_flux);
      auto divFlux = tr(dFluxdX);
      return (divFlux - internal_heat_source);
    };
    tm.setSource(heat_source, tm.mesh().entireBody());

    auto body_force = [=](auto X, auto /* time */) {
      typename MaterialType::State state{};

      auto X_val = get_value(X);
      auto theta = evalT(make_dual(X_val));
      auto dtheta_dX = gradT(X_val);
      auto du_dX = gradU(make_dual(X_val));

      auto [stress, heat_accumulation, internal_heat_source, heat_flux] = material(state, du_dX, theta, dtheta_dX);
      auto dPdX = get_gradient(stress);
      tensor<double, dim> divP{};
      for (int i = 0; i < dim; ++i) {
        divP[i] = tr(dPdX[i]);
      }
      return (-1.0 * divP);
    };
    tm.addBodyForce(body_force, tm.mesh().entireBody());
  }

 private:
  int p;
  tensor<double, dim, dim> A_;
  tensor<double, dim> B_, C_;
  double D_;
};

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

/**
 * @brief Solve problem and compare numerical solution to exact answer
 *
 * @tparam element_type type describing element geometry and polynomial order to use for this test
 *
 * @param bc Specifier for boundary condition type to test
 * @param coef Coefficients for manufactured solutions
 * @return double L2 norm (continuous) of error in computed solution
 */
template <typename ElementType>
std::array<double, 2> SolutionError(
    PatchBoundaryCondition temp_bc, PatchBoundaryCondition disp_bc,
    std::tuple<tensor<double, dimension_of(ElementType::geometry), dimension_of(ElementType::geometry)>,
               tensor<double, dimension_of(ElementType::geometry)>, tensor<double, dimension_of(ElementType::geometry)>,
               double>
        coef,
    double alpha = 1e-3)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermomech_patch_test");

  constexpr int p = ElementType::order;
  constexpr int dim = dimension_of(ElementType::geometry);

  // Hanyu: polynomial order is set to 1 to pass all tests.
  //        same issue as the solid_statics_patch tests.
  //
  //        relevant issue: https://github.com/LLNL/smith/issues/926
  constexpr int solution_polynomial_order = 1;
  // constexpr int solution_polynomial_order = ElementType::order;
  auto exact_solution = ManufacturedSolution<dim>(solution_polynomial_order, std::get<0>(coef), std::get<1>(coef),
                                                  std::get<2>(coef), std::get<3>(coef));

  std::string meshdir = std::string(SMITH_REPO_DIR) + "/data/meshes/";
  std::string filename;
  switch (ElementType::geometry) {
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

  int serial_refinement;
  if (dim == 2)
    serial_refinement = 1;
  else if (dim == 3)
    serial_refinement = 0;

  const std::string mesh_tag = "mesh";
  auto mesh = std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, 0);

  auto linear_opts = thermomechanics::direct_linear_options;
  linear_opts.relative_tol = 1e-14;
  linear_opts.absolute_tol = 1e-14;
  auto nonlinear_opts = thermomechanics::default_nonlinear_options;
  nonlinear_opts.relative_tol = 1e-14;
  nonlinear_opts.absolute_tol = 1e-14;
  nonlinear_opts.print_level = 0;
  ThermomechanicsMonolithic<p, dim> thermomech_solver(nonlinear_opts, linear_opts, "thermomech_patch_test", mesh);

  double rho = 1.0;
  double E = 100.0;
  double nu = 0.25;
  double c = 1.0;
  double theta_ref = std::get<3>(coef);
  double k = 1.0;
  thermomechanics::GreenSaintVenantThermoelasticMaterial material{rho, E, nu, c, alpha, theta_ref, k};

  thermomech_solver.setMaterial(material, mesh->entireBody());

  mesh->addDomainOfBoundaryElements("temp_ess_bdr", by_attr<dim>(essentialBoundaryAttributes<dim>(temp_bc)));
  mesh->addDomainOfBoundaryElements("disp_ess_bdr", by_attr<dim>(essentialBoundaryAttributes<dim>(disp_bc)));
  exact_solution.applyLoads(material, thermomech_solver, mesh->domain("temp_ess_bdr"), mesh->domain("disp_ess_bdr"));

  // Finalize the data structures
  thermomech_solver.completeSetup();

  // Perform the quasi-static solve
  thermomech_solver.advanceTimestep(1.0);

  // Compute norm of error
  mfem::FunctionCoefficient exact_temperature_coef(exact_solution);
  mfem::VectorFunctionCoefficient exact_displacement_coef(dim, exact_solution);
  return {computeL2Error(thermomech_solver.temperature(), exact_temperature_coef),
          computeL2Error(thermomech_solver.displacement(), exact_displacement_coef)};
}

/////////////////////////////////Smoke Tests//////////////////////////////////////////////
TEST(Thermomechanics, SmokeVolumetricHeating)
{
  ThermomechHeatedDeform<1, 2>(
      std::set<int>{2, 4}, [](auto /* X */, auto /* time */) -> double { return 0.0; },
      [](auto /* X */, auto /* n */, auto /* time */, auto /* T */) { return 0.0; },
      [](auto /* X */, auto /* time */, auto /* T */, auto /* dT_dx */) { return 1.0; });
}

TEST(Thermomechanics, SmokeBCFluxHeating)
{
  ThermomechHeatedDeform<1, 2>(
      std::set<int>{2, 4}, [](auto /* X */, auto /* time */) -> double { return 0.0; },
      [](auto X, auto /* n */, auto /* time */, auto /* T */) {
        auto x = get<0>(X);
        if (x[0] == 0.0 || x[1] == 0.0)
          return -1.0;
        else
          return 0.0;
      },
      [](auto /* X */, auto /* time */, auto /* T */, auto /* dT_dx */) { return 0.0; });
}

///////////////////////////////////Patch Tests (Essential BC)/////////////////////////////
const double tol = 1e-12;
constexpr int LINEAR = 1;
constexpr int QUADRATIC = 2;
constexpr int CUBIC = 3;

TEST(Thermomechanics, Patch2dQ1TempEssentialDispConst)
{
  tensor<double, 2, 2> A{{{0.0, 0.0}, {0.0, 0.0}}};
  tensor<double, 2> B{{0.226645549201083, 0.39746067373029}};

  tensor<double, 2> C{{0.639707923874072, 0.3301482067142397}};
  double D = 0.1289548360319865;

  using triangle = finite_element<mfem::Geometry::TRIANGLE, H1<LINEAR>>;
  auto tri_err_arr = SolutionError<triangle>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                             std::tuple{A, B, C, D});
  EXPECT_LT(tri_err_arr[0], tol);
  EXPECT_LT(tri_err_arr[1], tol);

  using quadrilateral = finite_element<mfem::Geometry::SQUARE, H1<LINEAR>>;
  auto quad_err_arr = SolutionError<quadrilateral>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                                   std::tuple{A, B, C, D});
  EXPECT_LT(quad_err_arr[0], tol);
  EXPECT_LT(quad_err_arr[1], tol);
}

TEST(Thermomechanics, Patch2dQ1DispEssentialTempConst)
{
  tensor<double, 2, 2> A{{{0.110791568544027, 0.230421268325901}, {0.198344644470483, 0.060514559793513}}};
  tensor<double, 2> B{{0.226645549201083, 0.39746067373029}};

  tensor<double, 2> C{{0.0, 0.0}};
  double D = 0.1289548360319865;

  using triangle = finite_element<mfem::Geometry::TRIANGLE, H1<LINEAR>>;
  auto tri_err_arr = SolutionError<triangle>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                             std::tuple{A, B, C, D});
  EXPECT_LT(tri_err_arr[0], tol);
  EXPECT_LT(tri_err_arr[1], tol);

  using quadrilateral = finite_element<mfem::Geometry::SQUARE, H1<LINEAR>>;
  auto quad_err_arr = SolutionError<quadrilateral>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                                   std::tuple{A, B, C, D});
  EXPECT_LT(quad_err_arr[0], tol);
  EXPECT_LT(quad_err_arr[1], tol);
}

TEST(Thermomechanics, Patch2dQ1EssentialDecoupled)
{
  tensor<double, 2, 2> A{{{0.110791568544027, 0.230421268325901}, {0.198344644470483, 0.060514559793513}}};
  tensor<double, 2> B{{0.226645549201083, 0.39746067373029}};

  tensor<double, 2> C{{0.639707923874072, 0.3301482067142397}};
  double D = 0.1289548360319865;

  using triangle = finite_element<mfem::Geometry::TRIANGLE, H1<LINEAR>>;
  auto tri_err_arr = SolutionError<triangle>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                             std::tuple{A, B, C, D}, 0.0);
  EXPECT_LT(tri_err_arr[0], tol);
  EXPECT_LT(tri_err_arr[1], tol);

  using quadrilateral = finite_element<mfem::Geometry::SQUARE, H1<LINEAR>>;
  auto quad_err_arr = SolutionError<quadrilateral>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                                   std::tuple{A, B, C, D}, 0.0);
  EXPECT_LT(quad_err_arr[0], tol);
  EXPECT_LT(quad_err_arr[1], tol);
}

TEST(Thermomechanics, Patch2dQ1EssentialCoupled)
{
  tensor<double, 2, 2> A{{{0.110791568544027, 0.230421268325901}, {0.198344644470483, 0.060514559793513}}};
  tensor<double, 2> B{{0.226645549201083, 0.39746067373029}};

  tensor<double, 2> C{{0.639707923874072, 0.3301482067142397}};
  double D = 0.1289548360319865;

  using triangle = finite_element<mfem::Geometry::TRIANGLE, H1<LINEAR>>;
  auto tri_err_arr = SolutionError<triangle>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                             std::tuple{A, B, C, D});
  EXPECT_LT(tri_err_arr[0], tol);
  EXPECT_LT(tri_err_arr[1], tol);

  using quadrilateral = finite_element<mfem::Geometry::SQUARE, H1<LINEAR>>;
  auto quad_err_arr = SolutionError<quadrilateral>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                                   std::tuple{A, B, C, D});
  EXPECT_LT(quad_err_arr[0], tol);
  EXPECT_LT(quad_err_arr[1], tol);
}

TEST(Thermomechanics, Patch3dQ1EssentialCoupled)
{
  tensor<double, 3, 3> A{{{0.110791568544027, 0.230421268325901, 0.15167673653354},
                          {0.198344644470483, 0.060514559793513, 0.084137393813728},
                          {0.011544253485023, 0.060942846497753, 0.186383473579596}}};
  tensor<double, 3> B{{0.226645549201083, 0.39746067373029, 0.152779218111711}};

  tensor<double, 3> C{{0.639707923874072, 0.3301482067142397, 0.1098652982167019}};
  double D = 0.1289548360319865;

  using tetrahedron = finite_element<mfem::Geometry::TETRAHEDRON, H1<LINEAR>>;
  auto tet_err_arr = SolutionError<tetrahedron>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                                std::tuple{A, B, C, D});
  EXPECT_LT(tet_err_arr[0], tol);
  EXPECT_LT(tet_err_arr[1], tol);

  using hexahedron = finite_element<mfem::Geometry::CUBE, H1<LINEAR>>;
  auto hex_err_arr = SolutionError<hexahedron>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                               std::tuple{A, B, C, D});
  EXPECT_LT(hex_err_arr[0], tol);
  EXPECT_LT(hex_err_arr[1], tol);
}

TEST(Thermomechanics, Patch2dQ2EssentialCoupled)
{
  tensor<double, 2, 2> A{{{0.110791568544027, 0.230421268325901}, {0.198344644470483, 0.060514559793513}}};
  tensor<double, 2> B{{0.226645549201083, 0.39746067373029}};

  tensor<double, 2> C{{0.639707923874072, 0.3301482067142397}};
  double D = 0.1289548360319865;

  using triangle = finite_element<mfem::Geometry::TRIANGLE, H1<QUADRATIC>>;
  auto tri_err_arr = SolutionError<triangle>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                             std::tuple{A, B, C, D});
  EXPECT_LT(tri_err_arr[0], tol);
  EXPECT_LT(tri_err_arr[1], tol);

  using quadrilateral = finite_element<mfem::Geometry::SQUARE, H1<QUADRATIC>>;
  auto quad_err_arr = SolutionError<quadrilateral>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                                   std::tuple{A, B, C, D});
  EXPECT_LT(quad_err_arr[0], tol);
  EXPECT_LT(quad_err_arr[1], tol);
}

TEST(Thermomechanics, Patch3dQ2EssentialCoupled)
{
  tensor<double, 3, 3> A{{{0.110791568544027, 0.230421268325901, 0.15167673653354},
                          {0.198344644470483, 0.060514559793513, 0.084137393813728},
                          {0.011544253485023, 0.060942846497753, 0.186383473579596}}};
  tensor<double, 3> B{{0.226645549201083, 0.39746067373029, 0.152779218111711}};

  tensor<double, 3> C{{0.639707923874072, 0.3301482067142397, 0.1098652982167019}};
  double D = 0.1289548360319865;

  using tetrahedron = finite_element<mfem::Geometry::TETRAHEDRON, H1<QUADRATIC>>;
  auto tet_err_arr = SolutionError<tetrahedron>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                                std::tuple{A, B, C, D});
  EXPECT_LT(tet_err_arr[0], tol);
  EXPECT_LT(tet_err_arr[1], tol);

  using hexahedron = finite_element<mfem::Geometry::CUBE, H1<QUADRATIC>>;
  auto hex_err_arr = SolutionError<hexahedron>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                               std::tuple{A, B, C, D});
  EXPECT_LT(hex_err_arr[0], tol);
  EXPECT_LT(hex_err_arr[1], tol);
}

TEST(Thermomechanics, Patch2dQ3EssentialCoupled)
{
  tensor<double, 2, 2> A{{{0.110791568544027, 0.230421268325901}, {0.198344644470483, 0.060514559793513}}};
  tensor<double, 2> B{{0.226645549201083, 0.39746067373029}};

  tensor<double, 2> C{{0.639707923874072, 0.3301482067142397}};
  double D = 0.1289548360319865;

  using triangle = finite_element<mfem::Geometry::TRIANGLE, H1<CUBIC>>;
  auto tri_err_arr = SolutionError<triangle>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                             std::tuple{A, B, C, D});
  EXPECT_LT(tri_err_arr[0], tol);
  EXPECT_LT(tri_err_arr[1], tol);

  using quadrilateral = finite_element<mfem::Geometry::SQUARE, H1<CUBIC>>;
  auto quad_err_arr = SolutionError<quadrilateral>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                                   std::tuple{A, B, C, D});
  EXPECT_LT(quad_err_arr[0], tol);
  EXPECT_LT(quad_err_arr[1], tol);
}

TEST(Thermomechanics, Patch3dQ3EssentialCoupled)
{
  tensor<double, 3, 3> A{{{0.110791568544027, 0.230421268325901, 0.15167673653354},
                          {0.198344644470483, 0.060514559793513, 0.084137393813728},
                          {0.011544253485023, 0.060942846497753, 0.186383473579596}}};
  tensor<double, 3> B{{0.226645549201083, 0.39746067373029, 0.152779218111711}};

  tensor<double, 3> C{{0.639707923874072, 0.3301482067142397, 0.1098652982167019}};
  double D = 0.1289548360319865;

  using tetrahedron = finite_element<mfem::Geometry::TETRAHEDRON, H1<CUBIC>>;
  auto tet_err_arr = SolutionError<tetrahedron>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                                std::tuple{A, B, C, D});
  EXPECT_LT(tet_err_arr[0], tol);
  EXPECT_LT(tet_err_arr[1], tol);

  using hexahedron = finite_element<mfem::Geometry::CUBE, H1<CUBIC>>;
  auto hex_err_arr = SolutionError<hexahedron>(PatchBoundaryCondition::Essential, PatchBoundaryCondition::Essential,
                                               std::tuple{A, B, C, D});
  EXPECT_LT(hex_err_arr[0], tol);
  EXPECT_LT(hex_err_arr[1], tol);
}

//////////////////////////Patch Tests (Essential and natural BC)/////////////////////////////
TEST(Thermomechanics, Patch2dQ1TempEssentialAndNaturalDispConst)
{
  tensor<double, 2, 2> A{{{0.0, 0.0}, {0.0, 0.0}}};
  tensor<double, 2> B{{0.226645549201083, 0.39746067373029}};

  tensor<double, 2> C{{0.639707923874072, 0.3301482067142397}};
  double D = 0.1289548360319865;

  using triangle = finite_element<mfem::Geometry::TRIANGLE, H1<LINEAR>>;
  auto tri_err_arr = SolutionError<triangle>(PatchBoundaryCondition::EssentialAndNatural,
                                             PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(tri_err_arr[0], tol);
  EXPECT_LT(tri_err_arr[1], tol);

  using quadrilateral = finite_element<mfem::Geometry::SQUARE, H1<LINEAR>>;
  auto quad_err_arr = SolutionError<quadrilateral>(PatchBoundaryCondition::EssentialAndNatural,
                                                   PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(quad_err_arr[0], tol);
  EXPECT_LT(quad_err_arr[1], tol);
}

TEST(Thermomechanics, Patch2dQ1DispEssentialAndNaturalTempConst)
{
  tensor<double, 2, 2> A{{{0.110791568544027, 0.230421268325901}, {0.198344644470483, 0.060514559793513}}};
  tensor<double, 2> B{{0.226645549201083, 0.39746067373029}};

  tensor<double, 2> C{{0.0, 0.0}};
  double D = 0.1289548360319865;

  using triangle = finite_element<mfem::Geometry::TRIANGLE, H1<LINEAR>>;
  auto tri_err_arr = SolutionError<triangle>(PatchBoundaryCondition::EssentialAndNatural,
                                             PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(tri_err_arr[0], tol);
  EXPECT_LT(tri_err_arr[1], tol);

  using quadrilateral = finite_element<mfem::Geometry::SQUARE, H1<LINEAR>>;
  auto quad_err_arr = SolutionError<quadrilateral>(PatchBoundaryCondition::EssentialAndNatural,
                                                   PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(quad_err_arr[0], tol);
  EXPECT_LT(quad_err_arr[1], tol);
}

TEST(Thermomechanics, Patch2dQ1EssentialAndNaturalDecoupled)
{
  tensor<double, 2, 2> A{{{0.110791568544027, 0.230421268325901}, {0.198344644470483, 0.060514559793513}}};
  tensor<double, 2> B{{0.226645549201083, 0.39746067373029}};

  tensor<double, 2> C{{0.639707923874072, 0.3301482067142397}};
  double D = 0.1289548360319865;

  using triangle = finite_element<mfem::Geometry::TRIANGLE, H1<LINEAR>>;
  auto tri_err_arr = SolutionError<triangle>(PatchBoundaryCondition::EssentialAndNatural,
                                             PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D}, 0.0);
  EXPECT_LT(tri_err_arr[0], tol);
  EXPECT_LT(tri_err_arr[1], tol);

  using quadrilateral = finite_element<mfem::Geometry::SQUARE, H1<LINEAR>>;
  auto quad_err_arr =
      SolutionError<quadrilateral>(PatchBoundaryCondition::EssentialAndNatural,
                                   PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D}, 0.0);
  EXPECT_LT(quad_err_arr[0], tol);
  EXPECT_LT(quad_err_arr[1], tol);
}

TEST(Thermomechanics, Patch2dQ1EssentialAndNaturalCoupled)
{
  tensor<double, 2, 2> A{{{0.110791568544027, 0.230421268325901}, {0.198344644470483, 0.060514559793513}}};
  tensor<double, 2> B{{0.226645549201083, 0.39746067373029}};

  tensor<double, 2> C{{0.639707923874072, 0.3301482067142397}};
  double D = 0.1289548360319865;

  using triangle = finite_element<mfem::Geometry::TRIANGLE, H1<LINEAR>>;
  auto tri_err_arr = SolutionError<triangle>(PatchBoundaryCondition::EssentialAndNatural,
                                             PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(tri_err_arr[0], tol);
  EXPECT_LT(tri_err_arr[1], tol);

  using quadrilateral = finite_element<mfem::Geometry::SQUARE, H1<LINEAR>>;
  auto quad_err_arr = SolutionError<quadrilateral>(PatchBoundaryCondition::EssentialAndNatural,
                                                   PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(quad_err_arr[0], tol);
  EXPECT_LT(quad_err_arr[1], tol);
}

TEST(Thermomechanics, Patch3dQ1EssentialAndNaturalCoupled)
{
  tensor<double, 3, 3> A{{{0.110791568544027, 0.230421268325901, 0.15167673653354},
                          {0.198344644470483, 0.060514559793513, 0.084137393813728},
                          {0.011544253485023, 0.060942846497753, 0.186383473579596}}};
  tensor<double, 3> B{{0.226645549201083, 0.39746067373029, 0.152779218111711}};

  tensor<double, 3> C{{0.639707923874072, 0.3301482067142397, 0.1098652982167019}};
  double D = 0.1289548360319865;

  using tetrahedron = finite_element<mfem::Geometry::TETRAHEDRON, H1<LINEAR>>;
  auto tet_err_arr = SolutionError<tetrahedron>(PatchBoundaryCondition::EssentialAndNatural,
                                                PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(tet_err_arr[0], tol);
  EXPECT_LT(tet_err_arr[1], tol);

  using hexahedron = finite_element<mfem::Geometry::CUBE, H1<LINEAR>>;
  auto hex_err_arr = SolutionError<hexahedron>(PatchBoundaryCondition::EssentialAndNatural,
                                               PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(hex_err_arr[0], tol);
  EXPECT_LT(hex_err_arr[1], tol);
}

TEST(Thermomechanics, Patch2dQ2EssentialAndNaturalCoupled)
{
  tensor<double, 2, 2> A{{{0.110791568544027, 0.230421268325901}, {0.198344644470483, 0.060514559793513}}};
  tensor<double, 2> B{{0.226645549201083, 0.39746067373029}};

  tensor<double, 2> C{{0.639707923874072, 0.3301482067142397}};
  double D = 0.1289548360319865;

  using triangle = finite_element<mfem::Geometry::TRIANGLE, H1<QUADRATIC>>;
  auto tri_err_arr = SolutionError<triangle>(PatchBoundaryCondition::EssentialAndNatural,
                                             PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(tri_err_arr[0], tol);
  EXPECT_LT(tri_err_arr[1], tol);

  using quadrilateral = finite_element<mfem::Geometry::SQUARE, H1<QUADRATIC>>;
  auto quad_err_arr = SolutionError<quadrilateral>(PatchBoundaryCondition::EssentialAndNatural,
                                                   PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(quad_err_arr[0], tol);
  EXPECT_LT(quad_err_arr[1], tol);
}

TEST(Thermomechanics, Patch3dQ2EssentialAndNaturalCoupled)
{
  tensor<double, 3, 3> A{{{0.110791568544027, 0.230421268325901, 0.15167673653354},
                          {0.198344644470483, 0.060514559793513, 0.084137393813728},
                          {0.011544253485023, 0.060942846497753, 0.186383473579596}}};
  tensor<double, 3> B{{0.226645549201083, 0.39746067373029, 0.152779218111711}};

  tensor<double, 3> C{{0.639707923874072, 0.3301482067142397, 0.1098652982167019}};
  double D = 0.1289548360319865;

  using tetrahedron = finite_element<mfem::Geometry::TETRAHEDRON, H1<QUADRATIC>>;
  auto tet_err_arr = SolutionError<tetrahedron>(PatchBoundaryCondition::EssentialAndNatural,
                                                PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(tet_err_arr[0], tol);
  EXPECT_LT(tet_err_arr[1], tol);

  using hexahedron = finite_element<mfem::Geometry::CUBE, H1<QUADRATIC>>;
  auto hex_err_arr = SolutionError<hexahedron>(PatchBoundaryCondition::EssentialAndNatural,
                                               PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(hex_err_arr[0], tol);
  EXPECT_LT(hex_err_arr[1], tol);
}

TEST(Thermomechanics, Patch2dQ3EssentialAndNaturalCoupled)
{
  tensor<double, 2, 2> A{{{0.110791568544027, 0.230421268325901}, {0.198344644470483, 0.060514559793513}}};
  tensor<double, 2> B{{0.226645549201083, 0.39746067373029}};

  tensor<double, 2> C{{0.639707923874072, 0.3301482067142397}};
  double D = 0.1289548360319865;

  using triangle = finite_element<mfem::Geometry::TRIANGLE, H1<CUBIC>>;
  auto tri_err_arr = SolutionError<triangle>(PatchBoundaryCondition::EssentialAndNatural,
                                             PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(tri_err_arr[0], tol);
  EXPECT_LT(tri_err_arr[1], tol);

  using quadrilateral = finite_element<mfem::Geometry::SQUARE, H1<CUBIC>>;
  auto quad_err_arr = SolutionError<quadrilateral>(PatchBoundaryCondition::EssentialAndNatural,
                                                   PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(quad_err_arr[0], tol);
  EXPECT_LT(quad_err_arr[1], tol);
}

TEST(Thermomechanics, Patch3dQ3EssentialAndNaturalCoupled)
{
  tensor<double, 3, 3> A{{{0.110791568544027, 0.230421268325901, 0.15167673653354},
                          {0.198344644470483, 0.060514559793513, 0.084137393813728},
                          {0.011544253485023, 0.060942846497753, 0.186383473579596}}};
  tensor<double, 3> B{{0.226645549201083, 0.39746067373029, 0.152779218111711}};

  tensor<double, 3> C{{0.639707923874072, 0.3301482067142397, 0.1098652982167019}};
  double D = 0.1289548360319865;

  using tetrahedron = finite_element<mfem::Geometry::TETRAHEDRON, H1<CUBIC>>;
  auto tet_err_arr = SolutionError<tetrahedron>(PatchBoundaryCondition::EssentialAndNatural,
                                                PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(tet_err_arr[0], tol);
  EXPECT_LT(tet_err_arr[1], tol);

  using hexahedron = finite_element<mfem::Geometry::CUBE, H1<CUBIC>>;
  auto hex_err_arr = SolutionError<hexahedron>(PatchBoundaryCondition::EssentialAndNatural,
                                               PatchBoundaryCondition::EssentialAndNatural, std::tuple{A, B, C, D});
  EXPECT_LT(hex_err_arr[0], tol);
  EXPECT_LT(hex_err_arr[1], tol);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
