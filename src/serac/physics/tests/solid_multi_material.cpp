// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include <gtest/gtest.h>

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac {

TEST(Solid, MultiMaterial)
{
  /*
   * Checks multi material case with the following uniaxial problem:
   *              MATERIAL 1            MATERIAL 2
   *               E = 1                 E = 2
   * u = 0   --------------------|-------------------- stress = 1
   *
   * Solution:
   * strain =       1                    0.5
   *
   */
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 2;
  constexpr int dim = 3;
  int serial_refinement = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_mechanics_multimaterial");

  constexpr double L = 8.0;
  constexpr double W = 1.0;
  constexpr double H = 1.0;
  constexpr double VOLUME = L * W * H;

  auto mesh = mesh::refineAndDistribute(buildCuboidMesh(8, 1, 1, L, W, H), serial_refinement, parallel_refinement);

  const std::string mesh_tag{"mesh"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // identify the relevant boundary domains on this mesh
  Domain x_min = Domain::ofBoundaryElements(pmesh, by_attr<dim>(5));
  Domain y_min = Domain::ofBoundaryElements(pmesh, by_attr<dim>(2));
  Domain z_min = Domain::ofBoundaryElements(pmesh, by_attr<dim>(1));
  Domain end_face = Domain::ofBoundaryElements(pmesh, by_attr<dim>(3));

  // _solver_params_start
  serac::LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU};

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-12,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level = 1};

  SolidMechanics<p, dim> solid(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                               "solid_mechanics", mesh_tag);
  // _solver_params_end

  using Material = solid_mechanics::LinearIsotropic;

  constexpr double E_left = 1.0;
  constexpr double nu_left = 0.125;
  Material mat_left{.density = 1.0, .K = E_left / 3.0 / (1 - 2 * nu_left), .G = 0.5 * E_left / (1 + nu_left)};

  constexpr double E_right = 2.0 * E_left;
  constexpr double nu_right = 2 * nu_left;
  Material mat_right{.density = 1.0, .K = E_right / 3.0 / (1 - 2 * nu_right), .G = 0.5 * E_right / (1 + nu_right)};

  auto is_in_left = [](std::vector<tensor<double, dim>> coords, int /* attribute */) {
    return average(coords)[0] < 0.5 * L;
  };
  Domain left = Domain::ofElements(pmesh, is_in_left);
  Domain right = EntireDomain(pmesh) - left;

  solid.setMaterial(mat_left, left);
  solid.setMaterial(mat_right, right);

  constexpr double stress = 1.0;
  solid.setTraction(
      DependsOn<>{}, [stress](auto, auto n, auto) { return stress * n; }, end_face);

  solid.setFixedBCs(x_min, Component::X);
  solid.setFixedBCs(y_min, Component::Y);
  solid.setFixedBCs(z_min, Component::Z);

  solid.completeSetup();

  // Perform the quasi-static solve
  solid.advanceTimestep(1.0);
  solid.outputStateToDisk("paraview");

  // Define output functionals for verification

  constexpr double subdomain_volume = 0.5 * VOLUME;

  auto average_strain_integrand = [subdomain_volume](auto, auto, auto displacement) {
    auto strain = get<1>(displacement);
    return strain[0][0] / subdomain_volume;
  };

  Functional<double(H1<p, dim>)> average_strain_left({&solid.displacement().space()});
  average_strain_left.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, average_strain_integrand, left);

  Functional<double(H1<p, dim>)> average_strain_right({&solid.displacement().space()});
  average_strain_right.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, average_strain_integrand, right);

  EXPECT_NEAR(average_strain_left(solid.time(), solid.displacement()), stress / E_left, 1e-10);
  EXPECT_NEAR(average_strain_right(solid.time(), solid.displacement()), stress / E_right, 1e-10);
}

TEST(Solid, MultiMaterialWithState)
{
  /*
   * Checks multi-material case with the following uniaxial problem:
   *              MATERIAL 1            MATERIAL 2
   *               E = 2                 E = 2
   *               nu_1                 nu_2 = 0
   *                                   sigma_y = 0.5*stress
   *                                    Hi = E / 3.6 (linear hardening modulus)
   * x = 0                       x = L/2               x = L
   * u = 0   --------------------|-------------------- stress = 1
   *
   * The solution for the strain is:
   * strain = | stress/E,                            x in (0, L/2)
   *          | (Hi + E)/(Hi*E)*stress - sigma_y/Hi, x in (L/2, L)
   *
   * nu_1 is chosen so that the problem is uniaxial. That is, the lateral contraction in the
   * elastic side exactly matches the contraction in the plastic side. For the values given
   * above, this corresponds to nu_1 = 0.45.
   *
   * This problem checks that both materials models are correctly called for their subdomains, and
   * that the internal variables are correctly indexed for the multi-material case.
   *
   */
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 2;
  constexpr int dim = 3;
  int serial_refinement = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_mechanics_multimaterial_with_state");

  constexpr double L = 8.0;
  constexpr double W = 1.0;
  constexpr double H = 1.0;
  constexpr double VOLUME = L * W * H;

  constexpr double applied_stress = 1.0;

  auto mesh = mesh::refineAndDistribute(buildCuboidMesh(8, 1, 1, L, W, H), serial_refinement, parallel_refinement);

  const std::string mesh_tag{"mesh"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  Domain x_min_face = Domain::ofBoundaryElements(pmesh, by_attr<dim>(5));
  Domain y_min_face = Domain::ofBoundaryElements(pmesh, by_attr<dim>(2));
  Domain z_min_face = Domain::ofBoundaryElements(pmesh, by_attr<dim>(1));

  serac::LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU};

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-12,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level = 1};

  SolidMechanics<p, dim> solid(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                               "solid_mechanics", mesh_tag);

  auto is_in_left = [](std::vector<tensor<double, dim>> coords, int /* attribute */) {
    return average(coords)[0] < 0.5 * L;
  };
  Domain left = Domain::ofElements(pmesh, is_in_left);
  Domain right = EntireDomain(pmesh) - left;

  using Hardening = solid_mechanics::LinearHardening;
  using MaterialRight = solid_mechanics::J2SmallStrain<Hardening>;

  constexpr double E_right = 2.0;
  constexpr double nu_right = 0.0;
  constexpr double Hi = E_right / 3.6;
  constexpr double sigma_y = 0.75 * applied_stress;

  Hardening hardening{.sigma_y = sigma_y, .Hi = Hi};
  MaterialRight mat_right{
      .E = E_right,    // Young's modulus
      .nu = nu_right,  // Poisson's ratio
      .hardening = hardening,
      .Hk = 0.0,      // kinematic hardening constant
      .density = 1.0  // mass density
  };

  using MaterialLeft = solid_mechanics::LinearIsotropic;

  constexpr double E_left = 2.0;
  constexpr double nu_left = 0.5 * E_left / Hi * (1 - sigma_y / applied_stress);
  MaterialLeft mat_left{.density = 1.0, .K = E_left / 3.0 / (1 - 2 * nu_left), .G = 0.5 * E_left / (1 + nu_left)};

  MaterialRight::State initial_state{};
  auto qdata = solid.createQuadratureDataBuffer(initial_state, right);

  solid.setMaterial(mat_left, left);
  solid.setMaterial(mat_right, right, qdata);

  Domain end_face = Domain::ofBoundaryElements(pmesh, by_attr<dim>(3));
  solid.setTraction(
      DependsOn<>{}, [applied_stress](auto, auto n, auto) { return applied_stress * n; }, end_face);

  solid.setFixedBCs(x_min_face, Component::X);
  solid.setFixedBCs(y_min_face, Component::Y);
  solid.setFixedBCs(z_min_face, Component::Z);

  solid.completeSetup();

  std::cout << "setup complete " << std::endl;

  // Perform the quasi-static solve
  solid.advanceTimestep(1.0);
  solid.outputStateToDisk("paraview");

  // Define output functionals for verification

  constexpr double subdomain_volume = 0.5 * VOLUME;

  auto average_strain_integrand = [subdomain_volume](auto, auto, auto displacement) {
    auto strain = get<1>(displacement);
    return strain[0][0] / subdomain_volume;
  };

  Functional<double(H1<p, dim>)> average_strain_left({&solid.displacement().space()});
  average_strain_left.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, average_strain_integrand, left);

  Functional<double(H1<p, dim>)> average_strain_right({&solid.displacement().space()});
  average_strain_right.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, average_strain_integrand, right);

  EXPECT_NEAR(average_strain_left(solid.time(), solid.displacement()), applied_stress / E_left, 1e-10);

  double exact = (E_right + Hi) / (E_right * Hi) * applied_stress - sigma_y / Hi;
  EXPECT_NEAR(average_strain_right(solid.time(), solid.displacement()), exact, 1e-10);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);

  int result = RUN_ALL_TESTS();

  serac::exitGracefully(result);
}
