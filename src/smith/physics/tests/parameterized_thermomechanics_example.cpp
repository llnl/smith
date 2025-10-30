// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/numerics/functional/tests/check_gradient.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/differentiate_wrt.hpp"
#include "smith/numerics/functional/finite_element.hpp"  // for H1
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

using namespace smith;

template <typename T>
auto greenStrain(const tensor<T, 3, 3>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

struct ParameterizedThermoelasticMaterial {
  using State = Empty;

  static constexpr int VALUE = 0, GRADIENT = 1;

  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double theta_ref;  ///< datum temperature for thermal expansion

  template <typename T1, typename T2, typename T3>
  auto operator()(const State& /*state*/, const tensor<T1, 3, 3>& grad_u, T2 temperature,
                  T3 coefficient_of_thermal_expansion) const
  {
    auto theta = get<VALUE>(temperature);
    auto alpha = get<VALUE>(coefficient_of_thermal_expansion);

    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    static constexpr auto I = Identity<3>();
    auto F = grad_u + I;
    const auto Eg = greenStrain(grad_u);
    const auto trEg = tr(Eg);

    const auto S = 2.0 * G * dev(Eg) + K * (trEg - 3.0 * alpha * (theta - theta_ref)) * I;
    const auto P = dot(F, S);
    return dot(P, transpose(F));
  }
};

TEST(Thermomechanics, ParameterizedMaterial)
{
  constexpr int p = 1;
  constexpr int dim = 3;
  int serial_refinement = 0;
  int parallel_refinement = 0;

  double time = 0.0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "parameterized_thermomechanics");

  size_t radial_divisions = 3;
  size_t angular_divisions = 16;
  size_t vertical_divisions = 8;

  double inner_radius = 1.0;
  double outer_radius = 1.25;
  double height = 2.0;

  std::string mesh_tag{"mesh"};
  auto mesh = std::make_shared<smith::Mesh>(
      build_hollow_quarter_cylinder(radial_divisions, angular_divisions, vertical_divisions, inner_radius, outer_radius,
                                    height),
      mesh_tag, serial_refinement, parallel_refinement);

  NonlinearSolverOptions nonlinear_opts = solid_mechanics::default_nonlinear_options;
  LinearSolverOptions linear_opts = solid_mechanics::default_linear_options;

  nonlinear_opts.relative_tol = 1e-8;
  nonlinear_opts.absolute_tol = 1e-10;

#ifdef SMITH_USE_PETSC
  nonlinear_opts.nonlin_solver = NonlinearSolver::PetscNewton;
  linear_opts.linear_solver = LinearSolver::PetscGMRES;
  linear_opts.preconditioner = Preconditioner::Petsc;
  linear_opts.petsc_preconditioner = PetscPCType::HMG;
#endif

  SolidMechanics<p, dim, Parameters<H1<p>, H1<p>>> simulation(nonlinear_opts, linear_opts,
                                                              solid_mechanics::default_quasistatic_options,
                                                              "thermomechanics_simulation", mesh, {"theta", "alpha"});

  double density = 1.0;    ///< density
  double E = 1000.0;       ///< Young's modulus
  double nu = 0.25;        ///< Poisson's ratio
  double theta_ref = 0.0;  ///< datum temperature for thermal expansion

  ParameterizedThermoelasticMaterial material{density, E, nu, theta_ref};
  simulation.setMaterial(DependsOn<0, 1>{}, material, mesh->entireBody());

  double deltaT = 1.0;
  FiniteElementState temperature(mesh->mfemParMesh(), H1<p>{}, "theta");

  temperature = theta_ref;
  simulation.setParameter(0, temperature);

  double alpha0 = 1.0e-3;
  auto alpha_fec = std::unique_ptr<mfem::FiniteElementCollection>(new mfem::H1_FECollection(p, dim));
  FiniteElementState alpha(mesh->mfemParMesh(), H1<p>{}, "alpha");

  alpha = alpha0;
  simulation.setParameter(1, alpha);

  // set up essential boundary conditions
  mesh->addDomainOfBoundaryElements("x_equals_0", by_attr<dim>(4));
  mesh->addDomainOfBoundaryElements("y_equals_0", by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("z_equals_0", by_attr<dim>(1));

  simulation.setFixedBCs(mesh->domain("x_equals_0"), Component::X);
  simulation.setFixedBCs(mesh->domain("y_equals_0"), Component::Y);
  simulation.setFixedBCs(mesh->domain("z_equals_0"), Component::Z);

  // Finalize the data structures
  simulation.completeSetup();

  simulation.outputStateToDisk("paraview");

  // Perform the quasi-static solve
  temperature = theta_ref + deltaT;
  simulation.setParameter(0, temperature);
  simulation.advanceTimestep(1.0);

  simulation.outputStateToDisk("paraview");

  // create Domains to integrate over
  mesh->addDomainOfBoundaryElements("top_surface", [=](std::vector<vec3> vertices, int /* attr */) {
    // select the faces whose "average" z coordinate is close to the top of the mesh
    return average(vertices)[2] > 0.99 * height;
  });

  // define quantities of interest
  Functional<double(H1<p, dim>)> qoi({&simulation.displacement().space()});
  qoi.AddSurfaceIntegral(
      DependsOn<0>{},
      [=](double /*t*/, auto position, auto displacement) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = displacement;
        auto n = normalize(cross(dX_dxi));
        return dot(u, n);
      },
      mesh->domain("top_surface"));

  double initial_qoi = qoi(time, simulation.displacement());
  SLIC_INFO_ROOT(axom::fmt::format("vertical displacement integrated over the top surface: {}", initial_qoi));
  EXPECT_NEAR(initial_qoi, 0.000883477, 1e-5);

  Functional<double(H1<p, dim>)> area({&simulation.displacement().space()});
  area.AddSurfaceIntegral(
      DependsOn<>{}, [=](double /*t*/, auto /*position*/) { return 1.0; }, mesh->domain("top_surface"));

  double top_area = area(time, simulation.displacement());

  SLIC_INFO_ROOT(axom::fmt::format("total area of the top surface: {}", top_area));

  double exact_area = M_PI_4 * ((outer_radius * outer_radius) - (inner_radius * inner_radius));

  SLIC_INFO_ROOT(axom::fmt::format("exact area of the top surface: {}", exact_area));
  EXPECT_NEAR(top_area, exact_area, 1e-3);

  double avg_disp = qoi(time, simulation.displacement()) / area(time, simulation.displacement());

  SLIC_INFO_ROOT(axom::fmt::format("average vertical displacement: {}", avg_disp));

  double exact_avg_disp = alpha0 * deltaT * height;

  SLIC_INFO_ROOT(axom::fmt::format("expected average vertical displacement: {}", exact_avg_disp));
  EXPECT_NEAR(avg_disp, exact_avg_disp, 1e-5);

  smith::FiniteElementDual adjoint_load(simulation.displacement().space(), "adjoint_load");
  auto dqoi_du = get<1>(qoi(DifferentiateWRT<0>{}, time, simulation.displacement()));
  adjoint_load = *assemble(dqoi_du);

  check_gradient(qoi, time, simulation.displacement());

  simulation.setAdjointLoad({{"displacement", adjoint_load}});

  simulation.reverseAdjointTimestep();

  auto dqoi_dalpha = simulation.computeTimestepSensitivity(1);

  double epsilon = 1.0e-5;
  auto dalpha = alpha.CreateCompatibleVector();
  dalpha = 1.0;
  alpha.Add(epsilon, dalpha);
  simulation.setParameter(1, alpha);

  // rerun the simulation to the beginning,
  // but this time use perturbed values of alpha
  simulation.advanceTimestep(1.0);

  simulation.outputStateToDisk("paraview");

  double final_qoi = qoi(time, simulation.displacement());

  double adjoint_qoi_derivative = mfem::InnerProduct(dqoi_dalpha, dalpha);

  double fd_qoi_derivative = (final_qoi - initial_qoi) / epsilon;

  // compare the expected change in the QoI to the actual change:
  SLIC_INFO_ROOT(
      axom::fmt::format("directional derivative of QoI by adjoint-state method: {}", adjoint_qoi_derivative));
  SLIC_INFO_ROOT(axom::fmt::format("directional derivative of QoI by finite-difference:    {}", fd_qoi_derivative));

  EXPECT_NEAR(fd_qoi_derivative, adjoint_qoi_derivative, fd_qoi_derivative * 3.0e-5);
}

// output:
// vertical displacement integrated over the top surface: 0.000883477
// total area of the top surface: 0.441077
// exact area of the top surface: 0.441786
// average vertical displacement: 0.001999
// expected average vertical displacement: 0.002
// directional derivative of QoI by adjoint-state method: 0.8812734293294495
// directional derivative of QoI by finite-difference:    0.8812609461498273

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
