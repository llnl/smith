// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cstddef>
#include <algorithm>
#include <array>
#include <memory>
#include <set>
#include <string>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"

#include "smith/physics/thermomechanics_monolithic.hpp"
#include "smith/physics/materials/parameterized_thermoelastic_material.hpp"
#include "smith/smith_config.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/tests/paraview_helper.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"

#include "smith/physics/materials/thermoelastic_material.hpp"

#include "../../../../korner_examples/quasistatic_thermomechanics.hpp"
#include "../../korner_examples/calculate_reactions.hpp"

namespace smith {

void FiniteDifferenceParameter()
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 1;
  constexpr int dim = 3;
  int serial_refinement = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermomech_functional_parameterized_sensitivities");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/2d_plate.g"; //50x50x1 unit plate
  std::string mesh_tag("mesh");
  auto mesh = std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

  // Define boundary conditions
  std::set<int> temp_flux_bcs = {1, 2};
  std::set<int> disp_trac_bcs = {3, 4};
  std::set<int> temp_ess_bcs = {3, 4};
  std::set<int> disp_ess_bdr_y = {2};
  std::set<int> disp_ess_bdr_x = {1};

  mesh->addDomainOfBoundaryElements("flux_boundary", by_attr<dim>(temp_flux_bcs));
  mesh->addDomainOfBoundaryElements("trac_boundary", by_attr<dim>(disp_trac_bcs));
  mesh->addDomainOfBoundaryElements("temp_bdr", by_attr<dim>(temp_ess_bcs));
  mesh->addDomainOfBoundaryElements("ess_y_bdr", by_attr<dim>(disp_ess_bdr_y));
  mesh->addDomainOfBoundaryElements("ess_x_bdr", by_attr<dim>(disp_ess_bdr_x));

  // Initialize linear and nonlinear solver options
  smith::LinearSolverOptions heat_linear_options{.linear_solver = smith::LinearSolver::GMRES,
                                                 .relative_tol = 1e-8,
                                                 .absolute_tol = 1e-11,
                                                 .max_iterations = 50,
                                                 .print_level = 0};

  smith::LinearSolverOptions solid_linear_options{.linear_solver = smith::LinearSolver::GMRES,
                                                  .preconditioner = smith::Preconditioner::HypreAMG,
                                                  .relative_tol = 1e-8,
                                                  .absolute_tol = 1e-11,
                                                  .max_iterations = 10000,
                                                  .print_level = 0};

  smith::NonlinearSolverOptions solid_nonlinear_opts{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                     .relative_tol = 1.0e-8,
                                                     .absolute_tol = 1.0e-11,
                                                     .max_iterations = 500,
                                                     .print_level = 0};
  
  // Construct and initialize the linear and nonlinear solvers
  std::shared_ptr<DifferentiableSolver> d_heat_linear_solver =
      buildDifferentiableLinearSolve(heat_linear_options, *mesh);

  std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver =
      buildDifferentiableNonlinearSolve(solid_nonlinear_opts, solid_linear_options, *mesh);

  // Construct and initialize the time discretized weak form
  smith::SecondOrderTimeIntegrationRule backward_euler_heat(1.0);
  smith::SecondOrderTimeIntegrationRule backward_euler_solid(1.0);

  double theta_ref = 1.0;
  double external_heat_source = 1.0;

  double Km   = 0.5;         ///< matrix bulk modulus, MPa
  double Gm   = 0.0073976;   ///< matrix shear modulus, MPa
  double betam = 0.0;        ///< matrix volumetric thermal expansion coefficient
  double Ke    = 0.5;        ///< entanglement bulk modulus, MPa
  double Ge    = 0.225075;   ///< entanglement shear modulus, MPa
  double Cv    = 1.5;        ///< net volumetric heat capacity
  double kappa = 30.0;       ///< net thermal conductivity
  double Af   = 2.5e15;      ///< forward (low-high) exponential prefactor, 1/s
  double E_af = 1.5e5;       ///< forward (low-high) activation energy, J/mol
  double Ar   = 1.0e-21;     ///< reverse exponential prefactor, 1/s
  double E_ar = -1.55e5;     ///< reverse activation energy, J/mol
  double R    = 8.314;       ///< universal gas constant, J/mol/K
  double Tr   = 353.0;       ///< reference temperature, K
  double gw   = 0.2;         ///< particle weight fraction
  double wm   = 0.5;         ///< matrix mass fraction (set to 0.5, not real for now)

  using thermalStiffMaterial = thermomechanics::ParameterizedThermalStiffeningMaterial;
  thermalStiffMaterial material{Km, Gm, betam, Ke, Ge, Cv, kappa, Af, E_af, Ar, E_ar, R, Tr, gw, wm};

  // Build the physics and weak forms
  using ShapeDispSpace = H1<1, dim>;
  using VectorSpace = H1<p, dim>;
  using ScalarSpace = H1<p, 1>;
  using ScalarParameterSpace = L2<0>;
  std::string physics_name = "thermomech";
  auto [physics, solid_mechanics_weak_form, heat_transfer_weak_form, vector_bcs, scalar_bcs] =
      custom_physics::buildThermoMechanics<dim, ShapeDispSpace, VectorSpace, ScalarSpace, ScalarParameterSpace>(
          mesh, d_solid_nonlinear_solver, d_heat_linear_solver, backward_euler_solid, backward_euler_heat, physics_name, {"fictitious_density"});

  // Set the boundary conditions
  auto fixed_displacement = [](const double t, const smith::tensor<double, dim> & x){
    auto output = 0.0 * x;
    output[0] = 0.0*t;
    return output;
  };
  auto applied_temp = [&theta_ref](const double t, const smith::tensor<double, dim>){
    return theta_ref+0.0*t; //only the temperature differential here
  };

  vector_bcs->setVectorBCs<dim>(mesh->domain("ess_y_bdr"), {1}, fixed_displacement);
  vector_bcs->setVectorBCs<dim>(mesh->domain("ess_x_bdr"), {0}, fixed_displacement);
  vector_bcs->setScalarBCs<dim>(mesh->domain("temp_bdr"), applied_temp);

  // Add body integrals to the weak forms
  solid_mechanics_weak_form->addBodyIntegral(
    smith::DependsOn<0, 1, 2, 3>{}, mesh->entireBodyName(),
    [material, dim](const TimeInfo& time_info, auto /*X*/, auto u, auto v, auto /*a*/, auto theta, auto /*theta_dot*/,
                 auto /*theta_dot_dot*/, auto fictDens){
      thermalStiffMaterial::State<dim> state;
      auto [pk, C_v, s0, q0] = material(time_info.dt(), state, get<DERIVATIVE>(u),
         get<DERIVATIVE>(v), get<VALUE>(theta), get<DERIVATIVE>(theta), fictDens);
      return smith::tuple{smith::zero{}, pk};
    });

  heat_transfer_weak_form->addBodyIntegral(
    smith::DependsOn<0, 1, 2, 3>{}, mesh->entireBodyName(),
    [material, dim](const TimeInfo& time_info, auto /*X*/, auto theta, auto theta_dot, auto /*theta_dot_dot*/, auto u, auto v,
                 auto /*a*/, auto fictDens)  {
      thermalStiffMaterial::State<dim> state;
      auto [pk, C_v, s0, q0] = material(time_info.dt(), state, get<DERIVATIVE>(u), 
        get<DERIVATIVE>(v), get<VALUE>(theta), get<DERIVATIVE>(theta), fictDens);
      auto dT_dt = get<VALUE>(theta_dot);
      return smith::tuple{C_v * dT_dt - s0, -q0};
    });

  heat_transfer_weak_form->addBodySource(
      smith::DependsOn<0>(), mesh->entireBodyName(),
      [external_heat_source](auto /*t*/, auto /* x */, auto fictDens) { 
        return external_heat_source * fictDens; 
      });
  
  // Extract fields from the physics object
  auto shape_disp = physics->getShapeDispFieldState();
  auto params = physics->getFieldParams();
  auto states = physics->getInitialFieldStates();

  // Define ficitious density
  params[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) {
    double density = 0.5;
    return density;
  });

  // Construct the state advancer
  physics->resetStates();

  SLIC_INFO_ROOT("... Parameterized thermomechanics test setup complete");

  auto pv_writer = smith::createParaviewOutput(*mesh, physics->getFieldStatesAndParamStates(), "sol_param_thermal_stiff_test");
  pv_writer.write(0, physics->time(), physics->getFieldStatesAndParamStates());

  SLIC_INFO_ROOT("... Created paraview output writer");
  
  // Solve the forward problem
  size_t NumSteps = 10;
  double Total_Time = 50.0;
  double time_increment = Total_Time / (static_cast<double>(NumSteps));

  SLIC_INFO_ROOT("... Solving forward problem:");

  for (size_t i = 0; i < NumSteps; ++i) {
    SLIC_INFO_ROOT(axom::fmt::format("\n... Solving Step = {}", i));
    physics->advanceTimestep(time_increment);
    pv_writer.write(i + 1, physics->time(), physics->getFieldStatesAndParamStates());
  }

  ////////////////////////////////////////////////////////////
  // Compute the sensitivity of the QOI w.r.t. the parameter
  ////////////////////////////////////////////////////////////

  TimeInfo time_info(physics->time() - time_increment, time_increment);
  auto state_advancer = physics->getStateAdvancer();
  auto reactions = state_advancer->computeResultants(shape_disp, physics->getFieldStates(),
                                                     physics->getFieldStatesOld(), params, time_info);
  auto reaction_squared = 0.5 * innerProduct(reactions[0], reactions[0]);

  // smith::FieldState disp = states[0];
  // auto reaction_squared = smith::innerProduct(disp, disp);
  
  EXPECT_GT(checkGradWrt(reaction_squared, params[0], 1.0e-2, 1, true), 0.7);
}

TEST(ThermomechanicsFictDensParameter, Source) { FiniteDifferenceParameter(); }

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
