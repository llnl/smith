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

#include "smith/physics/materials/thermoelastic_material.hpp"

#include "../../../../korner_examples/quasistatic_thermomechanics.hpp"
#include "../../korner_examples/calculate_reactions.hpp"

namespace smith {

void FiniteDifferenceParameter(size_t sensitivity_parameter_index = 0)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 1;
  constexpr int dim = 2;
  int serial_refinement = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermomech_functional_parameterized_sensitivities");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";
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

  // // Construct a dummy adjoint load (this would come from a QOI downstream).
  // // This adjoint load is equivalent to a discrete L1 norm on the temperature and displacement.
  // mfem::ParLinearForm temp_adjoint_load_form(
  //     const_cast<mfem::ParFiniteElementSpace*>(&thermomech_solver.temperature().space()));
  // mfem::ParLinearForm disp_adjoint_load_form(
  //     const_cast<mfem::ParFiniteElementSpace*>(&thermomech_solver.displacement().space()));
  // temp_adjoint_load_form = 1.0;
  // disp_adjoint_load_form = 1.0;

  // FiniteElementDual temp_adjoint_load(thermomech_solver.temperature().space(), "temperature_adjoint_load");
  // FiniteElementDual disp_adjoint_load(thermomech_solver.displacement().space(), "displacement_adjoint_load");
  // temp_adjoint_load = 1.0;
  // disp_adjoint_load = 1.0;

  // // Solve the adjoint problem
  // thermomech_solver.setAdjointLoad({{"temperature", temp_adjoint_load}, {"displacement", disp_adjoint_load}});
  // thermomech_solver.reverseAdjointTimestep();

  // // Compute the sensitivity (d QOI/ d state * d state/d parameter) given the current adjoint solution
  // auto sensitivity = thermomech_solver.computeTimestepSensitivity(sensitivity_parameter_index);

  // // Perform finite difference on each parameter value
  // // to check if computed qoi sensitivity is consistent
  // // with finite difference on the temperature and displacement
  // double eps = 1.0e-5;
  // FiniteElementState* sensitivity_parameter = parameter_ptrs[sensitivity_parameter_index];
  // for (int i = 0; i < sensitivity_parameter->gridFunction().Size(); ++i) {
  //   // Perturb the parameter
  //   (*sensitivity_parameter)(i) = parameter_values[sensitivity_parameter_index] + eps;
  //   thermomech_solver.setTemperature(initTemp);
  //   thermomech_solver.setDisplacement(zeroVector);

  //   thermomech_solver.setParameter(sensitivity_parameter_index, *sensitivity_parameter);
  //   thermomech_solver.advanceTimestep(1.0);

  //   mfem::ParGridFunction temperature_plus = thermomech_solver.temperature().gridFunction();
  //   mfem::ParGridFunction displacement_plus = thermomech_solver.displacement().gridFunction();

  //   (*sensitivity_parameter)(i) = parameter_values[sensitivity_parameter_index] - eps;
  //   thermomech_solver.setTemperature(initTemp);
  //   thermomech_solver.setDisplacement(zeroVector);

  //   thermomech_solver.setParameter(sensitivity_parameter_index, *sensitivity_parameter);
  //   thermomech_solver.advanceTimestep(1.0);

  //   mfem::ParGridFunction temperature_minus = thermomech_solver.temperature().gridFunction();
  //   mfem::ParGridFunction displacement_minus = thermomech_solver.displacement().gridFunction();

  //   // Reset to the original parameter value
  //   (*sensitivity_parameter)(i) = parameter_values[sensitivity_parameter_index];

  //   // Finite difference to compute sensitivity of temperature and displacement w.r.t the parameter
  //   mfem::ParGridFunction dtemp_dparam(
  //       const_cast<mfem::ParFiniteElementSpace*>(&thermomech_solver.temperature().space()));
  //   for (int i2 = 0; i2 < temperature_plus.Size(); ++i2) {
  //     dtemp_dparam(i2) = (temperature_plus(i2) - temperature_minus(i2)) / (2.0 * eps);
  //   }
  //   mfem::ParGridFunction ddisp_dparam(
  //       const_cast<mfem::ParFiniteElementSpace*>(&thermomech_solver.displacement().space()));
  //   for (int i3 = 0; i3 < displacement_plus.Size(); ++i3) {
  //     ddisp_dparam(i3) = (displacement_plus(i3) - displacement_minus(i3)) / (2.0 * eps);
  //   }

  //   // Compute numerical value of sensitivity of qoi w.r.t the parameter
  //   // by taking the inner product between the adjoint load and temperature and displacement sensitivity
  //   double dqoi_dparam_1 = temp_adjoint_load_form(dtemp_dparam);
  //   double dqoi_dparam_2 = disp_adjoint_load_form(ddisp_dparam);
  //   double dqoi_dparam = dqoi_dparam_1 + dqoi_dparam_2;

  //   // See if these are similar
  //   SLIC_INFO(axom::fmt::format("dqoi_dparam: {}", dqoi_dparam));
  //   SLIC_INFO(axom::fmt::format("sensitivity: {}", sensitivity(i)));
  //   double relative_error = (sensitivity(i) - dqoi_dparam) / std::max(dqoi_dparam, 1.0e-2);
  //   EXPECT_NEAR(relative_error, 0.0, 5.0e-4);
}

TEST(ThermomechanicsFictDensParameter, Source) { FiniteDifferenceParameter(0); }

// /**
//  * @brief A driver for a shape sensitivity test
//  *
//  * This performs a finite difference check for the sensitivities of the shape displacements
//  * for various loading types. It can currently only run in serial.
//  *
//  * @param load The type of loading to apply to the problem
//  */
// void FiniteDifferenceShapeTest(LoadingType load)
// {
//   MPI_Barrier(MPI_COMM_WORLD);

//   int serial_refinement = 1;
//   int parallel_refinement = 0;

//   // Create DataStore
//   axom::sidre::DataStore datastore;
//   smith::StateManager::initialize(datastore, "thermomech_functional_shape_sensitivities");

//   // Construct the appropriate dimension mesh and give it to the data store
//   std::string filename = SMITH_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";
//   std::string mesh_tag("mesh");
//   auto mesh =
//       std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

//   constexpr int p = 1;
//   constexpr int dim = 2;

//   FiniteElementState shape_displacement(mesh->mfemParMesh(), H1<SHAPE_ORDER, dim>{});
//   double shape_displacement_value = 1.0;
//   shape_displacement = shape_displacement_value;

//   auto linear_opts = thermomechanics::direct_linear_options;
//   linear_opts.relative_tol = 1e-12;
//   linear_opts.absolute_tol = 1e-12;
//   auto nonlinear_opts = thermomechanics::default_nonlinear_options;
//   nonlinear_opts.relative_tol = 1e-12;
//   nonlinear_opts.absolute_tol = 1e-12;
//   nonlinear_opts.print_level = 0;

//   ThermomechanicsMonolithic<p, dim> thermomech_solver(nonlinear_opts, linear_opts, "thermomech_shape_sensitivities",
//                                                       mesh);
//   thermomech_solver.setShapeDisplacement(shape_displacement);

//   double rho = 1.0;
//   double E = 100.0;
//   double nu = 0.25;
//   double c = 1.0;
//   double alpha = 1.0e-1;
//   double theta_ref = 1.0;
//   double k = 1.0;
//   thermomechanics::GreenSaintVenantThermoelasticMaterial material{rho, E, nu, c, alpha, theta_ref, k};

//   thermomech_solver.setMaterial(DependsOn<>{}, material, mesh->entireBody());

//   std::set<int> temp_flux_bcs = {1, 2};
//   std::set<int> disp_trac_bcs = {3, 4};
//   mesh->addDomainOfBoundaryElements("flux_boundary", by_attr<dim>(temp_flux_bcs));
//   mesh->addDomainOfBoundaryElements("trac_boundary", by_attr<dim>(disp_trac_bcs));

//   std::set<int> temp_ess_bcs = {3, 4};
//   mesh->addDomainOfBoundaryElements("temp_bdr", by_attr<dim>(temp_ess_bcs));
//   thermomech_solver.setTemperatureBCs(temp_ess_bcs, [&theta_ref](const auto&, auto) { return theta_ref; });

//   auto initTemp = [&theta_ref](const mfem::Vector&, double) -> double { return theta_ref; };
//   thermomech_solver.setTemperature(initTemp);

//   std::set<int> disp_ess_bdr_y = {2};
//   std::set<int> disp_ess_bdr_x = {1};
//   mesh->addDomainOfBoundaryElements("ess_y_bdr", by_attr<dim>(disp_ess_bdr_y));
//   mesh->addDomainOfBoundaryElements("ess_x_bdr", by_attr<dim>(disp_ess_bdr_x));
//   thermomech_solver.setFixedBCs(mesh->domain("ess_y_bdr"), Component::Y);
//   thermomech_solver.setFixedBCs(mesh->domain("ess_x_bdr"), Component::X);

//   auto zeroVector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };
//   thermomech_solver.setDisplacement(zeroVector);

//   if (load == LoadingType::Source) {
//     auto source_function = [](auto /* X */, auto /* time */, auto /* T */, auto /* dT_dx */) { return 1.0; };
//     thermomech_solver.setSource(source_function, mesh->entireBody());
//   } else if (load == LoadingType::Flux) {
//     auto flux_bc_function = [](auto /* X */, auto /* n */, auto /* time */, auto /* T */) { return -1.0; };
//     thermomech_solver.setFluxBCs(flux_bc_function, mesh->domain("flux_boundary"));
//   } else if (load == LoadingType::BodyForce) {
//     auto body_force_function = [](auto /* X */, auto /* time */) {
//       tensor<double, dim> bf{};
//       for (int n = 0; n < dim; ++n) {
//         bf[n] = 100.0;
//       }
//       return bf;
//     };
//     thermomech_solver.addBodyForce(body_force_function, mesh->entireBody());
//   } else if (load == LoadingType::Traction) {
//     auto traction_bc_function = [](auto /* X */, auto n0, auto /* T */) { return -1.0 * n0; };
//     thermomech_solver.setTraction(traction_bc_function, mesh->domain("trac_boundary"));
//   }

//   thermomech_solver.completeSetup();

//   thermomech_solver.advanceTimestep(1.0);

//   // Output the sidre-based plot files
//   thermomech_solver.outputStateToDisk();

//   // Construct a dummy adjoint load (this would come from a QOI downstream).
//   // This adjoint load is equivalent to a discrete L1 norm on the temperature and displacement.
//   mfem::ParLinearForm temp_adjoint_load_form(
//       const_cast<mfem::ParFiniteElementSpace*>(&thermomech_solver.temperature().space()));
//   mfem::ParLinearForm disp_adjoint_load_form(
//       const_cast<mfem::ParFiniteElementSpace*>(&thermomech_solver.displacement().space()));
//   temp_adjoint_load_form = 1.0;
//   disp_adjoint_load_form = 1.0;

//   FiniteElementDual temp_adjoint_load(thermomech_solver.temperature().space(), "temperature_adjoint_load");
//   FiniteElementDual disp_adjoint_load(thermomech_solver.displacement().space(), "displacement_adjoint_load");
//   temp_adjoint_load = 1.0;
//   disp_adjoint_load = 1.0;

//   // Solve the adjoint problem
//   thermomech_solver.setAdjointLoad({{"temperature", temp_adjoint_load}, {"displacement", disp_adjoint_load}});
//   thermomech_solver.reverseAdjointTimestep();

//   // Compute the sensitivity (d QOI/ d state * d state/d shape) given the current adjoint solution
//   [[maybe_unused]] auto& sensitivity = thermomech_solver.computeTimestepShapeSensitivity();

//   // Perform finite difference on each shape displacement value
//   // to check if computed qoi sensitivity is consistent
//   // with finite difference on the displacement
//   double eps = 1.0e-5;
//   for (int i = 0; i < shape_displacement.Size(); ++i) {
//     // Perturb the shape field
//     shape_displacement(i) = shape_displacement_value + eps;
//     thermomech_solver.setTemperature(initTemp);
//     thermomech_solver.setDisplacement(zeroVector);

//     thermomech_solver.setShapeDisplacement(shape_displacement);
//     thermomech_solver.advanceTimestep(1.0);

//     mfem::ParGridFunction temperature_plus = thermomech_solver.temperature().gridFunction();
//     mfem::ParGridFunction displacement_plus = thermomech_solver.displacement().gridFunction();

//     shape_displacement(i) = shape_displacement_value - eps;
//     thermomech_solver.setTemperature(initTemp);
//     thermomech_solver.setDisplacement(zeroVector);

//     thermomech_solver.setShapeDisplacement(shape_displacement);
//     thermomech_solver.advanceTimestep(1.0);

//     mfem::ParGridFunction temperature_minus = thermomech_solver.temperature().gridFunction();
//     mfem::ParGridFunction displacement_minus = thermomech_solver.displacement().gridFunction();

//     // Reset to the original shape displacement value
//     shape_displacement(i) = shape_displacement_value;

//     // Finite difference to compute sensitivity of temperature and displacement w.r.t the shape displacement
//     mfem::ParGridFunction dtemp_dshape(
//         const_cast<mfem::ParFiniteElementSpace*>(&thermomech_solver.temperature().space()));
//     for (int i2 = 0; i2 < temperature_plus.Size(); ++i2) {
//       dtemp_dshape(i2) = (temperature_plus(i2) - temperature_minus(i2)) / (2.0 * eps);
//     }
//     mfem::ParGridFunction ddisp_dshape(
//         const_cast<mfem::ParFiniteElementSpace*>(&thermomech_solver.displacement().space()));
//     for (int i3 = 0; i3 < displacement_plus.Size(); ++i3) {
//       ddisp_dshape(i3) = (displacement_plus(i3) - displacement_minus(i3)) / (2.0 * eps);
//     }

//     // Compute numerical value of sensitivity of qoi w.r.t the shape displacement
//     // by taking the inner product between the adjoint load and temperature and displacement sensitivity
//     double dqoi_dshape_1 = temp_adjoint_load_form(dtemp_dshape);
//     double dqoi_dshape_2 = disp_adjoint_load_form(ddisp_dshape);
//     double dqoi_dshape = dqoi_dshape_1 + dqoi_dshape_2;

//     // See if these are similar
//     SLIC_INFO(axom::fmt::format("dqoi_dshape: {}", dqoi_dshape));
//     SLIC_INFO(axom::fmt::format("sensitivity: {}", sensitivity(i)));
//     double relative_error = (sensitivity(i) - dqoi_dshape) / std::max(dqoi_dshape, 1.0e-2);
//     EXPECT_NEAR(relative_error, 0.0, 5.0e-4);
//   }
// }

// TEST(ThermomechanicsShape, Source) { FiniteDifferenceShapeTest(LoadingType::Source); }
// TEST(ThermomechanicsShape, Flux) { FiniteDifferenceShapeTest(LoadingType::Flux); }
// TEST(ThermomechanicsShape, BodyForce) { FiniteDifferenceShapeTest(LoadingType::BodyForce); }
// TEST(ThermomechanicsShape, Traction) { FiniteDifferenceShapeTest(LoadingType::Traction); }
}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
