// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"

#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/thermo_mechanics_system.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/physics/functional_objective.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"

namespace smith {

/**
 * @brief Compute Green's strain from the displacement gradient
 */
template <typename T, int dim>
auto greenStrain(const tensor<T, dim, dim>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

/// @brief Green-Saint Venant isotropic thermoelastic model
/// This an unparametrized version of the model in green_saint_venant_thermoelastic.hpp
/// Another difference is that this implementation does not use 'State' as state is not supported in smith for
/// autodifferentiation
struct GreenSaintVenantThermoelasticMaterial {
  double density;    ///< density
  double E0;         ///< base Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha;      ///< thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa;      ///< thermal conductivity

  using State = Empty;

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the temperature (number-like)
   * @tparam T3 Type of the temperature gradient components (number-like)
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] theta Temperature
   * @param[in] grad_theta Temperature gradient
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * First Piola stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */
  template <typename T1, typename T2, typename T3, typename T4, typename T5, int dim>
  auto operator()(double, State&, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta, const T5& E_param) const
  {
    auto E = E0 + get<0>(E_param);
    const auto K = E / (3.0 * (1.0 - 2.0 * nu));
    const auto G = 0.5 * E / (1.0 + nu);
    const auto Eg = greenStrain(grad_u);
    const auto trEg = tr(Eg);

    // stress
    static constexpr auto I = Identity<dim>();
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - dim * alpha * (theta - theta_ref)) * I;
    auto F = grad_u + I;
    const auto Piola = dot(F, S);

    // internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate) + 0.0 * E;

    // heat flux
    const auto q0 = -kappa * grad_theta;

    return smith::tuple{Piola, C_v, s0, q0};
  }

  static constexpr int numParameters() { return 1; }
};

smith::LinearSolverOptions thermo_linear_options{.linear_solver = LinearSolver::SuperLU,
                                          .preconditioner = Preconditioner::None,
                                          .relative_tol = 1e-8,
                                          .absolute_tol = 1e-8,
                                          .max_iterations = 200,
                                          .print_level = 0};

smith::NonlinearSolverOptions nonlinear_opts{.nonlin_solver = NonlinearSolver::Newton,
                                             .relative_tol = 1.0e-10,
                                             .absolute_tol = 1.0e-10,
                                             .max_iterations = 1000,
                                             .max_line_search_iterations = 50,
                                             .print_level = 2};

static constexpr int dim = 3;
static constexpr int vdim = dim;
static constexpr int displacement_order = 1;
static constexpr int temperature_order = 1;

struct ThermoMechanicsMeshFixture : public testing::Test {
  double length = 1.0;
  double width = 0.04;
  int num_elements_x = 4;
  int num_elements_y = 1;
  int num_elements_z = 1;
  double elem_size = length / num_elements_x;

  void SetUp()
  {
    smith::StateManager::initialize(datastore_, "solid");
    auto mfem_shape = mfem::Element::QUADRILATERAL;
    mesh_ = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian3D(num_elements_x, num_elements_y, num_elements_z, mfem_shape, length, width, width),
        "mesh", 0, 0);
    mesh_->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
    mesh_->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));
  }

  axom::sidre::DataStore datastore_;
  std::shared_ptr<smith::Mesh> mesh_;
};

TEST_F(ThermoMechanicsMeshFixture, MonolithicVsStaggered)
{
  SMITH_MARK_FUNCTION;
  
  auto run_problem = [&](std::shared_ptr<SystemSolver> sys_solver, bool test_gradients) {
    double rho = 1.0;
    double E0 = 100.0;
    double nu = 0.25;
    double specific_heat = 1.0;
    double kappa = 0.1;
    // Use alpha = 0.01 to ensure coupling
    GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, 0.01, 1.0, kappa};

    FieldType<L2<0>> youngs_modulus("youngs_modulus");
    auto system =
        buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh_, sys_solver, youngs_modulus);
    system.setMaterial(material, mesh_->entireBodyName());

    system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

    system.disp_bc->setVectorBCs<dim>(mesh_->domain("left"), [](double t, smith::tensor<double, dim> X) {
      auto bc = 0.0 * X;
      bc[0] = 0.01 * t;
      return bc;
    });
    system.disp_bc->setFixedVectorBCs<dim>(mesh_->domain("right"));
    system.temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("left"));
    system.temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("right"));

    system.addThermalHeatSource(mesh_->entireBodyName(), [](auto /*t*/, auto /*x*/, auto /*u*/, auto /*v*/, auto /*T*/,
                                                            auto /*T_dot*/, auto /*E_param*/) { return 100.0; });

    double dt = 0.001;
    double time = 0.0;
    int cycle = 0;

    auto shape_disp = system.field_store->getShapeDisp();
    auto states = system.getStateFields();
    auto params = system.getParameterFields();
    std::vector<ReactionState> reactions;

    for (size_t step = 0; step < 2; ++step) {
      TimeInfo t_info(time, dt, step);
      std::tie(states, reactions) = system.advancer->advanceState(t_info, shape_disp, states, params);
      time += dt;
      cycle++;
    }

    // Check that reactions are zero for unconstrained DOFs (should be within solver tolerance)
    checkUnconstrainedReactions(*reactions[0].get(), system.disp_bc->getBoundaryConditionManager());
    checkUnconstrainedReactions(*reactions[1].get(), system.temperature_bc->getBoundaryConditionManager());

    if (test_gradients) {
      auto reaction_squared = innerProduct(reactions[0], reactions[0]);
      gretl::set_as_objective(reaction_squared);

      EXPECT_GT(checkGradWrt(reaction_squared, shape_disp, 1.1e-2, 4, true), 0.7);
      EXPECT_GT(checkGradWrt(reaction_squared, params[0], 6.2e-1, 4, true), 0.7);
    }

    // Return copies of the final fields
    auto disp_idx = system.field_store->getFieldIndex("displacement_predicted");
    auto temp_idx = system.field_store->getFieldIndex("temperature_predicted");
    return std::make_pair(mfem::Vector(*states[disp_idx].get()), mfem::Vector(*states[temp_idx].get()));
  };

  auto solver = buildDifferentiableSolver(nonlinear_opts, thermo_linear_options, *mesh_);

  // Run Monolithic
  auto mono_sys_solver = std::make_shared<SystemSolver>(solver);
  auto mono_result = run_problem(mono_sys_solver, true);

  // Reset StateManager for the next run
  smith::StateManager::reset();
  this->SetUp(); // Re-initialize StateManager and Mesh

  // Run Staggered
  int max_staggered_iterations = 10;
  auto stag_sys_solver = std::make_shared<SystemSolver>(max_staggered_iterations);
  stag_sys_solver->addStage({0}, solver);
  stag_sys_solver->addStage({1}, solver);
  auto stag_result = run_problem(stag_sys_solver, false); // Skip gradients for staggered for now

  // Compare results
  mfem::Vector diff_disp = mono_result.first;
  diff_disp -= stag_result.first;
  EXPECT_LT(diff_disp.Normlinf(), 1e-6);

  mfem::Vector diff_temp = mono_result.second;
  diff_temp -= stag_result.second;
  EXPECT_LT(diff_temp.Normlinf(), 1e-6);
}

TEST_F(ThermoMechanicsMeshFixture, TransientHeatEquationAnalytic)
{
  SMITH_MARK_FUNCTION;

  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, 0.0, 0.0, kappa};

  auto solver = buildDifferentiableSolver(nonlinear_opts, thermo_linear_options, *mesh_);
  FieldType<L2<0>> youngs_modulus("youngs_modulus");
  auto sys_solver = std::make_shared<SystemSolver>(solver);
  auto system =
      buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh_, sys_solver, youngs_modulus);
  system.setMaterial(material, mesh_->entireBodyName());

  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

  system.disp_bc->setFixedVectorBCs<dim>(mesh_->domain("left"));
  system.disp_bc->setFixedVectorBCs<dim>(mesh_->domain("right"));
  system.temperature_bc->setScalarBCs<dim>(mesh_->domain("left"), [](double, auto) { return 100.0; });
  system.temperature_bc->setScalarBCs<dim>(mesh_->domain("right"), [](double, auto) { return 100.0; });

  auto& temp_field = const_cast<FiniteElementState&>(*system.field_store->getField("temperature_predicted").get());
  temp_field.setFromFieldFunction([](tensor<double, dim> x) {
    using std::sin;
    return 100.0 + sin(M_PI * x[0]);
  });
  const_cast<FiniteElementState&>(*system.field_store->getField("temperature").get()) = temp_field;

  double dt = 0.01;
  double time = 0.0;
  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();
  auto params = system.getParameterFields();
  std::vector<ReactionState> reactions;

  double diffusivity = kappa / (rho * specific_heat);

  size_t num_steps = 2;
  for (size_t step = 0; step < num_steps; ++step) {
    TimeInfo t_info(time, dt, step);
    std::tie(states, reactions) = system.advancer->advanceState(t_info, shape_disp, states, params);
    time += dt;
  }

  // Check that reactions are zero for unconstrained DOFs
  checkUnconstrainedReactions(*reactions[0].get(), system.disp_bc->getBoundaryConditionManager());
  checkUnconstrainedReactions(*reactions[1].get(), system.temperature_bc->getBoundaryConditionManager());

  // Check nodal error against exact solution: T(x,t) = 100 + sin(pi*x) * exp(-diffusivity * pi^2 * t)
  auto temp_idx = system.field_store->getFieldIndex("temperature_predicted");
  FieldState final_temp = states[temp_idx];

  FiniteElementState exact_temp(final_temp.get()->space(), "exact_temp");
  exact_temp.setFromFieldFunction([&](tensor<double, dim> x) {
    return 100.0 + std::sin(M_PI * x[0]) * std::exp(-diffusivity * M_PI * M_PI * time);
  });
  mfem::Vector diff(final_temp.get()->Size());
  subtract(*final_temp.get(), exact_temp, diff);
  double max_nodal_error = diff.Normlinf();

  std::cout << "Transient Heat max nodal error: " << max_nodal_error << std::endl;
  EXPECT_LT(max_nodal_error, 1e-4);
}

TEST_F(ThermoMechanicsMeshFixture, StaticElasticityAnalytic)
{
  SMITH_MARK_FUNCTION;

  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, 0.0, 0.0, kappa};

  auto solver = buildDifferentiableSolver(nonlinear_opts, thermo_linear_options, *mesh_);
  FieldType<H1<1>> youngs_modulus("youngs_modulus");
  auto sys_solver = std::make_shared<SystemSolver>(solver);
  auto system =
      buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh_, sys_solver, youngs_modulus);
  system.setMaterial(material, mesh_->entireBodyName());

  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

  // Arbitrary affine displacement: u(X) = G * X, where G is a constant displacement gradient
  // Choose a small uniform deformation with both normal and shear components
  tensor<double, dim, dim> G;
  G[0][0] = 0.02;
  G[0][1] = 0.01;
  G[0][2] = 0.005;
  G[1][0] = 0.01;
  G[1][1] = 0.03;
  G[1][2] = 0.002;
  G[2][0] = 0.005;
  G[2][1] = 0.002;
  G[2][2] = 0.015;

  auto u_exact_func = [G](auto X) { return dot(G, X); };

  system.disp_bc->setVectorBCs<dim>(mesh_->entireBoundary(),
                                    [=](double, tensor<double, dim> X) { return u_exact_func(X); });
  system.temperature_bc->setFixedScalarBCs<dim>(mesh_->entireBoundary());

  double dt = 1.0;
  double time = 0.0;
  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();
  auto params = system.getParameterFields();
  std::vector<ReactionState> reactions;

  // Run 1 step
  TimeInfo t_info(time, dt, 0);
  auto states_and_reactions = system.advancer->advanceState(t_info, shape_disp, states, params);
  states = states_and_reactions.first;
  reactions = states_and_reactions.second;

  // Check that reactions are zero for unconstrained DOFs
  checkUnconstrainedReactions(*reactions[0].get(), system.disp_bc->getBoundaryConditionManager());
  checkUnconstrainedReactions(*reactions[1].get(), system.temperature_bc->getBoundaryConditionManager());

  // Check error - for affine displacement, FEM solution should be exact (up to machine precision)
  auto disp_idx = system.field_store->getFieldIndex("displacement_predicted");
  FieldState final_disp = states[disp_idx];

  FunctionalObjective<dim, Parameters<H1<displacement_order, vdim>>> error_obj("error", mesh_, spaces({final_disp}));
  error_obj.addBodyIntegral(DependsOn<0>{}, mesh_->entireBodyName(), [=](auto /*t*/, auto X, auto U) {
    auto u_val = get<VALUE>(U);
    auto x_val = get<0>(X);
    auto exact = u_exact_func(x_val);
    return inner(u_val - exact, u_val - exact);
  });

  double l2_error_sq =
      error_obj.evaluate(TimeInfo(time + dt, dt, 0), shape_disp.get().get(), getConstFieldPointers({final_disp}));
  double l2_error = std::sqrt(l2_error_sq);

  std::cout << "Static Elasticity L2 Error (affine patch test): " << l2_error << std::endl;
  EXPECT_LT(l2_error, 1e-11);  // Should be machine precision for affine displacement
}

TEST_F(ThermoMechanicsMeshFixture, TransientThermoMechanicsCompilation)
{
  SMITH_MARK_FUNCTION;

  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, 0.0, 1.0, kappa};

  auto fast_nonlinear_opts = nonlinear_opts;
  fast_nonlinear_opts.max_iterations = 5;
  auto solver = buildDifferentiableSolver(fast_nonlinear_opts, thermo_linear_options, *mesh_);

  FieldType<L2<0>> youngs_modulus("youngs_modulus");
  auto sys_solver = std::make_shared<SystemSolver>(solver);
  auto system =
      buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh_, sys_solver, youngs_modulus);
  system.setMaterial(material, mesh_->entireBodyName());

  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

  // Add Solid Body Force (Gravity)
  system.addSolidBodyForce(mesh_->entireBodyName(), [](double /*time*/, auto /*X*/, auto /*u*/, auto /*v*/, auto /*T*/,
                                                       auto /*T_dot*/, auto... /*params*/) {
    tensor<double, dim> f{};
    f[1] = -9.81;
    return f;
  });

  // Add Solid Traction
  system.addSolidTraction("right", [](double /*time*/, auto /*X*/, auto /*n*/, auto /*u*/, auto /*v*/, auto /*T*/,
                                      auto /*T_dot*/, auto... /*params*/) {
    tensor<double, dim> t{};
    t[0] = 1.0;
    return t;
  });

  // Add Thermal Heat Source
  system.addThermalHeatSource(mesh_->entireBodyName(),
                              [](double /*time*/, auto /*X*/, auto /*u*/, auto /*v*/, auto /*T*/, auto /*T_dot*/,
                                 auto... /*params*/) { return 10.0; });

  // Add Thermal Heat Flux
  system.addThermalHeatFlux("left", [](double /*time*/, auto /*X*/, auto /*n*/, auto /*u*/, auto /*v*/, auto /*T*/,
                                       auto /*T_dot*/, auto... /*params*/) {
    return 5.0;  // Flux into domain
  });

  system.disp_bc->setVectorBCs<dim>(mesh_->domain("left"),
                                    [](double /*t*/, smith::tensor<double, dim> X) { return 0.0 * X; });
  // Don't run anything, just make sure the templates build as expected.
}

TEST_F(ThermoMechanicsMeshFixture, PressureBC)
{
  SMITH_MARK_FUNCTION;

  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, 0.0, 1.0, kappa};

  auto solver = buildDifferentiableSolver(nonlinear_opts, thermo_linear_options, *mesh_);
  FieldType<L2<0>> youngs_modulus("youngs_modulus");
  auto sys_solver = std::make_shared<SystemSolver>(solver);
  auto system =
      buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh_, sys_solver, youngs_modulus);
  system.setMaterial(material, mesh_->entireBodyName());

  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

  // Fixed left side
  system.disp_bc->setFixedVectorBCs<dim>(mesh_->domain("left"));
  system.temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("left"));
  system.temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("right"));

  // Apply pressure on right side
  double pressure_mag = 0.1;
  system.addPressure("right", [pressure_mag](double, auto, auto, auto, auto, auto, auto...) { return pressure_mag; });

  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();
  auto params = system.getParameterFields();

  double dt = 0.001;
  double time = 0.0;
  size_t cycle = 0;
  // Advance one step
  TimeInfo t_info(time, dt, cycle);
  auto result = system.advancer->advanceState(t_info, shape_disp, states, params);
  states = result.first;

  // Check that we have some displacement
  auto disp_idx = system.field_store->getFieldIndex("displacement_predicted");
  FieldState final_disp = states[disp_idx];

  // We expect non-zero displacement
  double disp_norm = final_disp.get()->Norml2();
  EXPECT_GT(disp_norm, 1e-6);
}



TEST_F(ThermoMechanicsMeshFixture, StaggeredWithDifferentSolvers)
{
  SMITH_MARK_FUNCTION;
  // Use a material with actual thermal expansion
  double rho = 1.0;
  double E0 = 1000.0;
  double nu = 0.3;
  double specific_heat = 10.0;
  double alpha = 0.01;
  double kappa = 1.0;
  GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, alpha, 0.0, kappa};

  // Mechanics Solver: CG with HypreJacobi
  smith::LinearSolverOptions mech_lin_opts{.linear_solver = smith::LinearSolver::CG,
                                           .preconditioner = smith::Preconditioner::HypreJacobi,
                                           .relative_tol = 1e-6,
                                           .absolute_tol = 1e-6,
                                           .max_iterations = 200,
                                           .print_level = 0};
  smith::NonlinearSolverOptions mech_nonlin_opts = nonlinear_opts; // Newton Line Search
  auto mech_solver = buildDifferentiableSolver(mech_nonlin_opts, mech_lin_opts, *mesh_);

  // Thermal Solver: GMRES with HypreAMG
  smith::LinearSolverOptions therm_lin_opts{.linear_solver = smith::LinearSolver::GMRES,
                                            .preconditioner = smith::Preconditioner::HypreAMG,
                                            .relative_tol = 1e-6,
                                            .absolute_tol = 1e-6,
                                            .max_iterations = 200,
                                            .print_level = 0};
  smith::NonlinearSolverOptions therm_nonlin_opts = nonlinear_opts;
  auto therm_solver = buildDifferentiableSolver(therm_nonlin_opts, therm_lin_opts, *mesh_);

  FieldType<L2<0>> youngs_modulus("youngs_modulus");
  int max_staggered_iterations = 15;
  auto sys_solver = std::make_shared<SystemSolver>(max_staggered_iterations);
  sys_solver->addStage({0}, mech_solver);
  sys_solver->addStage({1}, therm_solver);

  auto system =
      buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh_, sys_solver, youngs_modulus);
  system.setMaterial(material, mesh_->entireBodyName());

  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

  // Fixed on the left
  system.disp_bc->setFixedVectorBCs<dim>(mesh_->domain("left"));
  // Heat up the left boundary
  system.temperature_bc->setScalarBCs<dim>(mesh_->domain("left"), [](double t, smith::tensor<double, dim>) {
    return 100.0 * std::min(t, 1.0); // Ramp up temperature
  });

  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();
  auto params = system.getParameterFields();
  std::vector<ReactionState> reactions;

  double dt = 0.1;
  double time = 0.0;

  // Solve for 5 steps to see expansion
  for (size_t step = 0; step < 5; ++step) {
    TimeInfo t_info(time, dt, step);
    std::tie(states, reactions) = system.advancer->advanceState(t_info, shape_disp, states, params);
    time += dt;
  }

  // Check that the bar expanded in the x-direction due to thermal heating
  auto disp_idx = system.field_store->getFieldIndex("displacement_predicted");
  FieldState final_disp = states[disp_idx];
  
  double disp_norm = final_disp.get()->Normlinf();
  EXPECT_GT(disp_norm, 1e-3);
  SLIC_INFO_ROOT("Max displacement after heating: " << disp_norm);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
