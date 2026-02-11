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
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/physics/functional_objective.hpp"
#include "gretl/strumm_walther_checkpoint_strategy.hpp"

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
struct GreenSaintVenantThermoelasticMaterial {
  double density;    ///< density
  double E;          ///< Young's modulus
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
  template <typename T1, typename T2, typename T3, typename T4, int dim>
  auto operator()(double, State&, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v, T3 theta,
                  const tensor<T4, dim>& grad_theta) const
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
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
    const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);

    // heat flux
    const auto q0 = -kappa * grad_theta;

    return smith::tuple{Piola, C_v, s0, q0};
  }
};

smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack,
                                          .relative_tol = 1e-8,
                                          .absolute_tol = 1e-8,
                                          .max_iterations = 200,
                                          .print_level = 0};

smith::NonlinearSolverOptions nonlinear_opts{.nonlin_solver = NonlinearSolver::NewtonLineSearch,
                                             .relative_tol = 1.9e-6,
                                             .absolute_tol = 1.0e-10,
                                             .max_iterations = 500,
                                             .max_line_search_iterations = 50,
                                             .print_level = 2};

static constexpr int dim = 3;
static constexpr int order = 1;

struct SolidMechanicsMeshFixture : public testing::Test {
  double length = 1.0;
  double width = 0.04;
  int num_elements_x = 12;
  int num_elements_y = 2;
  int num_elements_z = 2;
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

  static constexpr double total_simulation_time_ = 1.1;
  static constexpr size_t num_steps_ = 4;
  static constexpr double dt_ = total_simulation_time_ / num_steps_;

  axom::sidre::DataStore datastore_;
  std::shared_ptr<smith::Mesh> mesh_;
};


TEST_F(SolidMechanicsMeshFixture, RunThermoMechanicalCoupled)
{
  SMITH_MARK_FUNCTION;

  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  FieldType<H1<order, dim>> disp_type("displacement");
  FieldType<H1<order>> temperature_type("temperature");

  std::shared_ptr<DifferentiableBlockSolver> d_nonlinear_solver =
      buildDifferentiableNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh_);

  double rho = 1.0;
  double E = 100.0;
  double nu = 0.25;
  double c = 1.0;
  double alpha = 1.0e-3;
  double theta_ref = 0.0;
  double k = 1.0;
  GreenSaintVenantThermoelasticMaterial material{rho, E, nu, c, alpha, theta_ref, k};

  auto field_store = std::make_shared<FieldStore>(mesh_, 100);
  field_store->addShapeDisp(shape_disp_type);

  // register states associated with the displacement

  auto disp_time_rule = std::make_shared<QuasiStaticFirstOrderTimeIntegrationRule>();
  std::shared_ptr<DirichletBoundaryConditions> disp_bc = field_store->addIndependent(disp_type, disp_time_rule);
  disp_bc->setVectorBCs<dim>(mesh_->domain("left"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    bc[0] = 0.01 * t;
    return bc;
  });
  disp_bc->setFixedVectorBCs<dim, dim>(mesh_->domain("right"));
  
  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VALUE);
  
  auto temperature_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();

  std::shared_ptr<DirichletBoundaryConditions> temperature_bc = field_store->addIndependent(temperature_type, temperature_time_rule);
  temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("left"));
  temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("right"));

  auto temperature_old_type = field_store->addDependent(temperature_type, FieldStore::TimeDerivative::VALUE);

  auto solid_weak_form = createWeakForm<dim>("solid_force", disp_type, *field_store, disp_type, disp_old_type,
                                             temperature_type, temperature_old_type);

  solid_weak_form->addBodyIntegral(mesh_->entireBodyName(), [=](auto t_info, auto /*X*/, auto disp, auto disp_old,
                                                                auto temperature, auto temperature_old) {
    auto u = disp_time_rule->value(t_info, disp, disp_old);
    auto v = disp_time_rule->dot(t_info, disp, disp_old);
    auto T = temperature_time_rule->value(t_info, temperature, temperature_old);
    GreenSaintVenantThermoelasticMaterial::State state;
    auto [pk, C_v, s0, q0] =
        material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T), get<DERIVATIVE>(T));
    return smith::tuple{smith::zero{}, pk};
  });

  auto thermal_weak_form = createWeakForm<dim>("thermal_flux", temperature_type, *field_store, temperature_type,
                                               temperature_old_type, disp_type, disp_old_type);

  // native unknowns...
  thermal_weak_form->addBodyIntegral(mesh_->entireBodyName(), [=](auto t_info, auto /*X*/, auto temperature,
                                                                  auto temperature_old, auto disp, auto disp_old) {
    GreenSaintVenantThermoelasticMaterial::State state;
    auto u = disp_time_rule->value(t_info, disp, disp_old);
    auto v = disp_time_rule->dot(t_info, disp, disp_old);
    auto T = temperature_time_rule->value(t_info, temperature, temperature_old);
    auto T_dot = temperature_time_rule->dot(t_info, temperature, temperature_old);
    auto [pk, C_v, s0, q0] =
        material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T), get<DERIVATIVE>(T));
    auto dT_dt = get<VALUE>(T_dot);
    return smith::tuple{C_v * dT_dt - s0, -q0};
  });

  thermal_weak_form->addBodySource(smith::DependsOn<>(), mesh_->entireBodyName(),
                                   [](auto /*t*/, auto /* x */) { return 100.0; });

  std::vector<std::shared_ptr<WeakForm>> weak_forms{solid_weak_form, thermal_weak_form};
  MultiPhysicsTimeIntegrator advancer(field_store, weak_forms, d_nonlinear_solver);
  
  std::string pv_dir = "paraview_thermo_mechanics";
  auto pv_writer = smith::createParaviewWriter(*mesh_, field_store->getAllFields(), pv_dir);
  
  double dt = 0.001;
  double time = 0.0;
  int cycle = 0;

  auto shape_disp = field_store->getShapeDisp();
  auto states = field_store->getAllFields();

  pv_writer.write(cycle, time, states);
  for (size_t step = 0; step < 10; ++step) {
     TimeInfo t_info(time, dt, step);
     auto states_and_reactions = advancer.advanceState(t_info, shape_disp, states, {});
     states = states_and_reactions.first;
     time += dt;
     cycle++;
     pv_writer.write(cycle, time, states);
  }

  EXPECT_EQ(0, 0);
}

TEST_F(SolidMechanicsMeshFixture, TransientHeatEquationAnalytic)
{
  SMITH_MARK_FUNCTION;

  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  FieldType<H1<order, dim>> disp_type("displacement");
  FieldType<H1<order>> temperature_type("temperature");

  std::shared_ptr<DifferentiableBlockSolver> d_nonlinear_solver =
      buildDifferentiableNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh_);

  double rho = 1.0;
  double E = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double alpha = 0.0;  // Decoupled
  double theta_ref = 0.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E, nu, specific_heat, alpha, theta_ref, kappa};

  auto field_store = std::make_shared<FieldStore>(mesh_, 100);
  field_store->addShapeDisp(shape_disp_type);

  auto disp_time_rule = std::make_shared<QuasiStaticFirstOrderTimeIntegrationRule>();
  std::shared_ptr<DirichletBoundaryConditions> disp_bc = field_store->addIndependent(disp_type, disp_time_rule);
  disp_bc->setFixedVectorBCs<dim, dim>(mesh_->domain("left"));
  disp_bc->setFixedVectorBCs<dim, dim>(mesh_->domain("right"));

  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VALUE);

  auto temperature_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc =
      field_store->addIndependent(temperature_type, temperature_time_rule);
  // BCs: T = 100 at x=0 (left) and x=1 (right). Assuming mesh length is 1.0.
  temperature_bc->setScalarBCs<dim>(mesh_->domain("left"), [](double, auto) { return 100.0; });
  temperature_bc->setScalarBCs<dim>(mesh_->domain("right"), [](double, auto) { return 100.0; });

  auto temperature_old_type = field_store->addDependent(temperature_type, FieldStore::TimeDerivative::VALUE);

  auto solid_weak_form = createWeakForm<dim>("solid_force", disp_type, *field_store, disp_type, disp_old_type,
                                             temperature_type, temperature_old_type);

  solid_weak_form->addBodyIntegral(mesh_->entireBodyName(), [=](auto t_info, auto /*X*/, auto disp, auto disp_old,
                                                                auto temperature, auto temperature_old) {
    auto u = disp_time_rule->value(t_info, disp, disp_old);
    auto v = disp_time_rule->dot(t_info, disp, disp_old);
    auto T = temperature_time_rule->value(t_info, temperature, temperature_old);
    GreenSaintVenantThermoelasticMaterial::State state;
    auto [pk, C_v, s0, q0] =
        material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T), get<DERIVATIVE>(T));
    return smith::tuple{smith::zero{}, pk};
  });

  auto thermal_weak_form = createWeakForm<dim>("thermal_flux", temperature_type, *field_store, temperature_type,
                                               temperature_old_type, disp_type, disp_old_type);

  thermal_weak_form->addBodyIntegral(mesh_->entireBodyName(), [=](auto t_info, auto /*X*/, auto temperature,
                                                                  auto temperature_old, auto disp, auto disp_old) {
    GreenSaintVenantThermoelasticMaterial::State state;
    auto u = disp_time_rule->value(t_info, disp, disp_old);
    auto v = disp_time_rule->dot(t_info, disp, disp_old);
    auto T = temperature_time_rule->value(t_info, temperature, temperature_old);
    auto T_dot = temperature_time_rule->dot(t_info, temperature, temperature_old);
    auto [pk, C_v, s0, q0] =
        material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T), get<DERIVATIVE>(T));
    auto dT_dt = get<VALUE>(T_dot);
    return smith::tuple{C_v * dT_dt - s0, -q0};
  });

  std::vector<std::shared_ptr<WeakForm>> weak_forms{solid_weak_form, thermal_weak_form};
  MultiPhysicsTimeIntegrator advancer(field_store, weak_forms, d_nonlinear_solver);

  // Set Initial Condition
  auto& temp_field = const_cast<FiniteElementState&>(*field_store->getField("temperature").get());
  temp_field.setFromFieldFunction([](tensor<double, dim> x) {
    using std::sin;
    return 100.0 + sin(M_PI * x[0]);
  });
  // Also set old temperature to IC for consistency at t=0
  const_cast<FiniteElementState&>(*field_store->getField("temperature_old").get()) = temp_field;

  double dt = 0.01;
  double time = 0.0;
  auto shape_disp = field_store->getShapeDisp();
  auto states = field_store->getAllFields();

  double diffusivity = kappa / (rho * specific_heat);

  size_t num_steps = 10;
  for (size_t step = 0; step < num_steps; ++step) {
    TimeInfo t_info(time, dt, step);
    auto states_and_reactions = advancer.advanceState(t_info, shape_disp, states, {});
    states = states_and_reactions.first;
    time += dt;
  }

  // Check nodal error against exact solution: T(x,t) = 100 + sin(pi*x) * exp(-diffusivity * pi^2 * t)
  auto temp_idx = field_store->getFieldIndex("temperature");
  FieldState final_temp = states[temp_idx];

  FiniteElementState exact_temp(final_temp.get()->space(), "exact_temp");
  exact_temp.setFromFieldFunction([&](tensor<double, dim> x) {
    return 100.0 + std::sin(M_PI * x[0]) * std::exp(-diffusivity * M_PI * M_PI * time);
  });
  mfem::Vector diff(final_temp.get()->Size());
  subtract(*final_temp.get(), exact_temp, diff);
  double max_nodal_error = diff.Normlinf();

  std::cout << "Transient Heat max nodal error: " << max_nodal_error << std::endl;
  EXPECT_LT(max_nodal_error, 1e-3);
}

TEST_F(SolidMechanicsMeshFixture, StaticElasticityAnalytic)
{
  SMITH_MARK_FUNCTION;

  FieldType<H1<1, dim>> shape_disp_type("shape_displacement");
  FieldType<H1<order, dim>> disp_type("displacement");
  FieldType<H1<order>> temperature_type("temperature");

  std::shared_ptr<DifferentiableBlockSolver> d_nonlinear_solver =
      buildDifferentiableNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh_);

  double rho = 1.0;
  double E = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double alpha = 0.0;  // Decoupled
  double theta_ref = 0.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E, nu, specific_heat, alpha, theta_ref, kappa};

  auto field_store = std::make_shared<FieldStore>(mesh_, 100);
  field_store->addShapeDisp(shape_disp_type);

  auto disp_time_rule = std::make_shared<QuasiStaticFirstOrderTimeIntegrationRule>();
  std::shared_ptr<DirichletBoundaryConditions> disp_bc = field_store->addIndependent(disp_type, disp_time_rule);
  
  double lambda = 0.1;
  auto u_exact_func = [=](auto X) {
      return lambda * X;
  };

  // BCs: u = lambda * X on boundary
  disp_bc->setVectorBCs<dim>(mesh_->entireBoundary(), [=](double, tensor<double, dim> X) {
    return u_exact_func(X);
  });

  auto disp_old_type = field_store->addDependent(disp_type, FieldStore::TimeDerivative::VALUE);

  auto temperature_time_rule = std::make_shared<BackwardEulerFirstOrderTimeIntegrationRule>();
  std::shared_ptr<DirichletBoundaryConditions> temperature_bc =
      field_store->addIndependent(temperature_type, temperature_time_rule);
  temperature_bc->setFixedScalarBCs<dim>(mesh_->entireBoundary());

  auto temperature_old_type = field_store->addDependent(temperature_type, FieldStore::TimeDerivative::VALUE);

  auto solid_weak_form = createWeakForm<dim>("solid_force", disp_type, *field_store, disp_type, disp_old_type,
                                             temperature_type, temperature_old_type);

  solid_weak_form->addBodyIntegral(mesh_->entireBodyName(), [=](auto t_info, auto /*X*/, auto disp, auto disp_old,
                                                                auto temperature, auto temperature_old) {
    auto u = disp_time_rule->value(t_info, disp, disp_old);
    auto v = disp_time_rule->dot(t_info, disp, disp_old);
    auto T = temperature_time_rule->value(t_info, temperature, temperature_old);
    GreenSaintVenantThermoelasticMaterial::State state;
    auto [pk, C_v, s0, q0] =
        material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T), get<DERIVATIVE>(T));
    return smith::tuple{smith::zero{}, pk};
  });

  // Thermal part needed for existence of T field, but won't affect solid due to alpha=0
  auto thermal_weak_form = createWeakForm<dim>("thermal_flux", temperature_type, *field_store, temperature_type,
                                               temperature_old_type, disp_type, disp_old_type);
  thermal_weak_form->addBodyIntegral(mesh_->entireBodyName(), [=](auto t_info, auto /*X*/, auto temperature,
                                                                  auto temperature_old, auto disp, auto disp_old) {
    GreenSaintVenantThermoelasticMaterial::State state;
    auto u = disp_time_rule->value(t_info, disp, disp_old);
    auto v = disp_time_rule->dot(t_info, disp, disp_old);
    auto T = temperature_time_rule->value(t_info, temperature, temperature_old);
    auto T_dot = temperature_time_rule->dot(t_info, temperature, temperature_old);
    auto [pk, C_v, s0, q0] =
        material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(T), get<DERIVATIVE>(T));
    auto dT_dt = get<VALUE>(T_dot);
    return smith::tuple{C_v * dT_dt - s0, -q0};
  });

  std::vector<std::shared_ptr<WeakForm>> weak_forms{solid_weak_form, thermal_weak_form};
  MultiPhysicsTimeIntegrator advancer(field_store, weak_forms, d_nonlinear_solver);

  double dt = 1.0;
  double time = 0.0;
  auto shape_disp = field_store->getShapeDisp();
  auto states = field_store->getAllFields();

  // Run 1 step
  TimeInfo t_info(time, dt, 0);
  auto states_and_reactions = advancer.advanceState(t_info, shape_disp, states, {});
  states = states_and_reactions.first;

  // Check error
  auto disp_idx = field_store->getFieldIndex("displacement");
  FieldState final_disp = states[disp_idx];

  FunctionalObjective<dim, Parameters<H1<order, dim>>> error_obj("error", mesh_, spaces({final_disp}));
  error_obj.addBodyIntegral(DependsOn<0>{}, mesh_->entireBodyName(), [=](auto /*t*/, auto X, auto U) {
    auto u_val = get<VALUE>(U);
    auto x_val = get<0>(X);
    auto exact = u_exact_func(x_val);
    return inner(u_val - exact, u_val - exact);
  });

  double l2_error_sq = error_obj.evaluate(TimeInfo(time+dt, dt, 0), shape_disp.get().get(), getConstFieldPointers({final_disp}));
  double l2_error = std::sqrt(l2_error_sq);

  std::cout << "Static Elasticity L2 Error: " << l2_error << std::endl;
  EXPECT_LT(l2_error, 1e-6);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
