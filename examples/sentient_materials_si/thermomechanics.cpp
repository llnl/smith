// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include "mfem.hpp"
#include <string>
#include <tuple>
#include <vector>

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/functional_objective.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/field_store.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/system_base.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"

#include "smith/numerics/functional/tuple.hpp"
#include "smith/physics/weak_form.hpp"

namespace example_tm {

static constexpr int dim = 3;
static constexpr int vdim = dim;
static constexpr int displacement_order = 1;
static constexpr int temperature_order = 1;

using smith::cross;
using smith::dev;
using smith::dot;
using smith::get;
using smith::inner;
using smith::norm;
using smith::tr;
using smith::transpose;

/**
 * @brief Compute Green's strain from the displacement gradient.
 */
template <typename T, int d>
auto greenStrain(const smith::tensor<T, d, d>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

/// @brief Green-Saint Venant isotropic thermoelastic model (copied from tests).
struct GreenSaintVenantThermoelasticMaterial {
  double density;
  double E0;
  double nu;
  double C_v;
  double alpha;
  double theta_ref;
  double kappa;

  using State = smith::Empty;

  template <typename T1, typename T2, typename T3, typename T4, typename T5, int d>
  auto operator()(double, State&, const smith::tensor<T1, d, d>& grad_u, const smith::tensor<T2, d, d>& grad_v,
                  T3 theta, const smith::tensor<T4, d>& grad_theta, const T5& E_param) const
  {
    auto E = E0 + get<0>(E_param);
    const auto K = E / (3.0 * (1.0 - 2.0 * nu));
    const auto G = 0.5 * E / (1.0 + nu);
    const auto Eg = greenStrain<T1, d>(grad_u);
    const auto trEg = tr(Eg);

    static constexpr auto I = smith::Identity<d>();
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - d * alpha * (theta - theta_ref)) * I;
    auto F = grad_u + I;
    const auto Piola = dot(F, S);

    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto s0 = -d * K * alpha * (theta + 273.1) * tr(greenStrainRate) + 0.0 * E;

    const auto q0 = -kappa * grad_theta;

    return smith::tuple{Piola, C_v, s0, q0};
  }

  static constexpr int numParameters() { return 1; }
};

smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack,
                                          .relative_tol = 1e-8,
                                          .absolute_tol = 1e-8,
                                          .max_iterations = 200,
                                          .print_level = 0};

smith::NonlinearSolverOptions nonlinear_opts{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                             .relative_tol = 1.0e-10,
                                             .absolute_tol = 1.0e-10,
                                             .max_iterations = 500,
                                             .max_line_search_iterations = 50,
                                             .print_level = 2};

/**
 * @brief Local copy of the thermo-mechanics "physics reconstruction" from
 * `src/smith/differentiable_numerics/thermo_mechanics_system.hpp`.
 *
 * The point of this example is to show how the coupled physics is assembled:
 * fields + time integration + weak forms + sources/BCs + block solve.
 */
template <int d, int disp_order, int temp_order, typename... parameter_space>
struct ExampleThermoMechanicsSystem : public smith::SystemBase {
  using SolidWeakFormType = smith::TimeDiscretizedWeakForm<
      d, smith::H1<disp_order, d>,
      smith::Parameters<smith::H1<disp_order, d>, smith::H1<disp_order, d>, smith::H1<temp_order>,
                        smith::H1<temp_order>, parameter_space...>>;

  using ThermalWeakFormType = smith::TimeDiscretizedWeakForm<
      d, smith::H1<temp_order>,
      smith::Parameters<smith::H1<temp_order>, smith::H1<temp_order>, smith::H1<disp_order, d>,
                        smith::H1<disp_order, d>, parameter_space...>>;

  std::shared_ptr<SolidWeakFormType> solid_weak_form;
  std::shared_ptr<ThermalWeakFormType> thermal_weak_form;
  std::shared_ptr<smith::DirichletBoundaryConditions> disp_bc;
  std::shared_ptr<smith::DirichletBoundaryConditions> temperature_bc;
  std::shared_ptr<smith::QuasiStaticFirstOrderTimeIntegrationRule> disp_time_rule;
  std::shared_ptr<smith::BackwardEulerFirstOrderTimeIntegrationRule> temperature_time_rule;

  std::vector<smith::FieldState> getStateFields() const
  {
    std::vector<smith::FieldState> states;
    states.push_back(field_store->getField(prefix("displacement_predicted")));
    states.push_back(field_store->getField(prefix("displacement")));
    states.push_back(field_store->getField(prefix("temperature_predicted")));
    states.push_back(field_store->getField(prefix("temperature")));
    return states;
  }

  std::shared_ptr<smith::DifferentiablePhysics> createDifferentiablePhysics(std::string physics_name)
  {
    return std::make_shared<smith::DifferentiablePhysics>(
        field_store->getMesh(), field_store->graph(), field_store->getShapeDisp(), getStateFields(),
        getParameterFields(), advancer, std::move(physics_name),
        std::vector<std::string>{prefix("solid_force"), prefix("thermal_flux")});
  }

  template <typename MaterialType>
  void setMaterial(const MaterialType& material, const std::string& domain_name)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    solid_weak_form->addBodyIntegral(domain_name, [=](auto t_info, auto /*X*/, auto u, auto u_old, auto temperature,
                                                      auto temperature_old, auto... params) {
      auto u_current = captured_disp_rule->value(t_info, u, u_old);
      auto v_current = captured_disp_rule->dot(t_info, u, u_old);
      auto T = captured_temp_rule->value(t_info, temperature, temperature_old);

      typename MaterialType::State state;
      auto [pk, C_v, s0, q0] =
          material(t_info.dt(), state, get<smith::DERIVATIVE>(u_current), get<smith::DERIVATIVE>(v_current),
                   get<smith::VALUE>(T), get<smith::DERIVATIVE>(T), params...);
      return smith::tuple{smith::zero{}, pk};
    });

    thermal_weak_form->addBodyIntegral(
        domain_name, [=](auto t_info, auto /*X*/, auto T, auto T_old, auto disp, auto disp_old, auto... params) {
          auto T_current = captured_temp_rule->value(t_info, T, T_old);
          auto T_dot = captured_temp_rule->dot(t_info, T, T_old);
          auto u = captured_disp_rule->value(t_info, disp, disp_old);
          auto v = captured_disp_rule->dot(t_info, disp, disp_old);

          typename MaterialType::State state;
          auto [pk, C_v, s0, q0] = material(t_info.dt(), state, get<smith::DERIVATIVE>(u), get<smith::DERIVATIVE>(v),
                                            get<smith::VALUE>(T_current), get<smith::DERIVATIVE>(T_current), params...);
          auto dT_dt = get<smith::VALUE>(T_dot);
          return smith::tuple{C_v * dT_dt - s0, -q0};
        });
  }

  template <int... active_parameters, typename BodyForceType>
  void addSolidBodyForce(smith::DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                         BodyForceType force_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    solid_weak_form->addBodySource(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto u, auto u_old, auto temperature, auto temperature_old, auto... params) {
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto v_current = captured_disp_rule->dot(t_info, u, u_old);
          auto current_T = captured_temp_rule->value(t_info, temperature, temperature_old);
          auto T_dot = captured_temp_rule->dot(t_info, temperature, temperature_old);
          return force_function(t_info.time(), X, u_current, v_current, current_T, T_dot, params...);
        });
  }

  template <typename BodyForceType>
  void addSolidBodyForce(const std::string& domain_name, BodyForceType force_function)
  {
    addSolidBodyForceAllParams(domain_name, force_function, std::make_index_sequence<4 + sizeof...(parameter_space)>{});
  }

  template <int... active_parameters, typename SurfaceFluxType>
  void addSolidTraction(smith::DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                        SurfaceFluxType flux_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    solid_weak_form->addBoundaryFlux(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto n, auto u, auto u_old, auto temperature, auto temperature_old, auto... params) {
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto v_current = captured_disp_rule->dot(t_info, u, u_old);
          auto current_T = captured_temp_rule->value(t_info, temperature, temperature_old);
          auto T_dot = captured_temp_rule->dot(t_info, temperature, temperature_old);
          return flux_function(t_info.time(), X, n, u_current, v_current, current_T, T_dot, params...);
        });
  }

  template <typename SurfaceFluxType>
  void addSolidTraction(const std::string& domain_name, SurfaceFluxType flux_function)
  {
    addSolidTractionAllParams(domain_name, flux_function, std::make_index_sequence<4 + sizeof...(parameter_space)>{});
  }

  template <int... active_parameters, typename BodySourceType>
  void addThermalHeatSource(smith::DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                            BodySourceType source_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    thermal_weak_form->addBodySource(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto T, auto T_old, auto disp, auto disp_old, auto... params) {
          auto current_u = captured_disp_rule->value(t_info, disp, disp_old);
          auto v_current = captured_disp_rule->dot(t_info, disp, disp_old);
          auto T_current = captured_temp_rule->value(t_info, T, T_old);
          auto T_dot = captured_temp_rule->dot(t_info, T, T_old);
          return source_function(t_info.time(), X, current_u, v_current, T_current, T_dot, params...);
        });
  }

  template <typename BodySourceType>
  void addThermalHeatSource(const std::string& domain_name, BodySourceType source_function)
  {
    addThermalHeatSourceAllParams(domain_name, source_function,
                                  std::make_index_sequence<4 + sizeof...(parameter_space)>{});
  }

  template <int... active_parameters, typename SurfaceFluxType>
  void addThermalHeatFlux(smith::DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                          SurfaceFluxType flux_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    thermal_weak_form->addBoundaryFlux(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto n, auto T, auto T_old, auto disp, auto disp_old, auto... params) {
          auto current_u = captured_disp_rule->value(t_info, disp, disp_old);
          auto v_current = captured_disp_rule->dot(t_info, disp, disp_old);
          auto T_current = captured_temp_rule->value(t_info, T, T_old);
          auto T_dot = captured_temp_rule->dot(t_info, T, T_old);
          return -flux_function(t_info.time(), X, n, current_u, v_current, T_current, T_dot, params...);
        });
  }

  template <typename SurfaceFluxType>
  void addThermalHeatFlux(const std::string& domain_name, SurfaceFluxType flux_function)
  {
    addThermalHeatFluxAllParams(domain_name, flux_function, std::make_index_sequence<4 + sizeof...(parameter_space)>{});
  }

  template <int... active_parameters, typename PressureType>
  void addPressure(smith::DependsOn<active_parameters...> depends_on, const std::string& domain_name,
                   PressureType pressure_function)
  {
    auto captured_disp_rule = disp_time_rule;
    auto captured_temp_rule = temperature_time_rule;

    solid_weak_form->addBoundaryIntegral(
        depends_on, domain_name,
        [=](auto t_info, auto X, auto u, auto u_old, auto temperature, auto temperature_old, auto... params) {
          auto u_current = captured_disp_rule->value(t_info, u, u_old);
          auto v_current = captured_disp_rule->dot(t_info, u, u_old);
          auto T_current = captured_temp_rule->value(t_info, temperature, temperature_old);
          auto T_dot = captured_temp_rule->dot(t_info, temperature, temperature_old);

          auto x_current = X + u_current;
          auto n_deformed = cross(get<smith::DERIVATIVE>(x_current));
          auto n_shape_norm = norm(cross(get<smith::DERIVATIVE>(X)));

          auto pressure = pressure_function(t_info.time(), get<smith::VALUE>(X), u_current, v_current, T_current, T_dot,
                                            get<smith::VALUE>(params)...);

          return pressure * n_deformed * (1.0 / n_shape_norm);
        });
  }

  template <typename PressureType>
  void addPressure(const std::string& domain_name, PressureType pressure_function)
  {
    addPressureAllParams(domain_name, pressure_function, std::make_index_sequence<4 + sizeof...(parameter_space)>{});
  }

 private:
  template <typename BodyForceType, std::size_t... Is>
  void addSolidBodyForceAllParams(const std::string& domain_name, BodyForceType force_function,
                                  std::index_sequence<Is...>)
  {
    addSolidBodyForce(smith::DependsOn<static_cast<int>(Is)...>{}, domain_name, force_function);
  }

  template <typename SurfaceFluxType, std::size_t... Is>
  void addSolidTractionAllParams(const std::string& domain_name, SurfaceFluxType flux_function,
                                 std::index_sequence<Is...>)
  {
    addSolidTraction(smith::DependsOn<static_cast<int>(Is)...>{}, domain_name, flux_function);
  }

  template <typename PressureType, std::size_t... Is>
  void addPressureAllParams(const std::string& domain_name, PressureType pressure_function, std::index_sequence<Is...>)
  {
    addPressure(smith::DependsOn<static_cast<int>(Is)...>{}, domain_name, pressure_function);
  }

  template <typename BodySourceType, std::size_t... Is>
  void addThermalHeatSourceAllParams(const std::string& domain_name, BodySourceType source_function,
                                     std::index_sequence<Is...>)
  {
    addThermalHeatSource(smith::DependsOn<static_cast<int>(Is)...>{}, domain_name, source_function);
  }

  template <typename SurfaceFluxType, std::size_t... Is>
  void addThermalHeatFluxAllParams(const std::string& domain_name, SurfaceFluxType flux_function,
                                   std::index_sequence<Is...>)
  {
    addThermalHeatFlux(smith::DependsOn<static_cast<int>(Is)...>{}, domain_name, flux_function);
  }
};

template <int d, int disp_order, int temp_order, typename... parameter_space>
ExampleThermoMechanicsSystem<d, disp_order, temp_order, parameter_space...> buildExampleThermoMechanicsSystem(
    std::shared_ptr<smith::Mesh> mesh, std::shared_ptr<smith::NonlinearBlockSolverBase> solver,
    std::string prepend_name, smith::FieldType<parameter_space>... parameter_types)
{
  auto field_store = std::make_shared<smith::FieldStore>(mesh, 100);

  auto prefix = [&](const std::string& name) {
    if (prepend_name.empty()) {
      return name;
    }
    return prepend_name + "_" + name;
  };

  smith::FieldType<smith::H1<1, d>> shape_disp_type(prefix("shape_displacement"));
  field_store->addShapeDisp(shape_disp_type);

  auto disp_time_rule = std::make_shared<smith::QuasiStaticFirstOrderTimeIntegrationRule>();
  smith::FieldType<smith::H1<disp_order, d>> disp_type(prefix("displacement_predicted"));
  auto disp_bc = field_store->addIndependent(disp_type, disp_time_rule);
  auto disp_old_type =
      field_store->addDependent(disp_type, smith::FieldStore::TimeDerivative::VAL, prefix("displacement"));

  auto temperature_time_rule = std::make_shared<smith::BackwardEulerFirstOrderTimeIntegrationRule>();
  smith::FieldType<smith::H1<temp_order>> temperature_type(prefix("temperature_predicted"));
  auto temperature_bc = field_store->addIndependent(temperature_type, temperature_time_rule);
  auto temperature_old_type =
      field_store->addDependent(temperature_type, smith::FieldStore::TimeDerivative::VAL, prefix("temperature"));

  std::vector<smith::FieldState> parameter_fields;
  (field_store->addParameter(smith::FieldType<parameter_space>(prefix("param_" + parameter_types.name))), ...);
  (parameter_fields.push_back(field_store->getField(prefix("param_" + parameter_types.name))), ...);

  std::string solid_force_name = prefix("solid_force");
  auto solid_weak_form = std::make_shared<
      typename ExampleThermoMechanicsSystem<d, disp_order, temp_order, parameter_space...>::SolidWeakFormType>(
      solid_force_name, field_store->getMesh(), field_store->getField(disp_type.name).get()->space(),
      field_store->createSpaces(solid_force_name, disp_type.name, disp_type, disp_old_type, temperature_type,
                                temperature_old_type,
                                smith::FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  std::string thermal_flux_name = prefix("thermal_flux");
  auto thermal_weak_form = std::make_shared<
      typename ExampleThermoMechanicsSystem<d, disp_order, temp_order, parameter_space...>::ThermalWeakFormType>(
      thermal_flux_name, field_store->getMesh(), field_store->getField(temperature_type.name).get()->space(),
      field_store->createSpaces(thermal_flux_name, temperature_type.name, temperature_type, temperature_old_type,
                                disp_type, disp_old_type,
                                smith::FieldType<parameter_space>(prefix("param_" + parameter_types.name))...));

  std::vector<std::shared_ptr<smith::WeakForm>> weak_forms{solid_weak_form, thermal_weak_form};
  auto coupled_solver = std::make_shared<smith::CoupledSystemSolver>(solver);
  auto advancer = std::make_shared<smith::MultiphysicsTimeIntegrator>(field_store, weak_forms, coupled_solver);

  return ExampleThermoMechanicsSystem<d, disp_order, temp_order, parameter_space...>{
      {field_store, coupled_solver, advancer, parameter_fields, prepend_name},
      solid_weak_form,
      thermal_weak_form,
      disp_bc,
      temperature_bc,
      disp_time_rule,
      temperature_time_rule};
}

template <int d, int disp_order, int temp_order, typename... parameter_space>
ExampleThermoMechanicsSystem<d, disp_order, temp_order, parameter_space...> buildExampleThermoMechanicsSystem(
    std::shared_ptr<smith::Mesh> mesh, std::shared_ptr<smith::NonlinearBlockSolverBase> solver,
    smith::FieldType<parameter_space>... parameter_types)
{
  return buildExampleThermoMechanicsSystem<d, disp_order, temp_order, parameter_space...>(mesh, solver, "",
                                                                                          parameter_types...);
}

enum class Mode
{
  Coupled,
  Heat,
  Elastic,
  Pressure,
  Compilation
};

Mode parseMode(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--mode=coupled") return Mode::Coupled;
    if (arg == "--mode=heat") return Mode::Heat;
    if (arg == "--mode=elastic") return Mode::Elastic;
    if (arg == "--mode=pressure") return Mode::Pressure;
    if (arg == "--mode=compilation") return Mode::Compilation;
    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: thermomechanics [--mode=coupled|heat|elastic|pressure|compilation]\n";
      std::exit(0);
    }
  }
  return Mode::Coupled;
}

int runCoupled(const std::shared_ptr<smith::Mesh>& mesh)
{
  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, 0.0, 1.0, kappa};

  auto solver = smith::buildNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh);

  smith::FieldType<smith::L2<0>> youngs_modulus("youngs_modulus");
  auto system =
      buildExampleThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh, solver, youngs_modulus);
  system.setMaterial(material, mesh->entireBodyName());

  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

  system.disp_bc->setVectorBCs<dim>(mesh->domain("left"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    bc[0] = 0.01 * t;
    return bc;
  });
  system.disp_bc->template setFixedVectorBCs<dim, vdim>(mesh->domain("right"));
  system.temperature_bc->setFixedScalarBCs<dim>(mesh->domain("left"));
  system.temperature_bc->setFixedScalarBCs<dim>(mesh->domain("right"));

  system.addThermalHeatSource(mesh->entireBodyName(), [](auto /*t*/, auto /*x*/, auto /*u*/, auto /*v*/, auto /*T*/,
                                                         auto /*T_dot*/, auto /*E_param*/) { return 100.0; });

  std::string pv_dir = "paraview_thermomechanics";
  auto pv_writer = smith::createParaviewWriter(*mesh, system.getStateFields(), pv_dir);

  double dt = 0.001;
  double time = 0.0;
  int cycle = 0;

  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();
  auto params = system.getParameterFields();

  pv_writer.write(cycle, time, states);
  for (size_t step = 0; step < 2; ++step) {
    smith::TimeInfo t_info(time, dt, step);
    auto [new_states, reactions] = system.advancer->advanceState(t_info, shape_disp, states, params);
    states = std::move(new_states);

    double max_reaction_disp = reactions[0].get()->Normlinf();
    double max_reaction_temp = reactions[1].get()->Normlinf();
    std::cout << "step " << step << " max reaction (disp)=" << max_reaction_disp
              << " max reaction (temp)=" << max_reaction_temp << "\n";

    time += dt;
    cycle++;
    pv_writer.write(cycle, time, states);
  }

  std::cout << "Wrote ParaView output to '" << pv_dir << "'\n";
  return 0;
}

int runTransientHeatAnalytic(const std::shared_ptr<smith::Mesh>& mesh)
{
  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, 0.0, 0.0, kappa};

  auto solver = smith::buildNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh);
  smith::FieldType<smith::L2<0>> youngs_modulus("youngs_modulus");
  auto system =
      buildExampleThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh, solver, youngs_modulus);
  system.setMaterial(material, mesh->entireBodyName());

  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

  system.disp_bc->template setFixedVectorBCs<dim, vdim>(mesh->domain("left"));
  system.disp_bc->template setFixedVectorBCs<dim, vdim>(mesh->domain("right"));
  system.temperature_bc->setScalarBCs<dim>(mesh->domain("left"), [](double, auto) { return 100.0; });
  system.temperature_bc->setScalarBCs<dim>(mesh->domain("right"), [](double, auto) { return 100.0; });

  auto& temp_field =
      const_cast<smith::FiniteElementState&>(*system.field_store->getField("temperature_predicted").get());
  temp_field.setFromFieldFunction([](smith::tensor<double, dim> x) { return 100.0 + std::sin(M_PI * x[0]); });
  const_cast<smith::FiniteElementState&>(*system.field_store->getField("temperature").get()) = temp_field;

  double dt = 0.01;
  double time = 0.0;
  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();
  auto params = system.getParameterFields();

  double diffusivity = kappa / (rho * specific_heat);
  size_t num_steps = 2;
  for (size_t step = 0; step < num_steps; ++step) {
    smith::TimeInfo t_info(time, dt, step);
    states = system.advancer->advanceState(t_info, shape_disp, states, params).first;
    time += dt;
  }

  auto temp_idx = system.field_store->getFieldIndex("temperature_predicted");
  smith::FieldState final_temp = states[temp_idx];

  smith::FiniteElementState exact_temp(final_temp.get()->space(), "exact_temp");
  exact_temp.setFromFieldFunction([&](smith::tensor<double, dim> x) {
    return 100.0 + std::sin(M_PI * x[0]) * std::exp(-diffusivity * M_PI * M_PI * time);
  });

  mfem::Vector diff(final_temp.get()->Size());
  // mfem::subtract(*final_temp.get(), exact_temp, diff);
  diff = *final_temp.get();
  diff.Add(-1.0, exact_temp);
  double max_nodal_error = diff.Normlinf();

  std::cout << "Transient Heat max nodal error: " << max_nodal_error << "\n";
  return (max_nodal_error < 1e-4) ? 0 : 2;
}

int runStaticElasticityAnalytic(const std::shared_ptr<smith::Mesh>& mesh)
{
  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, 0.0, 0.0, kappa};

  auto solver = smith::buildNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh);
  smith::FieldType<smith::H1<1>> youngs_modulus("youngs_modulus");
  auto system =
      buildExampleThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh, solver, youngs_modulus);
  system.setMaterial(material, mesh->entireBodyName());

  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

  smith::tensor<double, dim, dim> G;
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
  system.disp_bc->setVectorBCs<dim>(mesh->entireBoundary(),
                                    [=](double, smith::tensor<double, dim> X) { return u_exact_func(X); });
  system.temperature_bc->setFixedScalarBCs<dim>(mesh->entireBoundary());

  double dt = 1.0;
  double time = 0.0;
  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();
  auto params = system.getParameterFields();

  smith::TimeInfo t_info(time, dt, 0);
  states = system.advancer->advanceState(t_info, shape_disp, states, params).first;

  auto disp_idx = system.field_store->getFieldIndex("displacement_predicted");
  smith::FieldState final_disp = states[disp_idx];

  smith::FunctionalObjective<dim, smith::Parameters<smith::H1<displacement_order, vdim>>> error_obj(
      "error", mesh, smith::spaces({final_disp}));
  error_obj.addBodyIntegral(smith::DependsOn<0>{}, mesh->entireBodyName(), [=](auto /*t*/, auto X, auto U) {
    auto u_val = get<smith::VALUE>(U);
    auto x_val = get<0>(X);
    auto exact = u_exact_func(x_val);
    return smith::inner(u_val - exact, u_val - exact);
  });

  double l2_error_sq = error_obj.evaluate(smith::TimeInfo(time + dt, dt, 0), shape_disp.get().get(),
                                          smith::getConstFieldPointers({final_disp}));
  double l2_error = std::sqrt(l2_error_sq);

  std::cout << "Static Elasticity L2 Error (affine patch test): " << l2_error << "\n";
  return (l2_error < 1e-11) ? 0 : 3;
}

int runPressureBC(const std::shared_ptr<smith::Mesh>& mesh)
{
  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, 0.0, 1.0, kappa};

  auto solver = smith::buildNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh);
  smith::FieldType<smith::L2<0>> youngs_modulus("youngs_modulus");
  auto system =
      buildExampleThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh, solver, youngs_modulus);
  system.setMaterial(material, mesh->entireBodyName());

  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

  system.disp_bc->setFixedVectorBCs<dim>(mesh->domain("left"));
  system.temperature_bc->setFixedScalarBCs<dim>(mesh->domain("left"));
  system.temperature_bc->setFixedScalarBCs<dim>(mesh->domain("right"));

  double pressure_mag = 0.1;
  system.addPressure("right", [pressure_mag](double, auto, auto, auto, auto, auto, auto...) { return pressure_mag; });

  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();
  auto params = system.getParameterFields();

  double dt = 0.001;
  double time = 0.0;
  size_t cycle = 0;
  smith::TimeInfo t_info(time, dt, cycle);
  states = system.advancer->advanceState(t_info, shape_disp, states, params).first;

  auto disp_idx = system.field_store->getFieldIndex("displacement_predicted");
  smith::FieldState final_disp = states[disp_idx];

  double disp_norm = final_disp.get()->Norml2();
  std::cout << "PressureBC displacement L2 norm: " << disp_norm << "\n";
  return (disp_norm > 1e-6) ? 0 : 4;
}

int runCompilationOnly(const std::shared_ptr<smith::Mesh>& mesh)
{
  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, 0.0, 1.0, kappa};

  auto fast_nonlinear_opts = nonlinear_opts;
  fast_nonlinear_opts.max_iterations = 5;
  auto solver = smith::buildNonlinearBlockSolver(fast_nonlinear_opts, linear_options, *mesh);

  smith::FieldType<smith::L2<0>> youngs_modulus("youngs_modulus");
  auto system =
      buildExampleThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh, solver, youngs_modulus);
  system.setMaterial(material, mesh->entireBodyName());

  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

  system.addSolidBodyForce(mesh->entireBodyName(), [](double, auto, auto, auto, auto, auto, auto...) {
    smith::tensor<double, dim> f{};
    f[1] = -9.81;
    return f;
  });

  system.addSolidTraction("right", [](double, auto, auto, auto, auto, auto, auto, auto...) {
    smith::tensor<double, dim> t{};
    t[0] = 1.0;
    return t;
  });

  system.addThermalHeatSource(mesh->entireBodyName(),
                              [](double, auto, auto, auto, auto, auto, auto...) { return 10.0; });

  system.addThermalHeatFlux("left", [](double, auto, auto, auto, auto, auto, auto, auto...) { return 5.0; });

  system.disp_bc->setVectorBCs<dim>(mesh->domain("left"), [](double, smith::tensor<double, dim> X) { return 0.0 * X; });

  std::cout << "Compilation-only mode: constructed weak forms, sources, and BCs.\n";
  return 0;
}

}  // namespace example_tm

int main(int argc, char** argv)
{
  smith::ApplicationManager applicationManager(argc, argv);

  const auto mode = example_tm::parseMode(argc, argv);

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid");

  double length = 1.0;
  double width = 0.04;
  int num_elements_x = 12;
  int num_elements_y = 2;
  int num_elements_z = 2;
  auto mfem_shape = mfem::Element::QUADRILATERAL;
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(num_elements_x, num_elements_y, num_elements_z, mfem_shape, length, width, width),
      "mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<example_tm::dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<example_tm::dim>(5));

  switch (mode) {
    case example_tm::Mode::Coupled:
      return example_tm::runCoupled(mesh);
    case example_tm::Mode::Heat:
      return example_tm::runTransientHeatAnalytic(mesh);
    case example_tm::Mode::Elastic:
      return example_tm::runStaticElasticityAnalytic(mesh);
    case example_tm::Mode::Pressure:
      return example_tm::runPressureBC(mesh);
    case example_tm::Mode::Compilation:
      return example_tm::runCompilationOnly(mesh);
  }

  return 0;
}
