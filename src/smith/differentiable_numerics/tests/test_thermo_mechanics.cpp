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
static constexpr int vdim = dim;
static constexpr int displacement_order = 1;
static constexpr int temperature_order = 1;

struct ThermoMechanicsMeshFixture : public testing::Test {
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

  axom::sidre::DataStore datastore_;
  std::shared_ptr<smith::Mesh> mesh_;
};

TEST_F(ThermoMechanicsMeshFixture, RunThermoMechanicalCoupled)
{
  SMITH_MARK_FUNCTION;

  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, 0.0, 1.0, kappa};

  auto solver = buildDifferentiableNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh_);

  FieldType<L2<0>> youngs_modulus("youngs_modulus");
  auto system = buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh_, solver, youngs_modulus);
  system.setMaterial(material, mesh_->entireBodyName());

  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

  system.disp_bc->setVectorBCs<dim>(mesh_->domain("left"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    bc[0] = 0.01 * t;
    return bc;
  });
  system.disp_bc->setFixedVectorBCs<dim, vdim>(mesh_->domain("right"));
  system.temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("left"));
  system.temperature_bc->setFixedScalarBCs<dim>(mesh_->domain("right"));

  system.thermal_weak_form->addBodySource(smith::DependsOn<>(), mesh_->entireBodyName(),
                                          [](auto /*t*/, auto /* x */) { return 100.0; });

  std::string pv_dir = "paraview_thermo_mechanics";
  auto pv_writer = smith::createParaviewWriter(*mesh_, system.getStateFields(), pv_dir);

  double dt = 0.001;
  double time = 0.0;
  int cycle = 0;

  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();
  auto params = system.getParameterFields();
  std::vector<ReactionState> reactions;

  pv_writer.write(cycle, time, states);
  for (size_t step = 0; step < 10; ++step) {
    TimeInfo t_info(time, dt, step);
    std::tie(states, reactions) = system.advancer->advanceState(t_info, shape_disp, states, params);
    time += dt;
    cycle++;
    pv_writer.write(cycle, time, states);
  }

  // Check that reactions are zero for unconstrained DOFs (should be within solver tolerance)
  std::vector<std::pair<std::string, const BoundaryConditionManager*>> reaction_checks = {
      {"Solid", &system.disp_bc->getBoundaryConditionManager()},
      {"Thermal", &system.temperature_bc->getBoundaryConditionManager()}};

  for (size_t i = 0; i < reactions.size(); ++i) {
    auto& reaction = *reactions[i].get();
    auto& bc_manager = *reaction_checks[i].second;

    FiniteElementState unconstrained_reactions(reaction.space(), "unconstrained_reactions");
    unconstrained_reactions = reaction;
    unconstrained_reactions.SetSubVector(bc_manager.allEssentialTrueDofs(), 0.0);

    double max_unconstrained = unconstrained_reactions.Normlinf();
    EXPECT_LT(max_unconstrained, 1e-8);  // Should be ~solver tolerance
  }

  auto reaction_squared = innerProduct(reactions[0], reactions[0]);
  gretl::set_as_objective(reaction_squared);

  EXPECT_GT(checkGradWrt(reaction_squared, shape_disp, 1.1e-2, 4, true), 0.7);
  EXPECT_GT(checkGradWrt(reaction_squared, params[0], 6.2e-1, 4, true), 0.7);
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

  auto solver = buildDifferentiableNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh_);
  FieldType<L2<0>> youngs_modulus("youngs_modulus");
  auto system = buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh_, solver, youngs_modulus);
  system.setMaterial(material, mesh_->entireBodyName());

  system.parameter_fields[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E0; });

  system.disp_bc->setFixedVectorBCs<dim, dim>(mesh_->domain("left"));
  system.disp_bc->setFixedVectorBCs<dim, dim>(mesh_->domain("right"));
  system.temperature_bc->setScalarBCs<dim>(mesh_->domain("left"), [](double, auto) { return 100.0; });
  system.temperature_bc->setScalarBCs<dim>(mesh_->domain("right"), [](double, auto) { return 100.0; });

  // Set Initial Condition: T(x,0) = 100 + sin(pi*x)
  auto& temp_field = const_cast<FiniteElementState&>(*system.field_store->getField("temperature").get());
  temp_field.setFromFieldFunction([](tensor<double, dim> x) {
    using std::sin;
    return 100.0 + sin(M_PI * x[0]);
  });
  const_cast<FiniteElementState&>(*system.field_store->getField("temperature_old").get()) = temp_field;

  double dt = 0.01;
  double time = 0.0;
  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();
  auto params = system.getParameterFields();
  std::vector<ReactionState> reactions;

  double diffusivity = kappa / (rho * specific_heat);

  size_t num_steps = 10;
  for (size_t step = 0; step < num_steps; ++step) {
    TimeInfo t_info(time, dt, step);
    std::tie(states, reactions) = system.advancer->advanceState(t_info, shape_disp, states, params);
    time += dt;
  }

  // Check nodal error against exact solution: T(x,t) = 100 + sin(pi*x) * exp(-diffusivity * pi^2 * t)
  auto temp_idx = system.field_store->getFieldIndex("temperature");
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

TEST_F(ThermoMechanicsMeshFixture, StaticElasticityAnalytic)
{
  SMITH_MARK_FUNCTION;

  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  GreenSaintVenantThermoelasticMaterial material{rho, E0, nu, specific_heat, 0.0, 0.0, kappa};

  auto solver = buildDifferentiableNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh_);
  FieldType<H1<1>> youngs_modulus("youngs_modulus");
  auto system = buildThermoMechanicsSystem<dim, displacement_order, temperature_order>(mesh_, solver, youngs_modulus);
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

  // Check error - for affine displacement, FEM solution should be exact (up to machine precision)
  auto disp_idx = system.field_store->getFieldIndex("displacement");
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
  EXPECT_LT(l2_error, 1e-10);  // Should be machine precision for affine displacement
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
