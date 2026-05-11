// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file composable_cooks_membrane_fbar.cpp
 * @brief Cook's membrane problem using a modified F-bar strategy with a projected J_bar field
 *        and an Augmented Lagrangian penalty to stabilize the staggered iterations.
 */

#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>

#include "smith/numerics/functional/tensor.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/state_variable_system.hpp"
#include "smith/differentiable_numerics/combined_system.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"

namespace {

template <typename StateSpace, typename InternalVarTimeRule>
auto registerCustomInternalVariableFields(std::shared_ptr<smith::FieldStore> field_store, const std::string& prefix)
{
  auto internal_variable_time_rule = std::make_shared<InternalVarTimeRule>();
  smith::FieldType<StateSpace> state_type(prefix + "_solve_state");
  field_store->addIndependent(state_type, internal_variable_time_rule);

  smith::FieldType<StateSpace> state_old_type(prefix + "_state");
  field_store->addDependent(state_type, smith::FieldStore::TimeDerivative::VAL, state_old_type.name);

  return smith::PhysicsFields<InternalVarTimeRule, StateSpace, StateSpace>{field_store, state_type, state_old_type};
}

struct ConstraintMetrics {
  double integral = 0.0;
  double element_l2_norm = 0.0;
  double penalty_objective = 0.0;
};

ConstraintMetrics computeConstraintMetrics(smith::FiniteElementState& u_state, smith::FiniteElementState& jbar_state,
                                           double penalty_kappa)
{
  mfem::ParGridFunction u_gf(&u_state.space());
  u_state.fillGridFunction(u_gf);

  mfem::ParGridFunction jbar_gf(&jbar_state.space());
  jbar_state.fillGridFunction(jbar_gf);

  auto& mesh = u_state.mesh();
  double local_integral = 0.0;
  double local_element_l2_squared = 0.0;

  for (int e = 0; e < mesh.GetNE(); ++e) {
    auto* trans = mesh.GetElementTransformation(e);
    const auto* fe = u_state.space().GetFE(e);
    const auto& ir = mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + 4);
    double element_constraint = 0.0;

    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const auto& ip = ir.IntPoint(q);
      trans->SetIntPoint(&ip);

      mfem::DenseMatrix grad_u(2, 2);
      u_gf.GetVectorGradient(*trans, grad_u);

      const double J_q = (1.0 + grad_u(0, 0)) * (1.0 + grad_u(1, 1)) - grad_u(0, 1) * grad_u(1, 0);
      const double Jbar = jbar_gf.GetValue(*trans, ip);
      const double c = J_q - Jbar;
      const double w = ip.weight * trans->Weight();

      element_constraint += w * c;
    }

    local_integral += element_constraint;
    local_element_l2_squared += element_constraint * element_constraint;
  }

  double global_integral = 0.0;
  double global_element_l2_squared = 0.0;
  MPI_Allreduce(&local_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, u_state.comm());
  MPI_Allreduce(&local_element_l2_squared, &global_element_l2_squared, 1, MPI_DOUBLE, MPI_SUM, u_state.comm());

  return {.integral = global_integral,
          .element_l2_norm = std::sqrt(global_element_l2_squared),
          .penalty_objective = 0.5 * penalty_kappa * global_element_l2_squared};
}

/**
 * @brief F-bar Neo-Hookean material using a blended volumetric Jacobian, stabilized by
 *        an Augmented Lagrangian term penalizing deviations of J_q from J_hat.
 */
struct FBarNeoHookeanWithAL {
  using State = smith::Empty;

  double K;
  double G;
  double density = 1.0;
  double gamma = 1.1;
  double penalty_kappa = 1e3;  // AL Penalty parameter

  double scale() const
  {
    double eta = G / K;
    return eta / 3.0 * (3.0 - 2.0 * eta + 2.0 * gamma * (3.0 + eta)) / (gamma * (3.0 + eta) - (3.0 - 2.0 * eta));
  }

  template <typename T, int dim, typename GradVType, typename JBarSS, typename JBar, typename PSS, typename P,
            typename... Rest>
  auto operator()(const smith::TimeInfo&, [[maybe_unused]] State& state, const smith::tensor<T, dim, dim>& grad_u,
                  const GradVType&, const JBarSS& jbar_ss, const JBar& /*jbar*/, const PSS& p_ss, const P& /*p*/,
                  const Rest&...) const
  {
    using std::pow;
    constexpr auto I = smith::Identity<dim>();
    auto F_q = I + grad_u;
    auto J_q = det(F_q);
    auto Jbar = smith::get<smith::VALUE>(jbar_ss);
    auto p_val = smith::get<smith::VALUE>(p_ss);

    double alpha = scale();
    double beta = (1.0 - alpha) / dim;

    // Modified deformation gradient
    auto F_hat = pow(Jbar / J_q, beta) * F_q;

    // Standard NeoHookean stress evaluated at the modified deformation gradient.
    using std::log1p;
    auto du_hat = F_hat - I;
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I_hat = dot(du_hat, transpose(du_hat)) + transpose(du_hat) + du_hat;
    auto logJ_hat = log1p(detApIm1(du_hat));
    auto TK_hat = lambda * logJ_hat * I + G * B_minus_I_hat;
    auto inv_F_hat_T = inv(transpose(F_hat));
    auto Piola_neo_hat = dot(TK_hat, inv_F_hat_T);

    // Trace of Kirchhoff stress
    auto trace_tau = TK_hat[0][0];
    for (int i = 1; i < dim; ++i) trace_tau = trace_tau + TK_hat[i][i];

    auto inv_F_q_T = inv(transpose(F_q));

    // Chain rule for partial W / partial F
    auto P_mod = pow(Jbar / J_q, beta) * Piola_neo_hat - beta * trace_tau * inv_F_q_T;

    // Augmented Lagrangian contribution
    auto Piola_AL = (p_val + penalty_kappa * (J_q - Jbar)) * J_q * inv_F_q_T;

    return P_mod + Piola_AL;
  }
};

}  // namespace

int main(int argc, char* argv[])
{
  smith::ApplicationManager application_manager(argc, argv);
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "cooks_membrane_fbar");

  constexpr int dim = 2;
  constexpr int order = 1;

  auto mfem_mesh = mfem::Mesh::MakeCartesian2D(32, 32, mfem::Element::QUADRILATERAL, false, 48.0, 44.0);
  mfem::Vector x_coords;
  mfem_mesh.GetNodes(x_coords);
  int num_nodes = mfem_mesh.GetNV();
  for (int i = 0; i < num_nodes; ++i) {
    double x = x_coords(i);
    double y_hat = x_coords(i + num_nodes);
    double y = y_hat * (1.0 - x / 48.0) + (44.0 + 16.0 * y_hat / 44.0) * (x / 48.0);
    x_coords(i + num_nodes) = y;
  }
  mfem_mesh.SetNodes(x_coords);

  auto mesh = std::make_shared<smith::Mesh>(std::move(mfem_mesh), "mesh");
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<dim>(1));

  auto field_store = std::make_shared<smith::FieldStore>(mesh, 10);

  // 1. Register Fields
  smith::SolidMechanicsOptions solid_options{.enable_stress_output = true, .output_cauchy_stress = true};
  auto solid_fields = smith::registerSolidMechanicsFields<dim, order, smith::QuasiStaticSecondOrderTimeIntegrationRule>(
      field_store, solid_options);

  using JBarSpace = smith::L2<0>;
  auto jbar_fields = registerCustomInternalVariableFields<JBarSpace, smith::QuasiStaticFirstOrderTimeIntegrationRule>(
      field_store, "jbar");

  auto p_fields = registerCustomInternalVariableFields<JBarSpace, smith::QuasiStaticFirstOrderTimeIntegrationRule>(
      field_store, "p");

  // 2. Build Systems
  auto jbar_system = smith::buildInternalVariableSystem<dim, JBarSpace>(nullptr, jbar_fields, solid_fields, p_fields);

  auto p_system = smith::buildInternalVariableSystem<dim, JBarSpace>(nullptr, p_fields, solid_fields, jbar_fields);

  constexpr double E = 250.0;
  constexpr double nu = 0.4999;
  constexpr double G = E / (2.0 * (1.0 + nu));
  constexpr double K = E / (3.0 * (1.0 - 2.0 * nu));
  constexpr double gamma = 1.1;
  constexpr double penalty_kappa = 2000 * E;

  FBarNeoHookeanWithAL mat{.K = K, .G = G, .gamma = gamma, .penalty_kappa = penalty_kappa};
  double alpha = mat.scale();
  double beta = (1.0 - alpha) / dim;
  std::cout << "alpha = " << alpha << "  (gamma = " << gamma << ")\n";

  jbar_system->addEvolution(
      mesh->entireBodyName(),
      [G, beta, penalty_kappa](auto /*t_info*/, auto Jbar, auto /*Jbar_dot*/, auto u_ss, auto /*u*/, auto /*v*/,
                               auto /*a*/, auto p_ss, auto /*p*/, auto&&...) {
        auto p_val = smith::get<smith::VALUE>(p_ss);
        auto grad_u = smith::get<smith::DERIVATIVE>(u_ss);
        auto F_q = smith::Identity<dim>() + grad_u;
        auto J_q = det(F_q);

        using std::pow;
        auto F_hat = pow(Jbar / J_q, beta) * F_q;

        using std::log1p;
        auto du_hat = F_hat - smith::Identity<dim>();
        auto lambda = K - (2.0 / 3.0) * G;
        auto B_minus_I_hat = dot(du_hat, transpose(du_hat)) + transpose(du_hat) + du_hat;
        auto logJ_hat = log1p(detApIm1(du_hat));
        auto TK_hat = lambda * logJ_hat * smith::Identity<dim>() + G * B_minus_I_hat;

        auto trace_tau = TK_hat[0][0];
        for (int i = 1; i < dim; ++i) trace_tau = trace_tau + TK_hat[i][i];

        // Stationary wrt Jbar: dW/dJbar - p - kappa*(J - Jbar) = 0
        // dW/dJbar = beta/Jbar * tr(tau_hat)
        return (beta / Jbar) * trace_tau - (p_val + penalty_kappa * (J_q - Jbar));
      });

  p_system->addEvolution(mesh->entireBodyName(),
                         [penalty_kappa](auto /*t_info*/, auto /*p*/, auto /*p_dot*/, auto u_ss, auto /*u*/, auto /*v*/,
                                         auto /*a*/, auto jbar_ss, auto /*jbar*/, auto&&...) {
                           auto grad_u = smith::get<smith::DERIVATIVE>(u_ss);
                           auto J_q = det(smith::Identity<dim>() + grad_u);
                           auto Jbar = smith::get<smith::VALUE>(jbar_ss);
                           return -penalty_kappa * (J_q - Jbar);
                         });

  auto solid_system =
      smith::buildSolidMechanicsSystem<dim, order>(nullptr, solid_options, solid_fields, jbar_fields, p_fields);
  solid_system->setMaterial(mat, mesh->entireBodyName());

  // 3. BCs
  solid_system->setDisplacementBC(mesh->domain("left"));
  solid_system->addTraction(smith::DependsOn<>{}, "right",
                            [](double /*t*/, auto /*X*/, auto /*n*/, auto /*u*/, auto /*v*/, auto /*a*/) {
                              smith::tensor<double, dim> traction{};
                              traction[1] = 0.1;
                              return traction;
                            });

  // 4. Solver and Combined System
  smith::LinearSolverOptions linear_opts{.linear_solver = smith::LinearSolver::CG,
                                         .preconditioner = smith::Preconditioner::HypreJacobi};

  smith::NonlinearSolverOptions nonlin_opts{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                            .relative_tol = 1e-8,
                                            .absolute_tol = 1e-7,
                                            .max_iterations = 30000,
                                            .print_level = 1};

  // One identity Newton update gives p_{m+1} = p_m + kappa*(J_q - J_hat).
  smith::LinearSolverOptions al_linear_opts{.linear_solver = smith::LinearSolver::None,
                                            .preconditioner = smith::Preconditioner::None};
  smith::NonlinearSolverOptions al_nonlin_opts{.nonlin_solver = smith::NonlinearSolver::Newton,
                                               .relative_tol = 1e-12,
                                               .absolute_tol = 1e-12,
                                               .max_iterations = 1,
                                               .print_level = 0};

  auto coupled_solver = std::make_shared<smith::SystemSolver>(200);
  coupled_solver->addSubsystemSolver({0, 1}, smith::buildNonlinearBlockSolver(nonlin_opts, linear_opts, *mesh), 1.0);
  coupled_solver->addSubsystemSolver({2}, smith::buildNonlinearBlockSolver(al_nonlin_opts, al_linear_opts, *mesh), 1.0);
  coupled_solver->setIterationCallback([](int iter, const std::vector<smith::FieldState>& states) {
    const auto metrics = computeConstraintMetrics(*states[0].get(), *states[1].get(), penalty_kappa);
    int rank = 0;
    MPI_Comm_rank(states[0].get()->comm(), &rank);
    if (rank == 0) {
      std::cout << "AL iter " << std::setw(3) << iter << ": int(J_q - Jbar) = " << metrics.integral
                << ", sqrt(sum_E C_E^2) = " << metrics.element_l2_norm
                << ", 0.5*kappa*sum_E C_E^2 = " << metrics.penalty_objective << std::endl;
    }
  });

  auto combined_system = smith::combineSystems(coupled_solver, solid_system, jbar_system, p_system);
  auto physics = smith::makeDifferentiablePhysics(combined_system, "cooks_membrane_fbar");

  auto writer = smith::createParaviewWriter(*mesh, field_store->getOutputFieldStates(), "paraview_cooks_membrane_fbar");
  writer.write(physics->cycle(), physics->time(), field_store->getOutputFieldStates());

  // 5. Run
  *field_store->getField(std::get<0>(solid_fields.fields).name).get() = 0.0;
  *field_store->getField(std::get<1>(solid_fields.fields).name).get() = 0.0;
  *field_store->getField(std::get<0>(jbar_fields.fields).name).get() = 1.0;
  *field_store->getField(std::get<1>(jbar_fields.fields).name).get() = 1.0;
  *field_store->getField(std::get<0>(p_fields.fields).name).get() = 0.0;
  *field_store->getField(std::get<1>(p_fields.fields).name).get() = 0.0;

  physics->advanceTimestep(1.0);

  auto jbar_field = field_store->getField(std::get<0>(jbar_fields.fields).name);
  auto p_field = field_store->getField(std::get<0>(p_fields.fields).name);
  std::cout << "J_bar L2 norm: " << jbar_field.get()->Norml2() << "\n";
  std::cout << "p L2 norm: " << p_field.get()->Norml2() << "\n";

  writer.write(physics->cycle(), physics->time(), field_store->getOutputFieldStates());

  std::cout << "Cook's membrane F-bar simulation completed.\n";

  return 0;
}
