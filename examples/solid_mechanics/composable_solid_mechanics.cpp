// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file composable_solid_mechanics.cpp
 * @brief Dynamic solid-mechanics example using composable differentiable numerics systems.
 */

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

// _includes_start
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/functional_objective.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/system_solver.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/time_info_solid_materials.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/evaluate_objective.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
// _includes_end

namespace {

constexpr size_t displacement_state_index = 0;
constexpr size_t initial_displacement_state_index = 1;
constexpr size_t velocity_state_index = 2;

struct TimeInfoYoungsModulusNeoHookean {
  using State = smith::solid_mechanics::NeoHookean::State;

  double density;
  double nu;

  template <typename T, int dim, typename GradVType, typename YoungsType>
  auto operator()(const smith::TimeInfo&, [[maybe_unused]] State& state, const smith::tensor<T, dim, dim>& grad_u,
                  const GradVType&, const YoungsType& youngs_modulus) const
  {
    using std::log1p;
    constexpr auto I = smith::Identity<dim>();
    auto E = smith::get<0>(youngs_modulus);
    auto G = E / (2.0 * (1.0 + nu));
    auto K = E / (3.0 * (1.0 - 2.0 * nu));
    auto lambda = K - (2.0 / dim) * G;
    auto B_minus_I = grad_u * transpose(grad_u) + transpose(grad_u) + grad_u;
    auto logJ = log1p(detApIm1(grad_u));
    auto TK = lambda * logJ * I + G * B_minus_I;
    auto F = grad_u + I;
    return dot(TK, inv(transpose(F)));
  }
};

std::vector<smith::FieldState> outputFields(const smith::FieldStore& field_store)
{
  return {field_store.getField(field_store.prefix("displacement")),
          field_store.getField(field_store.prefix("velocity")),
          field_store.getField(field_store.prefix("acceleration")), field_store.getField(field_store.prefix("stress"))};
}

std::vector<smith::FieldState> qoiFields(const std::vector<smith::FieldState>& states)
{
  if (states.size() <= velocity_state_index) {
    throw std::runtime_error("Unexpected solid state layout.");
  }
  return {states[displacement_state_index], states[velocity_state_index]};
}

}  // namespace

int main(int argc, char* argv[])
{
  // _init_start
  smith::ApplicationManager application_manager(argc, argv);
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "composable_solid_mechanics");
  // _init_end

  // _mesh_start
  constexpr int dim = 3;
  constexpr int order = 1;

  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(8, 2, 2, mfem::Element::HEXAHEDRON, 1.0, 0.1, 0.1), "mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));
  // _mesh_end

  // _solver_start
  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::CG,
                                            .preconditioner = smith::Preconditioner::HypreAMG,
                                            .relative_tol = 1e-6,
                                            .absolute_tol = 1e-10,
                                            .max_iterations = 80,
                                            .print_level = 0};
  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                  .relative_tol = 1e-7,
                                                  .absolute_tol = 1e-8,
                                                  .max_iterations = 15,
                                                  .print_level = 0};

  auto field_store = std::make_shared<smith::FieldStore>(mesh, 100);
  using DispRule = smith::ImplicitNewmarkSecondOrderTimeIntegrationRule;
  using DispSpace = smith::H1<order, dim>;

  smith::SolidMechanicsOptions solid_options{.enable_stress_output = true, .output_cauchy_stress = true};
  auto param_fields = smith::registerParameterFields(smith::FieldType<smith::L2<0>>("youngs_modulus"));
  auto solid_fields = smith::registerSolidMechanicsFields<dim, order, DispRule>(field_store, solid_options);
  // _solver_end

  // _build_start
  auto solid_solver =
      std::make_shared<smith::SystemSolver>(smith::buildNonlinearBlockSolver(nonlinear_options, linear_options, *mesh));

  auto solid_system =
      smith::buildSolidMechanicsSystem<dim, order>(solid_solver, solid_options, solid_fields, param_fields);

  constexpr double E = 100.0;
  constexpr double nu = 0.25;
  solid_system->setMaterial(TimeInfoYoungsModulusNeoHookean{.density = 1.0, .nu = nu}, mesh->entireBodyName());
  field_store->getParameterFields()[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) { return E; });

  auto initial_displacement = [](smith::tensor<double, dim> X) {
    auto displacement = 0.0 * X;
    displacement[0] = 1.0e-3 * X[0];
    return displacement;
  };
  auto initial_velocity = [](smith::tensor<double, dim> X) {
    auto velocity = 0.0 * X;
    velocity[1] = 2.0e-2 * X[0];
    return velocity;
  };

  field_store->getField(field_store->prefix("displacement_solve_state"))
      .get()
      ->setFromFieldFunction(initial_displacement);
  field_store->getField(field_store->prefix("displacement")).get()->setFromFieldFunction(initial_displacement);
  field_store->getField(field_store->prefix("velocity")).get()->setFromFieldFunction(initial_velocity);
  field_store->getField(field_store->prefix("acceleration"))
      .get()
      ->setFromFieldFunction([](smith::tensor<double, dim>) { return smith::tensor<double, dim>{}; });
  // _build_end

  // _bc_start
  solid_system->setDisplacementBC(mesh->domain("left"), std::vector<int>{0, 2});
  solid_system->addBodyForce(smith::DependsOn<>{}, mesh->entireBodyName(), [](double, auto X, auto, auto, auto) {
    auto body_force = 0.0 * X;
    body_force[1] = -0.02;
    return body_force;
  });
  solid_system->addTraction(smith::DependsOn<>{}, "right", [](double, auto X, auto, auto, auto, auto) {
    auto traction = 0.0 * X;
    traction[0] = -0.01;
    return traction;
  });
  // _bc_end

  // _run_start
  if (solid_system->cycle_zero_system == nullptr) {
    throw std::runtime_error("Expected cycle-zero solve for implicit dynamics.");
  }

  auto physics = smith::makeDifferentiablePhysics(solid_system, "composable_solid_mechanics");
  auto initial_states = physics->getInitialFieldStates();
  smith::FunctionalObjective<dim, smith::Parameters<DispSpace, DispSpace>> qoi(
      "solid_dynamic_energy_proxy", mesh, smith::spaces(qoiFields(initial_states)));
  qoi.addBodyIntegral(smith::DependsOn<0, 1>{}, mesh->entireBodyName(), [](double, auto, auto U, auto V) {
    auto u = smith::get<smith::VALUE>(U);
    auto v = smith::get<smith::VALUE>(V);
    return 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) + 0.05 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  });

  constexpr double dt = 0.25;
  constexpr int num_steps = 3;
  auto qoi_state =
      0.0 * smith::evaluateObjective(qoi, physics->getShapeDispFieldState(), qoiFields(initial_states),
                                     smith::TimeInfo(physics->time(), dt, static_cast<size_t>(physics->cycle())));
  for (int step = 0; step < num_steps; ++step) {
    physics->advanceTimestep(dt);
    qoi_state = qoi_state +
                smith::evaluateObjective(qoi, physics->getShapeDispFieldState(), qoiFields(physics->getFieldStates()),
                                         smith::TimeInfo(physics->time(), dt, static_cast<size_t>(physics->cycle())));
  }

  std::cout << "reaction norm: " << physics->getReactionStates().front().get()->Norml2() << '\n';
  gretl::set_as_objective(qoi_state);
  std::cout << "QoI value: " << qoi_state.get() << '\n';
  qoi_state.data_store().back_prop();
  auto shape_displacement = physics->getShapeDispFieldState();
  auto initial_displacement_state = initial_states[initial_displacement_state_index];
  auto initial_velocity_state = initial_states[velocity_state_index];
  auto youngs_modulus_state = field_store->getParameterFields()[0];
  std::cout << "dQoI/d(shape) norm: " << shape_displacement.get_dual()->Norml2() << '\n';
  std::cout << "dQoI/d(youngs_modulus) norm: " << youngs_modulus_state.get_dual()->Norml2() << '\n';
  std::cout << "dQoI/d(initial displacement) norm: " << initial_displacement_state.get_dual()->Norml2() << '\n';
  std::cout << "dQoI/d(initial velocity) norm: " << initial_velocity_state.get_dual()->Norml2() << '\n';
  std::cout << "shape FD rate: " << smith::checkGradWrt(qoi_state, shape_displacement, 1.0e-2, 4, false) << '\n';
  std::cout << "youngs_modulus FD rate: " << smith::checkGradWrt(qoi_state, youngs_modulus_state, 5.0e-2, 4, false)
            << '\n';
  std::cout << "initial displacement FD rate: "
            << smith::checkGradWrt(qoi_state, initial_displacement_state, 5.0e-3, 4, false) << '\n';
  std::cout << "initial velocity FD rate: " << smith::checkGradWrt(qoi_state, initial_velocity_state, 5.0e-3, 4, false)
            << '\n';
  // _run_end

  // _output_start
  auto writer = smith::createParaviewWriter(*mesh, outputFields(*field_store), "paraview_composable_solid_mechanics",
                                            smith::ParaviewWriter::Options{.write_duals = false});
  writer.write(physics->cycle(), physics->time(), outputFields(*field_store));
  std::cout << "ParaView output: paraview_composable_solid_mechanics\n";
  // _output_end

  return 0;
}
