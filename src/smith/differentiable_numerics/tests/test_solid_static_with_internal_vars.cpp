// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/differentiable_numerics/solid_mechanics_with_internal_vars_system.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/state_variable_system.hpp"
#include "smith/differentiable_numerics/combined_system.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"

namespace smith {

LinearSolverOptions solid_linear_options{.linear_solver = LinearSolver::SuperLU,
                                         .preconditioner = Preconditioner::None,
                                         .relative_tol = 1e-12,
                                         .absolute_tol = 1e-12,
                                         .max_iterations = 2000,
                                         .print_level = 0};

NonlinearSolverOptions solid_nonlinear_opts{.nonlin_solver = NonlinearSolver::NewtonLineSearch,
                                            .relative_tol = 1.0e-10,
                                            .absolute_tol = 1.0e-10,
                                            .max_iterations = 100,
                                            .max_line_search_iterations = 50,
                                            .print_level = 1};

static constexpr int dim = 3;
static constexpr int disp_order = 1;
static constexpr int state_order = 0;

using StateSpace = L2<state_order>;
using DispRule = QuasiStaticSecondOrderTimeIntegrationRule;
using InternalVariableRule = BackwardEulerFirstOrderTimeIntegrationRule;

struct SolidStaticWithInternalVarsFixture : public testing::Test {
  void SetUp() override
  {
    StateManager::initialize(datastore, "solid_static_with_internal_vars");
    mesh = std::make_shared<smith::Mesh>(mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0),
                                         "mesh", 0, 0);
    mesh->addDomainOfBoundaryElements("bottom", by_attr<dim>(1));  // z=0
    mesh->addDomainOfBoundaryElements("top", by_attr<dim>(6));     // z=1
  }

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
};

// Simple damage-like material: stiffness decreases with internal state variable (isv)
struct DamageMaterial {
  using State = smith::QOI;

  double E = 100.0;
  double nu = 0.3;
  double density = 1.0;

  template <typename StateType, typename DerivType, typename ISVType, typename... Params>
  SMITH_HOST_DEVICE auto operator()(StateType /*state*/, DerivType deriv_u, ISVType isv, Params... /*params*/) const
  {
    auto epsilon = sym(deriv_u);
    auto tr_eps = tr(epsilon);
    auto I = Identity<dim>();

    double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = E / (2.0 * (1.0 + nu));

    // Stiffness degradation: (1 - isv) * Stress_elastic
    auto damage = isv;
    if (damage > 1.0) damage = 1.0;
    if (damage < 0.0) damage = 0.0;

    auto factor = 1.0 - damage;
    auto sigma = factor * (lambda * tr_eps * I + 2.0 * mu * epsilon);

    return sigma;
  }
};

// Evolution law: isv_dot = (1 - isv) * eps_norm
//
// This ODE drives isv toward 1 as eps_norm increases.
// For constant eps_norm, the solution is: isv(t) = 1 - (1 - isv_0) * exp(-eps_norm * t)
// which asymptotes to 1 as eps_norm -> infinity (fast convergence) or t -> infinity.
//
// Residual form returned: isv_dot - (1 - isv) * eps_norm = 0
struct StrainNormEvolution {
  template <typename TimeInfo, typename ISVType, typename ISVDotType, typename DerivType, typename... Params>
  SMITH_HOST_DEVICE auto operator()(TimeInfo /*t_info*/, ISVType isv, ISVDotType isv_dot, DerivType deriv_u,
                                    Params... /*params*/) const
  {
    using std::sqrt;
    auto epsilon = sym(deriv_u);
    auto eps_norm = sqrt(inner(epsilon, epsilon) + 1e-16);
    return isv_dot - (1.0 - isv) * eps_norm;
  }
};

std::shared_ptr<SystemSolver> makeSystemSolver(const Mesh& mesh)
{
  return std::make_shared<SystemSolver>(buildNonlinearBlockSolver(solid_nonlinear_opts, solid_linear_options, mesh));
}

auto registerFields(const std::shared_ptr<FieldStore>& field_store)
{
  return std::tuple{registerSolidMechanicsFields<dim, disp_order, DispRule>(field_store),
                    registerInternalVariableFields<StateSpace, InternalVariableRule>(field_store)};
}

template <typename SolidFields, typename InternalVariableFields>
auto buildSystems(const std::shared_ptr<SystemSolver>& solid_solver,
                  const std::shared_ptr<SystemSolver>& internal_variable_solver, const SolidFields& solid_fields,
                  const InternalVariableFields& internal_variable_fields)
{
  return std::tuple{
      buildSolidMechanicsSystem<dim, disp_order, DispRule>(solid_solver, SolidMechanicsOptions{}, solid_fields,
                                                           internal_variable_fields),
      buildInternalVariableSystem<dim, StateSpace, InternalVariableRule>(internal_variable_solver,
                                                                         InternalVariableOptions{},
                                                                         internal_variable_fields, solid_fields)};
}

template <typename SolidSystemType, typename InternalVariableSystemType>
void setDamageCoupling(const std::shared_ptr<SolidSystemType>& solid,
                       const std::shared_ptr<InternalVariableSystemType>& internal_variables,
                       const std::shared_ptr<Mesh>& mesh)
{
  setCoupledSolidMechanicsInternalVariableMaterial(solid, internal_variables, DamageMaterial{}, mesh->entireBodyName());
  setCoupledInternalVariableMaterial(internal_variables, solid, StrainNormEvolution{}, mesh->entireBodyName());
}

template <typename SolidSystemType>
void setPullBoundaryConditions(const std::shared_ptr<SolidSystemType>& solid, const std::shared_ptr<Mesh>& mesh,
                               double pull_rate)
{
  solid->disp_bc->template setFixedVectorBCs<dim>(mesh->domain("bottom"));
  solid->disp_bc->template setVectorBCs<dim>(mesh->domain("top"), [pull_rate](double t, tensor<double, dim> /*X*/) {
    tensor<double, dim> u{};
    u[2] = pull_rate * t;
    return u;
  });
}

TEST_F(SolidStaticWithInternalVarsFixture, CoupledSolve)
{
  auto solid_solver = makeSystemSolver(*mesh);
  auto internal_variable_solver = makeSystemSolver(*mesh);
  auto nonlinear_block_solver = buildNonlinearBlockSolver(solid_nonlinear_opts, solid_linear_options, *mesh);
  auto coupled_solver = std::make_shared<SystemSolver>(nonlinear_block_solver);
  auto field_store = std::make_shared<FieldStore>(mesh, 100, "solid_static_with_internal_vars_");
  auto [solid_fields, internal_variable_fields] = registerFields(field_store);
  auto [solid, internal_variables] =
      buildSystems(solid_solver, internal_variable_solver, solid_fields, internal_variable_fields);

  auto system = combineSystems(coupled_solver, solid, internal_variables);

  setDamageCoupling(solid, internal_variables, mesh);
  setPullBoundaryConditions(solid, mesh, 0.05);

  auto physics = makeDifferentiablePhysics(system, "physics");

  // Create ParaView writer
  auto writer = createParaviewWriter(*mesh, system->field_store->getOutputFieldStates(), "solid_state_output");
  writer.write(0, 0.0, system->field_store->getOutputFieldStates());
  // Advance multiple steps
  for (int step = 1; step <= 5; ++step) {
    physics->advanceTimestep(1.0);
    writer.write(step, step * 1.0, system->field_store->getOutputFieldStates());
    SLIC_INFO("Completed step " << step);
  }
}

TEST_F(SolidStaticWithInternalVarsFixture, StaggeredSolveWithRelaxation)
{
  auto solid_solver = makeSystemSolver(*mesh);
  auto internal_variable_solver = makeSystemSolver(*mesh);
  auto disp_solver = buildNonlinearBlockSolver(solid_nonlinear_opts, solid_linear_options, *mesh);
  auto internal_variable_block_solver = buildNonlinearBlockSolver(solid_nonlinear_opts, solid_linear_options, *mesh);

  // Staggered solver: stage 0 solves displacement, stage 1 solves internal variable.
  // Use relaxation_factor = 0.5 on the displacement stage to exercise the relaxation path.
  auto staggered_solver = std::make_shared<SystemSolver>(20);
  staggered_solver->addSubsystemSolver({0}, disp_solver, 0.5);
  staggered_solver->addSubsystemSolver({1}, internal_variable_block_solver, 1.0);

  auto field_store = std::make_shared<FieldStore>(mesh, 100, "solid_staggered_relaxation_");
  auto [solid_fields, internal_variable_fields] = registerFields(field_store);
  auto [solid, internal_variables] =
      buildSystems(solid_solver, internal_variable_solver, solid_fields, internal_variable_fields);

  auto system = combineSystems(staggered_solver, solid, internal_variables);

  setDamageCoupling(solid, internal_variables, mesh);
  setPullBoundaryConditions(solid, mesh, 0.05);

  auto physics = makeDifferentiablePhysics(system, "physics_relaxed");
  for (int step = 1; step <= 3; ++step) {
    physics->advanceTimestep(1.0);
    SLIC_INFO("Staggered relaxation step " << step << " completed");
  }
}

TEST_F(SolidStaticWithInternalVarsFixture, BodyForceAndTraction)
{
  auto solid_solver = makeSystemSolver(*mesh);
  auto internal_variable_solver = makeSystemSolver(*mesh);
  auto nonlinear_block_solver = buildNonlinearBlockSolver(solid_nonlinear_opts, solid_linear_options, *mesh);
  auto coupled_solver = std::make_shared<SystemSolver>(nonlinear_block_solver);
  auto field_store = std::make_shared<FieldStore>(mesh, 100, "body_force_test_");
  auto [solid_fields, internal_variable_fields] = registerFields(field_store);
  auto [solid, internal_variables] =
      buildSystems(solid_solver, internal_variable_solver, solid_fields, internal_variable_fields);

  auto system = combineSystems(coupled_solver, solid, internal_variables);

  setDamageCoupling(solid, internal_variables, mesh);

  // Fix bottom face
  solid->disp_bc->setFixedVectorBCs<dim>(mesh->domain("bottom"));

  // Apply a gravity-like body force in the -z direction
  double body_force_mag = -0.01;
  solid->addBodyForce(mesh->entireBodyName(), [=](double, auto, auto, auto, auto, auto, auto) {
    tensor<double, dim> f{};
    f[2] = body_force_mag;
    return f;
  });

  // Apply a traction on the top face in the +z direction
  double traction_mag = 0.005;
  solid->addTraction("top", [=](double, auto, auto /*n*/, auto, auto, auto, auto, auto) {
    tensor<double, dim> t{};
    t[2] = traction_mag;
    return t;
  });

  auto physics = makeDifferentiablePhysics(system, "physics_bf");
  physics->advanceTimestep(1.0);

  // Check that the displacement field is non-zero (the body force + traction produced deformation)
  auto states = physics->getFieldStates();
  double disp_norm = norm(*states[0].get());
  EXPECT_GT(disp_norm, 1e-8) << "Body force + traction should produce nonzero displacement";
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
