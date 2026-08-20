// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_adjoint_workflow.cpp
 * @brief Exercises BasePhysics static adjoint port plumbing used by a host code.
 *
 * This test builds complete synthetic downstream gradient maps from stateNames() and dualNames(), runs repeated
 * static adjoint passes with one advertised state or dual port seeded at a time, and checks that all advertised
 * outputs and sensitivities are finite and repeatable. The unit seeds are not physics-specific QoIs; parameter and
 * shape sensitivity accuracy should be validated separately with finite-difference tests for concrete QoIs.
 */

#include <cmath>
#include <memory>
#include <string>
#include <tribol/common/Parameters.hpp>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "mfem.hpp"
#include "mpi.h"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/physics/base_physics.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/smith_config.hpp"

#ifdef SMITH_USE_TRIBOL
#include "smith/physics/contact/contact_config.hpp"
#include "smith/physics/solid_mechanics_contact.hpp"
#ifdef SMITH_USE_ENZYME
#include "tribol/interface/tribol.hpp"
#endif
#endif

namespace smith {
namespace {

constexpr int p = 1;
constexpr int dim = 2;
constexpr int contact_2d_dim = 2;
constexpr int contact_3d_dim = 3;

using ParametricSolid = SolidMechanics<p, dim, Parameters<H1<1>, H1<1>>>;
#ifdef SMITH_USE_TRIBOL
using EnergyContactSolid = SolidMechanicsContact<p, contact_2d_dim>;
using SingleMortarContactSolid = SolidMechanicsContact<p, contact_3d_dim>;
#endif

const std::string mesh_tag{"mesh"};
const std::string physics_prefix{"solid"};

struct AdjointWorkflowResult {
  std::vector<double> state_l2_norms;
  std::vector<double> dual_l2_norms;
  std::vector<double> parameter_l2_norms;
  double shape_l2_norm{};
};

struct StaticForwardResult {
  std::vector<double> state_l2_norms;
  std::vector<double> dual_l2_norms;
};

enum class AdjointSeedKind
{
  State,
  Dual
};

struct AdjointSeed {
  AdjointSeedKind kind;
  std::string name;
};

struct PhysicsCase {
  std::string name;
  std::unique_ptr<BasePhysics> physics;
};

#ifdef SMITH_USE_TRIBOL
struct EnergyMortarContactCaseConfig {
  std::string name;
  ContactOptions options;
  int interaction_id;
#ifdef SMITH_USE_ENZYME
  tribol::EnforcementLocation gap_option = tribol::EnforcementLocation::Nodal;
#endif
};
#endif

constexpr double time_step = 1.0;

std::shared_ptr<Mesh> createSolidMesh()
{
  const std::string filename = SMITH_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";
  return std::make_shared<Mesh>(buildMeshFromFile(filename), mesh_tag, 1, 0);
}

std::unique_ptr<BasePhysics> createSolidSolver(std::shared_ptr<Mesh> mesh)
{
  auto nonlinear_options = NonlinearSolverOptions{.nonlin_solver = NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-15,
                                                  .absolute_tol = 1.0e-15,
                                                  .max_iterations = 10,
                                                  .print_level = 0};

  auto solid = std::make_unique<ParametricSolid>(nonlinear_options, solid_mechanics::direct_linear_options,
                                                 solid_mechanics::default_quasistatic_options, physics_prefix, mesh,
                                                 std::vector<std::string>{"K", "G"});

  FiniteElementState bulk_modulus(mesh->mfemParMesh(), H1<p>{}, "bulk_modulus");
  bulk_modulus = 1.0;
  FiniteElementState shear_modulus(mesh->mfemParMesh(), H1<p>{}, "shear_modulus");
  shear_modulus = 0.5;
  solid->setParameter(0, bulk_modulus);
  solid->setParameter(1, shear_modulus);

  solid_mechanics::ParameterizedNeoHookeanSolid material{1.0, 0.0, 0.0};
  solid->setMaterial(DependsOn<0, 1>{}, material, mesh->entireBody());

  mesh->addDomainOfBoundaryElements("essential_boundary", by_attr<dim>(1));
  solid->setFixedBCs(mesh->domain("essential_boundary"));

  solid->addBodyForce(
      [](auto x, auto) {
        auto force = 0.0 * x;
        force[1] = 1.0e-3 * (1.0 + x[0]);
        return force;
      },
      mesh->entireBody());

  solid->completeSetup();
  return solid;
}

#ifdef SMITH_USE_TRIBOL
std::shared_ptr<Mesh> createSingleMortarContactMesh(const std::string& mesh_name)
{
  const std::string filename = SMITH_REPO_DIR "/data/meshes/contact_two_blocks.g";
  auto mesh = std::make_shared<Mesh>(buildMeshFromFile(filename), mesh_name, 0, 0);
  mesh->addDomainOfBoundaryElements("two", by_attr<contact_3d_dim>(2));
  mesh->addDomainOfBoundaryElements("four", by_attr<contact_3d_dim>(4));
  return mesh;
}

std::unique_ptr<BasePhysics> createSingleMortarContactSolver(std::shared_ptr<Mesh> mesh, const std::string& name,
                                                             int interaction_id)
{
  auto nonlinear_options = NonlinearSolverOptions{.nonlin_solver = NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-13,
                                                  .absolute_tol = 1.0e-13,
                                                  .max_iterations = 20,
                                                  .print_level = 0};

  auto solid = std::make_unique<SingleMortarContactSolid>(nonlinear_options, solid_mechanics::direct_linear_options,
                                                          solid_mechanics::default_quasistatic_options, name, mesh);

  solid_mechanics::NeoHookean material{1.0, 10.0, 0.25};
  solid->setMaterial(material, mesh->entireBody());
  solid->setFixedBCs(mesh->domain("two"));
  solid->setDisplacementBCs(
      [](tensor<double, contact_3d_dim> x, double) {
        auto displacement = 0.0 * x;
        displacement[1] = -0.1;
        return displacement;
      },
      mesh->domain("four"), Component::ALL);

  solid->addContactInteraction(interaction_id, {3}, {5},
                               ContactOptions{.method = ContactMethod::SingleMortar,
                                              .enforcement = ContactEnforcement::Penalty,
                                              .type = ContactType::TiedNormal,
                                              .penalty = 8.0e2,
                                              .jacobian = ContactJacobian::Exact});

  solid->completeSetup();
  return solid;
}

std::shared_ptr<Mesh> createEnergyMortarContactMesh(const std::string& mesh_name)
{
  // clang-format off
  auto mesh = std::make_shared<Mesh>(shared::MeshBuilder::Unify({
      shared::MeshBuilder::SquareMesh(1, 1)
          .translate({0.0, 0.999})
          .updateBdrAttrib(4, 7)
          .updateBdrAttrib(3, 9)
          .updateBdrAttrib(1, 6),
      shared::MeshBuilder::SquareMesh(1, 1)
          .updateBdrAttrib(4, 7)
          .updateBdrAttrib(1, 8)
          .updateBdrAttrib(3, 5)}),
      mesh_name, 0, 0);
  // clang-format on
  mesh->addDomainOfBoundaryElements("x0_faces", by_attr<contact_2d_dim>(7));
  mesh->addDomainOfBoundaryElements("y0_faces", by_attr<contact_2d_dim>(8));
  mesh->addDomainOfBoundaryElements("ymax_face", by_attr<contact_2d_dim>(9));
  return mesh;
}

std::unique_ptr<BasePhysics> createEnergyMortarContactSolver(std::shared_ptr<Mesh> mesh,
                                                             const EnergyMortarContactCaseConfig& config)
{
  auto nonlinear_options = NonlinearSolverOptions{.nonlin_solver = NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-13,
                                                  .absolute_tol = 1.0e-13,
                                                  .max_iterations = 20,
                                                  .print_level = 0};

  auto solid = std::make_unique<EnergyContactSolid>(nonlinear_options, solid_mechanics::direct_linear_options,
                                                    solid_mechanics::default_quasistatic_options, config.name, mesh);

  solid_mechanics::NeoHookean material{1.0, 10.0, 0.25};
  solid->setMaterial(material, mesh->entireBody());
  solid->setFixedBCs(mesh->domain("x0_faces"), Component::X);
  solid->setFixedBCs(mesh->domain("y0_faces"), Component::Y);
  solid->setDisplacementBCs(
      [](tensor<double, contact_2d_dim> x, double) {
        auto displacement = 0.0 * x;
        displacement[1] = -1.0e-2;
        return displacement;
      },
      mesh->domain("ymax_face"), Component::Y);

  solid->addContactInteraction(config.interaction_id, {3}, {5}, config.options);
#ifdef SMITH_USE_ENZYME
  if (config.options.method == ContactMethod::EnergyMortar) {
    tribol::setEnforcementLocation(config.interaction_id, config.gap_option);
  }
#endif

  solid->completeSetup();
  return solid;
}

std::vector<EnergyMortarContactCaseConfig> energyMortarContactCaseConfigs()
{
  std::vector<EnergyMortarContactCaseConfig> configs;
#ifdef SMITH_USE_ENZYME
  configs.push_back(EnergyMortarContactCaseConfig{.name = "solid_mechanics_contact_energy_mortar_nodal_gaps",
                                                  .options = ContactOptions{.method = ContactMethod::EnergyMortar,
                                                                            .enforcement = ContactEnforcement::Penalty,
                                                                            .type = ContactType::Frictionless,
                                                                            .penalty = 1.0e-1,
                                                                            .jacobian = ContactJacobian::Exact},
                                                  .interaction_id = 1,
                                                  .gap_option = tribol::EnforcementLocation::Nodal});
  configs.push_back(EnergyMortarContactCaseConfig{.name = "solid_mechanics_contact_energy_mortar_quadrature_point_gaps",
                                                  .options = ContactOptions{.method = ContactMethod::EnergyMortar,
                                                                            .enforcement = ContactEnforcement::Penalty,
                                                                            .type = ContactType::Frictionless,
                                                                            .penalty = 1.0e-1,
                                                                            .jacobian = ContactJacobian::Exact},
                                                  .interaction_id = 2,
                                                  .gap_option = tribol::EnforcementLocation::QuadraturePoint});
#endif
  return configs;
}
#endif

void updateStaticInputs(BasePhysics& physics)
{
  // Re-apply upstream parameter fields, matching what a host code would do before a solve.
  // This generic test deep-copies the existing fields because arbitrary physics may have different valid parameter
  // values.
  const auto parameter_names = physics.parameterNames();
  for (std::size_t i = 0; i < parameter_names.size(); ++i) {
    FiniteElementState parameter_value(physics.parameter(i));
    physics.setParameter(i, parameter_value);
  }

  FiniteElementState shape_value(physics.shapeDisplacement());
  physics.setShapeDisplacement(shape_value);
}

void setStaticStateGuesses(BasePhysics& physics)
{
  // Zero out each state field for the static solve's initial guess.
  for (const auto& state_name : physics.stateNames()) {
    FiniteElementState state_guess(physics.state(state_name).space(), state_name + "_state_guess");
    state_guess = 0.0;
    physics.setState(state_name, state_guess);
  }
}

std::string seedLabel(const AdjointSeed& seed)
{
  return (seed.kind == AdjointSeedKind::State ? "state:" : "dual:") + seed.name;
}

std::vector<AdjointSeed> adjointSeeds(const BasePhysics& physics)
{
  std::vector<AdjointSeed> seeds;
  for (const auto& state_name : physics.stateNames()) {
    seeds.push_back(AdjointSeed{.kind = AdjointSeedKind::State, .name = state_name});
  }
  for (const auto& dual_name : physics.dualNames()) {
    seeds.push_back(AdjointSeed{.kind = AdjointSeedKind::Dual, .name = dual_name});
  }
  return seeds;
}

StaticForwardResult runStaticForwardPass(BasePhysics& physics)
{
  updateStaticInputs(physics);
  physics.resetStates();
  setStaticStateGuesses(physics);
  physics.advanceTimestep(time_step);
  EXPECT_EQ(1, physics.cycle());
  EXPECT_NEAR(time_step, physics.time(), 1.0e-14);

  StaticForwardResult result;

  // Make sure the forward solve did not produce NaNs or Infs in any advertised output port.
  for (const auto& state_name : physics.stateNames()) {
    result.state_l2_norms.push_back(physics.state(state_name).Norml2());
    EXPECT_TRUE(std::isfinite(result.state_l2_norms.back())) << state_name;
  }

  for (const auto& dual_name : physics.dualNames()) {
    result.dual_l2_norms.push_back(physics.dual(dual_name).Norml2());
    EXPECT_TRUE(std::isfinite(result.dual_l2_norms.back())) << dual_name;
  }

  return result;
}

void expectNearL2Norms(const std::vector<double>& first, const std::vector<double>& second)
{
  ASSERT_EQ(first.size(), second.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    EXPECT_NEAR(first[i], second[i], 1.0e-12);
  }
}

AdjointWorkflowResult runStaticAdjointPass(BasePhysics& physics, const AdjointSeed& seed)
{
  updateStaticInputs(physics);
  physics.resetAdjointStates();
  EXPECT_EQ(1, physics.cycle());

  for (const auto& state_name : physics.stateNames()) {
    const auto checkpointed_state = physics.loadCheckpointedState(state_name, physics.cycle());
    EXPECT_TRUE(std::isfinite(checkpointed_state.Norml2())) << state_name;
  }
  for (const auto& dual_name : physics.dualNames()) {
    const auto checkpointed_dual = physics.loadCheckpointedDual(dual_name, physics.cycle());
    EXPECT_TRUE(std::isfinite(checkpointed_dual.Norml2())) << dual_name;
  }

  std::vector<std::unique_ptr<FiniteElementDual>> state_adjoint_loads;
  std::unordered_map<std::string, const FiniteElementDual&> state_adjoint_load_refs;
  for (const auto& state_name : physics.stateNames()) {
    auto load = std::make_unique<FiniteElementDual>(physics.state(state_name).space(), state_name + "_adjoint_load");
    *load = 0.0;
    if (seed.kind == AdjointSeedKind::State && state_name == seed.name) {
      *load = 1.0;
    }
    state_adjoint_load_refs.insert({state_name, *load});
    state_adjoint_loads.push_back(std::move(load));
  }

  std::vector<std::unique_ptr<FiniteElementState>> dual_adjoint_loads;
  std::unordered_map<std::string, const FiniteElementState&> dual_adjoint_load_refs;
  for (const auto& dual_name : physics.dualNames()) {
    auto load = std::make_unique<FiniteElementState>(physics.dual(dual_name).space(), dual_name + "_adjoint_load");
    *load = 0.0;
    if (seed.kind == AdjointSeedKind::Dual && dual_name == seed.name) {
      *load = 1.0;
    }
    dual_adjoint_load_refs.insert({dual_name, *load});
    dual_adjoint_loads.push_back(std::move(load));
  }

  // These unit DOF seeds are synthetic downstream gradients using the same FE spaces a host code would discover from
  // stateNames() and dualNames(). They test port plumbing, not physical sensitivity accuracy; parameter and shape
  // sensitivity values should be validated with separate finite-difference checks for physics-specific scalar QoIs.
  physics.setAdjointLoad(state_adjoint_load_refs);
  if (!dual_adjoint_load_refs.empty()) {
    physics.setDualAdjointBcs(dual_adjoint_load_refs);
  }
  physics.reverseAdjointTimestep();
  EXPECT_EQ(0, physics.cycle());

  const auto& shape_sensitivity = physics.computeTimestepShapeSensitivity();

  AdjointWorkflowResult result;
  for (const auto& state_name : physics.stateNames()) {
    result.state_l2_norms.push_back(physics.state(state_name).Norml2());
  }
  for (const auto& dual_name : physics.dualNames()) {
    result.dual_l2_norms.push_back(physics.dual(dual_name).Norml2());
  }
  for (std::size_t parameter_index = 0; parameter_index < physics.parameterNames().size(); ++parameter_index) {
    result.parameter_l2_norms.push_back(physics.computeTimestepSensitivity(parameter_index).Norml2());
  }
  result.shape_l2_norm = shape_sensitivity.Norml2();
  return result;
}

std::vector<PhysicsCase> createPhysicsCases()
{
  std::vector<PhysicsCase> cases;

  auto solid_mesh = createSolidMesh();
  cases.push_back(PhysicsCase{.name = "solid_mechanics", .physics = createSolidSolver(solid_mesh)});

#ifdef SMITH_USE_TRIBOL
  const std::string single_mortar_name = "solid_mechanics_contact_single_mortar";
  auto single_mortar_mesh = createSingleMortarContactMesh(single_mortar_name + "_mesh");
  cases.push_back(PhysicsCase{.name = single_mortar_name,
                              .physics = createSingleMortarContactSolver(single_mortar_mesh, single_mortar_name, 0)});

  for (const auto& config : energyMortarContactCaseConfigs()) {
    auto contact_mesh = createEnergyMortarContactMesh(config.name + "_mesh");
    cases.push_back(PhysicsCase{.name = config.name, .physics = createEnergyMortarContactSolver(contact_mesh, config)});
  }
#endif

  return cases;
}

}  // namespace

TEST(AdjointWorkflow, QuasistaticStaticAdjointSolveCanRepeatForBasePhysicsTypes)
{
  MPI_Barrier(MPI_COMM_WORLD);
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "adjoint_workflow_static_repeat");

  for (auto& test_case : createPhysicsCases()) {
    SCOPED_TRACE(test_case.name);
    auto& physics = *test_case.physics;

    // A host code might reuse the last static solution as a warm-start guess. This generic repeatability check
    // intentionally resets to zero guesses each time so both forward passes start from the same cold state.
    const auto first_forward = runStaticForwardPass(physics);
    const auto second_forward = runStaticForwardPass(physics);
    expectNearL2Norms(first_forward.state_l2_norms, second_forward.state_l2_norms);
    expectNearL2Norms(first_forward.dual_l2_norms, second_forward.dual_l2_norms);

    for (const auto& seed : adjointSeeds(physics)) {
      SCOPED_TRACE(seedLabel(seed));
      const auto first = runStaticAdjointPass(physics, seed);
      const auto second = runStaticAdjointPass(physics, seed);

      expectNearL2Norms(first.state_l2_norms, second.state_l2_norms);
      expectNearL2Norms(first.dual_l2_norms, second.dual_l2_norms);
      expectNearL2Norms(first.parameter_l2_norms, second.parameter_l2_norms);
      EXPECT_NEAR(first.shape_l2_norm, second.shape_l2_norm, 1.0e-12);

      for (const auto state_norm : first.state_l2_norms) {
        EXPECT_TRUE(std::isfinite(state_norm));
      }
      for (const auto dual_norm : first.dual_l2_norms) {
        EXPECT_TRUE(std::isfinite(dual_norm));
      }
      for (const auto parameter_norm : first.parameter_l2_norms) {
        EXPECT_TRUE(std::isfinite(parameter_norm));
      }
      EXPECT_TRUE(std::isfinite(first.shape_l2_norm));
    }
  }
}

double computeQoI(const BasePhysics& physics, const AdjointSeed& seed)
{
  double q = 0.0;
  if (seed.kind == AdjointSeedKind::State) {
    const auto& state = physics.state(seed.name);
    FiniteElementDual ones(state.space());
    ones = 1.0;
    q = innerProduct(state, ones);
  } else {
    const auto& dual = physics.dual(seed.name);
    FiniteElementState ones(dual.space());
    ones = 1.0;
    q = innerProduct(dual, ones);
  }
  return q;
}

double fd_param_sensitivity(BasePhysics& physics, const AdjointSeed& seed, size_t param_idx)
{
  const double epsilon = 1.0e-5;
  const auto& param = physics.parameter(param_idx);
  FiniteElementState p0(param);
  FiniteElementState ones(p0.space());
  ones = 1.0;

  FiniteElementState p_plus(p0);
  p_plus.Add(epsilon, ones);
  physics.setParameter(param_idx, p_plus);
  runStaticForwardPass(physics);
  double q_plus = computeQoI(physics, seed);

  FiniteElementState p_minus(p0);
  p_minus.Add(-epsilon, ones);
  physics.setParameter(param_idx, p_minus);
  runStaticForwardPass(physics);
  double q_minus = computeQoI(physics, seed);

  physics.setParameter(param_idx, p0);
  return (q_plus - q_minus) / (2.0 * epsilon);
}

double fd_shape_sensitivity(BasePhysics& physics, const AdjointSeed& seed)
{
  const double epsilon = 1.0e-5;
  const auto& shape_disp = physics.shapeDisplacement();
  FiniteElementState shape0(shape_disp);
  FiniteElementState ones(shape0.space());
  ones = 1.0;

  FiniteElementState shape_plus(shape0);
  shape_plus.Add(epsilon, ones);
  physics.setShapeDisplacement(shape_plus);
  runStaticForwardPass(physics);
  double q_plus = computeQoI(physics, seed);

  FiniteElementState shape_minus(shape0);
  shape_minus.Add(-epsilon, ones);
  physics.setShapeDisplacement(shape_minus);
  runStaticForwardPass(physics);
  double q_minus = computeQoI(physics, seed);

  physics.setShapeDisplacement(shape0);
  return (q_plus - q_minus) / (2.0 * epsilon);
}

TEST(AdjointWorkflow, QuasistaticStaticAdjointSolveFiniteDifference)
{
  MPI_Barrier(MPI_COMM_WORLD);
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "adjoint_workflow_static_fd");

  for (auto& test_case : createPhysicsCases()) {
    SCOPED_TRACE(test_case.name);
    auto& physics = *test_case.physics;

    for (const auto& seed : adjointSeeds(physics)) {
      SCOPED_TRACE(seedLabel(seed));

      runStaticForwardPass(physics);
      runStaticAdjointPass(physics, seed);

      std::vector<double> adj_param_sens(physics.parameterNames().size());
      for (std::size_t param_idx = 0; param_idx < physics.parameterNames().size(); ++param_idx) {
        const auto& param_sens = physics.computeTimestepSensitivity(param_idx);
        FiniteElementState ones(param_sens.space());
        ones = 1.0;
        adj_param_sens[param_idx] = innerProduct(param_sens, ones);
      }

      const auto& shape_sens = physics.computeTimestepShapeSensitivity();
      FiniteElementState shape_ones(shape_sens.space());
      shape_ones = 1.0;
      double adj_shape_sens = innerProduct(shape_sens, shape_ones);

      for (std::size_t param_idx = 0; param_idx < physics.parameterNames().size(); ++param_idx) {
        double fd_sens = fd_param_sensitivity(physics, seed, param_idx);

        double diff = std::abs(adj_param_sens[param_idx] - fd_sens);
        if (std::abs(fd_sens) > 1.0e-5) {
          EXPECT_LT(diff / std::abs(fd_sens), 5.0e-3);
        } else {
          EXPECT_LT(diff, 1.0e-5);
        }
      }

      double fd_shape_sens_val = fd_shape_sensitivity(physics, seed);

      double shape_diff = std::abs(adj_shape_sens - fd_shape_sens_val);
      if (std::abs(fd_shape_sens_val) > 1.0e-5) {
        EXPECT_LT(shape_diff / std::abs(fd_shape_sens_val), 5.0e-3);
      } else {
        EXPECT_LT(shape_diff, 1.0e-5);
      }
    }
  }
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
