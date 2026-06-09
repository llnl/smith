// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
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
#endif

namespace smith {
namespace {

constexpr int p = 1;
constexpr int dim = 2;
constexpr int contact_dim = 3;

using ParametricSolid = SolidMechanics<p, dim, Parameters<H1<1>, H1<1>>>;
#ifdef SMITH_USE_TRIBOL
using ContactSolid = SolidMechanicsContact<p, contact_dim>;
#endif

const std::string mesh_tag {"mesh"};
const std::string physics_prefix {"solid"};

struct AdjointWorkflowResult {
  double displacement_norm {};
  double reaction_norm {};
  double parameter0_norm {};
  double parameter1_norm {};
  double shape_norm {};
};

struct PhysicsCase {
  std::string name;
  std::shared_ptr<Mesh> mesh;
  std::unique_ptr<BasePhysics> physics;
  std::string adjoint_dual_name;
  std::string adjoint_dual_domain_name;
  int adjoint_dual_direction {};
  std::string preferred_validation_dual_name;
  bool check_parameter_sensitivities {};
};

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
                                                 std::vector<std::string>{"E", "v"});

  FiniteElementState bulk_modulus(mesh->mfemParMesh(), H1<p>{}, "bulk_modulus");
  bulk_modulus = 1.0;
  FiniteElementState poisson_ratio(mesh->mfemParMesh(), H1<p>{}, "poisson_ratio");
  poisson_ratio = 0.3;
  solid->setParameter(0, bulk_modulus);
  solid->setParameter(1, poisson_ratio);

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
std::shared_ptr<Mesh> createContactMesh()
{
  const std::string filename = SMITH_REPO_DIR "/data/meshes/contact_two_blocks.g";
  auto mesh = std::make_shared<Mesh>(buildMeshFromFile(filename), "contact_mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("two", by_attr<contact_dim>(2));
  mesh->addDomainOfBoundaryElements("four", by_attr<contact_dim>(4));
  return mesh;
}

std::unique_ptr<BasePhysics> createContactSolver(std::shared_ptr<Mesh> mesh)
{
  auto nonlinear_options = NonlinearSolverOptions{.nonlin_solver = NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-13,
                                                  .absolute_tol = 1.0e-13,
                                                  .max_iterations = 20,
                                                  .print_level = 0};

  auto solid = std::make_unique<ContactSolid>(nonlinear_options, solid_mechanics::direct_linear_options,
                                              solid_mechanics::default_quasistatic_options, "contact_solid", mesh);

  solid_mechanics::NeoHookean material{1.0, 10.0, 0.25};
  solid->setMaterial(material, mesh->entireBody());
  solid->setFixedBCs(mesh->domain("two"));
  solid->setDisplacementBCs(
      [](tensor<double, contact_dim> x, double) {
        auto displacement = 0.0 * x;
        displacement[1] = -0.1;
        return displacement;
      },
      mesh->domain("four"), Component::ALL);

  solid->addContactInteraction(0, {3}, {5},
                               ContactOptions{.method = ContactMethod::SingleMortar,
                                              .enforcement = ContactEnforcement::Penalty,
                                              .type = ContactType::TiedNormal,
                                              .penalty = 8.0e2,
                                              .jacobian = ContactJacobian::Exact});

  solid->completeSetup();
  return solid;
}
#endif

FiniteElementState createDualDirection(const BasePhysics& physics, std::shared_ptr<Mesh> mesh, const std::string& dual_name,
                                       const std::string& domain_name, int direction)
{
  const FiniteElementDual& dual = physics.dual(dual_name);

  FiniteElementState dual_direction(dual.space(), "dual_direction");
  dual_direction = 0.0;

  const int vdim = dual.space().GetVDim();
  mfem::VectorFunctionCoefficient direction_coefficient(vdim, [direction](const mfem::Vector& /*x*/, mfem::Vector& value) {
    value = 0.0;
    value[direction] = 1.0;
  });

  dual_direction.project(direction_coefficient, mesh->domain(domain_name));

  return dual_direction;
}

void updateStaticInputs(BasePhysics& physics)
{
  const auto parameter_names = physics.parameterNames();
  for (std::size_t i = 0; i < parameter_names.size(); ++i) {
    FiniteElementState parameter_value(physics.parameter(i));
    physics.setParameter(i, parameter_value);
  }

  FiniteElementState zero_shape(physics.shapeDisplacement().space(), "zero_shape");
  zero_shape = 0.0;
  physics.setShapeDisplacement(zero_shape);
}

void setStaticStateGuesses(BasePhysics& physics)
{
  for (const auto& state_name : physics.stateNames()) {
    FiniteElementState state_guess(physics.state(state_name).space(), state_name + "_state_guess");
    state_guess = 0.0;
    physics.setState(state_name, state_guess);
  }
}

std::string validationDualName(const BasePhysics& physics, const PhysicsCase& test_case)
{
  const auto dual_names = physics.dualNames();
  EXPECT_FALSE(dual_names.empty());
  if (dual_names.empty()) {
    return "";
  }

  if (std::find(dual_names.begin(), dual_names.end(), test_case.preferred_validation_dual_name) != dual_names.end()) {
    return test_case.preferred_validation_dual_name;
  }

  return dual_names.front();
}

AdjointWorkflowResult runStaticAdjointPass(BasePhysics& physics, const PhysicsCase& test_case)
{
  updateStaticInputs(physics);
  physics.resetAdjointStates();
  EXPECT_EQ(1, physics.cycle());

  const auto validation_dual_name = validationDualName(physics, test_case);
  const auto& displacement = physics.state("displacement");
  const auto& validation_dual = physics.dual(validation_dual_name);
  const auto checkpointed_displacement = physics.loadCheckpointedState("displacement", physics.cycle());
  const auto checkpointed_validation_dual = physics.loadCheckpointedDual(validation_dual_name, physics.cycle());
  EXPECT_GT(checkpointed_displacement.Norml2(), 0.0);
  EXPECT_GT(checkpointed_validation_dual.Norml2(), 0.0);

  std::vector<std::unique_ptr<FiniteElementDual>> state_adjoint_loads;
  std::unordered_map<std::string, const FiniteElementDual&> state_adjoint_load_refs;
  for (const auto& state_name : physics.stateNames()) {
    auto load = std::make_unique<FiniteElementDual>(physics.state(state_name).space(), state_name + "_adjoint_load");
    *load = 0.0;
    if (state_name == "displacement") {
      *load = 1.0;
    }
    state_adjoint_load_refs.insert({state_name, *load});
    state_adjoint_loads.push_back(std::move(load));
  }

  auto dual_adjoint_load = createDualDirection(physics, test_case.mesh, test_case.adjoint_dual_name,
                                              test_case.adjoint_dual_domain_name, test_case.adjoint_dual_direction);

  std::vector<std::unique_ptr<FiniteElementState>> dual_adjoint_loads;
  std::unordered_map<std::string, const FiniteElementState&> dual_adjoint_load_refs;
  for (const auto& dual_name : physics.dualNames()) {
    auto load = std::make_unique<FiniteElementState>(physics.dual(dual_name).space(), dual_name + "_adjoint_load");
    *load = 0.0;
    if (dual_name == test_case.adjoint_dual_name) {
      *load = dual_adjoint_load;
    }
    dual_adjoint_load_refs.insert({dual_name, *load});
    dual_adjoint_loads.push_back(std::move(load));
  }

  physics.setAdjointLoad(state_adjoint_load_refs);
  physics.setDualAdjointBcs(dual_adjoint_load_refs);
  physics.reverseAdjointTimestep();
  EXPECT_EQ(0, physics.cycle());

  const auto& shape_sensitivity = physics.computeTimestepShapeSensitivity();

  AdjointWorkflowResult result;
  result.displacement_norm = displacement.Norml2();
  result.reaction_norm = validation_dual.Norml2();
  if (physics.parameterNames().size() > 0) {
    result.parameter0_norm = physics.computeTimestepSensitivity(0).Norml2();
  }
  if (physics.parameterNames().size() > 1) {
    result.parameter1_norm = physics.computeTimestepSensitivity(1).Norml2();
  }
  result.shape_norm = shape_sensitivity.Norml2();
  return result;
}

std::vector<PhysicsCase> createPhysicsCases()
{
  std::vector<PhysicsCase> cases;

  auto solid_mesh = createSolidMesh();
  cases.push_back(PhysicsCase{.name = "solid_mechanics",
                              .mesh = solid_mesh,
                              .physics = createSolidSolver(solid_mesh),
                              .adjoint_dual_name = "reactions",
                              .adjoint_dual_domain_name = "essential_boundary",
                              .adjoint_dual_direction = 1,
                              .preferred_validation_dual_name = "reactions",
                              .check_parameter_sensitivities = true});

#ifdef SMITH_USE_TRIBOL
  auto contact_mesh = createContactMesh();
  cases.push_back(PhysicsCase{.name = "solid_mechanics_contact",
                              .mesh = contact_mesh,
                              .physics = createContactSolver(contact_mesh),
                              .adjoint_dual_name = "reactions",
                              .adjoint_dual_domain_name = "two",
                              .adjoint_dual_direction = 1,
                              .preferred_validation_dual_name = "contact_force_0",
                              .check_parameter_sensitivities = false});
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

    updateStaticInputs(physics);
    physics.resetStates();
    setStaticStateGuesses(physics);
    physics.advanceTimestep(time_step);
    EXPECT_EQ(1, physics.cycle());
    EXPECT_NEAR(time_step, physics.time(), 1.0e-14);
    EXPECT_GT(physics.state("displacement").Norml2(), 0.0);

    const auto dual_names = physics.dualNames();
    EXPECT_FALSE(dual_names.empty());
    for (const auto& dual_name : dual_names) {
      EXPECT_TRUE(std::isfinite(physics.dual(dual_name).Norml2())) << dual_name;
    }

    const auto first = runStaticAdjointPass(physics, test_case);
    const auto second = runStaticAdjointPass(physics, test_case);

    EXPECT_NEAR(first.displacement_norm, second.displacement_norm, 1.0e-12);
    EXPECT_NEAR(first.reaction_norm, second.reaction_norm, 1.0e-12);
    EXPECT_NEAR(first.parameter0_norm, second.parameter0_norm, 1.0e-12);
    EXPECT_NEAR(first.parameter1_norm, second.parameter1_norm, 1.0e-12);
    EXPECT_NEAR(first.shape_norm, second.shape_norm, 1.0e-12);

    EXPECT_TRUE(std::isfinite(first.displacement_norm));
    EXPECT_TRUE(std::isfinite(first.reaction_norm));
    EXPECT_TRUE(std::isfinite(first.parameter0_norm));
    EXPECT_TRUE(std::isfinite(first.parameter1_norm));
    EXPECT_TRUE(std::isfinite(first.shape_norm));
    if (test_case.check_parameter_sensitivities) {
      EXPECT_GT(first.parameter0_norm + first.parameter1_norm, 0.0);
    }
    EXPECT_GT(first.shape_norm, 0.0);
  }
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
