// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <memory>
#include <string>
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

using ParametricSolid = SolidMechanics<p, dim, Parameters<H1<1>, H1<1>>>;

const std::string mesh_tag {"mesh"};
const std::string physics_prefix {"solid"};

struct AdjointWorkflowResult {
  double displacement_norm {};
  double reaction_norm {};
  double parameter0_norm {};
  double parameter1_norm {};
  double shape_norm {};
};

constexpr int num_time_steps = 3;
constexpr double time_step = 1.0;

std::shared_ptr<Mesh> createMesh()
{
  const std::string filename = SMITH_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";
  return std::make_shared<Mesh>(buildMeshFromFile(filename), mesh_tag, 1, 0);
}

std::unique_ptr<ParametricSolid> createSolidSolver(std::shared_ptr<Mesh> mesh)
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

FiniteElementState createReactionDirection(const BasePhysics& solid_solver, std::shared_ptr<Mesh> mesh, int direction)
{
  const FiniteElementDual& reactions = solid_solver.dual("reactions");

  FiniteElementState reaction_directions(reactions.space(), "reaction_directions");
  reaction_directions = 0.0;

  mfem::VectorFunctionCoefficient reaction_direction(dim, [direction](const mfem::Vector& /*x*/, mfem::Vector& value) {
    value = 0.0;
    value[direction] = 1.0;
  });

  reaction_directions.project(reaction_direction, mesh->domain("essential_boundary"));

  return reaction_directions;
}

AdjointWorkflowResult runSolidSequence(ParametricSolid& solid, std::shared_ptr<Mesh> mesh)
{
  solid.resetStates();
  for (int step = 0; step < num_time_steps; ++step) {
    solid.advanceTimestep(time_step);
    EXPECT_EQ(step + 1, solid.cycle());
    EXPECT_GT(solid.state("displacement").Norml2(), 0.0);
    EXPECT_GT(solid.dual("reactions").Norml2(), 0.0);
  }

  solid.resetAdjointStates();

  const auto& displacement = solid.state("displacement");
  const auto& reactions = solid.dual("reactions");

  FiniteElementDual displacement_adjoint_load(displacement.space(), "displacement_adjoint_load");
  displacement_adjoint_load = 1.0;

  auto reaction_adjoint_load = createReactionDirection(solid, mesh, 1);

  FiniteElementDual parameter0_sensitivity(solid.parameter(0).space(), "parameter0_sensitivity");
  parameter0_sensitivity = 0.0;
  FiniteElementDual parameter1_sensitivity(solid.parameter(1).space(), "parameter1_sensitivity");
  parameter1_sensitivity = 0.0;
  FiniteElementDual shape_sensitivity(solid.shapeDisplacement().space(), "shape_sensitivity");
  shape_sensitivity = 0.0;

  for (int step = num_time_steps; step > 0; --step) {
    const auto& checkpointed_displacement = solid.loadCheckpointedState("displacement", solid.cycle());
    const auto& checkpointed_reactions = solid.loadCheckpointedDual("reactions", solid.cycle());
    EXPECT_GT(checkpointed_displacement.Norml2(), 0.0);
    EXPECT_GT(checkpointed_reactions.Norml2(), 0.0);

    solid.setAdjointLoad({{"displacement", displacement_adjoint_load}});
    solid.setDualAdjointBcs({{"reactions", reaction_adjoint_load}});
    solid.reverseAdjointTimestep();

    parameter0_sensitivity += solid.computeTimestepSensitivity(0);
    parameter1_sensitivity += solid.computeTimestepSensitivity(1);
    shape_sensitivity += solid.computeTimestepShapeSensitivity();
    EXPECT_EQ(step - 1, solid.cycle());
  }

  AdjointWorkflowResult result;
  result.displacement_norm = displacement.Norml2();
  result.reaction_norm = reactions.Norml2();
  result.parameter0_norm = parameter0_sensitivity.Norml2();
  result.parameter1_norm = parameter1_sensitivity.Norml2();
  result.shape_norm = shape_sensitivity.Norml2();
  return result;
}

#ifdef SMITH_USE_TRIBOL
using ContactSolid = SolidMechanicsContact<p, 3>;

std::shared_ptr<Mesh> createContactMesh()
{
  const std::string filename = SMITH_REPO_DIR "/data/meshes/contact_two_blocks.g";
  return std::make_shared<Mesh>(buildMeshFromFile(filename), "contact_mesh", 0, 0);
}

std::unique_ptr<ContactSolid> createContactSolver(std::shared_ptr<Mesh> mesh)
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

  mesh->addDomainOfBoundaryElements("support", by_attr<3>(2));
  mesh->addDomainOfBoundaryElements("driven_surface", by_attr<3>(4));
  solid->setFixedBCs(mesh->domain("support"));
  solid->setDisplacementBCs(
      [](tensor<double, 3>, double) {
        tensor<double, 3> value {};
        value[1] = -0.1;
        return value;
      },
      mesh->domain("driven_surface"), Component::ALL);

  solid->addContactInteraction(0, {3}, {5},
                               ContactOptions{.method = ContactMethod::SingleMortar,
                                              .enforcement = ContactEnforcement::Penalty,
                                              .type = ContactType::TiedNormal,
                                              .penalty = 8.0e2,
                                              .jacobian = ContactJacobian::Exact});

  solid->completeSetup();
  return solid;
}

AdjointWorkflowResult runContactSequence(ContactSolid& solid, std::shared_ptr<Mesh> mesh)
{
  solid.resetStates();
  for (int step = 0; step < num_time_steps; ++step) {
    solid.advanceTimestep(time_step);
    EXPECT_EQ(step + 1, solid.cycle());
    EXPECT_GT(solid.state("displacement").Norml2(), 0.0);
    EXPECT_GT(solid.dual("contact_force_0").Norml2(), 0.0);
  }

  solid.resetAdjointStates();

  const auto& displacement = solid.state("displacement");
  const auto& contact_force = solid.dual("contact_force_0");

  FiniteElementDual displacement_adjoint_load(displacement.space(), "displacement_adjoint_load");
  displacement_adjoint_load = 1.0;

  FiniteElementState contact_force_adjoint_load(contact_force.space(), "contact_force_adjoint_load");
  contact_force_adjoint_load = 0.0;
  mfem::VectorFunctionCoefficient contact_force_direction(3, [](const mfem::Vector&, mfem::Vector& value) {
    value = 0.0;
    value[1] = 1.0;
  });
  contact_force_adjoint_load.project(contact_force_direction, mesh->domain("driven_surface"));

  FiniteElementDual shape_sensitivity(solid.shapeDisplacement().space(), "shape_sensitivity");
  shape_sensitivity = 0.0;

  for (int step = num_time_steps; step > 0; --step) {
    const auto& checkpointed_displacement = solid.loadCheckpointedState("displacement", solid.cycle());
    EXPECT_GT(checkpointed_displacement.Norml2(), 0.0);

    solid.setAdjointLoad({{"displacement", displacement_adjoint_load}});
    solid.setDualAdjointBcs({{"contact_force_0", contact_force_adjoint_load}});
    solid.reverseAdjointTimestep();

    shape_sensitivity += static_cast<BasePhysics&>(solid).computeTimestepShapeSensitivity();
    EXPECT_EQ(step - 1, solid.cycle());
  }

  AdjointWorkflowResult result;
  result.displacement_norm = displacement.Norml2();
  result.reaction_norm = contact_force.Norml2();
  result.shape_norm = shape_sensitivity.Norml2();
  return result;
}
#endif

}  // namespace

TEST(AdjointWorkflow, QuasistaticSolidMechanics)
{
  MPI_Barrier(MPI_COMM_WORLD);
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "adjoint_workflow_solid");

  auto mesh = createMesh();
  auto solid = createSolidSolver(mesh);

  const auto first = runSolidSequence(*solid, mesh);
  const auto second = runSolidSequence(*solid, mesh);

  EXPECT_NEAR(first.displacement_norm, second.displacement_norm, 1.0e-12);
  EXPECT_NEAR(first.reaction_norm, second.reaction_norm, 1.0e-12);
  EXPECT_NEAR(first.parameter0_norm, second.parameter0_norm, 1.0e-12);
  EXPECT_NEAR(first.parameter1_norm, second.parameter1_norm, 1.0e-12);
  EXPECT_NEAR(first.shape_norm, second.shape_norm, 1.0e-12);
  EXPECT_GT(first.parameter0_norm + first.parameter1_norm, 0.0);
  EXPECT_GT(first.shape_norm, 0.0);
}

#ifdef SMITH_USE_TRIBOL
TEST(AdjointWorkflow, ContactSolidMechanics)
{
  MPI_Barrier(MPI_COMM_WORLD);
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "adjoint_workflow_contact");

  auto mesh = createContactMesh();
  auto solid = createContactSolver(mesh);

  const auto first = runContactSequence(*solid, mesh);
  const auto second = runContactSequence(*solid, mesh);

  EXPECT_NEAR(first.displacement_norm, second.displacement_norm, 1.0e-12);
  EXPECT_NEAR(first.reaction_norm, second.reaction_norm, 1.0e-12);
  EXPECT_NEAR(first.shape_norm, second.shape_norm, 1.0e-12);
  EXPECT_GT(first.reaction_norm, 0.0);
  EXPECT_GT(first.shape_norm, 0.0);
}
#endif

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
