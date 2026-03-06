// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>
#include <memory>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/finite_element.hpp"  // for H1
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/base_physics.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {

constexpr int dim = 2;
constexpr int p = 1;

using SolidMechanicsType = SolidMechanics<p, dim, Parameters<H1<1>, H1<1>>>;

const std::string mesh_tag = "mesh";
const std::string physics_prefix = "solid";

using SolidMaterial = solid_mechanics::ParameterizedNeoHookeanSolid;
// using SolidMaterial = solid_mechanics::ParameterizedLinearIsotropicSolid;

constexpr double boundary_disp = 0.013;
constexpr double shear_modulus_value = 1.0;
constexpr double bulk_modulus_value = 1.0;

std::unique_ptr<SolidMechanicsType> createNonlinearSolidMechanicsSolver(std::shared_ptr<smith::Mesh> mesh,
                                                                        const NonlinearSolverOptions& nonlinear_opts,
                                                                        const SolidMaterial& mat)
{
  static int iter = 0;
  auto solid = std::make_unique<SolidMechanicsType>(
      nonlinear_opts, solid_mechanics::direct_linear_options, solid_mechanics::default_quasistatic_options,
      physics_prefix + std::to_string(iter++), mesh, std::vector<std::string>{"shear modulus", "bulk modulus"});

  // Construct and initialized the user-defined moduli to be used as a differentiable parameter in
  // the solid physics module.
  FiniteElementState user_defined_shear_modulus(mesh->mfemParMesh(), H1<p>{}, "parameterized_shear");
  user_defined_shear_modulus = shear_modulus_value;

  FiniteElementState user_defined_bulk_modulus(mesh->mfemParMesh(), H1<p>{}, "parameterized_bulk");
  user_defined_bulk_modulus = bulk_modulus_value;

  solid->setParameter(0, user_defined_bulk_modulus);
  solid->setParameter(1, user_defined_shear_modulus);

  solid->setMaterial(DependsOn<0, 1>{}, mat, mesh->entireBody());

  solid->addBodyForce(
      [](auto X, auto /* t */) {
        auto Y = X;
        Y[0] = 0.1 + 0.1 * X[0] + 0.3 * X[1];
        Y[1] = -0.05 - 0.2 * X[0] + 0.15 * X[1];
        return 0.1 * X + Y;
      },
      mesh->entireBody());

  mesh->addDomainOfBoundaryElements("essential_bdr", by_attr<dim>(1));
  auto applied_displacement = [](vec2, double) { return boundary_disp * vec2{{1.0, 1.0}}; };

  solid->setDisplacementBCs(applied_displacement, mesh->domain("essential_bdr"));

  solid->completeSetup();

  return solid;
}

FiniteElementState createReactionDirection(const BasePhysics& solid_solver, int direction,
                                           std::shared_ptr<smith::Mesh> mesh)
{
  const FiniteElementDual& reactions = solid_solver.dual("reactions");

  FiniteElementState reactionDirections(reactions.space(), "reaction_directions");
  reactionDirections = 0.0;

  mfem::VectorFunctionCoefficient func(dim, [direction](const mfem::Vector& /*x*/, mfem::Vector& u) {
    u = 0.0;
    u[direction] = 1.0;
  });

  reactionDirections.project(func, mesh->domain("essential_bdr"));

  return reactionDirections;
}

double computeSolidMechanicsQoi(BasePhysics& solid_solver, std::shared_ptr<smith::Mesh> mesh)
{
  solid_solver.resetStates();
  solid_solver.advanceTimestep(0.1);

  const FiniteElementDual& reactions = solid_solver.dual("reactions");
  auto reactionDirections = createReactionDirection(solid_solver, 0, mesh);
  // reactionDirections = solid_solver.dual("reactions");

  const FiniteElementState& displacements = solid_solver.state("displacement");
  return innerProduct(reactions, reactionDirections) + 0.05 * innerProduct(displacements, displacements);
}

auto computeSolidMechanicsQoiSensitivities(BasePhysics& solid_solver, std::shared_ptr<smith::Mesh> mesh)
{
  double qoi = computeSolidMechanicsQoi(solid_solver, mesh);

  FiniteElementDual shape_sensitivity(solid_solver.shapeDisplacement().space(), "shape sensitivity");
  shape_sensitivity = 0.0;

  FiniteElementDual shear_modulus_sensitivity(mesh->mfemParMesh(), H1<p>{}, "shear modulus sensitivity");
  shear_modulus_sensitivity = 0.0;

  auto reaction_adjoint_load = createReactionDirection(solid_solver, 0, mesh);

  FiniteElementDual displacement_adjoint_load(solid_solver.state("displacement").space(), "displacement_adjoint_load");
  displacement_adjoint_load = solid_solver.state("displacement");
  displacement_adjoint_load *= 0.1;

  solid_solver.setAdjointLoad({{"displacement", displacement_adjoint_load}});
  solid_solver.setDualAdjointBcs({{"reactions", reaction_adjoint_load}});
  solid_solver.reverseAdjointTimestep();

  shear_modulus_sensitivity += solid_solver.computeTimestepSensitivity(0);
  shape_sensitivity += solid_solver.computeTimestepShapeSensitivity();

  return std::make_tuple(qoi, shear_modulus_sensitivity, shape_sensitivity);
}

double computeSolidMechanicsQoiAdjustingShape(BasePhysics& solid_solver,
                                              const FiniteElementState& shape_derivative_direction, double pertubation,
                                              std::shared_ptr<smith::Mesh> mesh)
{
  FiniteElementState shape_disp(mesh->mfemParMesh(), H1<SHAPE_ORDER, dim>{}, "input_shape_displacement");
  SLIC_ASSERT_MSG(shape_disp.Size() == shape_derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  shape_disp.Add(pertubation, shape_derivative_direction);
  solid_solver.setShapeDisplacement(shape_disp);

  return computeSolidMechanicsQoi(solid_solver, mesh);
}

double computeSolidMechanicsQoiAdjustingShearModulus(BasePhysics& solid_solver,
                                                     const FiniteElementState& shear_modulus_derivative_direction,
                                                     double pertubation, std::shared_ptr<smith::Mesh> mesh)
{
  FiniteElementState user_defined_shear_modulus(mesh->mfemParMesh(), H1<p>{}, "parameterized_shear");
  user_defined_shear_modulus = shear_modulus_value;

  SLIC_ASSERT_MSG(user_defined_shear_modulus.Size() == shear_modulus_derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  user_defined_shear_modulus.Add(pertubation, shear_modulus_derivative_direction);
  solid_solver.setParameter(0, user_defined_shear_modulus);

  return computeSolidMechanicsQoi(solid_solver, mesh);
}

struct SolidMechanicsSensitivityFixture : public ::testing::Test {
  void SetUp() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
    StateManager::initialize(dataStore, "solid_mechanics_solve");
    std::string filename = std::string(SMITH_REPO_DIR) + "/data/meshes/patch2D_quads.mesh";
    mesh = std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, 2, 1);
    mat.density = 1.0;
    mat.K0 = 1.0;
    mat.G0 = 0.1;
  }

  void fillDirection(FiniteElementState& direction) const
  {
    auto sz = direction.Size();
    for (int i = 0; i < sz; ++i) {
      direction(i) = -1.2 + 2.02 * (double(i) / sz);
    }
  }

  axom::sidre::DataStore dataStore;
  std::shared_ptr<smith::Mesh> mesh;

  NonlinearSolverOptions nonlinear_opts{.relative_tol = 1.0e-15, .absolute_tol = 1.0e-15};

  bool dispBc = true;

  SolidMaterial mat;

  static constexpr double eps = 2e-7;
};

TEST_F(SolidMechanicsSensitivityFixture, ReactionShapeSensitivities)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver(mesh, nonlinear_opts, mat);
  auto [qoi_base, _, shape_sensitivity] = computeSolidMechanicsQoiSensitivities(*solid_solver, mesh);

  solid_solver->resetStates();
  FiniteElementState derivative_direction(shape_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  double qoi_plus = computeSolidMechanicsQoiAdjustingShape(*solid_solver, derivative_direction, eps, mesh);
  double directional_deriv = innerProduct(derivative_direction, shape_sensitivity);

  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

TEST_F(SolidMechanicsSensitivityFixture, ReactionParameterSensitivities)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver(mesh, nonlinear_opts, mat);
  auto [qoi_base, shear_modulus_sensitivity, _] = computeSolidMechanicsQoiSensitivities(*solid_solver, mesh);

  solid_solver->resetStates();
  FiniteElementState derivative_direction(shear_modulus_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  double qoi_plus = computeSolidMechanicsQoiAdjustingShearModulus(*solid_solver, derivative_direction, eps, mesh);
  double directional_deriv = innerProduct(derivative_direction, shear_modulus_sensitivity);

  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
