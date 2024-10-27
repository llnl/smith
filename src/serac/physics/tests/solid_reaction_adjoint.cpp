// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <functional>
#include <set>
#include <string>

#include "serac/physics/solid_mechanics.hpp"
//#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac {

constexpr int dim = 2;
constexpr int p   = 1;

using SolidMechanicsType = SolidMechanics<p, dim, Parameters<H1<1>, H1<1>>>;

const std::string mesh_tag       = "mesh";
const std::string physics_prefix = "solid";

using SolidMaterial = solid_mechanics::ParameterizedNeoHookeanSolid;
// using SolidMaterial = solid_mechanics::ParameterizedLinearIsotropicSolid;
auto geoNonlinear = GeometricNonlinearities::Off;

constexpr double boundary_disp       = 0.013;
constexpr double shear_modulus_value = 1.0;
constexpr double bulk_modulus_value  = 1.0;

std::unique_ptr<SolidMechanicsType> createNonlinearSolidMechanicsSolver(mfem::ParMesh&,
                                                                        const NonlinearSolverOptions& nonlinear_opts,
                                                                        const SolidMaterial&          mat)
{
  static int iter  = 0;
  auto       solid = std::make_unique<SolidMechanicsType>(nonlinear_opts, solid_mechanics::direct_linear_options,
                                                    solid_mechanics::default_quasistatic_options, geoNonlinear,
                                                    physics_prefix + std::to_string(iter++), mesh_tag,
                                                    std::vector<std::string>{"shear modulus", "bulk modulus"});

  // Construct and initialized the user-defined moduli to be used as a differentiable parameter in
  // the solid physics module.
  FiniteElementState user_defined_shear_modulus(StateManager::mesh(mesh_tag), H1<p>{}, "parameterized_shear");
  user_defined_shear_modulus = shear_modulus_value;

  FiniteElementState user_defined_bulk_modulus(StateManager::mesh(mesh_tag), H1<p>{}, "parameterized_bulk");
  user_defined_bulk_modulus = bulk_modulus_value;

  solid->setParameter(0, user_defined_bulk_modulus);
  solid->setParameter(1, user_defined_shear_modulus);

  solid->setMaterial(DependsOn<0, 1>{}, mat);
  solid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& disp) { disp = boundary_disp; });
  solid->addBodyForce([](auto X, auto /* t */) {
    auto Y = X;
    Y[0]   = 0.1 + 0.1 * X[0] + 0.3 * X[1];
    Y[1]   = -0.05 - 0.2 * X[0] + 0.15 * X[1];
    return 0.1 * X + Y;
  });
  solid->completeSetup();

  return solid;
}

template <int dim>
tensor<double, dim> average(std::vector<tensor<double, dim>>& positions)
{
  tensor<double, dim> total{};
  for (auto x : positions) {
    total += x;
  }
  return total / double(positions.size());
}

FiniteElementState createReactionDirection(const BasePhysics& solid_solver, int direction)
{
  const FiniteElementDual& reactions = solid_solver.dual("reactions");

  FiniteElementState reactionDirections(reactions.space(), "reaction_directions");
  reactionDirections = 0.0;

  Domain essential_boundary = Domain::ofBoundaryElements(StateManager::mesh(mesh_tag), by_attr<dim>(1));

  mfem::VectorFunctionCoefficient func(dim, [direction](const mfem::Vector& /*x*/, mfem::Vector& u) {
    u            = 0.0;
    u[direction] = 1.0;
  });

  reactionDirections.project(func, essential_boundary);

  return reactionDirections;
}

double computeSolidMechanicsQoi(BasePhysics& solid_solver)
{
  solid_solver.resetStates();
  solid_solver.advanceTimestep(0.1);

  const FiniteElementDual& reactions          = solid_solver.dual("reactions");
  auto                     reactionDirections = createReactionDirection(solid_solver, 0);
  // reactionDirections = solid_solver.dual("reactions");

  const FiniteElementState& displacements = solid_solver.state("displacement");
  return innerProduct(reactions, reactionDirections) + 0.05 * innerProduct(displacements, displacements);
}

auto computeSolidMechanicsQoiSensitivities(BasePhysics& solid_solver)
{
  double qoi = computeSolidMechanicsQoi(solid_solver);

  FiniteElementDual shape_sensitivity(solid_solver.shapeDisplacement().space(), "shape sensitivity");
  shape_sensitivity = 0.0;

  FiniteElementDual shear_modulus_sensitivity(StateManager::mesh(mesh_tag), H1<p>{}, "shear modulus sensitivity");
  shear_modulus_sensitivity = 0.0;

  auto reaction_adjoint_load = createReactionDirection(solid_solver, 0);

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

double computeSolidMechanicsQoiAdjustingShape(BasePhysics&              solid_solver,
                                              const FiniteElementState& shape_derivative_direction, double pertubation)
{
  FiniteElementState shape_disp(StateManager::mesh(mesh_tag), H1<SHAPE_ORDER, dim>{}, "input_shape_displacement");
  SLIC_ASSERT_MSG(shape_disp.Size() == shape_derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  shape_disp.Add(pertubation, shape_derivative_direction);
  solid_solver.setShapeDisplacement(shape_disp);

  return computeSolidMechanicsQoi(solid_solver);
}

double computeSolidMechanicsQoiAdjustingShearModulus(BasePhysics&              solid_solver,
                                                     const FiniteElementState& shear_modulus_derivative_direction,
                                                     double                    pertubation)
{
  FiniteElementState user_defined_shear_modulus(StateManager::mesh(mesh_tag), H1<p>{}, "parameterized_shear");
  user_defined_shear_modulus = shear_modulus_value;

  SLIC_ASSERT_MSG(user_defined_shear_modulus.Size() == shear_modulus_derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  user_defined_shear_modulus.Add(pertubation, shear_modulus_derivative_direction);
  solid_solver.setParameter(0, user_defined_shear_modulus);

  return computeSolidMechanicsQoi(solid_solver);
}

struct SolidMechanicsSensitivityFixture : public ::testing::Test {
  void SetUp() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
    StateManager::initialize(dataStore, "solid_mechanics_solve");
    std::string filename = std::string(SERAC_REPO_DIR) + "/data/meshes/patch2D_quads.mesh";
    mesh        = &StateManager::setMesh(mesh::refineAndDistribute(buildMeshFromFile(filename), 2, 1), mesh_tag);
    mat.density = 1.0;
    mat.K0      = 1.0;
    mat.G0      = 0.1;
  }

  void fillDirection(FiniteElementState& direction) const
  {
    auto sz = direction.Size();
    for (int i = 0; i < sz; ++i) {
      direction(i) = -1.2 + 2.02 * (double(i) / sz);
    }
  }

  axom::sidre::DataStore dataStore;
  mfem::ParMesh*         mesh;

  NonlinearSolverOptions nonlinear_opts{.relative_tol = 1.0e-15, .absolute_tol = 1.0e-15};

  bool dispBc = true;

  SolidMaterial mat;

  static constexpr double eps = 2e-7;
};

TEST_F(SolidMechanicsSensitivityFixture, ReactionShapeSensitivities)
{
  auto solid_solver                     = createNonlinearSolidMechanicsSolver(*mesh, nonlinear_opts, mat);
  auto [qoi_base, _, shape_sensitivity] = computeSolidMechanicsQoiSensitivities(*solid_solver);

  solid_solver->resetStates();
  FiniteElementState derivative_direction(shape_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  double qoi_plus          = computeSolidMechanicsQoiAdjustingShape(*solid_solver, derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, shape_sensitivity);

  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

TEST_F(SolidMechanicsSensitivityFixture, ReactionParameterSensitivities)
{
  auto solid_solver                             = createNonlinearSolidMechanicsSolver(*mesh, nonlinear_opts, mat);
  auto [qoi_base, shear_modulus_sensitivity, _] = computeSolidMechanicsQoiSensitivities(*solid_solver);

  solid_solver->resetStates();
  FiniteElementState derivative_direction(shear_modulus_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  double qoi_plus          = computeSolidMechanicsQoiAdjustingShearModulus(*solid_solver, derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, shear_modulus_sensitivity);

  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  serac::exitGracefully(result);
  return result;
}
