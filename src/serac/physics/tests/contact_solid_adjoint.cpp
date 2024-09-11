// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <functional>
#include <set>
#include <string>

#include "serac/physics/solid_mechanics_contact.hpp"
#include "serac/physics/materials/solid_material.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac {

constexpr int dim = 3;
constexpr int p   = 1;

const std::string mesh_tag       = "mesh";
const std::string physics_prefix = "solid";

using SolidMaterial = solid_mechanics::NeoHookean;

struct TimeSteppingInfo {
  TimeSteppingInfo() : dts({0.0, 0.2, 0.4, 0.24, 0.12, 0.0}) {}

  int numTimesteps() const { return dts.Size() - 2; }

  mfem::Vector dts;
};

constexpr double disp_target = -0.34;

// MRT: add explicit velocity dependence
double computeStepQoi(const FiniteElementState& displacement)
{
  FiniteElementState displacement_error(displacement);
  displacement_error = -disp_target;
  displacement_error.Add(1.0, displacement);
  return 0.5 * innerProduct(displacement_error, displacement_error);
}

void computeStepAdjointLoad(const FiniteElementState& displacement, FiniteElementDual& d_qoi_d_displacement)
{
  d_qoi_d_displacement = -disp_target;
  d_qoi_d_displacement.Add(1.0, displacement);
}

using SolidMechT = serac::SolidMechanicsContact<p, dim>;

std::unique_ptr<SolidMechT> createContactSolver(const NonlinearSolverOptions& nonlinear_opts,
                                                const TimesteppingOptions& dyn_opts, const SolidMaterial& mat)
{
  static int iter = 0;

  auto solid =
      std::make_unique<SolidMechT>(nonlinear_opts, solid_mechanics::direct_linear_options, dyn_opts,
                                   physics_prefix + std::to_string(iter++), mesh_tag);
  solid->setMaterial(mat);

  solid->setDisplacementBCs({2}, [](const mfem::Vector& /*X*/, double /*t*/, mfem::Vector& disp) { disp = 0.0; });
  solid->setDisplacementBCs({4}, [](const mfem::Vector& /*X*/, double /*t*/, mfem::Vector& disp) {
    disp    = 0.0;
    disp[1] = -0.1;
  });

  auto   contact_type   = serac::ContactEnforcement::Penalty;
  double element_length = 1.0;
  double penalty        = 105.1 * mat.K / element_length;

  serac::ContactOptions contact_options{.method      = serac::ContactMethod::SingleMortar,
                                        .enforcement = contact_type,
                                        .type        = serac::ContactType::Frictionless,
                                        .penalty     = penalty};
  auto                  contact_interaction_id = 0;
  solid->addContactInteraction(contact_interaction_id, {3}, {5}, contact_options);

  solid->completeSetup();

  return solid;
}

double computeSolidMechanicsQoi(BasePhysics& solid_solver, const TimeSteppingInfo& ts_info)
{
  auto dts = ts_info.dts;
  solid_solver.resetStates();
  solid_solver.outputStateToDisk("paraview_contact");
  solid_solver.advanceTimestep(1.0);
  solid_solver.outputStateToDisk("paraview_contact");
  return computeStepQoi(solid_solver.state("displacement"));
}

auto computeContactQoiSensitivities(BasePhysics& solid_solver, const TimeSteppingInfo& ts_info)
{
  EXPECT_EQ(0, solid_solver.cycle());

  double qoi = computeSolidMechanicsQoi(solid_solver, ts_info);

  FiniteElementDual shape_sensitivity(solid_solver.shapeDisplacement().space(), "shape sensitivity");
  FiniteElementDual adjoint_load(solid_solver.state("displacement").space(), "adjoint_displacement_load");

  auto previous_displacement = solid_solver.loadCheckpointedState("displacement", solid_solver.cycle());
  computeStepAdjointLoad(previous_displacement, adjoint_load);
  EXPECT_EQ(1, solid_solver.cycle());
  solid_solver.setAdjointLoad({{"displacement", adjoint_load}});
  solid_solver.reverseAdjointTimestep();
  shape_sensitivity = solid_solver.computeTimestepShapeSensitivity();
  EXPECT_EQ(0, solid_solver.cycle());

  return std::make_tuple(qoi, shape_sensitivity);
}

double computeSolidMechanicsQoiAdjustingShape(SolidMechanics<p, dim>& solid_solver, const TimeSteppingInfo& ts_info,
                                              const FiniteElementState& shape_derivative_direction, double pertubation)
{
  FiniteElementState shape_disp(shape_derivative_direction.space(), "input_shape_displacement");

  shape_disp.Add(pertubation, shape_derivative_direction);
  solid_solver.setShapeDisplacement(shape_disp);

  return computeSolidMechanicsQoi(solid_solver, ts_info);
}

struct ContactSensitivityFixture : public ::testing::Test {
  void SetUp() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
    StateManager::initialize(dataStore, "contact_solve");
    std::string filename = std::string(SERAC_REPO_DIR) + "/data/meshes/contact_two_blocks.g";
    mesh                 = &StateManager::setMesh(mesh::refineAndDistribute(buildMeshFromFile(filename), 0), mesh_tag);

    mat.density = 1.0;
    mat.K       = 1.0;
    mat.G       = 0.1;
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

  NonlinearSolverOptions nonlinear_opts{.relative_tol = 1.0e-10, .absolute_tol = 1.0e-12};

  bool                dispBc = true;
  TimesteppingOptions dyn_opts{.timestepper = TimestepMethod::QuasiStatic};

  SolidMaterial    mat;
  TimeSteppingInfo tsInfo;

  static constexpr double eps = 2e-7;
};

TEST_F(ContactSensitivityFixture, WhenShapeSensitivitiesCalledTwice_GetSameObjectiveAndGradient)
{
  auto solid_solver               = createContactSolver(nonlinear_opts, dyn_opts, mat);
  auto [qoi1, shape_sensitivity1] = computeContactQoiSensitivities(*solid_solver, tsInfo);

  solid_solver->resetStates();
  FiniteElementState derivative_direction(shape_sensitivity1.space(), "derivative_direction");
  fillDirection(derivative_direction);

  auto [qoi2, shape_sensitivity2] = computeContactQoiSensitivities(*solid_solver, tsInfo);

  EXPECT_EQ(qoi1, qoi2);

  double directional_deriv1 = innerProduct(derivative_direction, shape_sensitivity1);
  double directional_deriv2 = innerProduct(derivative_direction, shape_sensitivity2);
  EXPECT_EQ(directional_deriv1, directional_deriv2);
}

TEST_F(ContactSensitivityFixture, QuasiStaticShapeSensitivities)
{
  auto solid_solver                  = createContactSolver(nonlinear_opts, dyn_opts, mat);
  auto [qoi_base, shape_sensitivity] = computeContactQoiSensitivities(*solid_solver, tsInfo);

  solid_solver->resetStates();
  FiniteElementState derivative_direction(shape_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  double qoi_plus = computeSolidMechanicsQoiAdjustingShape(*solid_solver, tsInfo, derivative_direction, eps);

  double directional_deriv    = innerProduct(derivative_direction, shape_sensitivity);
  double directional_deriv_fd = (qoi_plus - qoi_base) / eps;
  // std::cout << "qoi, derivs = " << qoi_base << " " << directional_deriv << " " << directional_deriv_fd << std::endl;
  EXPECT_NEAR(directional_deriv, directional_deriv_fd, 0.2);  // These are very large tolerances
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
