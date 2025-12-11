// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
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

#include "smith/physics/solid_mechanics_contact.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/base_physics.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/contact/contact_config.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {

constexpr int dim = 3;
constexpr int p = 1;

const std::string mesh_tag = "mesh";
const std::string physics_prefix = "solid";

using SolidMaterial = solid_mechanics::NeoHookean;

struct TimeSteppingInfo {
  TimeSteppingInfo() : dts({0.0, 0.2, 0.4, 0.24, 0.12, 0.0}) {}
  int numTimesteps() const { return dts.Size() - 2; }
  mfem::Vector dts;
};

constexpr double disp_target = -0.34;

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

using SolidMechT = smith::SolidMechanicsContact<p, dim>;
// using SolidMechT = smith::SolidMechanics<p, dim>;

std::unique_ptr<SolidMechT> createContactSolver(std::shared_ptr<Mesh> mesh,
                                                const NonlinearSolverOptions& nonlinear_opts,
                                                const TimesteppingOptions& dyn_opts, const SolidMaterial& mat)
{
  static int iter = 0;

  auto solid = std::make_unique<SolidMechT>(nonlinear_opts, solid_mechanics::direct_linear_options, dyn_opts,
                                            physics_prefix + std::to_string(iter++), mesh, std::vector<std::string>{},
                                            0, 0.0, false, true);

  solid->setMaterial(mat, mesh->entireBody());
  solid->setFixedBCs(mesh->domain("two"));
  solid->setDisplacementBCs(
      [](tensor<double, dim> x, double) {
        auto r = 0.0 * x;
        r[1] -= 0.1;
        return r;
      },
      mesh->domain("four"), Component::ALL);

#if 1
  auto contact_type = smith::ContactEnforcement::Penalty;
  double element_length = 1.0;
  double penalty = 105.1 * mat.K / element_length;

  smith::ContactOptions contact_options{.method = smith::ContactMethod::SingleMortar,
                                        .enforcement = contact_type,
                                        .type = smith::ContactType::TiedNormal,
                                        .penalty = penalty,
                                        .jacobian = smith::ContactJacobian::Exact};
  auto contact_interaction_id = 0;
  solid->addContactInteraction(contact_interaction_id, {3}, {5}, contact_options);
#endif

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
    std::string filename = std::string(SMITH_REPO_DIR) + "/data/meshes/contact_two_blocks.g";

    mesh = std::make_shared<Mesh>(filename, mesh_tag, 0, 0);
    mesh->addDomainOfBoundaryElements("two", by_attr<dim>(2));
    mesh->addDomainOfBoundaryElements("four", by_attr<dim>(4));

    mat.density = 1.0;
    mat.K = 1.0;
    mat.G = 0.1;
  }

  void fillDirection(const SolidMechT& solid_solver, FiniteElementState& direction) const
  {
    auto sz = direction.Size();
    for (int i = 0; i < sz; ++i) {
      direction(i) = -1.2 + 2.02 * (double(i) / sz);
      // direction(i) = 0.0;
    }
    solid_solver.zeroEssentials(direction);
  }

  axom::sidre::DataStore dataStore;
  std::shared_ptr<Mesh> mesh;

  NonlinearSolverOptions nonlinear_opts{.relative_tol = 1.0e-10, .absolute_tol = 1.0e-12};

  bool dispBc = true;
  TimesteppingOptions dyn_opts{.timestepper = TimestepMethod::QuasiStatic};

  SolidMaterial mat;
  TimeSteppingInfo tsInfo;

#ifdef SMITH_USE_ENZYME
  static constexpr double eps = 2e-7;
#else
  static constexpr double eps = 0.2;
#endif
};

TEST_F(ContactSensitivityFixture, WhenShapeSensitivitiesCalledTwice_GetSameObjectiveAndGradient)
{
  auto solid_solver = createContactSolver(mesh, nonlinear_opts, dyn_opts, mat);
  auto [qoi1, shape_sensitivity1] = computeContactQoiSensitivities(*solid_solver, tsInfo);
  auto [qoi2, shape_sensitivity2] = computeContactQoiSensitivities(*solid_solver, tsInfo);

  EXPECT_EQ(qoi1, qoi2);

  solid_solver->resetStates();
  FiniteElementState derivative_direction(shape_sensitivity1.space(), "derivative_direction");
  fillDirection(*solid_solver, derivative_direction);

  double directional_deriv1 = innerProduct(derivative_direction, shape_sensitivity1);
  double directional_deriv2 = innerProduct(derivative_direction, shape_sensitivity2);
  EXPECT_EQ(directional_deriv1, directional_deriv2);
}

TEST_F(ContactSensitivityFixture, QuasiStaticShapeSensitivities)
{
  auto solid_solver = createContactSolver(mesh, nonlinear_opts, dyn_opts, mat);
  auto [qoi_base, shape_sensitivity] = computeContactQoiSensitivities(*solid_solver, tsInfo);

  solid_solver->resetStates();
  FiniteElementState derivative_direction(shape_sensitivity.space(), "derivative_direction");
  fillDirection(*solid_solver, derivative_direction);

  double qoi_plus = computeSolidMechanicsQoiAdjustingShape(*solid_solver, tsInfo, derivative_direction, eps);

  double directional_deriv = innerProduct(derivative_direction, shape_sensitivity);
  double directional_deriv_fd = (qoi_plus - qoi_base) / eps;
  EXPECT_NEAR(directional_deriv, directional_deriv_fd, eps);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  int result = RUN_ALL_TESTS();

  return result;
}
