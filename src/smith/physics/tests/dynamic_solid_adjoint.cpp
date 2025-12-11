// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cstddef>
#include <string>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/base_physics.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {

constexpr int dim = 2;
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
constexpr double boundary_disp = 0.00013;

constexpr double initial_interior_disp = 0.001;
constexpr double initial_interior_velo = 0.04;

double computeStepQoi(const FiniteElementState& displacement, const FiniteElementDual& reactions, double dt)
{
  FiniteElementState displacement_error(displacement);
  displacement_error = disp_target;
  displacement_error.Add(1.0, displacement);
  return 0.5 * dt * innerProduct(displacement_error, displacement_error) +
         0.05 * dt * innerProduct(reactions, reactions);
}

void computeStepAdjointLoad(const FiniteElementState& displacement, const FiniteElementDual& reactions,
                            FiniteElementDual& d_qoi_d_displacement, FiniteElementState& d_qoi_d_reactions, double dt)
{
  d_qoi_d_displacement = disp_target;
  d_qoi_d_displacement.Add(1.0, displacement);
  d_qoi_d_displacement *= dt;

  d_qoi_d_reactions = reactions;
  d_qoi_d_reactions *= 0.1 * dt;
}

void applyInitialAndBoundaryConditions(SolidMechanics<p, dim>& solid_solver)
{
  FiniteElementState velo = solid_solver.velocity();
  velo = initial_interior_velo;
  solid_solver.zeroEssentials(velo);
  solid_solver.setVelocity(velo);

  FiniteElementState disp = solid_solver.displacement();
  disp = initial_interior_disp;
  solid_solver.zeroEssentials(disp);

  FiniteElementState bDisp1 = disp;
  FiniteElementState bDisp2 = disp;
  bDisp1 = boundary_disp;
  bDisp2 = boundary_disp;
  solid_solver.zeroEssentials(bDisp2);

  disp += bDisp1;
  disp -= bDisp2;

  solid_solver.setDisplacement(disp);
}

double computeSolidMechanicsQoi(BasePhysics& solid_solver, const TimeSteppingInfo& ts_info)
{
  auto dts = ts_info.dts;

  std::string outname = "paraview_solid";
  solid_solver.advanceTimestep(dts(0));  // advance by 0.0 seconds to get initial acceleration
  solid_solver.outputStateToDisk(outname);

  FiniteElementState dispForObjective = solid_solver.state("displacement");

  FiniteElementDual reactionsForObjective = solid_solver.dual("reactions");
  double qoi = computeStepQoi(dispForObjective, reactionsForObjective, 0.5 * (dts(0) + dts(1)));

  for (int i = 1; i <= ts_info.numTimesteps(); ++i) {
    solid_solver.advanceTimestep(dts(i));
    solid_solver.outputStateToDisk(outname);

    dispForObjective = solid_solver.state("displacement");
    reactionsForObjective = solid_solver.dual("reactions");
    qoi += computeStepQoi(dispForObjective, reactionsForObjective, 0.5 * (dts(i) + dts(i + 1)));
  }
  return qoi;
}

std::tuple<double, FiniteElementDual, FiniteElementDual, FiniteElementDual> computeSolidMechanicsQoiSensitivities(
    BasePhysics& solid_solver, const TimeSteppingInfo& ts_info)
{
  EXPECT_EQ(0, solid_solver.cycle());

  double qoi = computeSolidMechanicsQoi(solid_solver, ts_info);

  FiniteElementDual initial_displacement_sensitivity(solid_solver.state("displacement").space(),
                                                     "init_displacement_sensitivity");
  initial_displacement_sensitivity = 0.0;
  FiniteElementDual initial_velocity_sensitivity(solid_solver.state("velocity").space(), "init_velocity_sensitivity");
  initial_velocity_sensitivity = 0.0;
  FiniteElementDual shape_sensitivity(solid_solver.shapeDisplacement().space(), "shape sensitivity");
  shape_sensitivity = 0.0;

  FiniteElementDual adjoint_load(solid_solver.state("displacement").space(), "adjoint_displacement_load");
  FiniteElementState adjoint_bcs(solid_solver.dual("reactions").space(), "adjoint_reaction_bcs");

  // for solids, we go back to time = 0, because there is an extra hidden implicit solve at the start
  // consider unifying the interface between solids and thermal
  for (int i = solid_solver.cycle(); i > 0; --i) {
    auto previous_displacement = solid_solver.loadCheckpointedState("displacement", solid_solver.cycle());
    auto previous_reactions = solid_solver.loadCheckpointedDual("reactions", solid_solver.cycle());

    computeStepAdjointLoad(
        previous_displacement, previous_reactions, adjoint_load, adjoint_bcs,
        0.5 * (solid_solver.getCheckpointedTimestep(i - 1) + solid_solver.getCheckpointedTimestep(i)));
    EXPECT_EQ(i, solid_solver.cycle());
    solid_solver.setAdjointLoad({{"displacement", adjoint_load}});
    solid_solver.setDualAdjointBcs({{"reactions", adjoint_bcs}});
    solid_solver.reverseAdjointTimestep();

    shape_sensitivity += solid_solver.computeTimestepShapeSensitivity();
    EXPECT_EQ(i - 1, solid_solver.cycle());
  }

  EXPECT_EQ(0, solid_solver.cycle());  // we are back to the start
  auto initialConditionSensitivities = solid_solver.computeInitialConditionSensitivity();
  auto initialDisplacementSensitivityIter = initialConditionSensitivities.find("displacement");
  auto initialVelocitySensitivityIter = initialConditionSensitivities.find("velocity");
  SLIC_ASSERT_MSG(initialDisplacementSensitivityIter != initialConditionSensitivities.end(),
                  "Could not find displacement in the computed initial condition sensitivities.");
  SLIC_ASSERT_MSG(initialVelocitySensitivityIter != initialConditionSensitivities.end(),
                  "Could not find velocity in the computed initial condition sensitivities.");
  initial_displacement_sensitivity = initialDisplacementSensitivityIter->second;
  initial_velocity_sensitivity = initialVelocitySensitivityIter->second;

  return std::make_tuple(qoi, initial_displacement_sensitivity, initial_velocity_sensitivity, shape_sensitivity);
}

std::tuple<double, std::vector<FiniteElementDual>> computeSolidMechanicsQoiParameterSensitivities(
    BasePhysics& solid_solver, const TimeSteppingInfo& ts_info)
{
  EXPECT_EQ(0, solid_solver.cycle());

  double qoi = computeSolidMechanicsQoi(solid_solver, ts_info);

  std::vector<FiniteElementDual> parameter_sensitivities;
  size_t num_params = solid_solver.parameterNames().size();

  for (size_t ip = 0; ip < num_params; ++ip) {
    parameter_sensitivities.emplace_back(solid_solver.parameter(ip).space(), "param_" + std::to_string(ip));
    parameter_sensitivities.back() = 0.0;
  }

  FiniteElementDual adjoint_load(solid_solver.state("displacement").space(), "adjoint_displacement_load");
  FiniteElementState adjoint_bcs(solid_solver.dual("reactions").space(), "adjoint_reaction_bcs");

  // for solids, we go back to time = 0, because there is an extra hidden implicit solve at the start
  // consider unifying the interface between solids and thermal
  for (int i = solid_solver.cycle(); i > 0; --i) {
    auto previous_displacement = solid_solver.loadCheckpointedState("displacement", solid_solver.cycle());
    auto previous_reactions = solid_solver.loadCheckpointedDual("reactions", solid_solver.cycle());

    computeStepAdjointLoad(
        previous_displacement, previous_reactions, adjoint_load, adjoint_bcs,
        0.5 * (solid_solver.getCheckpointedTimestep(i - 1) + solid_solver.getCheckpointedTimestep(i)));
    EXPECT_EQ(i, solid_solver.cycle());
    solid_solver.setAdjointLoad({{"displacement", adjoint_load}});
    solid_solver.setDualAdjointBcs({{"reactions", adjoint_bcs}});
    solid_solver.reverseAdjointTimestep();

    for (size_t ip = 0; ip < num_params; ++ip) {
      parameter_sensitivities[ip] += solid_solver.computeTimestepSensitivity(ip);
    }

    EXPECT_EQ(i - 1, solid_solver.cycle());
  }

  EXPECT_EQ(0, solid_solver.cycle());  // we are back to the start

  return std::make_tuple(qoi, parameter_sensitivities);
}

double computeSolidMechanicsQoiAdjustingShape(BasePhysics& solid_solver, const TimeSteppingInfo& ts_info,
                                              const FiniteElementState& shape_derivative_direction, double pertubation)
{
  FiniteElementState shape_disp(StateManager::mesh(mesh_tag), H1<SHAPE_ORDER, dim>{}, "input_shape_displacement");
  SLIC_ASSERT_MSG(shape_disp.Size() == shape_derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  shape_disp.Add(pertubation, shape_derivative_direction);
  solid_solver.setShapeDisplacement(shape_disp);

  return computeSolidMechanicsQoi(solid_solver, ts_info);
}

double computeSolidMechanicsQoiAdjustingInitialDisplacement(BasePhysics& solid_solver, const TimeSteppingInfo& ts_info,
                                                            const FiniteElementState& derivative_direction,
                                                            double pertubation)
{
  FiniteElementState disp = solid_solver.state("displacement");
  SLIC_ASSERT_MSG(disp.Size() == derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  disp.Add(pertubation, derivative_direction);
  solid_solver.setState("displacement", disp);

  return computeSolidMechanicsQoi(solid_solver, ts_info);
}

double computeSolidMechanicsQoiAdjustingInitialVelocity(BasePhysics& solid_solver, const TimeSteppingInfo& ts_info,
                                                        const FiniteElementState& derivative_direction,
                                                        double pertubation)
{
  FiniteElementState velo = solid_solver.state("velocity");
  SLIC_ASSERT_MSG(velo.Size() == derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  velo.Add(pertubation, derivative_direction);
  solid_solver.setState("velocity", velo);

  return computeSolidMechanicsQoi(solid_solver, ts_info);
}

double computeSolidMechanicsQoiAdjustingParam(BasePhysics& solid_solver, const TimeSteppingInfo& ts_info,
                                              const FiniteElementState& derivative_direction, size_t param_index,
                                              double pertubation)
{
  std::string param_name = solid_solver.parameterNames()[param_index];
  FiniteElementState param = solid_solver.parameter(param_name);
  SLIC_ASSERT_MSG(param.Size() == derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  param.Add(pertubation, derivative_direction);
  solid_solver.setParameter(param_index, param);

  auto qoi = computeSolidMechanicsQoi(solid_solver, ts_info);

  param.Add(-pertubation, derivative_direction);
  solid_solver.setParameter(param_index, param);

  return qoi;
}

struct SolidMechanicsSensitivityFixture : public ::testing::Test {
  void SetUp() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
    StateManager::initialize(dataStore, "solid_mechanics_solve");
    std::string filename = std::string(SMITH_REPO_DIR) + "/data/meshes/star.mesh";
    mesh = std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag);
    mat.density = 1.0;
    mat.K = 1.0;
    mat.G = 0.1;
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

  NonlinearSolverOptions nonlinear_opts{
      .nonlin_solver = NonlinearSolver::TrustRegion, .relative_tol = 1.0e-15, .absolute_tol = 1.0e-14};

  LinearSolverOptions linear_opts = {.linear_solver = LinearSolver::CG,
                                     .preconditioner = Preconditioner::HypreJacobi,
                                     .relative_tol = 1.0e-10,
                                     .absolute_tol = 1.0e-16,
                                     .max_iterations = 2000,
                                     .print_level = 0};

  bool dispBc = true;
  TimesteppingOptions dyn_opts{.timestepper = TimestepMethod::Newmark,
                               .enforcement_method = dispBc ? DirichletEnforcementMethod::DirectControl
                                                            : DirichletEnforcementMethod::RateControl};

  SolidMaterial mat;
  TimeSteppingInfo tsInfo;

  static constexpr double eps = 1e-7;

  std::unique_ptr<SolidMechanics<p, dim>> createNonlinearSolidMechanicsSolver()
  {
    static int iter = 0;

    bool checkpoint_to_disk = true;
    auto solid = std::make_unique<SolidMechanics<p, dim>>(
        nonlinear_opts, linear_opts, dyn_opts, physics_prefix + std::to_string(iter++), mesh,
        std::vector<std::string>{}, 0, 0.0, checkpoint_to_disk, false);
    solid->setMaterial(mat, mesh->entireBody());

    auto applied_displacement = [](tensor<double, dim>, double t) {
      auto u = make_tensor<dim>([t](int) { return (1.0 + 10 * t) * boundary_disp; });
      return u;
    };
    auto applied_displacement_surface = Domain::ofBoundaryElements(solid->mfemParMesh(), by_attr<dim>(1));
    solid->setDisplacementBCs(applied_displacement, applied_displacement_surface);
    solid->addBodyForce(
        [](auto X, auto t) {
          auto Y = X;
          Y[0] = 0.1 + 0.1 * X[0] + 0.3 * X[1] - 0.2 * t;
          Y[1] = -0.05 - 0.08 * X[0] + 0.15 * X[1] + 0.3 * t;
          return 0.4 * X + Y;
        },
        mesh->entireBody());
    solid->completeSetup();

    applyInitialAndBoundaryConditions(*solid);

    return solid;
  }
};

TEST_F(SolidMechanicsSensitivityFixture, InitialDisplacementSensitivities)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver();
  auto [qoi_base, init_disp_sensitivity, _, __] = computeSolidMechanicsQoiSensitivities(*solid_solver, tsInfo);

  solid_solver->resetStates();
  applyInitialAndBoundaryConditions(*solid_solver);
  FiniteElementState derivative_direction(solid_solver->displacement().space(), "derivative_direction");
  fillDirection(derivative_direction);
  solid_solver->zeroEssentials(derivative_direction);

  double qoi_plus =
      computeSolidMechanicsQoiAdjustingInitialDisplacement(*solid_solver, tsInfo, derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, init_disp_sensitivity);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, 50 * eps);
}

TEST_F(SolidMechanicsSensitivityFixture, InitialVelocitySensitivities)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver();
  auto [qoi_base, _, init_velo_sensitivity, __] = computeSolidMechanicsQoiSensitivities(*solid_solver, tsInfo);

  solid_solver->resetStates();
  applyInitialAndBoundaryConditions(*solid_solver);
  FiniteElementState derivative_direction(solid_solver->velocity().space(), "derivative_direction");
  fillDirection(derivative_direction);
  solid_solver->zeroEssentials(derivative_direction);

  double qoi_plus = computeSolidMechanicsQoiAdjustingInitialVelocity(*solid_solver, tsInfo, derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, init_velo_sensitivity);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, 55 * eps);
}

TEST_F(SolidMechanicsSensitivityFixture, ShapeSensitivities)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver();
  auto [qoi_base, _, __, shape_sensitivity] = computeSolidMechanicsQoiSensitivities(*solid_solver, tsInfo);

  solid_solver->resetStates();
  applyInitialAndBoundaryConditions(*solid_solver);
  FiniteElementState derivative_direction(shape_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  double qoi_plus = computeSolidMechanicsQoiAdjustingShape(*solid_solver, tsInfo, derivative_direction, eps);

  double directional_deriv = innerProduct(derivative_direction, shape_sensitivity);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, 10 * eps);
}

TEST_F(SolidMechanicsSensitivityFixture, QuasiStaticShapeSensitivities)
{
  dyn_opts.timestepper = TimestepMethod::QuasiStatic;
  auto solid_solver = createNonlinearSolidMechanicsSolver();
  auto [qoi_base, _, __, shape_sensitivity] = computeSolidMechanicsQoiSensitivities(*solid_solver, tsInfo);

  solid_solver->resetStates();
  applyInitialAndBoundaryConditions(*solid_solver);
  FiniteElementState derivative_direction(shape_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  double qoi_plus = computeSolidMechanicsQoiAdjustingShape(*solid_solver, tsInfo, derivative_direction, eps);

  double directional_deriv = innerProduct(derivative_direction, shape_sensitivity);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, 10 * eps);
}

TEST_F(SolidMechanicsSensitivityFixture, WhenShapeSensitivitiesCalledTwice_GetSameObjectiveAndGradient)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver();
  auto [qoi1, shape_unused1, shape_unused2, shape_sensitivity1] =
      computeSolidMechanicsQoiSensitivities(*solid_solver, tsInfo);

  solid_solver->resetStates();
  applyInitialAndBoundaryConditions(*solid_solver);
  FiniteElementState derivative_direction(shape_sensitivity1.space(), "derivative_direction");
  fillDirection(derivative_direction);

  auto [qoi2, shape_unused3, shape_unused4, shape_sensitivity2] =
      computeSolidMechanicsQoiSensitivities(*solid_solver, tsInfo);

  EXPECT_EQ(qoi1, qoi2);

  double directional_deriv1 = innerProduct(derivative_direction, shape_sensitivity1);
  double directional_deriv2 = innerProduct(derivative_direction, shape_sensitivity2);
  EXPECT_EQ(directional_deriv1, directional_deriv2);
}

struct BucklingSensitivityFixture : public ::testing::Test {
  void SetUp() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
    StateManager::initialize(dataStore, "solid_mechanics_solve");
    int Nx = 24;
    int Ny = 12;
    // double Lx = 1.0; double Ly = 5.0;
    double Lx = 5.0;
    double Ly = 5.0;
    mesh = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian2D(Nx, Ny, mfem::Element::QUADRILATERAL, true, Lx, Ly), mesh_tag, 0, 0);
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

  NonlinearSolverOptions nonlinear_opts{.nonlin_solver = NonlinearSolver::TrustRegion,
                                        .relative_tol = 1.0e-15,
                                        .absolute_tol = 1.0e-13,
                                        .max_iterations = 200,
                                        .print_level = 0};

  LinearSolverOptions linear_opts{.linear_solver = LinearSolver::CG,
                                  .preconditioner = Preconditioner::HypreAMG,
                                  .relative_tol = 1.0e-9,
                                  .absolute_tol = 1.0e-14,
                                  .max_iterations = 2000,
                                  .print_level = 0};

  bool dispBc = true;
  TimesteppingOptions dyn_opts{.timestepper = TimestepMethod::Newmark,
                               .enforcement_method = dispBc ? DirichletEnforcementMethod::DirectControl
                                                            : DirichletEnforcementMethod::RateControl};

  using ParSolidMaterial = solid_mechanics::ParameterizedNeoHookeanSolid;
  ParSolidMaterial mat;
  TimeSteppingInfo tsInfo;

  static constexpr double eps = 4e-7;

  std::string kname = "bulk";
  std::string gname = "shear";

  template <typename DensitySpace>
  auto createBucklingSolidMechanicsSolver()
  {
    static int iter = 0;

    bool checkpoint_to_disk = false;
    auto solid = std::make_unique<SolidMechanics<p, dim, Parameters<DensitySpace, DensitySpace>>>(
        nonlinear_opts, linear_opts, dyn_opts, physics_prefix + std::to_string(iter++), mesh,
        std::vector<std::string>{kname, gname}, 0, 0.0, checkpoint_to_disk, false);

    solid->setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());
    return solid;
  }
};

TEST_F(BucklingSensitivityFixture, QuasiStaticShapeSensitivities)
{
  dyn_opts.timestepper = TimestepMethod::QuasiStatic;

  using DensitySpace = smith::L2<0>;

  auto solid_solver = createBucklingSolidMechanicsSolver<DensitySpace>();

  auto applied_displacement_surface = Domain::ofBoundaryElements(solid_solver->mfemParMesh(), by_attr<dim>(1));
  auto applied_traction_surface = Domain::ofBoundaryElements(solid_solver->mfemParMesh(), by_attr<dim>(3));
  double load = 0.1;

  solid_solver->setTraction([&](auto, auto n, auto t) { return -load * t * n; }, applied_traction_surface);
  solid_solver->setFixedBCs(applied_displacement_surface);

  auto K = smith::StateManager::newState(DensitySpace{}, kname, mesh_tag);
  K.setFromFieldFunction([=](smith::tensor<double, dim> x) {
    double scaling = ((x[0] < 3) && (x[0] > 2)) ? 0.99 : 0.001;
    return scaling * mat.K0;
  });

  auto G = smith::StateManager::newState(DensitySpace{}, gname, mesh_tag);
  G.setFromFieldFunction([=](smith::tensor<double, dim> x) {
    double scaling = ((x[0] <= 3) && (x[0] >= 2)) ? 0.99 : 0.001;
    return scaling * mat.G0;
  });

  solid_solver->setParameter(0, K);
  solid_solver->setParameter(1, G);

  solid_solver->completeSetup();

  auto [qoi_base, parameter_sensitivities] = computeSolidMechanicsQoiParameterSensitivities(*solid_solver, tsInfo);

  for (size_t ip = 0; ip < solid_solver->parameterNames().size(); ++ip) {
    auto param_name = solid_solver->parameterNames()[ip];
    auto param_field = solid_solver->parameter(ip);

    solid_solver->resetStates();
    FiniteElementState derivative_direction(param_field.space(), "derivative_direction");
    fillDirection(derivative_direction);

    double qoi_plus = computeSolidMechanicsQoiAdjustingParam(*solid_solver, tsInfo, derivative_direction, ip, eps);

    double directional_deriv = innerProduct(derivative_direction, parameter_sensitivities[ip]);
    EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, std::pow(300, ip + 1) * eps);
  }
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
