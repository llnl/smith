// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include "smith/differentiable_numerics/thermal_system.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/infrastructure/application_manager.hpp"

using namespace smith;

namespace {

bool allNodesAtX(const std::vector<vec2>& nodes, double x)
{
  constexpr double tol = 1.0e-12;
  return std::all_of(nodes.begin(), nodes.end(), [=](const vec2& node) { return std::abs(node[0] - x) < tol; });
}

}  // namespace

struct ThermalStaticFixture : public testing::Test {
  axom::sidre::DataStore datastore;
  std::shared_ptr<Mesh> mesh;

  void setupMesh(int n_elements, std::string tag)
  {
    StateManager::reset();
    StateManager::initialize(datastore, "thermal_static_" + tag);
    auto mfem_mesh = mfem::Mesh::MakeCartesian2D(n_elements, n_elements, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    mesh = std::make_shared<Mesh>(std::move(mfem_mesh), "thermal_mesh_" + tag);
  }

  void SetUp() override
  {
    // Default mesh for single-mesh tests
    setupMesh(32, "default");
  }

  void TearDown() override { StateManager::reset(); }

  template <int temp_order>
  double run_thermal_solve()
  {
    auto solver_options = NonlinearSolverOptions();
    solver_options.relative_tol = 1e-12;
    auto linear_options = LinearSolverOptions();

    auto thermal_system = buildThermalSystem<2, temp_order>(solver_options, linear_options, ThermalOptions{}, mesh);

    double k = 1.0;
    // Material returns {heat_capacity, heat_flux} consistent with heat_transfer.hpp convention.
    // heat_flux is the physical flux (Fourier's law): q = -k * grad(T).
    // The system negates it to form the weak form integral(k * grad(T) . grad(v)).
    thermal_system->setMaterial([=](const TimeInfo& /*t_info*/, auto /*temperature*/,
                                    auto grad_temperature) { return smith::tuple{0.0, -k * grad_temperature}; },
                                "entire_body");

    thermal_system->addHeatSource("entire_body", [=](auto /*t*/, auto X, auto /*T*/, auto... /*args*/) {
      auto x = X[0];
      auto y = X[1];
      return 2.0 * k * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
    });

    thermal_system->setTemperatureBC(mesh->entireBoundary(), [](double /*t*/, tensor<double, 2> /*X*/) { return 0.0; });

    TimeInfo t_info(0.0, 1.0);
    auto [new_states, reactions] = makeAdvancer(thermal_system)
                                       ->advanceState(t_info, thermal_system->field_store->getShapeDisp(),
                                                      thermal_system->field_store->getAllFields(),
                                                      thermal_system->field_store->getParameterFields());

    for (size_t i = 0; i < new_states.size(); ++i) {
      thermal_system->field_store->setField(i, new_states[i]);
    }

    auto temperature = thermal_system->field_store->getField("temperature");

    auto exact_sol_func = [](const mfem::Vector& X, mfem::Vector& T) {
      double x = X(0);
      double y = X(1);
      T(0) = std::sin(M_PI * x) * std::sin(M_PI * y);
    };
    mfem::VectorFunctionCoefficient exact_sol_coeff(1, exact_sol_func);

    return computeL2Error(*temperature.get(), exact_sol_coeff);
  }
};

TEST_F(ThermalStaticFixture, ManufacturedSolutionOrder1)
{
  double error = run_thermal_solve<1>();
  SLIC_INFO("L2 Error (Order 1, h=1/32): " << error);
  EXPECT_LT(error, 1e-3);
}

TEST_F(ThermalStaticFixture, ManufacturedSolutionOrder2)
{
  double error = run_thermal_solve<2>();
  SLIC_INFO("L2 Error (Order 2, h=1/32): " << error);
  EXPECT_LT(error, 1e-5);
}

TEST_F(ThermalStaticFixture, ConvergenceOrder1)
{
  setupMesh(16, "conv1_h16");
  double e1 = run_thermal_solve<1>();
  setupMesh(32, "conv1_h32");
  double e2 = run_thermal_solve<1>();

  double rate = std::log2(e1 / e2);
  SLIC_INFO("Convergence Rate (Order 1): " << rate);
  EXPECT_NEAR(rate, 2.0, 0.1);
}

TEST_F(ThermalStaticFixture, ConvergenceOrder2)
{
  setupMesh(8, "conv2_h8");
  double e1 = run_thermal_solve<2>();
  setupMesh(16, "conv2_h16");
  double e2 = run_thermal_solve<2>();

  double rate = std::log2(e1 / e2);
  SLIC_INFO("Convergence Rate (Order 2): " << rate);
  EXPECT_NEAR(rate, 3.0, 0.1);
}

TEST_F(ThermalStaticFixture, HeatSourceWithDependsOn)
{
  // Test that DependsOn works: use a parameter field for conductivity
  auto solver_options = NonlinearSolverOptions();
  solver_options.relative_tol = 1e-12;
  auto linear_options = LinearSolverOptions();

  auto thermal_system = buildThermalSystem<2, 1>(solver_options, linear_options, ThermalOptions{}, mesh,
                                                 FieldType<L2<0>>("conductivity"));

  // Set the conductivity parameter field to k=1.0
  thermal_system->field_store->getParameterFields()[0].get()->setFromFieldFunction(
      [](tensor<double, 2>) { return 1.0; });

  // Material uses the parameter field for conductivity
  thermal_system->setMaterial(
      [](const TimeInfo& /*t_info*/, auto /*temperature*/, auto grad_temperature, auto k_param) {
        auto k = get<0>(k_param);
        return smith::tuple{0.0, -k * grad_temperature};
      },
      "entire_body");

  // DependsOn now indexes only trailing coupling/parameter args, so empty means "state-only".
  thermal_system->addHeatSource("entire_body", [=](auto /*t*/, auto X, auto /*T*/, auto... /*args*/) {
    auto x = X[0];
    auto y = X[1];
    return 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
  });

  thermal_system->setTemperatureBC(mesh->entireBoundary(), [](double /*t*/, tensor<double, 2> /*X*/) { return 0.0; });

  TimeInfo t_info(0.0, 1.0);
  auto [new_states, reactions] = makeAdvancer(thermal_system)
                                     ->advanceState(t_info, thermal_system->field_store->getShapeDisp(),
                                                    thermal_system->field_store->getAllFields(),
                                                    thermal_system->field_store->getParameterFields());

  for (size_t i = 0; i < new_states.size(); ++i) {
    thermal_system->field_store->setField(i, new_states[i]);
  }

  auto temperature = thermal_system->field_store->getField("temperature");

  auto exact_sol_func = [](const mfem::Vector& X, mfem::Vector& T) {
    double x = X(0);
    double y = X(1);
    T(0) = std::sin(M_PI * x) * std::sin(M_PI * y);
  };
  mfem::VectorFunctionCoefficient exact_sol_coeff(1, exact_sol_func);

  double error = computeL2Error(*temperature.get(), exact_sol_coeff);
  SLIC_INFO("L2 Error (DependsOn test): " << error);
  EXPECT_LT(error, 1e-3);
}

TEST_F(ThermalStaticFixture, PrescribedTemperatureParameterField)
{
  mesh->addDomainOfBoundaryElements("left", [](std::vector<vec2> nodes, int) { return allNodesAtX(nodes, 0.0); });
  mesh->addDomainOfBoundaryElements("right", [](std::vector<vec2> nodes, int) { return allNodesAtX(nodes, 1.0); });

  NonlinearSolverOptions solver_options;
  solver_options.relative_tol = 1.0e-12;
  LinearSolverOptions linear_options;

  auto thermal_system = buildThermalSystem<2, 1>(solver_options, linear_options, ThermalOptions{}, mesh,
                                                 FieldType<H1<1>>("prescribed_temperature"));

  double k = 1.0;
  thermal_system->setMaterial([=](const TimeInfo& /*t_info*/, auto /*temperature*/, auto grad_temperature,
                                  auto /*prescribed_temperature*/) { return smith::tuple{0.0, -k * grad_temperature}; },
                              "entire_body");

  FieldState prescribed_temperature = thermal_system->field_store->getParameterFields()[0];
  mfem::FunctionCoefficient prescribed_temperature_coefficient([](const mfem::Vector& X) {
    const double interior_perturbation = X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1]);
    return X[0] + interior_perturbation;
  });
  prescribed_temperature.get()->project(prescribed_temperature_coefficient, mesh->domain("right"));

  thermal_system->temperature_bc->setScalarBCs<2>(mesh->domain("left"), [](double, tensor<double, 2>) { return 0.0; });
  thermal_system->temperature_bc->registerPrescribedField(prescribed_temperature);
  thermal_system->temperature_bc->setScalarBCsFromPrescribedField(mesh->domain("right"));

  const auto& second_derivative_bcs = thermal_system->temperature_bc->getSecondDerivativeManager();
  mfem::Vector second_derivative(prescribed_temperature.get()->space().GetTrueVSize());
  second_derivative = 1.0;
  for (const auto& bc : second_derivative_bcs.essentials()) {
    bc.setDofs(second_derivative);
  }
  for (int dof : second_derivative_bcs.allEssentialTrueDofs()) {
    EXPECT_DOUBLE_EQ(second_derivative[dof], 0.0);
  }

  TimeInfo t_info(0.0, 1.0);
  auto [new_states, reactions] = makeAdvancer(thermal_system)
                                     ->advanceState(t_info, thermal_system->field_store->getShapeDisp(),
                                                    thermal_system->field_store->getAllFields(),
                                                    thermal_system->field_store->getParameterFields());

  const size_t temperature_index = thermal_system->field_store->getFieldIndex("temperature");
  auto temperature = new_states[temperature_index];
  mfem::FunctionCoefficient exact_temperature([](const mfem::Vector& X) { return X[0]; });

  EXPECT_LT(computeL2Error(*temperature.get(), exact_temperature), 1.0e-10);
}

TEST_F(ThermalStaticFixture, MultiplePrescribedBoundaryDomains)
{
  mesh->addDomainOfBoundaryElements("left", [](std::vector<vec2> nodes, int) { return allNodesAtX(nodes, 0.0); });
  mesh->addDomainOfBoundaryElements("right", [](std::vector<vec2> nodes, int) { return allNodesAtX(nodes, 1.0); });

  NonlinearSolverOptions solver_options;
  solver_options.relative_tol = 1.0e-12;
  LinearSolverOptions linear_options;
  auto thermal_system = buildThermalSystem<2, 1>(solver_options, linear_options, ThermalOptions{}, mesh,
                                                 FieldType<H1<1>>("prescribed_temperature"));

  thermal_system->setMaterial(
      [](const TimeInfo&, auto, auto grad_temperature, auto) { return smith::tuple{0.0, -grad_temperature}; },
      "entire_body");

  const auto& prescribed_fields = thermal_system->field_store->getParameterFields();
  mfem::FunctionCoefficient left_temperature([](const mfem::Vector&) { return -1.0; });
  mfem::FunctionCoefficient right_temperature([](const mfem::Vector&) { return 2.0; });
  prescribed_fields[0].get()->project(left_temperature, mesh->domain("left"));
  prescribed_fields[0].get()->project(right_temperature, mesh->domain("right"));

  thermal_system->temperature_bc->registerPrescribedField(prescribed_fields[0]);
  thermal_system->temperature_bc->setScalarBCsFromPrescribedField(mesh->domain("left"));
  thermal_system->temperature_bc->setScalarBCsFromPrescribedField(mesh->domain("right"));

  TimeInfo t_info(0.0, 1.0);
  auto [new_states, reactions] = makeAdvancer(thermal_system)
                                     ->advanceState(t_info, thermal_system->field_store->getShapeDisp(),
                                                    thermal_system->field_store->getAllFields(), prescribed_fields);

  auto temperature = new_states[thermal_system->field_store->getFieldIndex("temperature")];
  mfem::FunctionCoefficient exact_temperature([](const mfem::Vector& X) { return -1.0 + 3.0 * X[0]; });
  EXPECT_LT(computeL2Error(*temperature.get(), exact_temperature), 1.0e-10);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  ApplicationManager app(argc, argv);
  return RUN_ALL_TESTS();
}
