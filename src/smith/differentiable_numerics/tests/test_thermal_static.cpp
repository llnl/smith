// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <gtest/gtest.h>
#include "smith/differentiable_numerics/thermal_system.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/coupled_system_solver.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "axom/slic.hpp"
#include "axom/fmt.hpp"
#include "axom/sidre.hpp"

using namespace smith;

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
    auto nonlinear_block_solver = buildNonlinearBlockSolver(solver_options, linear_options, *mesh);

    auto coupled_solver = std::make_shared<CoupledSystemSolver>(nonlinear_block_solver);
    auto thermal_system = buildThermalSystem<2, temp_order>(mesh, coupled_solver);

    double k = 1.0;
    thermal_system.setThermalIntegrand("entire_body", [=](auto /*t_info*/, auto /*X*/, auto T) {
      auto gradT = smith::get<smith::DERIVATIVE>(T);
      return smith::tuple{smith::zero{}, k * gradT};
    });

    thermal_system.addBodyHeatSource("entire_body", [=](auto /*t*/, auto X, auto /*T*/) {
      auto x = X[0];
      auto y = X[1];
      double pi = 3.14159265358979323846;
      return 2.0 * k * pi * pi * sin(pi * x) * sin(pi * y);
    });

    thermal_system.temperature_bc->template setScalarBCs<2>(mesh->entireBoundary(),
                                                            [](double /*t*/, tensor<double, 2> /*X*/) { return 0.0; });

    TimeInfo t_info(0.0, 1.0);
    auto [new_states, reactions] = thermal_system.advancer->advanceState(
        t_info, thermal_system.field_store->getShapeDisp(), thermal_system.field_store->getAllFields(),
        thermal_system.getParameterFields());

    for (size_t i = 0; i < new_states.size(); ++i) {
      thermal_system.field_store->setField(i, new_states[i]);
    }

    auto temperature = thermal_system.field_store->getField(thermal_system.prefix("temperature"));

    auto exact_sol_func = [](const mfem::Vector& X, mfem::Vector& T) {
      double x = X(0);
      double y = X(1);
      double pi = 3.14159265358979323846;
      T(0) = std::sin(pi * x) * std::sin(pi * y);
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

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  ApplicationManager app(argc, argv);
  return RUN_ALL_TESTS();
}
