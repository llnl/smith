// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/state/state_manager.hpp"

namespace smith {

namespace {

struct AxisymmetricNeoHookean {
  using State = Empty;

  template <typename T>
  SMITH_HOST_DEVICE auto operator()(State& state, const tensor<T, 3, 3>& du_dX) const
  {
    return material(state, du_dX);
  }

  solid_mechanics::NeoHookean material{.density = 1.0, .K = 3.0, .G = 2.0};
  double density = 1.0;
};

std::shared_ptr<Mesh> makeAxisymmetricMesh()
{
  constexpr int nr = 2;
  constexpr int nz = 2;
  constexpr double r0 = 1.0;
  constexpr double z0 = 0.0;
  constexpr double length = 1.0;
  auto serial_mesh = mfem::Mesh::MakeCartesian2D(nr, nz, mfem::Element::QUADRILATERAL, true, length, length);

  for (int i = 0; i < serial_mesh.GetNV(); ++i) {
    auto* vertex = serial_mesh.GetVertex(i);
    vertex[0] += r0;
    vertex[1] += z0;
  }

  return std::make_shared<Mesh>(std::move(serial_mesh), "axisymmetric_mesh");
}

void expectJacobianSymmetric(mfem_ext::StdFunctionOperator& op)
{
  mfem::Vector u(op.Width());
  mfem::Vector v(op.Width());
  mfem::Vector w(op.Width());
  mfem::Vector Jv(op.Height());
  mfem::Vector Jw(op.Height());

  for (int i = 0; i < u.Size(); ++i) {
    u[i] = 0.01 * std::sin(0.2 * (i + 1));
    v[i] = std::sin(0.7 * (i + 1));
    w[i] = std::cos(0.5 * (i + 1));
  }

  auto& J = op.GetGradient(u);
  J.Mult(v, Jv);
  J.Mult(w, Jw);

  const double lhs = mfem::InnerProduct(v, Jw);
  const double rhs = mfem::InnerProduct(w, Jv);
  EXPECT_NEAR(lhs, rhs, 1.0e-10);
}

void expectAxisymmetricJacobianSymmetric(bool use_material, bool use_body_force, bool use_traction, bool use_pressure)
{
  MPI_Barrier(MPI_COMM_WORLD);

  auto mesh = makeAxisymmetricMesh();

  NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                           .relative_tol = 0.0,
                                           .absolute_tol = 1.0e-12,
                                           .max_iterations = 1,
                                           .print_level = 0};
  LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU};
  auto solver = std::make_unique<EquationSolver>(nonlinear_options, linear_options, mesh->getComm());

  constexpr int order = 1;
  constexpr int dim = 2;
  auto solid = std::make_unique<SolidMechanics<order, dim>>(
      std::move(solver), solid_mechanics::default_quasistatic_options, "axisymmetric_solid", mesh);

  if (use_material) {
    solid->setAxisymmetricMaterial(AxisymmetricNeoHookean{}, mesh->entireBody());
  }

  if (use_body_force) {
    solid->addAxisymmetricBodyForce([](auto, double) { return tensor<double, dim>{{0.25, -0.125}}; },
                                    mesh->entireBody());
  }

  if (use_traction) {
    solid->setAxisymmetricTraction([](auto, auto, double) { return tensor<double, dim>{{0.1, 0.05}}; },
                                   mesh->entireBoundary());
  }

  if (use_pressure) {
    solid->setAxisymmetricPressure([](auto, double) { return 0.1; }, mesh->entireBoundary());
  }

  solid->completeSetup();

  auto displacement = [](tensor<double, dim> X) {
    return tensor<double, dim>{{0.02 * X[0] + 0.01 * X[1], -0.015 * X[0] + 0.005 * X[1]}};
  };
  solid->setDisplacement(displacement);

  auto op = solid->buildQuasistaticOperator();
  expectJacobianSymmetric(*op);
}

}  // namespace

TEST(SolidMechanics, AxisymmetricMaterialJacobianSymmetry)
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "axisymmetric_material_symmetry");

  expectAxisymmetricJacobianSymmetric(true, false, false, false);
}

TEST(SolidMechanics, AxisymmetricLoadJacobianSymmetry)
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "axisymmetric_load_symmetry");

  expectAxisymmetricJacobianSymmetric(true, true, true, true);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
