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
#include "smith/physics/materials/parameterized_solid_material.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/state/state_manager.hpp"

namespace smith {

namespace {

constexpr double two_pi = 2.0 * M_PI;

struct HoopStressMaterial {
  using State = Empty;

  double density = 0.0;

  template <typename T, int dim>
  SMITH_HOST_DEVICE auto operator()(State&, const tensor<T, dim, dim>& du_dX) const
  {
    auto zero = 0.0 * du_dX[0][0];
    return make_tensor<dim, dim>([&](int i, int j) {
      if (i == 2 && j == 2) {
        return 1.0 + zero;
      }
      return zero;
    });
  }
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

std::unique_ptr<SolidMechanics<1, 2>> makeAxisymmetricSolidMechanics(std::shared_ptr<Mesh> mesh)
{
  NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                           .relative_tol = 0.0,
                                           .absolute_tol = 1.0e-12,
                                           .max_iterations = 1,
                                           .print_level = 0};
  LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU};
  auto solver = std::make_unique<EquationSolver>(nonlinear_options, linear_options, mesh->getComm());

  return std::make_unique<SolidMechanics<1, 2>>(std::move(solver), solid_mechanics::default_quasistatic_options,
                                                "axisymmetric_solid", mesh);
}

void expectAxisymmetricJacobianSymmetric(bool use_material, bool use_body_force, bool use_traction, bool use_pressure)
{
  MPI_Barrier(MPI_COMM_WORLD);

  auto mesh = makeAxisymmetricMesh();

  constexpr int dim = 2;
  auto solid = makeAxisymmetricSolidMechanics(mesh);

  if (use_material) {
    solid->setAxisymmetricMaterial(solid_mechanics::NeoHookean{.density = 1.0, .K = 3.0, .G = 2.0}, mesh->entireBody());
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

template <typename LoadFunction>
std::pair<std::shared_ptr<Mesh>, mfem::Vector> residualForLoad(LoadFunction add_load)
{
  MPI_Barrier(MPI_COMM_WORLD);

  auto mesh = makeAxisymmetricMesh();
  auto solid = makeAxisymmetricSolidMechanics(mesh);
  add_load(*solid, *mesh);
  solid->completeSetup();

  auto op = solid->buildQuasistaticOperator();
  mfem::Vector u(op->Width());
  mfem::Vector r(op->Height());
  u = 0.0;
  op->Mult(u, r);
  return {mesh, r};
}

void expectResultant(const std::pair<std::shared_ptr<Mesh>, mfem::Vector>& load_residual, tensor<double, 2> expected)
{
  FiniteElementState virtual_displacement(load_residual.first->mfemParMesh(), smith::H1<1, 2>{},
                                          "virtual_displacement");

  for (int component = 0; component < 2; ++component) {
    virtual_displacement.setFromFieldFunction([component](tensor<double, 2> X) {
      auto value = 0.0 * X;
      value[component] = 1.0;
      return value;
    });
    EXPECT_NEAR(mfem::InnerProduct(load_residual.second, virtual_displacement), expected[component], 1.0e-12);
  }
}

void addBoundaryDomains(Mesh& mesh)
{
  mesh.addDomainOfBoundaryElements("inner",
                                   [](std::vector<vec2> vertices, int) { return average(vertices)[0] < 1.01; });
  mesh.addDomainOfBoundaryElements("outer",
                                   [](std::vector<vec2> vertices, int) { return average(vertices)[0] > 1.99; });
  mesh.addDomainOfBoundaryElements("top", [](std::vector<vec2> vertices, int) { return average(vertices)[1] > 0.99; });
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

TEST(SolidMechanics, AxisymmetricHoopStressResultant)
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "axisymmetric_hoop_resultant");

  auto residual = residualForLoad(
      [](auto& solid, auto& mesh) { solid.setAxisymmetricMaterial(HoopStressMaterial{}, mesh.entireBody()); });

  expectResultant(residual, tensor<double, 2>{{two_pi, 0.0}});
}

TEST(SolidMechanics, AxisymmetricBodyForceResultant)
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "axisymmetric_body_resultant");

  constexpr tensor<double, 2> body_force{{0.25, -0.125}};
  auto residual = residualForLoad([body_force](auto& solid, auto& mesh) {
    solid.addAxisymmetricBodyForce([body_force](auto, double) { return body_force; }, mesh.entireBody());
  });

  constexpr double weighted_area = 1.5 * two_pi;
  expectResultant(residual, -weighted_area * body_force);
}

TEST(SolidMechanics, AxisymmetricTractionResultant)
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "axisymmetric_traction_resultant");

  constexpr tensor<double, 2> traction{{0.1, 0.05}};
  auto residual = residualForLoad([traction](auto& solid, auto& mesh) {
    addBoundaryDomains(mesh);
    solid.setAxisymmetricTraction([traction](auto, auto, double) { return traction; }, mesh.domain("top"));
  });

  constexpr double weighted_length = 1.5 * two_pi;
  expectResultant(residual, -weighted_length * traction);
}

TEST(SolidMechanics, AxisymmetricPressureOuterResultant)
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "axisymmetric_pressure_outer_resultant");

  constexpr double pressure = 0.1;
  auto residual = residualForLoad([pressure](auto& solid, auto& mesh) {
    addBoundaryDomains(mesh);
    solid.setAxisymmetricPressure([pressure](auto, double) { return pressure; }, mesh.domain("outer"));
  });

  constexpr double expected_radial = pressure * 2.0 * two_pi;
  expectResultant(residual, tensor<double, 2>{{expected_radial, 0.0}});
}

TEST(SolidMechanics, AxisymmetricPressureInnerResultant)
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "axisymmetric_pressure_inner_resultant");

  constexpr double pressure = 0.1;
  auto residual = residualForLoad([pressure](auto& solid, auto& mesh) {
    addBoundaryDomains(mesh);
    solid.setAxisymmetricPressure([pressure](auto, double) { return pressure; }, mesh.domain("inner"));
  });

  constexpr double expected_radial = -pressure * two_pi;
  expectResultant(residual, tensor<double, 2>{{expected_radial, 0.0}});
}

TEST(SolidMechanics, AxisymmetricActiveParameterOverloadCompiles)
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "axisymmetric_active_parameter_compile");

  auto mesh = makeAxisymmetricMesh();
  SolidMechanics<1, 2, Parameters<H1<1>, H1<1>>> solid(
      solid_mechanics::default_nonlinear_options, solid_mechanics::direct_linear_options,
      solid_mechanics::default_quasistatic_options, "axisymmetric_solid", mesh, {"bulk modulus", "shear modulus"});
  FiniteElementState bulk_modulus(mesh->mfemParMesh(), H1<1>{}, "bulk_modulus");
  FiniteElementState shear_modulus(mesh->mfemParMesh(), H1<1>{}, "shear_modulus");
  bulk_modulus = 0.0;
  shear_modulus = 0.0;
  solid.setParameter(0, bulk_modulus);
  solid.setParameter(1, shear_modulus);
  solid.setAxisymmetricMaterial(DependsOn<0, 1>{}, solid_mechanics::ParameterizedNeoHookeanSolid{1.0, 3.0, 2.0},
                                mesh->entireBody());
  solid.addAxisymmetricBodyForce(
      DependsOn<0>{}, [](auto, double, auto p) { return (1.0 + get<VALUE>(p)) * vec2{{0.01, 0.02}}; },
      mesh->entireBody());
  solid.setAxisymmetricTraction(
      DependsOn<1>{}, [](auto, auto, double, auto p) { return (1.0 + get<VALUE>(p)) * vec2{{0.03, 0.04}}; },
      mesh->entireBoundary());
  solid.setAxisymmetricPressure(
      DependsOn<0>{}, [](auto, double, auto p) { return 0.05 * (1.0 + get<VALUE>(p)); }, mesh->entireBoundary());
  solid.completeSetup();

  auto op = solid.buildQuasistaticOperator();
  expectJacobianSymmetric(*op);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
