// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "axom/slic.hpp"
#include "mfem.hpp"

#include "smith/infrastructure/accelerator.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/differentiable_operator_weak_form.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/tests/physics_test_utils.hpp"

namespace {

struct DifferentiableOperatorWeakFormFixture : public testing::Test {
  static constexpr int DIM = 2;
  static constexpr int ORDER = 1;

  using VectorSpace = smith::H1<ORDER, DIM>;
  using DensitySpace = smith::L2<0>;
  using WeakFormType =
      smith::DifferentiableOperatorWeakForm<DIM, VectorSpace,
                                            smith::Parameters<VectorSpace, VectorSpace, DensitySpace>>;

  void SetUp() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
    smith::StateManager::initialize(datastore, "differentiable_operator_weak_form");
    mesh = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian2D(4, 5, mfem::Element::QUADRILATERAL, true, 0.5, 2.0), "dfem_weak_form_mesh");

    states.emplace_back(smith::StateManager::newState(VectorSpace{}, "displacement", mesh->tag()));
    states.emplace_back(smith::StateManager::newState(VectorSpace{}, "velocity", mesh->tag()));
    parameters.emplace_back(smith::StateManager::newState(DensitySpace{}, "density", mesh->tag()));
    shape = std::make_unique<smith::FiniteElementState>(mesh->newShapeDisplacement());

    states[0].setFromFieldFunction([](smith::tensor<double, DIM> x) { return 0.1 * x; });
    states[1].setFromFieldFunction([](smith::tensor<double, DIM> x) { return -0.02 * x; });
    parameters[0] = 1.2;

    state_tangents = states;
    parameter_tangents = parameters;
    for (auto& tangent : state_tangents) pseudoRand(tangent);
    for (auto& tangent : parameter_tangents) pseudoRand(tangent);
    shape_tangent = std::make_unique<smith::FiniteElementState>(*shape);
    pseudoRand(*shape_tangent);

    state_sensitivities.emplace_back(states[0].space(), "displacement_sensitivity");
    state_sensitivities.emplace_back(states[1].space(), "velocity_sensitivity");
    parameter_sensitivities.emplace_back(parameters[0].space(), "density_sensitivity");
    shape_sensitivity = std::make_unique<smith::FiniteElementDual>(mesh->newShapeDisplacementDual());

    typename WeakFormType::SpacesT spaces{&states[0].space(), &states[1].space(), &parameters[0].space()};
    concrete_weak_form =
        std::make_shared<WeakFormType>("dfem_test", mesh, states[0].space(), std::move(spaces));
    concrete_weak_form->addBodyIntegral(
        smith::DependsOn<0>{}, mesh->entireBodyName(),
        [] SMITH_HOST_DEVICE(const smith::TimeInfo&, auto, auto displacement) {
          return smith::tuple{-smith::get<smith::VALUE>(displacement),
                              smith::get<smith::DERIVATIVE>(displacement)};
        });
    concrete_weak_form->addBodySource(
        smith::DependsOn<>{}, mesh->entireBodyName(),
        [] SMITH_HOST_DEVICE(const smith::TimeInfo&, auto x) { return 0.5 * x; });
    weak_form = concrete_weak_form;
  }

  std::vector<smith::ConstFieldPtr> inputFields() const
  {
    return smith::getConstFieldPointers(states, parameters);
  }

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
  std::shared_ptr<WeakFormType> concrete_weak_form;
  std::shared_ptr<smith::WeakForm> weak_form;
  std::unique_ptr<smith::FiniteElementState> shape;
  std::unique_ptr<smith::FiniteElementState> shape_tangent;
  std::unique_ptr<smith::FiniteElementDual> shape_sensitivity;
  std::vector<smith::FiniteElementState> states;
  std::vector<smith::FiniteElementState> parameters;
  std::vector<smith::FiniteElementState> state_tangents;
  std::vector<smith::FiniteElementState> parameter_tangents;
  std::vector<smith::FiniteElementDual> state_sensitivities;
  std::vector<smith::FiniteElementDual> parameter_sensitivities;
  smith::TimeInfo time_info{0.3, 0.1, 4};
};

TEST_F(DifferentiableOperatorWeakFormFixture, JvpConsistency)
{
  const auto fields = inputFields();
  const auto tangents = smith::getConstFieldPointers(state_tangents, parameter_tangents);
  smith::FiniteElementDual expected(states[0].space(), "expected_jvp");
  smith::FiniteElementDual actual(states[0].space(), "actual_jvp");

  for (size_t i = 0; i < fields.size(); ++i) {
    std::vector<double> weights(fields.size(), 0.0);
    weights[i] = 1.0;
    auto jacobian = weak_form->jacobian(time_info, shape.get(), fields, weights);
    ASSERT_NE(jacobian, nullptr);
    jacobian->Mult(*tangents[i], expected);

    auto selected_tangents = tangents;
    for (size_t j = 0; j < selected_tangents.size(); ++j) {
      if (j != i) selected_tangents[j] = nullptr;
    }
    actual = 7.0;
    weak_form->jvp(time_info, shape.get(), fields, {}, nullptr, selected_tangents, {}, &actual);
    expected -= actual;
    EXPECT_NEAR(expected.Norml2(), 0.0, 1.0e-11);
  }
}

TEST_F(DifferentiableOperatorWeakFormFixture, VjpConsistency)
{
  const auto fields = inputFields();
  smith::FiniteElementState seed(states[0].space(), "seed");
  pseudoRand(seed);
  auto sensitivities = smith::getFieldPointers(state_sensitivities, parameter_sensitivities);
  for (auto* sensitivity : sensitivities) *sensitivity = 0.0;

  weak_form->vjp(time_info, shape.get(), fields, {}, &seed, shape_sensitivity.get(), sensitivities, {});
  for (size_t i = 0; i < fields.size(); ++i) {
    std::vector<double> weights(fields.size(), 0.0);
    weights[i] = 1.0;
    auto jacobian = weak_form->jacobian(time_info, shape.get(), fields, weights);
    smith::FiniteElementDual expected(*sensitivities[i]);
    expected = 0.0;
    jacobian->MultTranspose(seed, expected);
    expected -= *sensitivities[i];
    EXPECT_NEAR(expected.Norml2(), 0.0, 1.0e-11);
  }
}

TEST_F(DifferentiableOperatorWeakFormFixture, ShapeJvpMatchesFiniteDifferenceAndVjp)
{
  const auto fields = inputFields();
  std::vector<smith::ConstFieldPtr> zero_field_tangents(fields.size(), nullptr);
  smith::FiniteElementDual shape_jvp(states[0].space(), "shape_jvp");
  weak_form->jvp(time_info, shape.get(), fields, {}, shape_tangent.get(), zero_field_tangents, {}, &shape_jvp);

  constexpr double epsilon = 1.0e-6;
  smith::FiniteElementState shape_plus(*shape);
  smith::FiniteElementState shape_minus(*shape);
  shape_plus.Add(epsilon, *shape_tangent);
  shape_minus.Add(-epsilon, *shape_tangent);
  mfem::Vector finite_difference = weak_form->residual(time_info, &shape_plus, fields);
  finite_difference -= weak_form->residual(time_info, &shape_minus, fields);
  finite_difference /= 2.0 * epsilon;
  finite_difference -= shape_jvp;
  EXPECT_NEAR(finite_difference.Norml2(), 0.0, 2.0e-7);

  smith::FiniteElementState seed(states[0].space(), "shape_seed");
  pseudoRand(seed);
  *shape_sensitivity = 0.0;
  std::vector<smith::DualFieldPtr> null_sensitivities(fields.size(), nullptr);
  weak_form->vjp(time_info, shape.get(), fields, {}, &seed, shape_sensitivity.get(), null_sensitivities, {});
  const double lhs = mfem::InnerProduct(mesh->getComm(), seed, shape_jvp);
  const double rhs = mfem::InnerProduct(mesh->getComm(), *shape_tangent, *shape_sensitivity);
  EXPECT_NEAR(lhs, rhs, 1.0e-10 * std::max({1.0, std::abs(lhs), std::abs(rhs)}));
}

TEST_F(DifferentiableOperatorWeakFormFixture, ForwardsOriginalTimeInfoToIntegrands)
{
  typename WeakFormType::SpacesT spaces{&states[0].space(), &states[1].space(), &parameters[0].space()};
  WeakFormType time_weak_form("time", mesh, states[0].space(), spaces);
  time_weak_form.addBodySource(
      smith::DependsOn<>{}, mesh->entireBodyName(),
      [] SMITH_HOST_DEVICE(const smith::TimeInfo& info, auto x) {
        const double mode = info.isCycleZeroEvaluation() ? 10.0 : 0.0;
        return (info.time() + info.dt() + static_cast<double>(info.cycle()) + mode) * x;
      });
  const auto fields = inputFields();
  const smith::TimeInfo regular(2.0, 0.25, 7);
  const smith::TimeInfo cycle_zero(2.0, 0.25, 7, smith::TimeInfo::EvaluationMode::CycleZero);
  const mfem::Vector regular_residual = time_weak_form.residual(regular, shape.get(), fields);
  const mfem::Vector cycle_zero_residual = time_weak_form.residual(cycle_zero, shape.get(), fields);
  EXPECT_GT(regular_residual.Norml2(), 0.0);
  EXPECT_NEAR(cycle_zero_residual.Norml2() / regular_residual.Norml2(), 19.5 / 9.5, 1.0e-12);
}

TEST_F(DifferentiableOperatorWeakFormFixture, RejectsIncorrectJacobianWeightCount)
{
  axom::slic::ScopedAbortToThrow abort_to_throw;
  EXPECT_THROW(weak_form->jacobian(time_info, shape.get(), inputFields(), {1.0}), axom::slic::SlicAbortException);
}

TEST_F(DifferentiableOperatorWeakFormFixture, RejectsQuadratureFields)
{
  axom::slic::ScopedAbortToThrow abort_to_throw;
  EXPECT_THROW(weak_form->residual(time_info, shape.get(), inputFields(), {nullptr}),
               axom::slic::SlicAbortException);
}

TEST_F(DifferentiableOperatorWeakFormFixture, RejectsArbitraryElementSubsets)
{
  mesh->addDomainOfBodyElements("partial_attribute", [](std::vector<smith::vec2> nodes, int) {
    smith::vec2 center{};
    for (const auto& node : nodes) center += node;
    center = (1.0 / static_cast<double>(nodes.size())) * center;
    return center[0] < 0.25;
  });

  axom::slic::ScopedAbortToThrow abort_to_throw;
  EXPECT_THROW(concrete_weak_form->addBodySource(
                   smith::DependsOn<>{}, "partial_attribute",
                   [] SMITH_HOST_DEVICE(const smith::TimeInfo&, auto x) { return x; }),
               axom::slic::SlicAbortException);
}

TEST_F(DifferentiableOperatorWeakFormFixture, HonorsExplicitIntegrationOrder)
{
  const auto fields = inputFields();
  typename WeakFormType::SpacesT spaces{&states[0].space(), &states[1].space(), &parameters[0].space()};
  WeakFormType default_order("default_order", mesh, states[0].space(), spaces);
  WeakFormType low_order("low_order", mesh, states[0].space(), spaces, 0);
  const auto source = [] SMITH_HOST_DEVICE(const smith::TimeInfo&, auto x) {
    return x[0] * x[0] * x;
  };
  default_order.addBodySource(smith::DependsOn<>{}, mesh->entireBodyName(), source);
  low_order.addBodySource(smith::DependsOn<>{}, mesh->entireBodyName(), source);

  mfem::Vector difference = default_order.residual(time_info, shape.get(), fields);
  difference -= low_order.residual(time_info, shape.get(), fields);
  EXPECT_GT(difference.Norml2(), 1.0e-10);
}

TEST_F(DifferentiableOperatorWeakFormFixture, SupportsL2InputsAndOutputs)
{
  using ScalarWeakForm =
      smith::DifferentiableOperatorWeakForm<DIM, DensitySpace, smith::Parameters<DensitySpace>>;
  typename ScalarWeakForm::SpacesT spaces{&parameters[0].space()};
  ScalarWeakForm scalar_weak_form("l2", mesh, parameters[0].space(), spaces);
  scalar_weak_form.addBodySource(
      mesh->entireBodyName(),
      [] SMITH_HOST_DEVICE(const smith::TimeInfo&, auto, auto density) { return density; });

  const std::vector<smith::ConstFieldPtr> fields{&parameters[0]};
  const mfem::Vector residual = scalar_weak_form.residual(time_info, shape.get(), fields);
  EXPECT_GT(residual.Norml2(), 0.0);
  {
    axom::slic::ScopedAbortToThrow abort_to_throw;
    EXPECT_THROW(scalar_weak_form.jacobian(time_info, shape.get(), fields, {1.0}),
                 axom::slic::SlicAbortException);
  }

  smith::FiniteElementDual jvp_result(parameters[0].space(), "l2_jvp");
  const std::vector<smith::ConstFieldPtr> tangents{&parameter_tangents[0]};
  scalar_weak_form.jvp(time_info, shape.get(), fields, {}, nullptr, tangents, {}, &jvp_result);

  constexpr double epsilon = 1.0e-6;
  smith::FiniteElementState plus(parameters[0]);
  smith::FiniteElementState minus(parameters[0]);
  plus.Add(epsilon, parameter_tangents[0]);
  minus.Add(-epsilon, parameter_tangents[0]);
  mfem::Vector finite_difference = scalar_weak_form.residual(time_info, shape.get(), {&plus});
  finite_difference -= scalar_weak_form.residual(time_info, shape.get(), {&minus});
  finite_difference /= 2.0 * epsilon;
  finite_difference -= jvp_result;
  EXPECT_NEAR(finite_difference.Norml2(), 0.0, 1.0e-9);

  smith::FiniteElementState seed(parameters[0].space(), "l2_seed");
  smith::FiniteElementDual sensitivity(parameters[0].space(), "l2_sensitivity");
  pseudoRand(seed);
  sensitivity = 0.0;
  scalar_weak_form.vjp(time_info, shape.get(), fields, {}, &seed, nullptr, {&sensitivity}, {});
  const double lhs = mfem::InnerProduct(mesh->getComm(), seed, jvp_result);
  const double rhs = mfem::InnerProduct(mesh->getComm(), parameter_tangents[0], sensitivity);
  EXPECT_NEAR(lhs, rhs, 1.0e-10 * std::max({1.0, std::abs(lhs), std::abs(rhs)}));
}

}  // namespace

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager application_manager(argc, argv);
  return RUN_ALL_TESTS();
}
