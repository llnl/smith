// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

#include "mfem.hpp"

#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/weak_form_block_operator.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/functional_weak_form.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "gretl/data_store.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"

namespace smith {
namespace {

using ShapeDispSpace = H1<1, 2>;
using ScalarSpace = H1<1>;
using ScalarWeakForm = FunctionalWeakForm<2, ScalarSpace, Parameters<ScalarSpace>>;

class WeakFormBlockOperatorTest : public testing::Test {
 protected:
  void initialize(const std::string& physics_name)
  {
    StateManager::initialize(datastore, physics_name);

    auto serial_mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, 1, 1.0, 1.0);
    mesh = std::make_shared<Mesh>(std::move(serial_mesh), physics_name + "_mesh");
    graph = std::make_shared<gretl::DataStore>(std::make_unique<gretl::WangCheckpointStrategy>(100));
  }

  axom::sidre::DataStore datastore;
  std::shared_ptr<gretl::DataStore> graph;
  std::shared_ptr<Mesh> mesh;
};

void addMassIntegral(ScalarWeakForm& weak_form, const std::shared_ptr<Mesh>& mesh)
{
  weak_form.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(), [](auto /* time_info */, auto /* x */, auto U) {
    auto u = get<VALUE>(U);
    return tuple{u, zero{}};
  });
}

void addQuadraticMassIntegral(ScalarWeakForm& weak_form, const std::shared_ptr<Mesh>& mesh)
{
  weak_form.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(), [](auto /* time_info */, auto /* x */, auto U) {
    auto u = get<VALUE>(U);
    return tuple{u * u, zero{}};
  });
}

// Verifies the flat utility assembles an operator directly from weak-form inputs.
TEST_F(WeakFormBlockOperatorTest, BuildsOperatorFromWeakForm)
{
  const std::string physics_name = "weak_form_block_operator_build";
  initialize(physics_name);
  auto shape_disp = createFieldState(*graph, ShapeDispSpace{}, physics_name + "_shape_displacement", mesh->tag());
  auto field = createFieldState(*graph, ScalarSpace{}, physics_name + "_field", mesh->tag());
  ScalarWeakForm weak_form("mass", mesh, space(field), spaces({field}));
  addMassIntegral(weak_form, mesh);

  auto op = buildWeakFormOperator(weak_form, shape_disp, {field}, {1.0}, TimeInfo(0.0, 1.0));

  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->Height(), field.get()->Size());
  EXPECT_EQ(op->Width(), field.get()->Size());
}

// Verifies the public fixed provider factory hides the implementation builder.
TEST_F(WeakFormBlockOperatorTest, FixedOverrideProvidesWeakFormOperator)
{
  const std::string physics_name = "weak_form_block_operator_fixed";
  initialize(physics_name);
  auto shape_disp = createFieldState(*graph, ShapeDispSpace{}, physics_name + "_shape_displacement", mesh->tag());
  auto field = createFieldState(*graph, ScalarSpace{}, physics_name + "_field", mesh->tag());
  ScalarWeakForm weak_form("mass", mesh, space(field), spaces({field}));
  addMassIntegral(weak_form, mesh);

  auto override = makeWeakFormBlockProviderOverride(0, weak_form, shape_disp, {field}, {1.0}, TimeInfo(0.0, 1.0));
  const auto& op = override.provider->currentOperator();

  EXPECT_EQ(op.Height(), field.get()->Size());
  EXPECT_EQ(op.Width(), field.get()->Size());
}

// Verifies state-dependent provider overrides refresh scratch fields without mutating graph-owned fields.
TEST_F(WeakFormBlockOperatorTest, StateDependentOverrideUpdatesFromStateBlock)
{
  const std::string physics_name = "weak_form_block_operator_state_dependent";
  initialize(physics_name);
  auto shape_disp = createFieldState(*graph, ShapeDispSpace{}, physics_name + "_shape_displacement", mesh->tag());
  auto field = createFieldState(*graph, ScalarSpace{}, physics_name + "_field", mesh->tag());
  *field.get() = 2.0;
  ScalarWeakForm weak_form("quadratic_mass", mesh, space(field), spaces({field}));
  addQuadraticMassIntegral(weak_form, mesh);

  auto override = makeStateDependentWeakFormBlockProviderOverride(
      0, weak_form, shape_disp, {field}, {1.0}, TimeInfo(0.0, 1.0), mfem::Array<int>(), {StateBlockBinding{0, 0}});

  mfem::Array<int> block_offsets(2);
  block_offsets[0] = 0;
  block_offsets[1] = field.get()->Size();

  mfem::Vector x(block_offsets[1]);
  mfem::Vector y(block_offsets[1]);
  x = 1.0;

  mfem::Vector state(block_offsets[1]);
  state = 3.0;
  override.provider->updateForState(state, block_offsets);
  override.provider->currentOperator().Mult(x, y);
  const double first_norm = y.Norml2();

  state = 5.0;
  override.provider->updateForState(state, block_offsets);
  override.provider->currentOperator().Mult(x, y);
  const double second_norm = y.Norml2();

  EXPECT_GT(first_norm, 0.0);
  EXPECT_GT(second_norm, first_norm);
  for (int i = 0; i < field.get()->Size(); ++i) {
    EXPECT_DOUBLE_EQ((*field.get())[i], 2.0);
  }
}

}  // namespace
}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
