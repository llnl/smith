// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <memory>

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

TEST(WeakFormBlockOperatorMPI, EliminatesEssentialDofsOwnedByOnlyOneRank)
{
  axom::sidre::DataStore datastore;
  StateManager::reset();
  StateManager::initialize(datastore, "weak_form_block_operator_mpi");

  auto serial_mesh = mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  auto mesh = std::make_shared<Mesh>(std::move(serial_mesh), "weak_form_block_operator_mpi_mesh");
  auto graph = std::make_shared<gretl::DataStore>(std::make_unique<gretl::WangCheckpointStrategy>(100));
  auto shape_disp = createFieldState(*graph, ShapeDispSpace{}, "shape_displacement", mesh->tag());
  auto field = createFieldState(*graph, ScalarSpace{}, "field", mesh->tag());

  ScalarWeakForm weak_form("mass", mesh, space(field), spaces({field}));
  weak_form.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(),
                            [](auto /* time_info */, auto /* x */, auto U) { return tuple{get<VALUE>(U), zero{}}; });

  constexpr HYPRE_BigInt constrained_global_true_dof = 0;
  const auto& field_space = field.get()->space();
  const HYPRE_BigInt local_true_dof_begin = field_space.GetMyTDofOffset();
  const HYPRE_BigInt local_true_dof_end = local_true_dof_begin + field_space.GetTrueVSize();
  mfem::Array<int> essential_true_dofs;
  if (local_true_dof_begin <= constrained_global_true_dof && constrained_global_true_dof < local_true_dof_end) {
    essential_true_dofs.Append(static_cast<int>(constrained_global_true_dof - local_true_dof_begin));
  }

  auto op = buildWeakFormOperator(weak_form, shape_disp, {field}, {1.0}, TimeInfo(0.0, 1.0), essential_true_dofs);
  ASSERT_NE(op, nullptr);

  mfem::Vector essential_vector(op->Width());
  essential_vector = 0.0;
  essential_vector.SetSubVector(essential_true_dofs, 1.0);

  mfem::Vector applied(op->Height());
  op->Mult(essential_vector, applied);
  applied -= essential_vector;
  EXPECT_LT(applied.Norml2(), 1.0e-14);

  op->MultTranspose(essential_vector, applied);
  applied -= essential_vector;
  EXPECT_LT(applied.Norml2(), 1.0e-14);

  StateManager::reset();
}

}  // namespace
}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager application_manager(argc, argv);
  return RUN_ALL_TESTS();
}
