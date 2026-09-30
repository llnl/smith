// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include <mpi.h>
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

  auto serial_mesh = mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL, 1, 1.0, 1.0);
  auto mesh = std::make_shared<Mesh>(std::move(serial_mesh), "weak_form_block_operator_mpi_mesh");
  auto graph = std::make_shared<gretl::DataStore>(std::make_unique<gretl::WangCheckpointStrategy>(100));
  auto shape_disp = createFieldState(*graph, ShapeDispSpace{}, "shape_displacement", mesh->tag());
  auto field = createFieldState(*graph, ScalarSpace{}, "field", mesh->tag());

  ScalarWeakForm weak_form("mass", mesh, space(field), spaces({field}));
  weak_form.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(),
                            [](auto /* time_info */, auto /* x */, auto U) { return tuple{get<VALUE>(U), zero{}}; });

  int rank = 0;
  MPI_Comm_rank(mesh->getComm(), &rank);

  const int rank_zero_has_owned_dof = rank == 0 && field.get()->Size() > 0 ? 1 : 0;
  int global_rank_zero_has_owned_dof = 0;
  MPI_Allreduce(&rank_zero_has_owned_dof, &global_rank_zero_has_owned_dof, 1, MPI_INT, MPI_SUM, mesh->getComm());
  ASSERT_EQ(global_rank_zero_has_owned_dof, 1);

  mfem::Array<int> essential_true_dofs;
  if (rank == 0) {
    essential_true_dofs.Append(0);
  }

  const int local_essential_dof_count = essential_true_dofs.Size();
  int global_essential_dof_count = 0;
  MPI_Allreduce(&local_essential_dof_count, &global_essential_dof_count, 1, MPI_INT, MPI_SUM, mesh->getComm());
  ASSERT_EQ(global_essential_dof_count, 1);

  auto op = buildWeakFormOperator(weak_form, shape_disp, {field}, {1.0}, TimeInfo(0.0, 1.0), essential_true_dofs);

  ASSERT_NE(op, nullptr);
  mfem::Vector diagonal;
  op->GetDiag(diagonal);
  if (rank == 0) {
    EXPECT_DOUBLE_EQ(diagonal[0], 1.0);
  }

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
