// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <set>

#include "mpi.h"
#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_helper.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {

TEST(BoundaryCond, SimpleRepeatedDofs)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int N = 15;
  constexpr int ATTR = 1;
  auto mesh = mfem::Mesh::MakeCartesian2D(N, N, mfem::Element::TRIANGLE);
  mfem::ParMesh par_mesh(MPI_COMM_WORLD, mesh);
  FiniteElementState state(par_mesh, H1<1>{});

  for (int i = 0; i < par_mesh.GetNBE(); i++) {
    par_mesh.GetBdrElement(i)->SetAttribute(ATTR);
  }

  BoundaryConditionManager bcs(par_mesh);
  auto coef = std::make_shared<mfem::ConstantCoefficient>(1);
  bcs.addEssential(std::set<int>{ATTR}, coef, state.space(), 1);
  const auto before_dofs = bcs.allEssentialTrueDofs();

  bcs.addEssential(std::set<int>{ATTR}, coef, state.space(), 1);
  const auto after_dofs = bcs.allEssentialTrueDofs();

  // Make sure that attempting to add a boundary condition
  // on already-used elements doesn't change the dofs
  EXPECT_EQ(before_dofs.Size(), after_dofs.Size());
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(BoundaryCond, DirectLocalDofs)
{
  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int N = 15;
  auto mesh = mfem::Mesh::MakeCartesian2D(N, N, mfem::Element::TRIANGLE);
  mfem::ParMesh par_mesh(MPI_COMM_WORLD, mesh);
  FiniteElementState state(par_mesh, H1<1>{});

  BoundaryConditionManager bcs(par_mesh);

  auto coef = std::make_shared<mfem::ConstantCoefficient>(1.0);

  mfem::Array<int> local_dofs;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    local_dofs.SetSize(1);
    local_dofs[0] = 1;
  } else if (rank == 1) {
    local_dofs.SetSize(2);
    local_dofs[0] = 5;
    local_dofs[1] = 48;
  }

  mfem::Array<int> true_dofs;
  for (int i = 0; i < local_dofs.Size(); i++) {
    true_dofs.Append(state.space().GetLocalTDofNumber(local_dofs[i]));
  }

  bcs.addEssential(local_dofs, coef, state.space());
  auto should_be_true_dofs = bcs.allEssentialTrueDofs();

  should_be_true_dofs.Sort();

  EXPECT_EQ(should_be_true_dofs.Size(), true_dofs.Size());

  for (int i = 0; i < should_be_true_dofs.Size(); i++) {
    EXPECT_EQ(should_be_true_dofs[i], true_dofs[i]);
  }
}

TEST(BoundaryCondHelper, ElementAttributeDofListScalar)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int attribute = 2;

  auto mesh = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
  mesh.SetAttribute(1, attribute);
  mesh.SetAttribute(31, attribute);
  mesh.SetAttributes();

  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

  mfem::Array<int> elem_attr_is_ess(pmesh.attributes.Max());
  elem_attr_is_ess = 0;
  elem_attr_is_ess[attribute - 1] = 1;

  mfem::Array<int> ess_tdof_list;

  auto [l2_fes, l2_fec] = smith::generateParFiniteElementSpace<L2<0>>(&pmesh);
  smith::mfem_ext::GetEssentialTrueDofsFromElementAttribute(*l2_fes, elem_attr_is_ess, ess_tdof_list);

  int local_num_tdof = ess_tdof_list.Size();
  int global_num_tdof;
  MPI_Allreduce(&local_num_tdof, &global_num_tdof, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  EXPECT_EQ(global_num_tdof, 2);

  int numRanks;
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  if (numRanks == 1) {
    EXPECT_EQ(ess_tdof_list[0], 1);
    EXPECT_EQ(ess_tdof_list[1], 31);
  }

  auto [h1_fes, h1_fec] = smith::generateParFiniteElementSpace<H1<1>>(&pmesh);
  smith::mfem_ext::GetEssentialTrueDofsFromElementAttribute(*h1_fes, elem_attr_is_ess, ess_tdof_list, -1);

  local_num_tdof = ess_tdof_list.Size();
  MPI_Allreduce(&local_num_tdof, &global_num_tdof, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_EQ(global_num_tdof, 16);

  if (numRanks == 1) {
    EXPECT_EQ(ess_tdof_list[0], 25);
    EXPECT_EQ(ess_tdof_list[1], 26);
    EXPECT_EQ(ess_tdof_list[2], 30);
    EXPECT_EQ(ess_tdof_list[3], 31);
    EXPECT_EQ(ess_tdof_list[4], 50);
    EXPECT_EQ(ess_tdof_list[5], 51);
    EXPECT_EQ(ess_tdof_list[6], 55);
    EXPECT_EQ(ess_tdof_list[7], 56);
    EXPECT_EQ(ess_tdof_list[8], 81);
    EXPECT_EQ(ess_tdof_list[9], 82);
    EXPECT_EQ(ess_tdof_list[10], 86);
    EXPECT_EQ(ess_tdof_list[11], 87);
    EXPECT_EQ(ess_tdof_list[12], 106);
    EXPECT_EQ(ess_tdof_list[13], 107);
    EXPECT_EQ(ess_tdof_list[14], 111);
    EXPECT_EQ(ess_tdof_list[15], 112);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(BoundaryCondHelper, ElementAttributeDofList)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int attribute = 2;
  constexpr int dim = 3;

  auto mesh = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
  mesh.SetAttribute(2, attribute);
  mesh.SetAttribute(3, attribute);
  mesh.SetAttributes();

  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
  int sdim = pmesh.SpaceDimension();

  mfem::Array<int> elem_attr_is_ess(pmesh.attributes.Max());
  elem_attr_is_ess = 0;
  elem_attr_is_ess[attribute - 1] = 1;

  mfem::Array<int> ess_tdof_list;

  // scalar space
  using scalar_space = L2<1>;
  auto [l2_scalar, l2_fec] = smith::generateParFiniteElementSpace<scalar_space>(&pmesh);

  smith::mfem_ext::GetEssentialTrueDofsFromElementAttribute(*l2_scalar, elem_attr_is_ess, ess_tdof_list);
  int local_num_tdof = ess_tdof_list.Size();
  int global_num_tdof;
  MPI_Allreduce(&local_num_tdof, &global_num_tdof, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_EQ(global_num_tdof, 16);  // grab only component (scalar)

  // set up a gridfunction keyed off element attribute
  double attr1value = 3.2;
  double attr2value = 1.7;
  mfem::Vector values(2);
  values[0] = attr1value;
  values[1] = attr2value;
  mfem::PWConstCoefficient attr_coef(values);
  mfem::ParGridFunction l2_scalar_gf(l2_scalar.get());
  l2_scalar_gf.ProjectCoefficient(attr_coef);

  mfem::Vector& true_values = l2_scalar_gf.GetTrueVector();
  for (int i = 0; i < true_values.Size(); i++) {
    if (ess_tdof_list.Find(i) > -1) {  // check if index is contrained in array
      EXPECT_EQ(true_values(i), attr2value);
    } else {
      EXPECT_EQ(true_values(i), attr1value);
    }
  }

  // vector space
  using vector_space = L2<1, dim>;
  auto [l2_vector, l2_fec_vec] = smith::generateParFiniteElementSpace<vector_space>(&pmesh);
  mfem::ParGridFunction l2_vector_gf(l2_vector.get());
  mfem::PWVectorCoefficient attr_vec_coef(sdim);
  mfem::Vector attr1vec({0.0, 1.0, 2.0});
  mfem::Vector attr2vec({3.0, 4.0, 5.0});
  mfem::VectorConstantCoefficient attr1vCoef(attr1vec);
  mfem::VectorConstantCoefficient attr2vCoef(attr2vec);
  attr_vec_coef.UpdateCoefficient(1, attr1vCoef);
  attr_vec_coef.UpdateCoefficient(2, attr2vCoef);
  l2_vector_gf.ProjectCoefficient(attr_vec_coef);
  mfem::Vector& true_vec_values = l2_vector_gf.GetTrueVector();

  smith::mfem_ext::GetEssentialTrueDofsFromElementAttribute(*l2_vector, elem_attr_is_ess, ess_tdof_list, -1);
  local_num_tdof = ess_tdof_list.Size();
  MPI_Allreduce(&local_num_tdof, &global_num_tdof, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_EQ(global_num_tdof, 16 * 3);  // grab all

  smith::mfem_ext::GetEssentialTrueDofsFromElementAttribute(*l2_vector, elem_attr_is_ess, ess_tdof_list, 0);
  local_num_tdof = ess_tdof_list.Size();
  MPI_Allreduce(&local_num_tdof, &global_num_tdof, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_EQ(global_num_tdof, 16);  // grab single component

  for (int i = 0; i < true_vec_values.Size(); i++) {
    if (ess_tdof_list.Find(i) > -1) {  // check if index is contrained in array
      EXPECT_EQ(true_vec_values(i), 3.0);
    }
  }

  smith::mfem_ext::GetEssentialTrueDofsFromElementAttribute(*l2_vector, elem_attr_is_ess, ess_tdof_list, 1);
  local_num_tdof = ess_tdof_list.Size();
  MPI_Allreduce(&local_num_tdof, &global_num_tdof, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_EQ(global_num_tdof, 16);  // grab single component

  for (int i = 0; i < true_vec_values.Size(); i++) {
    if (ess_tdof_list.Find(i) > -1) {  // check if index is contrained in array
      EXPECT_EQ(true_vec_values(i), 4.0);
    }
  }

  smith::mfem_ext::GetEssentialTrueDofsFromElementAttribute(*l2_vector, elem_attr_is_ess, ess_tdof_list, 2);
  local_num_tdof = ess_tdof_list.Size();
  MPI_Allreduce(&local_num_tdof, &global_num_tdof, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_EQ(global_num_tdof, 16);  // grab single component

  for (int i = 0; i < true_vec_values.Size(); i++) {
    if (ess_tdof_list.Find(i) > -1) {  // check if index is contrained in array
      EXPECT_EQ(true_vec_values(i), 5.0);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
