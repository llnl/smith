// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file test_state_manager.cpp
 */

#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "mpi.h"
#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/physics/state/state_manager.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/quadrature_data.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/smith_config.hpp"

namespace smith {

namespace detail {

template <typename QDataType, typename StateType>
void apply_function_to_quadrature_data_states(const double starting_value, std::shared_ptr<QDataType> qdata,
                                              std::function<void(double&, StateType&)>& apply_function)
{
  for (std::size_t i = 0; i < detail::qdata_geometries.size(); ++i) {
    auto geom_type = detail::qdata_geometries[i];

    // Check if geometry type has any data
    if ((*qdata).data.find(geom_type) != (*qdata).data.end()) {
      // Get axom::Array of states in map
      auto states = (*qdata)[geom_type];
      double curr_value = starting_value;
      for (auto& state : states) {
        apply_function(curr_value, state);
      }
    }
  }
}

template <typename T, int M, int N>
bool compare_tensors(const tensor<T, M, N>& a, const tensor<T, M, N>& b)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (a(i, j) != b(i, j)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace detail

TEST(state_manager, QuadratureData_Restart)
{
  // This test checks that the state manager can save and load
  // a quadrature data object. It does this by creating a
  // quadrature data object, populating it with some data,
  // saving it to disk, and then loading it back from disk.
  // It then checks that the loaded data matches the original
  // data.

  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int dim = 3;
  constexpr int order = 2;

  // Info about this test's QuadratureData State
  /*
    struct State {
        tensor<double, dim, dim> Fpinv = DenseIdentity<3>();  ///< inverse of plastic distortion tensor
        double                   accumulated_plastic_strain;  ///< uniaxial equivalent plastic strain
    };
  */
  using State = smith::solid_mechanics::J2<smith::solid_mechanics::LinearHardening>::State;

  //--------------------------------- Helper functions for this test
  // Lamda to check the state against a starting value which is incremented after each check
  std::function<void(double&, State&)> check_state = [](double& curr_value, State& state) {
    tensor<double, dim, dim> expected_tensor = make_tensor<dim, dim>([&](int i, int j) { return i + curr_value * j; });
    EXPECT_TRUE(detail::compare_tensors(state.Fpinv, expected_tensor));
    EXPECT_DOUBLE_EQ(state.accumulated_plastic_strain, curr_value);
    curr_value++;
  };

  // Lamda to fill the state against a starting value which is incremented after each check
  std::function<void(double&, State&)> fill_state = [](double& curr_value, State& state) {
    state.Fpinv = make_tensor<dim, dim>([&](int i, int j) { return i + curr_value * j; });
    state.accumulated_plastic_strain = curr_value;
    curr_value++;
  };
  //---------------------------------

  // Create DataStore
  std::string name = "basic";
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the StateManager
  std::string filename = SMITH_REPO_DIR "/data/meshes/ball.mesh";
  std::string mesh_tag = "ball_mesh";
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), 1, 0);
  StateManager::setMesh(std::move(mesh), mesh_tag);

  // Create and store the initial state of the QuadratureData in Sidre
  SLIC_INFO("Creating Quadrature Data with initial state");
  Domain domain = EntireDomain(StateManager::mesh(mesh_tag));
  State initial_state{};
  std::shared_ptr<QuadratureData<State>> qdata =
      StateManager::newQuadratureDataBuffer(mesh_tag, domain, order, dim, initial_state);

  // Change data
  SLIC_INFO("Populating QuadratureData");
  constexpr double good_starting_value = 1.0;
  detail::apply_function_to_quadrature_data_states(good_starting_value, qdata, fill_state);
  SLIC_INFO("Verifying populated Quadrature Data");
  detail::apply_function_to_quadrature_data_states(good_starting_value, qdata, check_state);

  // Save to disk and simulate a restart
  const int cycle = 1;
  const double time_saved = 1.5;
  SLIC_INFO(axom::fmt::format("Saving mesh restart '{0}' at cycle '{1}'", mesh_tag, cycle));
  StateManager::save(time_saved, cycle, mesh_tag);

  // Reset StateManager then load from disk
  SLIC_INFO("Clearing current and loading previously saved State Manager");
  StateManager::reset();
  axom::sidre::DataStore new_datastore;
  StateManager::initialize(new_datastore, name + "_data");
  StateManager::load(cycle, mesh_tag);

  // Load data from disk
  SLIC_INFO("Loading previously saved Quadrature Data");
  Domain new_domain = EntireDomain(StateManager::mesh(mesh_tag));
  std::shared_ptr<QuadratureData<State>> new_qdata =
      StateManager::newQuadratureDataBuffer(mesh_tag, new_domain, order, dim, initial_state);

  // Verify data has reloaded to restart data
  SLIC_INFO("Verifying loaded Quadrature Data");
  detail::apply_function_to_quadrature_data_states(good_starting_value, new_qdata, check_state);
}

TEST(StateManager, StoresHighOrderMeshes)
{
  // This test ensures that when high order meshes are given to
  // the state manager, it indeed stores the high order mesh, and
  // does not cast it down to first order.
  //
  // This test will break if you change the mesh file.
  // It relies on knowledge of the specific mesh
  // in "single_curved_quad.g".

  // The mesh has a single element with one curved edge.
  // It looks something like this:
  //
  //     curved edge on top
  //            7
  //        __--O--__
  //  0  O--         --O 3
  //     |             |
  //     |       O     |    straight edges on sides and bottom
  //  4  O       8     O 6
  //     |             |
  //     |             |
  //  1  O------O------O 2
  //            5

  constexpr int dim = 2;
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "curved_element_output_test");

  const std::string filename = SMITH_REPO_DIR "/data/meshes/single_curved_quad.g";
  int serial_refinement = 0;
  int parallel_refinement = 0;
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  auto& pmesh = smith::StateManager::setMesh(std::move(mesh), "mesh");

  ASSERT_EQ(dim, pmesh.SpaceDimension());

  // Make sure that the stored mesh maintained second order character
  EXPECT_EQ(pmesh.GetNodalFESpace()->GetMaxElementOrder(), 2);
  EXPECT_EQ(pmesh.GetNodalFESpace()->GetNDofs(), 9);

  // make sure that the curved boundary hasn't been replaced
  // with a straight edge

  const mfem::GridFunction* nodes = pmesh.GetNodes();

  // Get dofs on curved edge
  const int curved_boundary_element = 2;  // edge elem id of the curved edge
  mfem::Array<int> dofs;
  pmesh.GetNodalFESpace()->GetBdrElementDofs(curved_boundary_element, dofs);
  constexpr int num_nodes_on_edge = dim + 1;
  ASSERT_EQ(dofs.Size(), num_nodes_on_edge);

  // Get coordinates of curved edge nodes
  mfem::Array<tensor<double, dim>> edge_coords(num_nodes_on_edge);
  for (int k = 0; k < dofs.Size(); k++) {
    int d = dofs[k];
    for (int i = 0; i < dim; i++) {
      edge_coords[k][i] = (*nodes)(pmesh.GetNodalFESpace()->DofToVDof(d, i));
    }
  }

  // Make sure edge nodes are not colinear
  auto v1 = edge_coords[0] - edge_coords[1];
  auto v2 = edge_coords[0] - edge_coords[2];
  double area = std::abs(v1[0] * v2[1] - v1[1] * v2[0]);
  EXPECT_GT(area, 1e-6);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
