// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>
#include <functional>

#include "axom/slic/core/SimpleLogger.hpp"
#include "gtest/gtest.h"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/numerics/functional/domain.hpp"
#include "serac/numerics/functional/quadrature_data.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"

namespace serac {

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
      auto   states     = (*qdata)[geom_type];
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

TEST(state_manager, basic)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // TODO: are these right?
  constexpr int dim   = 3;
  constexpr int order = 2;

  // Info about this test's Quadrature data state
  /*
    struct State {
        tensor<double, dim, dim> Fpinv = DenseIdentity<3>();  ///< inverse of plastic distortion tensor
        double                   accumulated_plastic_strain;  ///< uniaxial equivalent plastic strain
    };
  */
  using State = serac::solid_mechanics::J2<serac::solid_mechanics::LinearHardening>::State;

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
    state.Fpinv                      = make_tensor<dim, dim>([&](int i, int j) { return i + curr_value * j; });
    state.accumulated_plastic_strain = curr_value;
    curr_value++;
  };
  //---------------------------------

  // Create DataStore
  std::string            name = "basic";
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the StateManager
  std::string filename = SERAC_REPO_DIR "/data/meshes/ball.mesh";
  std::string mesh_tag = "ball_mesh";
  auto        mesh     = mesh::refineAndDistribute(buildMeshFromFile(filename), 1, 0);
  StateManager::setMesh(std::move(mesh), mesh_tag);

  // Create and store the initial state of the quadrature data in sidre
  SLIC_INFO("Creating Quadrature Data with initial state");
  Domain                                 domain = EntireDomain(StateManager::mesh(mesh_tag));
  State                                  initial_state{};
  std::shared_ptr<QuadratureData<State>> qdata =
      StateManager::newQuadratureDataBuffer(mesh_tag, domain, order, dim, initial_state);

  // Change data
  SLIC_INFO("Populating Quadrature Data");
  constexpr double good_starting_value = 1.0;
  detail::apply_function_to_quadrature_data_states(good_starting_value, qdata, fill_state);
  SLIC_INFO("Verifying populated Quadrature Data");
  detail::apply_function_to_quadrature_data_states(good_starting_value, qdata, check_state);

  // Save to disk and simulate a restart
  const int    cycle      = 1;
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
  Domain                                 new_domain = EntireDomain(StateManager::mesh(mesh_tag));
  std::shared_ptr<QuadratureData<State>> new_qdata =
      StateManager::newQuadratureDataBuffer(mesh_tag, new_domain, order, dim, initial_state);

  // Verify data has reloaded to restart data
  SLIC_INFO("Verifying loaded Quadrature Data");
  detail::apply_function_to_quadrature_data_states(good_starting_value, new_qdata, check_state);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
