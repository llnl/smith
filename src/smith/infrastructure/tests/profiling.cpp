// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <array>
#include <cstring>
#include <exception>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "gtest/gtest.h"
#include "mpi.h"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/profiling.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/smith_config.hpp"

namespace smith {

TEST(Profiling, MeshRefinement)
{
  // profile mesh refinement
  SMITH_MARK_FUNCTION;
  MPI_Barrier(MPI_COMM_WORLD);
  smith::profiling::initialize();

  std::string mesh_file = std::string(SMITH_REPO_DIR) + "/data/meshes/bortel_echem.e";

  // the following string is a proxy for templated test names
  std::string test_name = "_profiling";

  SMITH_MARK_BEGIN(profiling::concat("RefineAndLoadMesh", test_name).c_str());
  auto pmesh = mesh::refineAndDistribute(buildMeshFromFile(mesh_file));
  SMITH_MARK_END(profiling::concat("RefineAndLoadMesh", test_name).c_str());

  SMITH_MARK_LOOP_BEGIN(refinement_loop, "refinement_loop");
  for (int i = 0; i < 2; i++) {
    SMITH_MARK_LOOP_ITERATION(refinement_loop, i);
    pmesh->UniformRefinement();
  }
  SMITH_MARK_LOOP_END(refinement_loop);

  // Refine once more and utilize SMITH_MARK_SCOPE
  {
    SMITH_MARK_SCOPE("RefineOnceMore");
    pmesh->UniformRefinement();
  }

  // Add metadata
  SMITH_SET_METADATA("test", "profiling");
  SMITH_SET_METADATA("mesh_file", mesh_file.c_str());
  SMITH_SET_METADATA("number_mesh_elements", pmesh->GetNE());

  // this number represents "llnl" as an unsigned integer
  unsigned int magic_uint = 1819176044;
  SMITH_SET_METADATA("magic_uint", magic_uint);

  // decode unsigned int back into char[4]
  std::array<char, sizeof(magic_uint) + 1> uint_string;
  std::fill(std::begin(uint_string), std::end(uint_string), 0);
  std::memcpy(uint_string.data(), &magic_uint, 4);
  std::cout << uint_string.data() << std::endl;

  // encode double with "llnl" bytes
  double magic_double;
  std::memcpy(&magic_double, "llnl", 4);
  SMITH_SET_METADATA("magic_double", magic_double);

  // decode the double and print
  std::array<char, sizeof(magic_double) + 1> double_string;
  std::fill(std::begin(double_string), std::end(double_string), 0);
  std::memcpy(double_string.data(), &magic_double, 4);
  std::cout << double_string.data() << std::endl;

  smith::profiling::finalize();

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Profiling, Exception)
{
  // profile mesh refinement
  MPI_Barrier(MPI_COMM_WORLD);
  smith::profiling::initialize();

  {
    SMITH_MARK_SCOPE("Non-exceptionScope");
    try {
      SMITH_MARK_SCOPE("Exception scope");
      throw std::runtime_error("Caliper to verify RAII");
    } catch (std::exception& e) {
      std::cout << e.what() << "\n";
    }
  }

  smith::profiling::finalize();

  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
