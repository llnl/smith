// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <exception>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>

#include "serac/infrastructure/cli.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/boundary_conditions/boundary_condition.hpp"
#include "serac/numerics/equation_solver.hpp"
#include "serac/serac_config.hpp"

class SlicErrorException : public std::exception {};

namespace serac {

// Only need to test this when AmgX is **not** available
#ifndef MFEM_USE_AMGX
TEST(ErrorHandling, EquationSolverAmgxNotAvailable)
{
  LinearSolverOptions options;
  NonlinearSolverOptions nonlin;
  options.preconditioner = Preconditioner::AMGX;
  EXPECT_THROW(EquationSolver(nonlin, options, MPI_COMM_WORLD), SlicErrorException);
}
#endif

TEST(ErrorHandling, BcOneComponentVectorCoef)
{
  mfem::Vector vec;
  auto coef = std::make_shared<mfem::VectorConstantCoefficient>(vec);

  auto mesh = mesh::refineAndDistribute(buildDiskMesh(10), 0, 0);

  auto [space, coll] = serac::generateParFiniteElementSpace<H1<1>>(mesh.get());

  EXPECT_THROW(BoundaryCondition(coef, 0, *space, std::set<int>{1}), SlicErrorException);
}

TEST(ErrorHandling, BcOneComponentVectorCoefDofs)
{
  mfem::Vector vec;
  auto coef = std::make_shared<mfem::VectorConstantCoefficient>(vec);
  mfem::Array<int> dofs(1);
  dofs[0] = 1;

  auto mesh = mesh::refineAndDistribute(buildDiskMesh(10), 0, 0);

  auto [space, coll] = serac::generateParFiniteElementSpace<H1<1>>(mesh.get());

  EXPECT_THROW(BoundaryCondition(coef, 0, *space, dofs), SlicErrorException);
}

TEST(ErrorHandling, InvalidCmdlineArg)
{
  // The command is actually --input-file
  char const* fake_argv[] = {"serac", "--file", "input.lua"};
  const int fake_argc = 3;
  EXPECT_THROW(cli::defineAndParse(fake_argc, const_cast<char**>(fake_argv), ""), SlicErrorException);
}

TEST(ErrorHandling, NonexistentMeshPath)
{
  std::string mesh_path = "nonexistent.mesh";
  std::string input_file_path = std::string(SERAC_REPO_DIR) + "/data/input_files/default.lua";
  EXPECT_THROW(input::findMeshFilePath(mesh_path, input_file_path), SlicErrorException);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  axom::slic::setAbortFunction([]() { throw SlicErrorException{}; });
  axom::slic::setAbortOnError(true);
  axom::slic::setAbortOnWarning(false);

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
