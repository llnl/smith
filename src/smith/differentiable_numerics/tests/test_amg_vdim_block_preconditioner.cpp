// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <mpi.h>

#include "mfem.hpp"
#include <HYPRE_parcsr_ls.h>

#include "smith/infrastructure/application_manager.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/numerics/block_preconditioner.hpp"
#include "smith/numerics/solver_with_preconditioner.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {
namespace {

HYPRE_Int GetBoomerAMGNumFunctions(const mfem::HypreBoomerAMG& amg)
{
  HYPRE_Solver h = amg;  // mfem::HypreBoomerAMG has operator HYPRE_Solver()
  HYPRE_Int dim = -1;
  HYPRE_BoomerAMGGetNumFunctions(h, &dim);
  return dim;
}

TEST(AMGVdimSetup, BlockDiagonalPreconditionerSetsPerBlockVdim)
{
  MPI_Comm comm = MPI_COMM_WORLD;

  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, 1, 1.0, 1.0);
  mfem::ParMesh pmesh(comm, mesh);
  mesh.Clear();

  const int order = 1;
  mfem::H1_FECollection fec(order, pmesh.Dimension());

  // Block 0: vdim=2
  mfem::ParFiniteElementSpace fes_v2(&pmesh, &fec, 2, smith::ordering);
  // Block 1: vdim=1
  mfem::ParFiniteElementSpace fes_v1(&pmesh, &fec, 1, smith::ordering);

  auto u0 = std::make_shared<FiniteElementState>(fes_v2, "u0");
  auto u1 = std::make_shared<FiniteElementState>(fes_v1, "u1");
  std::vector<NonlinearBlockSolverBase::FieldPtr> us{u0, u1};

  NonlinearSolverOptions nonlin_opts;
  nonlin_opts.nonlin_solver = NonlinearSolver::Newton;
  nonlin_opts.max_iterations = 1;
  nonlin_opts.print_level = 0;
  nonlin_opts.absolute_tol = 1e-12;
  nonlin_opts.relative_tol = 1e-10;

  LinearSolverOptions lin_opts;
  lin_opts.linear_solver = LinearSolver::GMRES;
  lin_opts.preconditioner = Preconditioner::BlockDiagonal;
  lin_opts.max_iterations = 1;
  lin_opts.print_level = 0;
  lin_opts.preconditioner_print_level = 0;

  LinearSolverOptions sub;
  sub.linear_solver = LinearSolver::GMRES;
  sub.preconditioner = Preconditioner::HypreAMG;
  sub.max_iterations = 1;
  sub.relative_tol = 0.99;
  sub.absolute_tol = 0.0;
  sub.print_level = 0;
  sub.preconditioner_print_level = 0;

  lin_opts.sub_block_linear_solver_options.push_back(sub);
  lin_opts.sub_block_linear_solver_options.push_back(sub);

  auto eq = std::make_unique<EquationSolver>(nonlin_opts, lin_opts, comm);
  mfem::Solver* top_prec = &eq->preconditioner();

  NonlinearBlockSolver nbs(std::move(eq), comm, nonlin_opts.absolute_tol, nonlin_opts.relative_tol, nonlin_opts,
                           lin_opts);
  nbs.completeSetup(us);

  auto* block_diag = dynamic_cast<smith::BlockDiagonalPreconditioner*>(top_prec);
  ASSERT_NE(block_diag, nullptr);
  ASSERT_GE(block_diag->numSubSolvers(), 2);

  // Block 0
  {
    auto* wrapped = dynamic_cast<smith::SolverWithPreconditioner*>(block_diag->subSolver(0));
    ASSERT_NE(wrapped, nullptr);

    auto* amg = dynamic_cast<mfem::HypreBoomerAMG*>(wrapped->preconditioner());
    ASSERT_NE(amg, nullptr);
    EXPECT_EQ(GetBoomerAMGNumFunctions(*amg), 2);
  }

  // Block 1
  {
    auto* wrapped = dynamic_cast<smith::SolverWithPreconditioner*>(block_diag->subSolver(1));
    ASSERT_NE(wrapped, nullptr);

    auto* amg = dynamic_cast<mfem::HypreBoomerAMG*>(wrapped->preconditioner());
    ASSERT_NE(amg, nullptr);
    EXPECT_EQ(GetBoomerAMGNumFunctions(*amg), 1);
  }
}

}  // namespace
}  // namespace smith

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
