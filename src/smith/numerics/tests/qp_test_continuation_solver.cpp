// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/smith_config.hpp"
#include "problems/Problems.hpp"
#include "solvers/Solvers.hpp"
#include "utilities.hpp"

using namespace smith;

/* convex quadratic-programming problem
 *
 * min_u (u^T u) / 2
 *  s.t.  u - u_l >= 0
 *
 *  solution u = max{0, ul} (component-wise)
 */
class QPTestProblem : public OptProblem {
 protected:
  mfem::Vector ul_;
  mfem::HypreParMatrix* dgdu_ = nullptr;
  mfem::HypreParMatrix* d2Edu2_ = nullptr;

 public:
  QPTestProblem(int n);
  double E(const mfem::Vector& u, int& eval_err);

  void DdE(const mfem::Vector& u, mfem::Vector& gradE);

  mfem::Operator* DddE(const mfem::Vector& u);

  void g(const mfem::Vector& u, mfem::Vector& gu, int& eval_err);

  mfem::Operator* Ddg(const mfem::Vector&);

  virtual ~QPTestProblem();
};

TEST(InteriorPointMethod, QuadraticProgramming)
{
  int n = 30;
  double outerSolveTol = 1.e-8;
  double linSolveTol = 1.e-10;
  int maxiter = 40;

  QPTestProblem opt_problem(n);

  InteriorPointSolver solver(&opt_problem);
  mfem::GMRESSolver linSolver(MPI_COMM_WORLD);
  linSolver.SetRelTol(linSolveTol);
  linSolver.SetMaxIter(1000);
  linSolver.SetPrintLevel(2);
  solver.SetLinearSolver(linSolver);

  int dimx = opt_problem.GetDimU();
  mfem::Vector x0(dimx);
  x0 = 0.0;
  mfem::Vector xf(dimx);
  xf = 0.0;

  solver.SetTol(outerSolveTol);
  solver.SetMaxIter(maxiter);

  solver.Mult(x0, xf);

  EXPECT_TRUE(solver.GetConverged());
}

TEST(HomotopyMethod, QuadraticProgramming)
{
  int n = 30;
  double outerSolveTol = 1.e-8;
  double linSolveTol = 1.e-10;
  int maxiter = 40;

  QPTestProblem opt_problem(n);
  OptNLMCProblem nlmc_problem(&opt_problem);

  HomotopySolver solver(&nlmc_problem);
  mfem::GMRESSolver linSolver(MPI_COMM_WORLD);
  linSolver.SetRelTol(linSolveTol);
  linSolver.SetMaxIter(1000);
  linSolver.SetPrintLevel(2);
  solver.SetLinearSolver(linSolver);

  int dimx = nlmc_problem.GetDimx();
  int dimy = nlmc_problem.GetDimy();
  mfem::Vector x0(dimx);
  x0 = 0.0;
  mfem::Vector y0(dimy);
  y0 = 0.0;
  mfem::Vector xf(dimx);
  xf = 0.0;
  mfem::Vector yf(dimy);
  yf = 0.0;

  solver.SetTol(outerSolveTol);
  solver.SetMaxIter(maxiter);

  solver.Mult(x0, y0, xf, yf);

  EXPECT_TRUE(solver.GetConverged());
}

// Ex1Problem
QPTestProblem::QPTestProblem(int n) : OptProblem()
{
  MFEM_VERIFY(n >= 1, "QPTestProblem::QPTestProblem -- problem must have nontrivial size");

  // generate parallel partition
  int nprocs = mfem::Mpi::WorldSize();
  int myid = mfem::Mpi::WorldRank();

  HYPRE_BigInt dofOffsets[2];
  HYPRE_BigInt constraintOffsets[2];
  if (n >= nprocs) {
    dofOffsets[0] = HYPRE_BigInt((myid * n) / nprocs);
    dofOffsets[1] = HYPRE_BigInt(((myid + 1) * n) / nprocs);
  } else {
    if (myid < n) {
      dofOffsets[0] = myid;
      dofOffsets[1] = myid + 1;
    } else {
      dofOffsets[0] = n;
      dofOffsets[1] = n;
    }
  }
  constraintOffsets[0] = dofOffsets[0];
  constraintOffsets[1] = dofOffsets[1];
  Init(dofOffsets, constraintOffsets);

  mfem::Vector temp(dimU);
  temp = 1.0;
  d2Edu2_ = GenerateHypreParMatrixFromDiagonal(dofOffsetsU, temp);
  dgdu_ = GenerateHypreParMatrixFromDiagonal(dofOffsetsU, temp);

  // random entries in [-1, 1]
  ul_.SetSize(dimM);
  ul_.Randomize(myid);
  ul_ *= 2.0;
  ul_ -= 1.0;
}

double QPTestProblem::E(const mfem::Vector& u, int& eval_err)
{
  eval_err = 0;
  double Eeval = 0.5 * mfem::InnerProduct(MPI_COMM_WORLD, u, u);
  return Eeval;
}

void QPTestProblem::DdE(const mfem::Vector& u, mfem::Vector& gradE) { gradE.Set(1.0, u); }

mfem::Operator* QPTestProblem::DddE(const mfem::Vector& /*u*/) { return d2Edu2_; }

void QPTestProblem::g(const mfem::Vector& u, mfem::Vector& gu, int& eval_err)
{
  eval_err = 0;
  gu = 0.0;
  gu.Set(1.0, u);
  gu.Add(-1.0, ul_);
}

mfem::Operator* QPTestProblem::Ddg(const mfem::Vector& /*u*/) { return dgdu_; }

QPTestProblem::~QPTestProblem()
{
  delete d2Edu2_;
  delete dgdu_;
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);

  return RUN_ALL_TESTS();
}
