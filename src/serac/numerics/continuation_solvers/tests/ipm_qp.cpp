// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

//#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/infrastructure/application_manager.hpp"
#include "serac/serac_config.hpp"
//#include "serac/mesh/mesh_utils_base.hpp"
//#include "serac/numerics/functional/functional.hpp"
//#include "serac/numerics/functional/shape_aware_functional.hpp"
//#include "serac/numerics/functional/tensor.hpp"
#include "serac/infrastructure/profiling.hpp"
//
//#include "serac/numerics/functional/tests/check_gradient.hpp"

#include "serac/numerics/continuation_solvers/problems/Problems.hpp"
#include "serac/numerics/continuation_solvers/solvers/IPSolver.hpp"
#include "serac/numerics/continuation_solvers/utilities.hpp"


using namespace serac;
using namespace serac::profiling;

int nsamples = 1;  // because mfem doesn't take in unsigned int

using namespace mfem;
/* convex quadratic-programming problem
 * 
 * min_u (u^T u) / 2
 *  s.t.  u - u_l >= 0
 *
 *  solution u = max{0, ul} (component-wise)
 */
class QPTestProblem : public ParOptProblem
{
protected:
   Vector ul;
   HypreParMatrix * dgdu;
   HypreParMatrix * d2Edu2;
public:
   QPTestProblem(int n);
   double E(const Vector & u, int & eval_err);

   void DdE(const Vector & u, Vector & gradE);

   HypreParMatrix * DddE(const Vector & u);

   void g(const Vector & u, Vector & gu, int & eval_err);

   HypreParMatrix * Ddg(const Vector &);

   virtual ~QPTestProblem();
};



TEST(InteriorPointMethod, QuadraticProgramming)
{
   int n = 30;
   double optTol = 1.e-8;
   int maxiter = 40;
   
   QPTestProblem problem(n);

   ParInteriorPointSolver solver(&problem);

   int dimx = problem.GetDimU();
   Vector x0(dimx); x0 = 0.0;
   Vector xf(dimx); xf = 0.0; 
   
   solver.SetTol(optTol);
   solver.SetMaxIter(maxiter);
   
   solver.Mult(x0, xf);
   
   EXPECT_TRUE(solver.GetConverged()); 
}



int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  

  return RUN_ALL_TESTS();
}


// Ex1Problem
QPTestProblem::QPTestProblem(int n) : ParOptProblem(), 
	dgdu(nullptr), d2Edu2(nullptr)
{
  MFEM_VERIFY(n >= 1, "QPTestProblem::QPTestProblem -- problem must have nontrivial size");
	
  // generate parallel partition  
  int nprocs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  
  HYPRE_BigInt * dofOffsets = new HYPRE_BigInt[2];
  HYPRE_BigInt * constraintOffsets = new HYPRE_BigInt[2];
  if (n >= nprocs)
  {
     dofOffsets[0] = HYPRE_BigInt((myid * n) / nprocs);
     dofOffsets[1] = HYPRE_BigInt(((myid + 1) * n) / nprocs);
  }
  else
  {
     if (myid < n)
     {
        dofOffsets[0] = myid;
        dofOffsets[1] = myid + 1;
     }
     else
     {
        dofOffsets[0] = n;
	dofOffsets[1] = n;
     }
  }
  constraintOffsets[0] = dofOffsets[0];
  constraintOffsets[1] = dofOffsets[1];
  Init(dofOffsets, constraintOffsets);
  delete[] dofOffsets;
  delete[] constraintOffsets;

  Vector temp(dimU); temp = 1.0;
  d2Edu2 = GenerateHypreParMatrixFromDiagonal(dofOffsetsU, temp);
  // deep copy?
  dgdu = GenerateHypreParMatrixFromDiagonal(dofOffsetsU, temp);
  
  
  // random entries in [-1, 1]
  ul.SetSize(dimM);
  ul.Randomize(myid);
  ul *= 2.0;
  ul -= 1.0;
}

double QPTestProblem::E(const Vector & u, int & eval_err)
{
   eval_err = 0;
   double Eeval = 0.5 * InnerProduct(MPI_COMM_WORLD, u, u);
   return Eeval;
}

void QPTestProblem::DdE(const Vector & u, Vector & gradE)
{
  gradE.Set(1.0, u);
}

HypreParMatrix * QPTestProblem::DddE(const Vector & u)
{
   return d2Edu2;
}

void QPTestProblem::g(const Vector & u, Vector & gu, int & eval_err)
{
   eval_err = 0;
   gu = 0.0;
   gu.Set(1.0, u);
   gu.Add(-1.0, ul);
}

HypreParMatrix * QPTestProblem::Ddg(const Vector & u)
{
   return dgdu;
}


QPTestProblem::~QPTestProblem()
{
   delete d2Edu2;
   delete dgdu;
}


