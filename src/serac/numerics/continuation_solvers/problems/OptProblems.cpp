#include "mfem.hpp"
#include "OptProblems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



ParGeneralOptProblem::ParGeneralOptProblem() : block_offsetsx(3) { label = 0; }

void ParGeneralOptProblem::Init(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_)
{
  dofOffsetsU = new HYPRE_BigInt[2];
  dofOffsetsM = new HYPRE_BigInt[2];
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsU[i] = dofOffsetsU_[i];
    dofOffsetsM[i] = dofOffsetsM_[i];
  }
  dimU = dofOffsetsU[1] - dofOffsetsU[0];
  dimM = dofOffsetsM[1] - dofOffsetsM[0];
  dimC = dimM;
  
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();

  MPI_Allreduce(&dimU, &dimUglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimM, &dimMglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

double ParGeneralOptProblem::CalcObjective(const BlockVector &x)
{
  int eval_err; // throw away
  return CalcObjective(x, eval_err);
}

void ParGeneralOptProblem::CalcObjectiveGrad(const BlockVector &x, BlockVector &y)
{
   Duf(x, y.GetBlock(0));
   Dmf(x, y.GetBlock(1));
}

void ParGeneralOptProblem::c(const BlockVector &x, Vector &y)
{
  int eval_err; // throw-away
  return c(x, y, eval_err);
}

ParGeneralOptProblem::~ParGeneralOptProblem()
{
   block_offsetsx.DeleteAll();
}


// min E(d) s.t. g(d) >= 0
// min_(d,s) E(d) s.t. c(d,s) := g(d) - s = 0, s >= 0
ParOptProblem::ParOptProblem() : ParGeneralOptProblem()
{
}

void ParOptProblem::Init(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_)
{
  dofOffsetsU = new HYPRE_BigInt[2];
  dofOffsetsM = new HYPRE_BigInt[2];
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsU[i] = dofOffsetsU_[i];
    dofOffsetsM[i] = dofOffsetsM_[i];
  }

  dimU = dofOffsetsU[1] - dofOffsetsU[0];
  dimM = dofOffsetsM[1] - dofOffsetsM[0];
  dimC = dimM;
  
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();

  MPI_Allreduce(&dimU, &dimUglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimM, &dimMglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  ml.SetSize(dimM); ml = 0.0;
  Vector negIdentDiag(dimM);
  negIdentDiag = -1.0;
  Ih = GenerateHypreParMatrixFromDiagonal(dofOffsetsM, negIdentDiag);
}


double ParOptProblem::CalcObjective(const BlockVector &x, int & eval_err)
{ 
   return E(x.GetBlock(0), eval_err); 
}


void ParOptProblem::Duf(const BlockVector &x, Vector &y) { DdE(x.GetBlock(0), y); }

void ParOptProblem::Dmf([[maybe_unused]] const BlockVector &x, Vector &y) { y = 0.0; }

HypreParMatrix * ParOptProblem::Duuf(const BlockVector &x) 
{ 
   return DddE(x.GetBlock(0)); 
}

HypreParMatrix * ParOptProblem::Dumf([[maybe_unused]] const BlockVector &x) { return nullptr; }

HypreParMatrix * ParOptProblem::Dmuf([[maybe_unused]] const BlockVector &x) { return nullptr; }

HypreParMatrix * ParOptProblem::Dmmf([[maybe_unused]] const BlockVector &x) { return nullptr; }

void ParOptProblem::c(const BlockVector &x, Vector &y, int & eval_err) // c(u,m) = g(u) - m 
{
   g(x.GetBlock(0), y, eval_err);
   y.Add(-1.0, x.GetBlock(1));  
}


HypreParMatrix * ParOptProblem::Duc(const BlockVector &x) 
{ 
   return Ddg(x.GetBlock(0)); 
}

HypreParMatrix * ParOptProblem::Dmc([[maybe_unused]] const BlockVector &x) 
{ 
   return Ih;
} 

ParOptProblem::~ParOptProblem() 
{
  delete[] dofOffsetsU;
  delete[] dofOffsetsM;
  delete Ih;
}






ReducedProblem::ReducedProblem(ParOptProblem * problem_, HYPRE_Int * constraintMask)
{
  problem = problem_;
  J = nullptr;
  P = nullptr;
  
  // int nprocs = Mpi::WorldSize();
  int myrank = Mpi::WorldRank();

  HYPRE_BigInt * dofOffsets = problem->GetDofOffsetsU();

  // given a constraint mask, lets update the constraintOffsets
  // from the original problem
  int nLocConstraints = 0;
  int nProblemConstraints = problem->GetDimM();
  for (int i = 0; i < nProblemConstraints; i++)
  {
    if (constraintMask[i] == 1)
    {
      nLocConstraints += 1;
    }
  }

  HYPRE_BigInt * constraintOffsets_reduced;
  constraintOffsets_reduced = offsetsFromLocalSizes(nLocConstraints);


  for (int i = 0; i < 2; i++)
  {
    cout << "constraintOffsetsReduced_" << i << " = " << constraintOffsets_reduced[i] << ", (rank = " << myrank << ")\n";
  }

  HYPRE_BigInt * constraintOffsets;
  constraintOffsets = offsetsFromLocalSizes(nProblemConstraints);
  for (int i = 0; i < 2; i++)
  {
    cout << "constraintOffsets_" << i << " = " << constraintOffsets[i] << ", (rank = " << myrank << ")\n";
  }
  
  
  P = GenerateProjector(constraintOffsets, constraintOffsets_reduced, constraintMask);

  Init(dofOffsets, constraintOffsets_reduced);
  delete[] constraintOffsets_reduced;
  delete[] constraintOffsets;
}

ReducedProblem::ReducedProblem(ParOptProblem * problem_, HypreParVector & constraintMask)
{
  problem = problem_;
  J = nullptr;
  P = nullptr;
  
  // int nprocs = Mpi::WorldSize();
  int myrank = Mpi::WorldRank();

  HYPRE_BigInt * dofOffsets = problem->GetDofOffsetsU();

  // given a constraint mask, lets update the constraintOffsets
  // from the original problem
  int nLocConstraints = 0;
  int nProblemConstraints = problem->GetDimM();
  for (int i = 0; i < nProblemConstraints; i++)
  {
    if (constraintMask[i] == 1)
    {
      nLocConstraints += 1;
    }
  }

  HYPRE_BigInt * constraintOffsets_reduced;
  constraintOffsets_reduced = offsetsFromLocalSizes(nLocConstraints);


  for (int i = 0; i < 2; i++)
  {
    cout << "constraintOffsetsReduced_" << i << " = " << constraintOffsets_reduced[i] << ", (rank = " << myrank << ")\n";
  }

  HYPRE_BigInt * constraintOffsets;
  constraintOffsets = offsetsFromLocalSizes(nProblemConstraints);
  for (int i = 0; i < 2; i++)
  {
    cout << "constraintOffsets_" << i << " = " << constraintOffsets[i] << ", (rank = " << myrank << ")\n";
  }
  
  
  P = GenerateProjector(constraintOffsets, constraintOffsets_reduced, constraintMask);

  Init(dofOffsets, constraintOffsets_reduced);
  delete[] constraintOffsets_reduced;
  delete[] constraintOffsets;
}

// energy objective E(d)
double ReducedProblem::E(const Vector &d, int & eval_err)
{
  return problem->E(d, eval_err);
}


// gradient of energy objective
void ReducedProblem::DdE(const Vector &d, Vector & gradE)
{
  problem->DdE(d, gradE);
}


HypreParMatrix * ReducedProblem::DddE(const Vector &d)
{
  return problem->DddE(d);
}

void ReducedProblem::g(const Vector &d, Vector &gd, int & eval_err)
{
  Vector gdfull(problem->GetDimM()); gdfull = 0.0;
  problem->g(d, gdfull, eval_err);
  P->Mult(gdfull, gd);
}


HypreParMatrix * ReducedProblem::Ddg(const Vector &d)
{
  HypreParMatrix * Jfull = problem->Ddg(d);
  if (J != nullptr)
  {
    delete J; J = nullptr;
  }
  J = ParMult(P, Jfull, true);
  return J;
}

ReducedProblem::~ReducedProblem()
{
  delete P;
  if (J != nullptr)
  {
    delete J;
  }
}


