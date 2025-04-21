#include "mfem.hpp"
#include "NLMCProblems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



GeneralNLMCProblem::GeneralNLMCProblem() 
{ 
  dofOffsetsx = nullptr;
  dofOffsetsy = nullptr;
}

void GeneralNLMCProblem::Init(HYPRE_BigInt * dofOffsetsx_, HYPRE_BigInt * dofOffsetsy_)
{
  dofOffsetsx = new HYPRE_BigInt[2];
  dofOffsetsy = new HYPRE_BigInt[2];
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsx[i] = dofOffsetsx_[i];
    dofOffsetsy[i] = dofOffsetsy_[i];
  }
  dimx = dofOffsetsx[1] - dofOffsetsx[0];
  dimy = dofOffsetsy[1] - dofOffsetsy[0];
  
  MPI_Allreduce(&dimx, &dimxglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimy, &dimyglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}



GeneralNLMCProblem::~GeneralNLMCProblem() 
{ 
   if (dofOffsetsx != nullptr)
   {
      delete[] dofOffsetsx;
   }
   if (dofOffsetsy != nullptr)
   {
      delete[] dofOffsetsy;
   }
}


// ------------------------------------


OptNLMCProblem::OptNLMCProblem(ParOptProblem * optproblem_)
{
   optproblem = optproblem_;
   
   // x = dual variable
   // y = primal variable
   Init(optproblem->GetDofOffsetsM(), optproblem->GetDofOffsetsU());

   {
      Vector temp(dimx); 
      temp = 0.0;
      dFdx = GenerateHypreParMatrixFromDiagonal(dofOffsetsx, temp);
   }
   dFdy = nullptr;
   dQdx = nullptr;
   dQdy = nullptr; 
}

// F(x, y) = g(y)
void OptNLMCProblem::F(const Vector & x, const Vector & y, Vector & feval, int & eval_err) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && feval.Size() == dimx, "OptNLMCProblem::F -- Inconsistent dimensions");
  optproblem->g(y, feval, eval_err);
}




// Q(x, y) = \nabla_y L(y, x) = \nabla_y E(y) - (dg(y)/ dy)^T x
void OptNLMCProblem::Q(const Vector & x, const Vector & y, Vector & qeval, int &eval_err) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimy, "OptNLMCProblem::Q -- Inconsistent dimensions");
  
  optproblem->DdE(y, qeval);
  
  HypreParMatrix * J = optproblem->Ddg(y);
  Vector temp(dimy); temp = 0.0;
  J->MultTranspose(x, temp);
  
  eval_err = 0;
  qeval.Add(-1.0, temp);
}


// dF/dx = 0
HypreParMatrix * OptNLMCProblem::DxF([[maybe_unused]] const Vector & x, [[maybe_unused]] const Vector & y)
{
   return dFdx;
}

// dF/dy = dg/dy
HypreParMatrix * OptNLMCProblem::DyF([[maybe_unused]] const Vector & x, const Vector & y)
{
   return optproblem->Ddg(y);
}


// dQ/dx = -(dg/dy)^T
HypreParMatrix * OptNLMCProblem::DxQ([[maybe_unused]] const Vector & x, const Vector & y)
{
   // HypreParMatrix data is not owned by J
   HypreParMatrix * J = optproblem->Ddg(y);
   if (dQdx != nullptr)
   {
      delete dQdx;
   }
   dQdx = J->Transpose();
   Vector temp(dimy); temp = -1.0;
   dQdx->ScaleRows(temp);
   return dQdx;
}


// dQdy = Hessian(E) - second order derivaives in g
HypreParMatrix * OptNLMCProblem::DyQ([[maybe_unused]] const Vector & x, const Vector & y)
{
   return optproblem->DddE(y);
}



OptNLMCProblem::~OptNLMCProblem()
{
   delete dFdx;
   if (dQdx != nullptr)
   {
      delete dQdx;
   }
}

