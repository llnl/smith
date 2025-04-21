#include "mfem.hpp"
#include "OptProblems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#ifndef PROBLEM_DEFS
#define PROBLEM_DEFS

/* Abstract GeneralNLMCProblem class
 * to describe the nonlinear mixed complementarity problem
 * 0 <= x \perp F(x, y) >= 0
 *              Q(x, y)  = 0
 * where NLMC stands for nonlinear mixed complementarity 
 */
class GeneralNLMCProblem
{
protected:
   int dimx, dimy;
   HYPRE_BigInt dimxglb, dimyglb;
   HYPRE_BigInt * dofOffsetsx;
   HYPRE_BigInt * dofOffsetsy;
public:
   GeneralNLMCProblem();
   virtual void Init(HYPRE_BigInt * dofOffsetsx_, HYPRE_BigInt * dofOffsetsy_);
   virtual void F(const Vector &x, const Vector &y, Vector &feval, int &eval_err) const = 0;
   virtual void Q(const Vector &x, const Vector &y, Vector &qeval, int &eval_err) const = 0;
   virtual HypreParMatrix * DxF(const Vector &x, const Vector &y) = 0;
   virtual HypreParMatrix * DyF(const Vector &x, const Vector &y) = 0;
   virtual HypreParMatrix * DxQ(const Vector &x, const Vector &y) = 0;
   virtual HypreParMatrix * DyQ(const Vector &x, const Vector &y) = 0;
   int GetDimx() const { return dimx; };
   int GetDimy() const { return dimy; }; 
   HYPRE_BigInt GetDimxGlb() const { return dimxglb; };
   HYPRE_BigInt GetDimyGlb() const { return dimyglb; };
   HYPRE_BigInt * GetDofOffsetsx() const { return dofOffsetsx; };
   HYPRE_BigInt * GetDofOffsetsy() const { return dofOffsetsy; }; 
   ~GeneralNLMCProblem();
};


class OptNLMCProblem : public GeneralNLMCProblem
{
protected:
   ParOptProblem * optproblem;
   HypreParMatrix * dFdx;
   HypreParMatrix * dFdy;
   HypreParMatrix * dQdx;
   HypreParMatrix * dQdy;
public:
   OptNLMCProblem(ParOptProblem * problem_);
   void F(const Vector &x, const Vector &y, Vector &feval, int &eval_err) const;
   void Q(const Vector &x, const Vector &y, Vector &qeval, int &eval_err) const;
   HypreParMatrix * DxF(const Vector &x, const Vector &y);
   HypreParMatrix * DyF(const Vector &x, const Vector &y);
   HypreParMatrix * DxQ(const Vector &x, const Vector &y);
   HypreParMatrix * DyQ(const Vector &x, const Vector &y);
   ParOptProblem * GetOptProblem() { return optproblem;  };
   ~OptNLMCProblem();    
};



#endif
