#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../utilities.hpp"

using namespace std;
using namespace mfem;

#ifndef PARPROBLEM_DEFS
#define PARPROBLEM_DEFS

// abstract ParGeneralOptProblem class
// of the form
// min_(u,m) f(u,m) s.t. c(u,m)=0 and m>=ml
// the primal variable (u, m) is represented as a BlockVector
class ParGeneralOptProblem
{
protected:
    int dimU, dimM, dimC;
    int dimUglb, dimMglb;
    HYPRE_BigInt * dofOffsetsU;
    HYPRE_BigInt * dofOffsetsM;
    Array<int> block_offsetsx;
    Vector ml;
    int label;
public:
    ParGeneralOptProblem();
    virtual void Init(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_);
    virtual double CalcObjective(const BlockVector &, int &) = 0;
    double CalcObjective(const BlockVector &) ;
    virtual void Duf(const BlockVector &, Vector &) = 0;
    virtual void Dmf(const BlockVector &, Vector &) = 0;
    void CalcObjectiveGrad(const BlockVector &, BlockVector &);
    virtual HypreParMatrix * Duuf(const BlockVector &) = 0;
    virtual HypreParMatrix * Dumf(const BlockVector &) = 0;
    virtual HypreParMatrix * Dmuf(const BlockVector &) = 0;
    virtual HypreParMatrix * Dmmf(const BlockVector &) = 0;
    virtual HypreParMatrix * Duc(const BlockVector &) = 0;
    virtual HypreParMatrix * Dmc(const BlockVector &) = 0;
    virtual void c(const BlockVector &, Vector &, int &) = 0;
    void c(const BlockVector &, Vector &) ;
    int GetDimU() const { return dimU; };
    int GetDimM() const { return dimM; }; 
    int GetDimC() const { return dimC; };
    int GetDimUGlb() const { return dimUglb; };
    int GetDimMGlb() const { return dimMglb; };
    HYPRE_BigInt * GetDofOffsetsU() const { return dofOffsetsU; };
    HYPRE_BigInt * GetDofOffsetsM() const { return dofOffsetsM; }; 
    Vector Getml() const { return ml; };
    void setProblemLabel(int label_) { label = label_; };
    int getProblemLabel() { return label; };
    ~ParGeneralOptProblem();
};


// abstract ContactProblem class
// of the form
// min_d e(d) s.t. g(d) >= 0
class ParOptProblem : public ParGeneralOptProblem
{
protected:
    HypreParMatrix * Ih;
public:
    ParOptProblem();
    void Init(HYPRE_BigInt *, HYPRE_BigInt *);
    
    // ParGeneralOptProblem methods are defined in terms of
    // ParOptProblem specific methods: E, DdE, DddE, g, Ddg
    double CalcObjective(const BlockVector &, int &) ; 
    void Duf(const BlockVector &, Vector &) ;
    void Dmf(const BlockVector &, Vector &) ;
    HypreParMatrix * Duuf(const BlockVector &);
    HypreParMatrix * Dumf(const BlockVector &);
    HypreParMatrix * Dmuf(const BlockVector &);
    HypreParMatrix * Dmmf(const BlockVector &);
    void c(const BlockVector &, Vector &, int &) ;
    HypreParMatrix * Duc(const BlockVector &);
    HypreParMatrix * Dmc(const BlockVector &);
    
    // ParOptProblem specific methods:
    
    // energy objective function e(d)
    // input: d an mfem::Vector
    // output: e(d) a double
    virtual double E(const Vector &d, int &) = 0;
    // gradient of energy objective De / Dd
    // input: d an mfem::Vector,
    //        gradE an mfem::Vector, which will be the gradient of E at d
    // output: none    
    virtual void DdE(const Vector &d, Vector &gradE) = 0;
  
    // Hessian of energy objective D^2 e / Dd^2
    // input:  d, an mfem::Vector
    // output: The Hessian of the energy objective at d, a pointer to a HypreParMatrix
    virtual HypreParMatrix * DddE(const Vector &d) = 0;

    // Constraint function g(d) >= 0, e.g., gap function
    // input: d, an mfem::Vector,
    //       gd, an mfem::Vector, which upon successfully calling the g method will be
    //                            the evaluation of the function g at d
    // output: none
    virtual void g(const Vector &d, Vector &gd, int &) = 0;
    // Jacobian of constraint function Dg / Dd, e.g., gap function Jacobian
    // input:  d, an mfem::Vector,
    // output: The Jacobain of the constraint function g at d, a pointer to a HypreParMatrix
    virtual HypreParMatrix * Ddg(const Vector &) = 0;
    virtual ~ParOptProblem();
};






class ReducedProblem : public ParOptProblem
{
protected:
  HypreParMatrix *J;
  HypreParMatrix *P; // projector
  ParOptProblem  *problem;
public:
  ReducedProblem(ParOptProblem *problem, HYPRE_Int * constraintMask);
  ReducedProblem(ParOptProblem *problem, HypreParVector & constraintMask);
  double E(const Vector &, int &);
  void DdE(const Vector &, Vector &);
  HypreParMatrix * DddE(const Vector &);
  void g(const Vector &, Vector &, int &);
  HypreParMatrix * Ddg(const Vector &);
  ParOptProblem * GetProblem() {  return problem; }
  virtual ~ReducedProblem();
};




#endif
