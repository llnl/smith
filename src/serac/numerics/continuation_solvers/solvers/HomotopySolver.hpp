#include "mfem.hpp"
#include "../problems/NLMCProblems.hpp"
#include "../utilities.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#ifndef HomotopySOLVER
#define HomotopySOLVER

class HomotopySolver
{
protected:
   // problem and sizes of local/global vectors
   GeneralNLMCProblem * problem;
   int dimx, dimy;
   HYPRE_BigInt dimxglb, dimyglb;
   HYPRE_BigInt * dofOffsetsx, * dofOffsetsy; // owned by problem
   
   double tol = 1.e-4;


   // BlockVector X;
   Array<int> block_offsets_xsy;
   

   // pointers to various HypreParMatrix
   // solver will not own these pointers
   // memory management should be handled by 
   // problem class 
   HypreParMatrix * dFdx, * dFdy, * dQdx, * dQdy;

   HypreParMatrix * JGxx, * JGxs, * JGsx, * JGss, * JGsy, * JGyx,  * JGyy; 
   // Homotopy variable/parameters (eq 12.)
   double theta0 = 0.9;
   const double p = 1.5;
   const double q = 1.0;
   Vector gammax, gammay;
   Vector ax, bx, cx, cy;

   // filter
   Array<Vector *> filter;
   
   double gammaf = 1.e-4;

   const double delta0 = 1.0;
   const double delta_MAX = 1.e5;
   const double kappa_delta = 1.e2;
   const double eta1 = 0.2;
   const double eta2 = 0.5;
   // neighborhood parameters
   const double f0 = 100.0;
   const double fbeta = 100.0;
   
   double beta0 = 1.e5;
   double beta1 = fbeta * beta0;
   
   bool useNeighborhood1 = false;
   
   const double alg_nu = 0.75;
   const double alg_rho = 1.75;


   const double epsgrad = 1.e-4;
   const double deltabnd = 1.e10;
   const double feps = 1.e6;
   bool converged;
   int max_outer_iter = 100;
   int jOpt;

   // linear algebra
   // flag which controls which linear solver is used
   // to solve the unsymmetric Newton system
   int linSolveOption = 0;

   int MyRank;
   bool iAmRoot;
public:
   HomotopySolver(GeneralNLMCProblem * problem_);
   void Mult(const Vector & x0, const Vector & y0, Vector & xf, Vector & yf);
   bool GetConverged() const {  return converged;  };
   double E(const BlockVector & X, int & Eeval_err);
   void G(const BlockVector & X, const double theta, BlockVector & GX, int &Geval_err);
   void Residual(const BlockVector & X, const double theta, BlockVector & r, int &reval_err);
   void ResidualFromG(const BlockVector & GX, const double theta, BlockVector & r);
   void PredictorResidual(const BlockVector & X, const double theta, const double thetaplus, BlockVector & r, int & reval_err);
   void JacG(const BlockVector & X, const double theta, BlockOperator & JacG);
   void NewtonSolve(BlockOperator & JkOp, const BlockVector & rk, BlockVector & dXN);
   void DogLeg(const BlockOperator & JkOp, const BlockVector & gk, const double delta, const BlockVector & dXN, BlockVector & dXtr);
   bool FilterCheck(const Vector & r_comp_norm);
   void UpdateFilter(const Vector & r_comp_norm);
   void ClearFilter();
   bool NeighborhoodCheck(const BlockVector & X, const BlockVector & r, const double theta, const double beta, double & betabar_);
   bool NeighborhoodCheck_1(const BlockVector & X, const BlockVector & r, const double theta, const double beta, double & betabar_);
   bool NeighborhoodCheck_2(const BlockVector & X, const BlockVector & r, const double theta, const double beta, double & betabar_);
   void SetTol(double tol_) { tol = tol_; };
   void SetMaxIter(int max_outer_iter_) { max_outer_iter = max_outer_iter_; };
   virtual ~HomotopySolver();
};


#endif
