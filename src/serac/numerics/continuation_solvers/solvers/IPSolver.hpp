#include "mfem.hpp"
#include "../problems/OptProblems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#ifndef PARIPSOLVER
#define PARIPSOLVER

class ParInteriorPointSolver {
 protected:
  ParGeneralOptProblem* problem;
  double OptTol;
  int max_iter;
  double mu_k;  // \mu_k
  Vector lk, zlk;

  double sMax, kSig, tauMin, eta, thetaMin, delta, sTheta, sPhi, kMu, thetaMu;
  double thetaMax, kSoc, gTheta, gPhi, kEps;

  // filter
  Array<double> F1, F2;

  // quantities computed in lineSearch
  double alpha, alphaz;
  double thx0, thxtrial;
  double phx0, phxtrial;
  bool descentDirection, switchCondition, sufficientDecrease, lineSearchSuccess, inFilterRegion;

  int dimU, dimM, dimC;
  int dimUGlb, dimMGlb, dimCGlb;
  Array<int> block_offsetsumlz, block_offsetsuml, block_offsetsx;
  Vector ml;

  HypreParMatrix *Huu, *Hum, *Hmu, *Hmm, *Wmm, *D, *Ju, *Jm, *JuT, *JmT;

  Solver* linSolver;

  int jOpt;
  bool converged;

  int MyRank;
  bool iAmRoot;

  bool saveLogBarrierIterates;
  bool savedLogBarrierSol;
  double muLogBarrierSol;
  Vector uLogBarrierSol, mLogBarrierSol, lLogBarrierSol, zlLogBarrierSol;

  bool initializedm;
  bool initializedl;
  bool initializedzl;
  Vector minit, linit, zlinit;

 public:
  ParInteriorPointSolver(ParGeneralOptProblem*);
  double MaxStepSize(Vector&, Vector&, Vector&, double);
  double MaxStepSize(Vector&, Vector&, double);
  void Mult(const BlockVector&, BlockVector&);
  void Mult(const Vector&, Vector&);
  void GetLagrangeMultiplier(Vector&);
  void FormIPNewtonMat(BlockVector&, Vector&, Vector&, BlockOperator&);
  void IPNewtonSolve(BlockVector&, Vector&, Vector&, Vector&, BlockVector&, double);
  void lineSearch(BlockVector&, BlockVector&, double);
  void projectZ(const Vector&, Vector&, double);
  void filterCheck(double, double);
  double E(const BlockVector&, const Vector&, const Vector&, double, bool);
  double E(const BlockVector&, const Vector&, const Vector&, bool);
  bool GetConverged() const;
  double theta(const BlockVector&, int&);
  double phi(const BlockVector&, double, int&);
  double theta(const BlockVector&);
  double phi(const BlockVector&, double);
  void Dxphi(const BlockVector&, double, BlockVector&);
  double L(const BlockVector&, const Vector&, const Vector&);
  void DxL(const BlockVector&, const Vector&, const Vector&, BlockVector&);
  void SetTol(double);
  void SetMaxIter(int);
  void SetBarrierParameter(double);
  void GetNumIterations(int&);
  void InitializeM(Vector&);
  void InitializeL(Vector&);
  void InitializeZl(Vector&);
  void GetLogBarrierU(Vector&);
  void GetLogBarrierM(Vector&);
  void GetLogBarrierL(Vector&);
  void GetLogBarrierZl(Vector&);
  void GetLogBarrierMu(double&);
  void SetLogBarrierMu(double&);
  void SetLinearSolver(Solver&);
  virtual ~ParInteriorPointSolver();
};

#endif
