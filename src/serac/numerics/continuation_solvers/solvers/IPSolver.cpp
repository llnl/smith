#include "mfem.hpp"
#include "IPSolver.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace mfem;


ParInteriorPointSolver::ParInteriorPointSolver(ParGeneralOptProblem * problem_) 
                     : problem(problem_), 
                       block_offsetsumlz(5), block_offsetsuml(4), block_offsetsx(3),
                       Huu(nullptr), Hum(nullptr), Hmu(nullptr), 
                       Hmm(nullptr), Wmm(nullptr), D(nullptr), 
                       Ju(nullptr), Jm(nullptr), JuT(nullptr), JmT(nullptr), 
                       saveLogBarrierIterates(false)
{
   OptTol  = 1.e-2;
   max_iter = 20;
   mu_k     = 1.0;

   sMax     = 1.e2;
   kSig     = 1.e10;   // control deviation from primal Hessian
   tauMin   = 0.99;     // control rate at which iterates can approach the boundary
   eta      = 1.e-4;   // backtracking constant
   thetaMin = 1.e-4;   // allowed violation of the equality constraints

   // constants in line-step A-5.4
   delta    = 1.0;
   sTheta   = 1.1;
   sPhi     = 2.3;

   // control the rate at which the penalty parameter is decreased
   kMu     = 0.2;
   thetaMu = 1.5;

   thetaMax = 1.e6; // maximum constraint violation
   // data for the second order correction
   kSoc     = 0.99;

   // equation (18)
   gTheta = 1.e-5;
   gPhi   = 1.e-5;

   kEps   = 1.e1;

   dimU = problem->GetDimU();
   dimM = problem->GetDimM();
   dimC = problem->GetDimC();
   ckSoc.SetSize(dimC);
  
   block_offsetsumlz[0] = 0;
   block_offsetsumlz[1] = dimU; // u
   block_offsetsumlz[2] = dimM; // m
   block_offsetsumlz[3] = dimC; // lambda
   block_offsetsumlz[4] = dimM; // zl
   block_offsetsumlz.PartialSum();
  
   MPI_Allreduce(&dimU, &dimUGlb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&dimM, &dimMGlb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&dimC, &dimCGlb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   for(int i = 0; i < block_offsetsuml.Size(); i++)  
   { 
      block_offsetsuml[i] = block_offsetsumlz[i]; 
   }
   for(int i = 0; i < block_offsetsx.Size(); i++)    
   { 
      block_offsetsx[i] = block_offsetsuml[i] ; 
   }

  
   ml = problem->Getml();
  
   lk.SetSize(dimC);  lk  = 0.0;
   zlk.SetSize(dimM); zlk = 0.0;

   savedLogBarrierSol = false;
   muLogBarrierSol = 1.e-4;
   uLogBarrierSol.SetSize(dimU);
   mLogBarrierSol.SetSize(dimM);
   lLogBarrierSol.SetSize(dimC);
   zlLogBarrierSol.SetSize(dimM);
   initializedm = false;
   initializedl = false;
   initializedzl = false;
   minit.SetSize(dimM); linit.SetSize(dimC); zlinit.SetSize(dimM);

   linSolver = 0;
   linSolveTol = 1.e-8;
   MyRank = Mpi::WorldRank();
   iAmRoot = MyRank == 0 ? true : false;
}

double ParInteriorPointSolver::MaxStepSize(Vector &x, Vector &xl, Vector &xhat, double tau)
{
   double alphaMaxloc = 1.0;
   double alphaTmp;
   for(int i = 0; i < x.Size(); i++)
   {   
      if( xhat(i) < 0. )
      {
         alphaTmp = -1. * tau * (x(i) - xl(i)) / xhat(i);
         alphaMaxloc = min(alphaMaxloc, alphaTmp);
      } 
   }

   // alphaMaxloc is the local maximum step size which is
   // distinct on each MPI process. Need to compute
   // the global maximum step size 
   double alphaMaxglb;
   MPI_Allreduce(&alphaMaxloc, &alphaMaxglb, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   return alphaMaxglb;
}

double ParInteriorPointSolver::MaxStepSize(Vector &x, Vector &xhat, double tau)
{
   Vector zero(x.Size()); zero = 0.0;
   return MaxStepSize(x, zero, xhat, tau);
}


void ParInteriorPointSolver::Mult(const Vector &x0, Vector &xf)
{
   BlockVector x0block(block_offsetsx); x0block = 0.0;
   x0block.GetBlock(0).Set(1.0, x0);
   if (!(initializedm))
   {
       ParOptProblem * OptProblem = dynamic_cast<ParOptProblem *>(problem);
//    if (dimM > 0)
       {
          if (OptProblem == nullptr)
          {
              // fixed initialization     
              x0block.GetBlock(1) = 1.e2;
              x0block.GetBlock(1).Add(1.0, ml);
          }
          else
          {
             // use c(u, m) = g(u) - m
	     // if it is possible (with respect to the constraint m >= min_minit)
	     // choose m such that c(u, m) = 0
             double min_minit = 1.0;
	     x0block.GetBlock(1) = 0.0;
             x0block.GetBlock(1).Add(1.0, ml);
             Vector c0(dimC); c0 = 0.0;
             problem->c(x0block, c0);
             Vector dm(dimM); dm = 0.0;
             for (int i = 0; i < dimM; i++)
             {
                dm(i) = max(min_minit - x0block(dimU + i), c0(i));
             }
             x0block.GetBlock(1).Add(1.0, dm);
          }
       }
   }
// else
// {
//    x0block.GetBlock(1).Set(1.0, minit);
// }
   BlockVector xfblock(block_offsetsx); xfblock = 0.0;
   Mult(x0block, xfblock);
   xf.Set(1.0, xfblock.GetBlock(0));
}


void ParInteriorPointSolver::Mult(const BlockVector &x0, BlockVector &xf)
{
   converged = false;
   if(iAmRoot)
   {
     IPNewtonKrylovIters.open("data/IPNewtonKrylovIters.dat", ios::out | ios::trunc);
     muStream.open("data/MuHistory.dat", ios::out | ios::trunc);
     optHistory_mu.open("data/OptimalityHistory_mu.dat", ios::out | ios::trunc);
     optHistory_0.open("data/OptimalityHistory_0.dat", ios::out | ios::trunc);
   }
   
   BlockVector xk(block_offsetsx), xhat(block_offsetsx); xk = 0; xhat = 0.0;
   BlockVector Xk(block_offsetsumlz), Xhat(block_offsetsumlz); Xk = 0.0; Xhat = 0.0;
   BlockVector Xhatuml(block_offsetsuml); Xhatuml = 0.0;
   Vector zlhat(dimM); zlhat = 0.0;

   xk.GetBlock(0).Set(1.0, x0.GetBlock(0));
   xk.GetBlock(1).Set(1.0, x0.GetBlock(1));
   
   // running estimate of the final values of the Lagrange multipliers
   if (initializedl)
   {
      lk.Set(1.0, linit);
   }
   else
   {
      lk = 0.0;
   }
   if (initializedzl)
   {
      zlk.Set(1.0, zlinit);
   }
   else
   {
      for(int i = 0; i < dimM; i++)
      {
         zlk(i) = 1.e1 * mu_k / (xk(i+dimU) - ml(i));
      }
   }

   Xk.GetBlock(0).Set(1.0, xk.GetBlock(0));
   Xk.GetBlock(1).Set(1.0, xk.GetBlock(1));
   Xk.GetBlock(2).Set(1.0, lk);
   Xk.GetBlock(3).Set(1.0, zlk);

  /* set theta0 = theta(x0)
   *     thetaMin
   *     thetaMax
   * when theta(xk) < thetaMin and the switching condition holds
   * then we ask for the Armijo sufficient decrease of the barrier
   * objective to be satisfied, in order to accept the trial step length alphakl
   * 
   * thetaMax controls how the filter is initialized for each log-barrier subproblem
   * F0 = {(th, phi) s.t. th > thetaMax}
   * that is the filter does not allow for iterates where the constraint violation
   * is larger than that of thetaMax
   */
   double theta0 = theta(xk);
   thetaMin = 1.e-4 * max(1.0, theta0);
   thetaMax = 1.e8  * thetaMin; // 1.e4 * max(1.0, theta0)

   double Eeval, maxBarrierSolves, Eevalmu0;
   bool printOptimalityError; // control optimality error print to console for log-barrier subproblems
   
   maxBarrierSolves = 10;

   for(jOpt = 0; jOpt < max_iter; jOpt++)
   {
      if(iAmRoot)
      {
         cout << "interior-point solve step " << jOpt << endl;
      }
      // A-2. Check convergence of overall optimization problem
      printOptimalityError = false;
      Eevalmu0 = E(xk, lk, zlk, printOptimalityError);
      if(Eevalmu0 < OptTol)
      {
         converged = true;
	 if(iAmRoot)
         {
            cout << "solved optimization problem :)\n";
	    cout << "to abs tol " << OptTol << endl;
	    IPNewtonKrylovIters.close();
	    muStream.close();
	    optHistory_mu.close();
	    optHistory_0.close();
         }
         break;
      }
      
      if(jOpt > 0) { maxBarrierSolves = 1; }
      
      for(int i = 0; i < maxBarrierSolves; i++)
      {
         // A-3. Check convergence of the barrier subproblem
         printOptimalityError = true;
         Eeval = E(xk, lk, zlk, mu_k, printOptimalityError);
         if(iAmRoot)
         {
            cout << "E = " << Eeval << endl;
         }
         if(Eeval < kEps * mu_k)
         {
            if(iAmRoot)
            {
               cout << "solved barrier subproblem :), for mu = " << mu_k << endl;
            }
            // A-3.1. Recompute the barrier parameteri
            double mu_k_new = max(OptTol / 10., min(kMu * mu_k, pow(mu_k, thetaMu)));
	    //mu_k  = max(OptTol / 10., min(kMu * mu_k, pow(mu_k, thetaMu)));
	    if ( mu_k_new < muLogBarrierSol && !(savedLogBarrierSol))
	    {
	       uLogBarrierSol.Set(1.0, xk.GetBlock(0));
	       mLogBarrierSol.Set(1.0, xk.GetBlock(1));
	       lLogBarrierSol.Set(1.0, lk);
	       zlLogBarrierSol.Set(1.0, zlk);
	       savedLogBarrierSol = true;
	       muLogBarrierSol = mu_k;
	    }
	    mu_k = mu_k_new;
            // A-3.2. Re-initialize the filter
            F1.DeleteAll();
            F2.DeleteAll();
         }
         else
         {
            break;
         }
      }
    
      // A-4. Compute the search direction
      // solve for (uhat, mhat, lhat)
      if(iAmRoot)
      {
         cout << "\n** A-4. IP-Newton solve **\n";
      }
      zlhat = 0.0; Xhatuml = 0.0;
      IPNewtonSolve(xk, lk, zlk, zlhat, Xhatuml, mu_k, false); 

      // assign data stack, X = (u, m, l, zl)
      Xk = 0.0;
      Xk.GetBlock(0).Set(1.0, xk.GetBlock(0));
      Xk.GetBlock(1).Set(1.0, xk.GetBlock(1));
      Xk.GetBlock(2).Set(1.0, lk);
      Xk.GetBlock(3).Set(1.0, zlk);

      // assign data stack, Xhat = (uhat, mhat, lhat, zlhat)
      Xhat = 0.0;
      for(int i = 0; i < 3; i++)
      {
         Xhat.GetBlock(i).Set(1.0, Xhatuml.GetBlock(i));
      }
      Xhat.GetBlock(3).Set(1.0, zlhat);

      // A-5. Backtracking line search.
      if(iAmRoot)
      {
         cout << "\n** A-5. Linesearch **\n";
         cout << "mu = " << mu_k << endl;
      }
      lineSearch(Xk, Xhat, mu_k);

      if(lineSearchSuccess)
      {
         if(iAmRoot)
         {
            cout << "lineSearch successful :)\n";
         }
         if(!switchCondition || !sufficientDecrease)
         {
            F1.Append( (1. - gTheta) * thx0);
            F2.Append( phx0 - gPhi * thx0);
         }
         // ----- A-6: Accept the trial point
         xk.GetBlock(0).Add(alpha, Xhat.GetBlock(0));
         xk.GetBlock(1).Add(alpha, Xhat.GetBlock(1));
         lk.Add(alpha,   Xhat.GetBlock(2));
         zlk.Add(alphaz, Xhat.GetBlock(3));
         projectZ(xk, zlk, mu_k);
      }
      else
      {
         if(iAmRoot)
         {
            cout << "lineSearch not successful :(\n";
            cout << "attempting feasibility restoration with theta = " << thx0 << endl;
            cout << "no feasibility restoration implemented, exiting now \n";
	    IPNewtonKrylovIters.close();
	    muStream.close();
	    optHistory_mu.close();
	    optHistory_0.close();
         }
         break;
      }
      if(jOpt + 1 == max_iter && iAmRoot) 
      {  
         cout << "maximum optimization iterations :(\n";
	 IPNewtonKrylovIters.close();
	 muStream.close();
	 optHistory_mu.close();
	 optHistory_0.close();
      }
   }
   // done with optimization routine, just reassign data to xf reference so
   // that the application code has access to the optimal point
   xf = 0.0;
   xf.GetBlock(0).Set(1.0, xk.GetBlock(0));
   xf.GetBlock(1).Set(1.0, xk.GetBlock(1));
}

void ParInteriorPointSolver::FormIPNewtonMat(BlockVector & x, [[maybe_unused]] Vector & l, Vector &zl, BlockOperator &Ak)
{
   // WARNING: Huu, Hum, Hmu, Hmm should all be Hessian terms of the Lagrangian, currently we 
   //          them by Hessian terms of the objective function and neglect the Hessian of l^T c

   Huu = problem->Duuf(x); 
   Hum = problem->Dumf(x);
   Hmu = problem->Dmuf(x);
   Hmm = problem->Dmmf(x);

   Vector DiagLogBar(dimM); DiagLogBar = 0.0;
   for(int ii = 0; ii < dimM; ii++)
   {
      DiagLogBar(ii) = zl(ii) / (x(ii+dimU) - ml(ii));
   }
   if(iAmRoot)
   {
      std::ofstream diagStream;
      char diagString[100];
      snprintf(diagString, 100, "data/D%d.dat", jOpt);
      diagStream.open(diagString, ios::out | ios::trunc);
      for(int ii = 0; ii < dimM; ii++)
      {
         diagStream << setprecision(30) << DiagLogBar(ii) << endl;
      }
      diagStream.close();
      
      std::ofstream sStream;
      char sString[100];
      snprintf(sString, 100, "data/s%d.dat", jOpt);
      sStream.open(sString, ios::out | ios::trunc);
      for(int ii = 0; ii < dimM; ii++)
      {
         sStream << setprecision(30) << x(dimU+ii) << endl;
      }
      sStream.close();
   } 

   D = GenerateHypreParMatrixFromDiagonal(problem->GetDofOffsetsM(), DiagLogBar);
  
   if(Hmm != nullptr)
   {
      Wmm = Hmm;
      Wmm->Add(1.0, *D);
   }
   else
   {
      Wmm = D;
   }

   Ju = problem->Duc(x); JuT = Ju->Transpose();
   Jm = problem->Dmc(x); JmT = Jm->Transpose();
   //         IP-Newton system matrix
   //    Ak = [[H_(u,u)  H_(u,m)   J_u^T]
   //          [H_(m,u)  W_(m,m)   J_m^T]
   //          [ J_u      J_m       0  ]]

   Ak.SetBlock(0, 0, Huu);                         Ak.SetBlock(0, 2, JuT);
                           Ak.SetBlock(1, 1, Wmm); Ak.SetBlock(1, 2, JmT);
   Ak.SetBlock(2, 0,  Ju); Ak.SetBlock(2, 1,  Jm);

   if(Hum != nullptr) { Ak.SetBlock(0, 1, Hum); Ak.SetBlock(1, 0, Hmu); }
}

// perturbed KKT system solve
// determine the search direction
void ParInteriorPointSolver::IPNewtonSolve(BlockVector &x, Vector &l, Vector &zl, Vector &zlhat, BlockVector &Xhat, double mu, [[maybe_unused]] bool socSolve)
{
   int nKrylovIts = -1;
   // solve A x = b, where A is the IP-Newton matrix
   BlockOperator A(block_offsetsuml, block_offsetsuml); BlockVector b(block_offsetsuml); b = 0.0;
   FormIPNewtonMat(x, l, zl, A);

   //       [grad_u phi + Ju^T l]
   // b = - [grad_m phi + Jm^T l]
   //       [          c        ]
   BlockVector gradphi(block_offsetsx); gradphi = 0.0;
   BlockVector JTl(block_offsetsx); JTl = 0.0;
   Dxphi(x, mu, gradphi);
   
   (A.GetBlock(0,2)).Mult(l, JTl.GetBlock(0));
   (A.GetBlock(1,2)).Mult(l, JTl.GetBlock(1));

   for(int ii = 0; ii < 2; ii++)
   {
      b.GetBlock(ii).Set(1.0, gradphi.GetBlock(ii));
      b.GetBlock(ii).Add(1.0, JTl.GetBlock(ii));
   }
   //if(!socSolve) 
   //{
   problem->c(x, b.GetBlock(2));
   //}
   //else
   //{
   //   b.GetBlock(2).Set(1.0, ckSoc);
   //}
   b *= -1.0; 
   Xhat = 0.0;


   // Direct solver (default)
   if(linSolver == 0)
   {
      Array2D<const HypreParMatrix *> ABlockMatrix(3,3);
      for(int ii = 0; ii < 3; ii++)
      {
      for(int jj = 0; jj < 3; jj++)
      {
         if(!A.IsZeroBlock(ii, jj))
         {
            ABlockMatrix(ii, jj) = dynamic_cast<HypreParMatrix *>(
			           const_cast<Operator *>(&(A.GetBlock(ii, jj))));
         }
         else
         {
            ABlockMatrix(ii, jj) = nullptr;
         }
      }
      }
      
      HypreParMatrix * Ah = HypreParMatrixFromBlocks(ABlockMatrix);   
      // HypreParMatrix * Huu_print = dynamic_cast<HypreParMatrix *>(
      //   	                const_cast<Operator*>(&(A.GetBlock(0, 0))));
      // HypreParMatrix * Ju_print  = dynamic_cast<HypreParMatrix *>(
      //   	                const_cast<Operator*>(&(A.GetBlock(2, 0))));
      //std::ostringstream Huu_file_name;
      //Huu_file_name << "data/Huu" << problem->getProblemLabel();
      //Huu_print->Print(Huu_file_name.str());

      //std::ostringstream Ju_file_name;
      //Ju_file_name << "data/Ju" << problem->getProblemLabel();
      //Ju_print->Print(Ju_file_name.str());


      /* direct solve of the 3x3 IP-Newton linear system */
      #ifdef MFEM_USE_MUMPS
        MUMPSSolver ASolver;
        ASolver.SetPrintLevel(0);
        ASolver.SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_INDEFINITE);
        ASolver.SetOperator(*Ah);
        ASolver.Mult(b, Xhat);
      #else 
        #ifdef MFEM_USE_MKL_CPARDISO
          CPardisoSolver ASolver(MPI_COMM_WORLD);
          ASolver.SetOperator(*Ah);
          ASolver.Mult(b, Xhat);
	#else
	  MFEM_VERIFY(false, "linSolver 0 will not work unless compiled with MUMPS or MKL");
        #endif
      #endif

      delete Ah;
   }
   else if(linSolver == 1 || linSolver == 2)
   {
      // TO DO: print Huu and Ju with two time-steps.
      // output: solution as well
      // form A = Huu + Ju^T D Ju, Wmm = D for contact
      HypreParMatrix * Huuloc = dynamic_cast<HypreParMatrix *>(
		                const_cast<Operator*>(&(A.GetBlock(0, 0))));
      HypreParMatrix * Wmmloc = dynamic_cast<HypreParMatrix *>(
		                const_cast<Operator*>(&(A.GetBlock(1, 1))));
      HypreParMatrix * Juloc  = dynamic_cast<HypreParMatrix *>(
		                const_cast<Operator*>(&(A.GetBlock(2, 0))));
      HypreParMatrix * JuTloc = dynamic_cast<HypreParMatrix *>(
		                const_cast<Operator*>(&(A.GetBlock(0, 2))));
      
      
      HypreParMatrix *JuTDJu   = RAP(Wmmloc, Juloc);     // Ju^T D Ju
      HypreParMatrix *Areduced = ParAdd(Huuloc, JuTDJu);  // Huu + Ju^T D Ju
      /* prepare the reduced rhs */
      // breduced = bu + Ju^T (bm + Wmm bl)
      Vector breduced(dimU); breduced = 0.0;
      Vector tempVec(dimM); tempVec = 0.0;
      Wmmloc->Mult(b.GetBlock(2), tempVec);
      tempVec.Add(1.0, b.GetBlock(1));
      JuTloc->Mult(tempVec, breduced);
      breduced.Add(1.0, b.GetBlock(0));
      
      if(linSolver == 1)
      {
         // setup the solver for the reduced linear system
         #ifdef MFEM_USE_MUMPS
	   MUMPSSolver AreducedSolver;   
           AreducedSolver.SetPrintLevel(0);
           AreducedSolver.SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_INDEFINITE);
	   AreducedSolver.SetOperator(*Areduced);
	   AreducedSolver.Mult(breduced, Xhat.GetBlock(0));
         #else 
           #ifdef MFEM_USE_MKL_CPARDISO
	     CPardisoSolver AreducedSolver(MPI_COMM_WORLD);
	     AreducedSolver.SetOperator(*Areduced);
	     AreducedSolver.Mult(breduced, Xhat.GetBlock(0));
           #else
	     MFEM_VERIFY(false, "linSolver 1 will not work unless compiled with MUMPS or MKL");
           #endif
         #endif
      }
      else
      {
         HyprePCG AreducedSolver(MPI_COMM_WORLD);
         //HypreGMRES AreducedSolver(MPI_COMM_WORLD);
	 AreducedSolver.SetOperator(*Areduced);
	 HypreBoomerAMG AreducedPrec;
         AreducedPrec.SetSystemsOptions(3, false);
	 AreducedPrec.SetRelaxType(8);
	 AreducedSolver.SetTol(linSolveTol);
         AreducedSolver.SetMaxIter(1000);
         AreducedSolver.SetPreconditioner(AreducedPrec);
         //AreducedSolver.SetResidualConvergenceOptions(); // convergence criteria based on residual norm
	 AreducedSolver.SetPrintLevel(2);
         AreducedSolver.Mult(breduced, Xhat.GetBlock(0));
	 AreducedSolver.GetNumIterations(nKrylovIts);
      }

      // now propagate solved uhat to obtain mhat and lhat
      // xm = Ju xu - bl
      Juloc->Mult(Xhat.GetBlock(0), Xhat.GetBlock(1));
      Xhat.GetBlock(1).Add(-1.0, b.GetBlock(2));

      // xl = Wmm xm - bm
      Wmmloc->Mult(Xhat.GetBlock(1), Xhat.GetBlock(2));
      Xhat.GetBlock(2).Add(-1.0, b.GetBlock(1));

      delete JuTDJu;
      delete Areduced;
   }

   /* backsolve to determine zlhat */
   for(int ii = 0; ii < dimM; ii++)
   {
      zlhat(ii) = -1.*(zl(ii) + (zl(ii) * Xhat(ii + dimU) - mu) / (x(ii + dimU) - ml(ii)) );
   }
   
   if (iAmRoot)
   {
     IPNewtonKrylovIters << nKrylovIts << endl;
     muStream << setprecision(30) << mu << endl; 
   }

   // free memory
   delete D;
   delete JuT;
   delete JmT;
   if(Hmm != nullptr)
   {
      delete Wmm;
   }


   // check to see if this is a small step
   int locLargeStep = 0;
   int glbLargeStep = 0;
   for(int ii = 0; ii < dimU; ii++)
   {
     if(abs(Xhat(ii)) / (1. + abs(x(ii))) > 1.e-15)
     {
        locLargeStep = 1;
	break;
     }	     
   }
   if (locLargeStep == 0)
   {
      for(int ii = 0; ii < dimM; ii++)
      {
         if(abs(Xhat(ii+dimU)) / (1. + abs(x(ii+dimU))) > 1.e-15)
	 {
	    locLargeStep = 1;
	    break;
	 }
      }
   }

   MPI_Allreduce(&locLargeStep, &glbLargeStep, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   if (iAmRoot && glbLargeStep < 1)
   {
      cout << "SMALL STEP" << endl;
   } 
   double xnorm_inf = GlobalLpNorm(infinity(), x.Normlinf(), MPI_COMM_WORLD); 
   double xhat_norm_inf = max(GlobalLpNorm(infinity(), Xhat.GetBlock(0).Normlinf(), MPI_COMM_WORLD),
		              GlobalLpNorm(infinity(), Xhat.GetBlock(1).Normlinf(), MPI_COMM_WORLD));

   if (iAmRoot)
   {
      cout << "||x||_inf = " << xnorm_inf << endl;
      cout << "||xhat||_inf = " << xhat_norm_inf << endl;
   }
}

// here Xhat, X will be BlockVectors w.r.t. the 4 partitioning X = (u, m, l, zl)

void ParInteriorPointSolver::lineSearch(BlockVector& X0, BlockVector& Xhat, double mu)
{
   //double tau  = max(tauMin, 1.0 - mu);
   double tau = tauMin;
   Vector u0   = X0.GetBlock(0);
   Vector m0   = X0.GetBlock(1);
   Vector l0   = X0.GetBlock(2);
   Vector z0   = X0.GetBlock(3);
   Vector uhat = Xhat.GetBlock(0);
   Vector mhat = Xhat.GetBlock(1);
   Vector lhat = Xhat.GetBlock(2);
   Vector zhat = Xhat.GetBlock(3);
   double alphaMax  = MaxStepSize(m0, ml, mhat, tau);
   double alphaMaxz = MaxStepSize(z0, zhat, tau);
   alphaz = alphaMaxz;

   BlockVector x0(block_offsetsx); x0 = 0.0;
   x0.GetBlock(0).Set(1.0, u0);
   x0.GetBlock(1).Set(1.0, m0);
   
   BlockVector xhat(block_offsetsx); xhat = 0.0;
   xhat.GetBlock(0).Set(1.0, uhat);
   xhat.GetBlock(1).Set(1.0, mhat);
   
   BlockVector xtrial(block_offsetsx); xtrial = 0.0;
   BlockVector Dxphi0(block_offsetsx); Dxphi0 = 0.0;
   int maxBacktrack = 20;
   alpha = alphaMax;

   Vector ck0(dimC); ck0 = 0.0;
   Vector zhatsoc(dimM); zhatsoc = 0.0;
   BlockVector Xhatumlsoc(block_offsetsuml); Xhatumlsoc = 0.0;
   BlockVector xhatsoc(block_offsetsx); xhatsoc = 0.0;
   Vector uhatsoc(dimU); uhatsoc = 0.0;
   Vector mhatsoc(dimM); mhatsoc = 0.0;

   // ParOptProblem * optProblem = dynamic_cast<ParOptProblem *>(problem);
// if (optProblem != nullptr)
// {
//    Vector d0(dimU); Vector dhat(dimU); Vector dtrial(dimU);
//    d0.Set(1.0, x0.GetBlock(0));
//    dhat.Set(1.0, xhat.GetBlock(0));
//    dtrial = 0.0;
//
//    double E0, Eplus;
//    Vector gradE(dimU);
//    int eval_err;
//    E0 = optProblem->E(d0, eval_err); 
//    optProblem->DdE(d0, gradE);
//    
//    double gradE_dhat = InnerProduct(MPI_COMM_WORLD, dhat, gradE);
//    double eps_step = 1.0;
//    double fd_err = 0.0;
//      for(int iii = 0; iii < 31; iii++)
//      {
//         if( iii == 30 ) eps_step = 1.0e-40; 
//         dtrial.Set(1.0, d0);
//         dtrial.Add(eps_step, dhat);
//         Eplus = optProblem->E(dtrial, eval_err);
//         fd_err = abs((eps_step*gradE_dhat - (Eplus - E0) ) / eps_step);
//         if (iAmRoot)
//         {
//            double diff = abs(eps_step*((Eplus-E0)/eps_step-gradE_dhat));
//            cout << std::scientific;
//            cout << std::setprecision(14);
//            if( iii < 10 ) {
//              cout << "iii  " << iii << " eps " << eps_step << " eps*gradE_dhat " << eps_step*gradE_dhat << " Eplus " << Eplus << " E0 " << E0 << " absdiff " << diff <<  endl;
//            } else {
//              cout << "iii "  << iii << " eps " << eps_step << " eps*gradE_dhat " << eps_step*gradE_dhat << " Eplus " << Eplus << " E0 " << E0 << " absdiff " << diff <<  endl;
//            }
//          //cout << "|grad(E)^T dhat - (E(d0 + eps * dhat) - E(d0))/eps| = " << fd_err << endl;
//          //cout << "grad(E)^T dhat = " << gradE_dhat << endl;
//          //cout << "(E(d0 + eps * dhat) - E(d0)) / eps = " << (Eplus - E0) / eps_step << endl;
//          //cout << "E(d0) = " << E0 << endl;
//          //cout << "E(d0 + eps * dhat) = " << Eplus << endl;
//          //cout << "E(d0 + eps * dhat) - E(d0) = " << Eplus - E0 << endl;
//          //cout << "eps = " << eps_step << "\n\n";
//         }
//         eps_step /= 2.0;
//      }
//   }
   Dxphi(x0, mu, Dxphi0);

   int th_eval_err; int ph_eval_err;
   Dxphi0_xhat = InnerProduct(MPI_COMM_WORLD, Dxphi0, xhat);
   double xhat_norm = sqrt(InnerProduct(MPI_COMM_WORLD, xhat, xhat));
   double Dxphi0_norm = sqrt(InnerProduct(MPI_COMM_WORLD, Dxphi0, Dxphi0));
   descentDirection = Dxphi0_xhat < 0. ? true : false;
   if (iAmRoot)
   {
      cout << "grad(phi)^T xhat / (||grad(phi)|| * ||xhat||) = " << Dxphi0_xhat / (xhat_norm * Dxphi0_norm) << endl;
      cout << "|grad(phi)^T xhat| = " << abs(Dxphi0_xhat) << endl;
      if(descentDirection)
      {
         cout << "is a descent direction for the log-barrier objective\n";
      }
      else
      {
         cout << "is not a descent direction for the log-barrier objective\n";
      }
   }
   thx0 = theta(x0);
   phx0 = phi(x0, mu);

   lineSearchSuccess = false;
   for(int i = 0; i < maxBacktrack; i++)
   {
      if (iAmRoot)
      {
         cout << "\n--------- alpha = " << alpha << " ---------\n";
      }
      // ----- A-5.2. Compute trial point: xtrial = x0 + alpha_i xhat
      xtrial.Set(1.0, x0);
      xtrial.Add(alpha, xhat);

      // ------ A-5.3. if not in filter region go to A.5.4 otherwise go to A-5.5.
      thxtrial = theta(xtrial, th_eval_err);
      phxtrial = phi(xtrial, mu, ph_eval_err);
      if (iAmRoot)
      {
         cout << "| grad(phi)^xhat - (phi(x0 + alpha xhat) - phi(x0)) / alpha | = " << abs(Dxphi0_xhat - (phxtrial - phx0) / alpha) << ", alpha = " << alpha << endl;
      }

      if (!(th_eval_err == 0 && ph_eval_err == 0))
      {
        if(iAmRoot)
	{
	  cout << "BAD STEP: reducing step length\n";
	}
	alpha *= 0.5;
	continue;
      }
      filterCheck(thxtrial, phxtrial);    
      if(!inFilterRegion)
      {
         if (iAmRoot)
         {
            cout << "not in filter region :)\n";
         }
         // ------ A.5.4: Check sufficient decrease
         if(!descentDirection)
         {
            switchCondition = false;
         }
         else
         {
            switchCondition = (alpha * pow(abs(Dxphi0_xhat), sPhi) > delta * pow(thx0, sTheta)) ? true : false;
         }
         if (iAmRoot)
         {
            cout << setprecision(30) << "theta(x0) = "     << thx0     << ", thetaMin = "                  << thetaMin             << endl;
            cout << "theta(xtrial) = " << thxtrial << ", (1-gTheta) *theta(x0) = "     << (1. - gTheta) * thx0 << endl;
            cout << "phi(xtrial) = "   << phxtrial << ", phi(x0) = " << phx0 << ", phi(x0) - gPhi *theta(x0) = " << phx0 - gPhi * thx0   << endl;
         }      
         // Case I      
         if(thx0 <= thetaMin && switchCondition)
         {
            sufficientDecrease = (phxtrial <= phx0 + eta * alpha * Dxphi0_xhat) ? true : false;
            if(sufficientDecrease)
            {
               if(iAmRoot) { cout << "Line search successful: sufficient decrease in log-barrier objective.\n"; } 
               // accept the trial step
               lineSearchSuccess = true;
               break;
            }
            else
            {
               if(iAmRoot) { cout << "sufficient decrease not achieved in log-barrier objective.\n";}
            }
         }
         else
         {
            if(thxtrial <= (1. - gTheta) * thx0 || phxtrial <= phx0 - gPhi * thx0)
            {
               if(iAmRoot) { cout << "Line search successful: infeasibility or log-barrier objective decreased.\n"; } 
               // accept the trial step
               lineSearchSuccess = true;
               break;
            }
         }
         // A-5.5: Initialize the second-order correction
         //if((!(thx0 < thxtrial)) && i == 0)
         //{
         //   if (iAmRoot)
         //   {
         //      cout << "second order correction\n";
         //   }
         //   problem->c(xtrial, ckSoc);
         //   problem->c(x0, ck0);
         //   ckSoc.Add(alphaMax, ck0);
         //   // A-5.6 Compute the second-order correction.
         //   IPNewtonSolve(x0, l0, z0, zhatsoc, Xhatumlsoc, mu, true);
         //   mhatsoc.Set(1.0, Xhatumlsoc.GetBlock(1));
         //   //WARNING: not complete but currently solver isn't entering this region
         //}
      }
      else
      {
         if (iAmRoot)
         {
            cout << "in filter region :(\n"; 
         }
      }
      alpha *= 0.5;

   } 
}


void ParInteriorPointSolver::projectZ(const Vector &x, Vector &z, double mu)
{
   double zi;
   double mudivmml;
   for(int i = 0; i < dimM; i++)
   {
      zi = z(i);
      mudivmml = mu / (x(i + dimU) - ml(i));
      z(i) = max(min(zi, kSig * mudivmml), mudivmml / kSig);
   }
}

void ParInteriorPointSolver::filterCheck(double th, double ph)
{
   inFilterRegion = false;
   if(th > thetaMax)
   {
      inFilterRegion = true;
   }
   else
   {
      for(int i = 0; i < F1.Size(); i++)
      {
         if(th >= F1[i] && ph >= F2[i])
         {
            inFilterRegion = true;
            break;
         }
      }
   }
}

double ParInteriorPointSolver::E(const BlockVector &x, const Vector &l, const Vector &zl, double mu, bool printEeval)
{
   double E1, E2, E3;
   double sc, sd;
   BlockVector gradL(block_offsetsx); gradL = 0.0; // stationarity grad L = grad f + J^T l - z
   Vector cx(dimC); cx = 0.0;     // feasibility c = c(x)
   Vector comp(dimM); comp = 0.0; // complementarity M Z - mu 1

   DxL(x, l, zl, gradL);
   E1 = GlobalLpNorm(infinity(), gradL.Normlinf(), MPI_COMM_WORLD);

   problem->c(x, cx);
   E2 = GlobalLpNorm(infinity(), cx.Normlinf(), MPI_COMM_WORLD); 

   for(int ii = 0; ii < dimM; ii++) 
   { 
      comp(ii) = x(dimU + ii) * zl(ii) - mu;
   }
   E3 = GlobalLpNorm(infinity(), comp.Normlinf(), MPI_COMM_WORLD); 
   double ll1, zl1;

   zl1 = GlobalLpNorm(1, zl.Norml1(), MPI_COMM_WORLD); 
   ll1 = GlobalLpNorm(1, l.Norml1(), MPI_COMM_WORLD);
   sc = max(sMax, zl1 / (double(dimMGlb)) ) / sMax;
   sd = max(sMax, (ll1 + zl1) / (double(dimCGlb + dimMGlb))) / sMax;
   if(iAmRoot && printEeval)
   {
      cout << "evaluating optimality error for mu = " << mu << endl;
      cout << "stationarity measure = "    << E1 / sd << endl;
      cout << "feasibility measure  = "    << E2      << endl;
      cout << "complimentarity measure = " << E3 / sc << endl;
   }
   if (iAmRoot)
   {
      
      if(mu < 1.e-14)
      {
         optHistory_0 << E1 << " " << E2 << " " << E3 << endl;
      }
      else
      {
         optHistory_mu << E1 << " " << E2 << " " << E3 << endl;
      }
   }
   return max(max(E1 / sd, E2), E3 / sc);
}

double ParInteriorPointSolver::E(const BlockVector &x, const Vector &l, const Vector &zl, bool printEeval)
{
  return E(x, l, zl, 0.0, printEeval);
}

double ParInteriorPointSolver::theta(const BlockVector &x, int & eval_err)
{
  Vector cx(dimC); cx = 0.0;
  problem->c(x, cx, eval_err);
  return GlobalLpNorm(2, cx.Norml2(), MPI_COMM_WORLD);
}

double ParInteriorPointSolver::theta(const BlockVector &x)
{
  int eval_err; // throw away
  return theta(x, eval_err);
}

// log-barrier objective
double ParInteriorPointSolver::phi(const BlockVector &x, double mu, int & eval_err)
{
   double fx = problem->CalcObjective(x, eval_err); 
   double logBarrierLoc = 0.0;
   for(int i = 0; i < dimM; i++) 
   { 
     logBarrierLoc += log(x(dimU+i) - ml(i));
   }
   double logBarrierGlb;
   MPI_Allreduce(&logBarrierLoc, &logBarrierGlb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   return fx - mu * logBarrierGlb;
}

double ParInteriorPointSolver::phi(const BlockVector &x, double mu)
{
  int eval_err; // throw away
  return phi(x, mu, eval_err);
}

// gradient of log-barrier objective with respect to x = (u, m)
void ParInteriorPointSolver::Dxphi(const BlockVector &x, double mu, BlockVector &y)
{
   problem->CalcObjectiveGrad(x, y);
   
   for(int i = 0; i < dimM; i++) 
   { 
      y(dimU + i) -= mu / (x(dimU + i));
   } 
}

// Lagrangian function evaluation
// L(x, l, zl) = f(x) + l^T c(x) - zl^T m
double ParInteriorPointSolver::L(const BlockVector &x, const Vector &l, const Vector &zl)
{
   double fx = problem->CalcObjective(x);
   Vector cx(dimC); problem->c(x, cx);
   return (fx + InnerProduct(MPI_COMM_WORLD, cx, l) - InnerProduct(MPI_COMM_WORLD, x.GetBlock(1), zl));
}

void ParInteriorPointSolver::DxL(const BlockVector &x, const Vector &l, const Vector &zl, BlockVector &y)
{
   // evaluate the gradient of the objective with respect to the primal variables x = (u, m)
   BlockVector gradxf(block_offsetsx); gradxf = 0.0;
   problem->CalcObjectiveGrad(x, gradxf);
   
   HypreParMatrix *Jacu, *Jacm, *JacuT, *JacmT;
   Jacu = problem->Duc(x); 
   Jacm = problem->Dmc(x);
   JacuT = Jacu->Transpose();
   JacmT = Jacm->Transpose();
   
   JacuT->Mult(l, y.GetBlock(0));
   JacmT->Mult(l, y.GetBlock(1));
   
   delete JacuT;
   delete JacmT;
   
   y.Add(1.0, gradxf);
   (y.GetBlock(1)).Add(-1.0, zl);
}

bool ParInteriorPointSolver::GetConverged() const
{
   return converged;
}

void ParInteriorPointSolver::SetTol(double Tol)
{
   OptTol = Tol;
}

void ParInteriorPointSolver::SetMaxIter(int max_it)
{
   max_iter = max_it;
}

void ParInteriorPointSolver::SetBarrierParameter(double mu_0)
{
   mu_k = mu_0;
}

void ParInteriorPointSolver::SaveLogBarrierHessianIterates(bool save)
{
   MFEM_ASSERT(MyRank == 0 || save == false, "currently can only save logbarrier hessian in serial codes");
   saveLogBarrierIterates = save;
}

void ParInteriorPointSolver::SetLinearSolver(int LinSolver)
{
   linSolver = LinSolver;
}

void ParInteriorPointSolver::SetLinearSolveTol(double Tol)
{
  linSolveTol = Tol;
}

void ParInteriorPointSolver::GetLagrangeMultiplier(Vector & y)
{
  y.SetSize(dimM); y = 0.;
  y.Set(1.0, zlk);
}

void ParInteriorPointSolver::InitializeM(Vector & m0)
{
  minit.Set(1.0, m0);
  initializedm = true;
}

void ParInteriorPointSolver::InitializeL(Vector & l0)
{
  linit.Set(1.0, l0);
  initializedl = true;
}

void ParInteriorPointSolver::InitializeZl(Vector & z0)
{
  zlinit.Set(1.0, z0);
  initializedzl = true;
}

void ParInteriorPointSolver::GetLogBarrierU(Vector & uLogBar)
{
  uLogBar.Set(1.0, uLogBarrierSol);
}

void ParInteriorPointSolver::GetLogBarrierM(Vector & mLogBar)
{
  mLogBar.Set(1.0, mLogBarrierSol);
}

void ParInteriorPointSolver::GetLogBarrierL(Vector & lLogBar)
{
  lLogBar.Set(1.0, lLogBarrierSol);
}

void ParInteriorPointSolver::GetLogBarrierZl(Vector & zlLogBar)
{
  zlLogBar.Set(1.0, zlLogBarrierSol);
}

void ParInteriorPointSolver::GetNumIterations(int & its)
{
  its = jOpt;
}

void ParInteriorPointSolver::GetLogBarrierMu(double & mu)
{
  mu = muLogBarrierSol;
}

void ParInteriorPointSolver::SetLogBarrierMu(double mu)
{
  muLogBarrierSol = mu;
}

ParInteriorPointSolver::~ParInteriorPointSolver() 
{
   F1.DeleteAll();
   F2.DeleteAll();
   block_offsetsx.DeleteAll();
   block_offsetsumlz.DeleteAll();
   block_offsetsuml.DeleteAll();
   ml.SetSize(0);
}
