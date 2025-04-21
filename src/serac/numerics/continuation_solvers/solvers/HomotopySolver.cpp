#include "mfem.hpp"
#include "HomotopySolver.hpp"
#include <fstream>
#include <iostream>
#ifdef MFEM_USE_STRUMPACK
#include <StrumpackOptions.hpp>
#include <mfem/linalg/strumpack.hpp>
#endif

using namespace std;
using namespace mfem;


HomotopySolver::HomotopySolver(GeneralNLMCProblem * problem_) : problem(problem_), block_offsets_xsy(4),
	dFdx(nullptr), dFdy(nullptr), dQdx(nullptr), dQdy(nullptr), JGxx(nullptr), JGxs(nullptr), JGsx(nullptr),
   JGss(nullptr), JGsy(nullptr), JGyx(nullptr), JGyy(nullptr), filter(0)
{
   dimx = problem->GetDimx();
   dimy = problem->GetDimy();
   dimxglb = problem->GetDimxGlb();
   dimyglb = problem->GetDimyGlb();
   dofOffsetsx = problem->GetDofOffsetsx();
   dofOffsetsy = problem->GetDofOffsetsy();

   block_offsets_xsy[0] = 0;
   block_offsets_xsy[1] = dimx; // x
   block_offsets_xsy[2] = dimx; // s
   block_offsets_xsy[3] = dimy; // y
   block_offsets_xsy.PartialSum();

   gammax.SetSize(dimx); gammax = 1.0;
   gammay.SetSize(dimy); gammay = 1.0;
   ax.SetSize(dimx); ax = 1.0;
   bx.SetSize(dimx); bx = 1.e-6;
   cx.SetSize(dimx); cx = 0.0;
   cy.SetSize(dimy); cy = 0.0;

   converged = false; 
    
   MyRank = Mpi::WorldRank();
   iAmRoot = MyRank == 0 ? true : false;
}


void HomotopySolver::Mult(const Vector & x0, const Vector & y0, Vector & xf, Vector & yf)
{
   BlockVector  Xk(block_offsets_xsy);  Xk = 0.0;
   BlockVector  Xtrial(block_offsets_xsy); Xtrial = 0.0;
   BlockVector dXNk(block_offsets_xsy); dXNk = 0.0;
   BlockVector dXtrk(block_offsets_xsy); dXtrk = 0.0;
   BlockVector GX(block_offsets_xsy); GX = 0.0;
   BlockVector GX0(block_offsets_xsy); GX0 = 0.0;
   BlockVector rk(block_offsets_xsy); rk = 0.0;
   BlockVector rktrial(block_offsets_xsy); rktrial = 0.0;
   BlockVector rklinear(block_offsets_xsy); rklinear = 0.0;
   BlockVector gk(block_offsets_xsy); gk = 0.0; // grad(mk) = grad_z(||rk + Jk * z||)|_{z=0} = 2 Jk^T rk
   BlockOperator JGX(block_offsets_xsy, block_offsets_xsy);
   Xk.GetBlock(0).Set(1.0, x0);
   Xk.GetBlock(1).Set(0.0, x0); // s0 = 0
   Xk.GetBlock(2).Set(1.0, y0);
   

   jOpt = 0;
   double theta = theta0;
   double delta = delta0;
   double rhok  = 0.0;
   bool inFilterRegion = false;
   bool inNeighborhood = false;
   const int max_tr_centering = 30;
   bool tr_centering;
   
   int Geval_err;
   int reval_err;
   int Feval_err;
   int Qeval_err;
   int Eeval_err;
   
   // in Cosmin's matlab code this is fx and fy
   Vector F0(dimx); F0 = 0.0;
   Vector Q0(dimy); Q0 = 0.0;
   problem->F(Xk.GetBlock(0), Xk.GetBlock(2), F0, Feval_err);
   problem->Q(Xk.GetBlock(0), Xk.GetBlock(2), Q0, Qeval_err);
   MFEM_VERIFY(Feval_err == 0 && Qeval_err == 0, "unsuccessful evaluation of F and/or Q at initial point of Homotopy solver");

   double cx_scale = GlobalLpNorm(2, F0.Norml2(), MPI_COMM_WORLD);
   double cy_scale = GlobalLpNorm(2, Q0.Norml2(), MPI_COMM_WORLD);
   cx_scale = max(1.0, sqrt(cx_scale));
   cy_scale = max(1.0, sqrt(cy_scale));
   cx = cx_scale;
   cy = cy_scale;
   
   double opt_err; // optimality error
   double betabar;
   while (jOpt < max_outer_iter)
   {
      opt_err = E(Xk, Eeval_err);
      MFEM_VERIFY(Eeval_err == 0, "error in evaluation of optimality error E, should not occur\n");
      if (iAmRoot)
      {
         cout << "-----------------\n";
         cout << "jOpt = " << jOpt << endl;
         cout << "optimality error = " << opt_err << endl;
      }
      if (opt_err < tol)
      {
         if (iAmRoot)
	 {
            cout << "NMCP solver converged!\n";
	 }
	 converged = true;
	 break;
      }
      tr_centering = true;
      
      G(Xk, theta, GX, Geval_err);
      ResidualFromG(GX, theta, rk); 
      JacG(Xk, theta, JGX);
      
      if (iAmRoot)
      {
	 cout << "delta = " << delta << endl;
	 cout << "||rk||_2 = " << rk.Norml2() << ", (theta = " << theta << ")\n";
      }
      
      NewtonSolve(JGX, rk, dXNk); // Newton direction, associated to equation rk = 0
     
      JGX.MultTranspose(rk, gk); gk *= 2.0; // gradient of quadratic-model (\nabla_{dX}(||rk + Jk * dX||_2)^2)_{|dX=0}= 2 Jk^T rk
      DogLeg(JGX, gk, kappa_delta * delta_MAX, dXNk, dXtrk);
      
      // compute trial point
      Xtrial.Set(1.0, Xk);
      Xtrial.Add(1.0, dXtrk);


      Residual(Xtrial, theta, rktrial, reval_err);
      Vector rktrial_comp_norm(3); rktrial_comp_norm = 0.0;
      for (int i = 0; i < 3; i++)
      {
         rktrial_comp_norm(i) = GlobalLpNorm(2, rktrial.GetBlock(i).Norml2(), MPI_COMM_WORLD);
      }
      inFilterRegion = FilterCheck(rktrial_comp_norm);
      inNeighborhood = NeighborhoodCheck(Xtrial, rktrial, theta, beta1, betabar);
      if (iAmRoot)
      {
         if (inFilterRegion)
         {
            cout << "cenGN -- trial point in filter region\n";
         }
         if (inNeighborhood)
         {
            cout << "cenGN -- trial point in beta1 neighborhood\n";
         }
	 if (!inFilterRegion && inNeighborhood)
	 {
	    cout << "cenGN -- skipping TR-centering\n";
	 }
      }
      
      if (!inFilterRegion && inNeighborhood)
      {
	 UpdateFilter(rktrial_comp_norm);
	 tr_centering = false;
	 Xk.Set(1.0, Xtrial);
      }

      for (int itr_centering = 0; itr_centering < max_tr_centering; itr_centering++)
      {
	 MFEM_VERIFY(delta > 1.e-30, "loss of accuracy in dog-leg (TR radius too small)");
         if (!tr_centering)
	 {
	    break;
	 }
	 DogLeg(JGX, gk, delta, dXNk, dXtrk);
	 
	 
	 Xtrial.Set(1.0, Xk);
	 Xtrial.Add(1.0, dXtrk);
	 Residual(Xtrial, theta, rktrial, reval_err);
	 if (reval_err > 0)
	 {
	    if (iAmRoot)
	    {
	       cout << "TRcen -- bad evaluation of residual\n";
	    }   
	    delta *= 0.5;
	    continue;
	 }
         
	 // linearized residual, rk + Jk * dX 
	 JGX.Mult(dXtrk, rklinear);
         rklinear.Add(1.0, rk);
         /* evaluate the reduction in the objective
	  * (|| r(x) ||_2)^2
	  * and the reduction predicted from the
	  * linearized form
	  * (|| rk + Jk * dx||_2)^2 
	  */	 
	 double rk_sqrnorm       = InnerProduct(MPI_COMM_WORLD, rk, rk);
         double rktrial_sqrnorm  = InnerProduct(MPI_COMM_WORLD, rktrial, rktrial);
         double rklinear_sqrnorm = InnerProduct(MPI_COMM_WORLD, rklinear, rklinear);
         double pred_decrease   = rk_sqrnorm - rklinear_sqrnorm; // || rk ||_2^2 - || rk + Jk * dx||_2^2
	 double actual_decrease = rk_sqrnorm - rktrial_sqrnorm;  // || r(Xk) ||_2^2 - || r(Xk + dX) ||_2^2
	 rhok = actual_decrease / pred_decrease;
	 if (iAmRoot)
	 {
	    cout << "-*-*-*-*-*-*-*-*-*\n";
	    cout << "TRcen -- delta = " << delta << endl;
	    cout << "TRcen -- predicted decrease = " << pred_decrease << endl;
	    cout << "TRcen -- actual decrease = " << actual_decrease << endl;
	 }
	 MFEM_VERIFY(pred_decrease > 0., "Loss of accuracy in dog-leg");

	 if (rhok < eta1)
	 {
	    for (int i = 0; i < 3; i++)
	    {
	       rktrial_comp_norm(i) = GlobalLpNorm(2, rktrial.GetBlock(i).Norml2(), MPI_COMM_WORLD);
	    }
            inFilterRegion = FilterCheck(rktrial_comp_norm);
	    inNeighborhood = NeighborhoodCheck(Xtrial, rktrial, theta, beta1, betabar);
	    if (inFilterRegion && iAmRoot)
	    {
	       cout << "TRcen -- in filter region\n";
	    }
	    if (!inNeighborhood && iAmRoot)
	    {
	       cout << "TRcen -- not in beta1 neighborhood\n";
	    }
	    if (!inFilterRegion && inNeighborhood)
	    {
	       UpdateFilter(rktrial_comp_norm);
	       delta *= 0.5;
	       Xk.Set(1.0, Xtrial);
	       if (iAmRoot)
	       {
	          cout << "TRcen -- accepted trial point, decreasing TR-radius\n";
	       }
	       break;
	    }
	    else
	    {
	       delta *= 0.5;
	       if (iAmRoot)
	       {
	          cout << "TRcen -- rejected trial point, decreasing TR-radius\n";
	       }
	       continue;
	    }
	 }
	 else if (rhok < eta2)
	 {
	    Xk.Set(1.0, Xtrial);
	    if (iAmRoot)
	    {
	       cout << "TRcen -- accepted trial point\n";
	    }	       
	    break;
	 }
	 else
	 {
	    if (iAmRoot)
	    {
	       cout << "TRcen -- accepted trial point, potentially increasing TR-radius\n";
	    }
	    delta = min(2.0 * delta, delta_MAX);
	    Xk.Set(1.0, Xtrial);
	    break;
	 }
      } 
      
      jOpt += 1;
      JacG(Xk, theta, JGX);

      Residual(Xk, theta, rk, reval_err);
      MFEM_VERIFY(reval_err == 0, "bad residual evaluation, this should have been caught\n");
      // Centrality management and targeting predictor step
      // zk \in N(theta, beta0)
      inNeighborhood = NeighborhoodCheck(Xk, rk, theta, beta0, betabar);
      if (inNeighborhood)
      {
         if (iAmRoot)
	 {
	    cout << "CenManagement -- reducing homotopy parameter\n";
	 }
	 double thetaplus = min(alg_nu * theta, pow(theta, alg_rho)); 
	 double t = 1.0;
	 double theta_t;
	 // compute predictor direction
         BlockVector rkp(block_offsets_xsy); rkp = 0.0;
	 BlockVector dXp(block_offsets_xsy); dXp = 0.0;
	 BlockVector Xtrialp(block_offsets_xsy); Xtrialp = 0.0;
	 
	 PredictorResidual(Xk, theta, thetaplus, rkp, reval_err);
	 MFEM_VERIFY(reval_err == 0, "bad residual evaluation, this should have been caught\n");
         NewtonSolve(JGX, rkp, dXp);
         // line search
	 bool linesearch_converged = false;
	 int max_linesearch_steps = 60;
	 int i_linesearch = 0;
	 while ( !(linesearch_converged) && i_linesearch < max_linesearch_steps)
	 {
	    theta_t = (1.0 - t) * theta + t * thetaplus;
	    if (t < 1.e-8)
	    {
	       if (iAmRoot)
	       {      
	          cout << "CenManagement -- predictor step length too small\n";
	       }
	       theta = 0.9 * theta;
	       break;
	    }
	    Xtrialp.Set(1.0, Xk);
	    Xtrialp.Add(t, dXp);
	    Residual(Xtrialp, theta_t, rktrial, reval_err);
	    if (reval_err > 0)
	    {
	       if (iAmRoot)
	       {
	           cout << "CenManagement -- bad evaluation of residual\n";
	       }   
	       continue;
	    }
	    inNeighborhood = NeighborhoodCheck(Xtrialp, rktrial, theta_t, beta1, betabar);
	    if (inNeighborhood) 
	    {
	       // accept the trial point
	       linesearch_converged = true;
	       Xk.Set(1.0, Xtrialp);
	       theta = theta_t;
	       delta = delta_MAX;
	       ClearFilter();
	       if (iAmRoot)
	       {
	          cout << "CenManagement -- accepted linesearch trial point\n";
	          cout << "CenManagement -- theta = " << theta << endl;
	       }
	    }
	    else
	    {
	       t = 0.995 * pow(t, 3.0);
	       if (iAmRoot)
	       {
	          cout << "CenManagement -- not in neighborhood\n";
	          cout << "CenManagement -- reducing t\n";
	          cout << "CenManagement -- t = " << t << endl;
	          cout << "CenManagement -- thetaplus = " << thetaplus << endl;
	       }
	    }
	    i_linesearch += 1;
	 }
      }
      else
      {
         if (iAmRoot)
	 {
	    cout << "CenManagement -- skipping\n";
	    cout << "CenManagement -- applying heuristics for quick termination resolution\n";
	 }
	 beta0 = fbeta * betabar;
	 beta1 = fbeta * beta0;
	 // compute grad(||r||^2) = ...
         JGX.MultTranspose(rk, gk); gk *= 2.0; // gradient of quadratic-model (\nabla_{dX}(||rk + Jk * dX||_2)^2)_{|dX=0}= 2 Jk^T rk
         double gk_norm = GlobalLpNorm(2, gk.Norml2(), MPI_COMM_WORLD);
	 if (gk_norm < theta * epsgrad)
	 {
	    if (iAmRoot)
	    {
	       cout << "Exiting -- converged to a local stationary point of ||rk||_2^2\n";
	    }
	    break;
	 }
	 else 
	 {	 
	    double Xk_norm = GlobalLpNorm(infinity(), Xk.Normlinf(), MPI_COMM_WORLD);
	    if (Xk_norm > deltabnd)
            {
	       if (iAmRoot)
	       {
	          cout << "Exiting -- iterates are unbounded\n";
	       }
	       break;
	    }
	    else
	    {
	       G(Xk, 0., GX0, Geval_err);
	       double GX0_norm = GlobalLpNorm(2, GX0.Norml2(), MPI_COMM_WORLD);
	       if (GX0_norm > feps * tol && theta < tol)
	       {
	          if (iAmRoot)
		  {
		     cout << "Exiting -- convergence to a non-interior point\n";
		  } 
		  break;
	       }
	    }
	 }

      }
   }
   xf.Set(1.0, Xk.GetBlock(0));
   yf.Set(1.0, Xk.GetBlock(2));
}



/* 
 *                     [ x + s - \sqrt( (x - s)^2 + 4(\theta * a)^q) ]
 * G(x, s, y; theta) = [ s - (F(x, y) + (\theta * \gamma_x)^p x)     ]
 *                     [ Q(x, y) + (\theta * \gamma_y)^p y           ] 
 */


void HomotopySolver::G(const BlockVector & X, const double theta, BlockVector & GX, int &Geval_err)
{
   int Feval_err = 0;
   int Qeval_err = 0;
   Vector tempx(dimx); tempx = 0.0;
   // compute sqrt((x-s)^2 + 4(theta a)^q) term
   for (int i = 0; i < dimx; i++)
   {
      tempx(i) = sqrt(pow(X(i) - X(i+dimx), 2) + 4.0 * pow(theta * ax(i), q)); 
   }
   GX.GetBlock(0).Set( 1.0, X.GetBlock(0));
   GX.GetBlock(0).Add( 1.0, X.GetBlock(1));
   GX.GetBlock(0).Add(-1.0, tempx);

   tempx = 0.0;
   problem->F(X.GetBlock(0), X.GetBlock(2), tempx, Feval_err);
   for (int i = 0; i < dimx; i++)
   {
      tempx(i) += pow(theta * gammax(i), p) * X(i);
   }
   GX.GetBlock(1).Set( 1.0, X.GetBlock(1));
   GX.GetBlock(1).Add(-1.0, tempx);
   
   Vector tempy(dimy); tempy = 0.0;
   problem->Q(X.GetBlock(0), X.GetBlock(2), tempy, Qeval_err);
   for (int i = 0; i < dimy; i++)
   {
      tempy(i) += pow(theta * gammay(i), p) * X.GetBlock(2)(i);
   }
   GX.GetBlock(2).Set(1.0, tempy);
   Geval_err = max(Feval_err, Qeval_err);
}

double HomotopySolver::E(const BlockVector &X, int & Eeval_err)
{
   Vector x(dimx); x = 0.0;
   Vector s(dimx); s = 0.0;
   x.Set(1.0, X.GetBlock(0));
   s.Set(1.0, X.GetBlock(1));
   Vector xsc(dimx); xsc = 0.0;
   Vector ssc(dimx); ssc = 0.0;
   for (int i = 0; i < dimx; i++)
   {
      xsc(i) = max(1.0, abs(x(i)) / f0);
      ssc(i) = max(1.0, abs(s(i)) / f0);
   }
   double xsc_1norm = GlobalLpNorm(1, xsc.Norml1(), MPI_COMM_WORLD);
   double ssc_1norm = GlobalLpNorm(1, ssc.Norml1(), MPI_COMM_WORLD);

   double xsc_infnorm = GlobalLpNorm(infinity(), xsc.Normlinf(), MPI_COMM_WORLD);
   double ssc_infnorm = GlobalLpNorm(infinity(), ssc.Normlinf(), MPI_COMM_WORLD);
   double Msc = max(xsc_infnorm, ssc_infnorm);
   double fz = f0;
   if( dimxglb > 0 ) {
     fz = max(xsc_1norm, ssc_1norm) / ( static_cast<double>(dimxglb) ); 
   }

   BlockVector r0(block_offsets_xsy); r0 = 0.0;
   Residual(X, 0.0, r0, Eeval_err); // residual at \theta = 0

   Array<double> r0_infnorms(3);
   for (int i = 0; i < 3; i++)
   {
      r0_infnorms[i] = GlobalLpNorm(infinity(), r0.GetBlock(i).Normlinf(), MPI_COMM_WORLD);
   }
   Vector xs(dimx); xs = 0.0;
   xs.Set(1.0, x);
   xs *= s;
   double xs_infnorm = GlobalLpNorm(infinity(), xs.Normlinf(), MPI_COMM_WORLD);
   
   double Err = 0.;
   if (dimxglb > 0)
   {
      Err = max(min(r0_infnorms[0] / fz, xs_infnorm / Msc), max(r0_infnorms[1], r0_infnorms[2]) / fz);
   }
   else
   {
      Err = r0_infnorms[2];
   }
   return Err;
}



void HomotopySolver::Residual(const BlockVector & X, const double theta, BlockVector & r, int & reval_err)
{
   G(X, theta, r, reval_err);
   r.GetBlock(0).Add(-theta, bx);
   r.GetBlock(1).Add(-theta, cx);
   r.GetBlock(2).Add(-theta, cy);
}

void HomotopySolver::ResidualFromG(const BlockVector & GX, const double theta, BlockVector & r)
{
   r.Set(1.0, GX);
   r.GetBlock(0).Add(-theta, bx);
   r.GetBlock(1).Add(-theta, cx);
   r.GetBlock(2).Add(-theta, cy);
}


void HomotopySolver::PredictorResidual(const BlockVector & X, const double theta, const double thetaplus, BlockVector & r, int & reval_err)
{
   G(X, theta, r, reval_err);
   r.GetBlock(0).Add(-thetaplus, bx);
   r.GetBlock(1).Add(-thetaplus, cx);
   r.GetBlock(2).Add(-thetaplus, cy);

   Vector tempx(dimx); tempx = 0.0;
   for (int i = 0; i < dimx; i++)
   {
      tempx(i) = 2.0 * q * pow(theta * ax(i), q - 1.0);
   }
   Vector temps(dimx); temps = 0.0;
   for (int i = 0; i < dimx; i++)
   {
      temps(i) = p * pow(theta * gammax(i), p - 1) * X.GetBlock(0)(i);
   }
   Vector tempy(dimy); tempy = 0.0;
   for (int i = 0; i < dimy; i++)
   {
      tempy(i) = -p * pow(theta * gammay(i), p - 1) * X.GetBlock(2)(i);
   }
   r.GetBlock(0).Add(theta - thetaplus, tempx);
   r.GetBlock(1).Add(theta - thetaplus, temps);
   r.GetBlock(2).Add(theta - thetaplus, tempy);
}



/*                       [dG_(1,1)   dG_(1,2)   0       ]
 * \nabla G_t(x, s, y) = [dG_(2,1)   dG_(2,2)   dG_(2,3)]
 *                       [dG_(3,1)     0        dG_(3,3)]
 */
void HomotopySolver::JacG(const BlockVector &X, const double theta, BlockOperator & JacG)
{
   // I - diag( (x - s) / sqrt( (x - s)^2 + 4 * (\theta a)^q)
   Vector diagJacGxx(dimx); diagJacGxx = 0.0;
   for (int i = 0; i < dimx; i++)
   {
      diagJacGxx(i) = 1.0 - (X(i) - X(i+dimx)) / sqrt(
		     pow(X(i) - X(i+dimx), 2) + 4. * pow(theta * ax(i), q)); 
   }
   if (JGxx)
   {
      delete JGxx; JGxx = nullptr;
   }
   JGxx = GenerateHypreParMatrixFromDiagonal(dofOffsetsx, diagJacGxx);

   // I + diag( (x - s) / sqrt( (x - s)^2 + 4 (\theta a)^q)
   // = 2 I - [I - diag( (x - s) / sqrt( (x - s)^2 + 4 * (\theta a)^q)]
   Vector diagJacGxs(dimx);
   diagJacGxs = 2.0;
   diagJacGxs.Add(-1.0, diagJacGxx);
   if (JGxs)
   {
      delete JGxs; JGxs = nullptr;
   }
   JGxs = GenerateHypreParMatrixFromDiagonal(dofOffsetsx, diagJacGxs);
   
   // d / dx (G_t)_2 = -dF/dx - (t \gamma_x)^p
   if (JGsx)
   {
      delete JGsx; JGsx = nullptr;
   }
   dFdx = problem->DxF(X.GetBlock(0), X.GetBlock(2));
   Vector diagtgx(dimx); diagtgx = 0.0;
   for (int i = 0; i < dimx; i++)
   {
      diagtgx(i) = pow(theta * gammax(i), p);
   }
   HypreParMatrix * Dtgx = GenerateHypreParMatrixFromDiagonal(dofOffsetsx, diagtgx);
   JGsx = Add(-1.0, *dFdx, -1.0, *Dtgx);
   delete Dtgx;

   // d / ds (G_t)_2 = I
   Vector one(dimx); one = 1.0;
   if (JGss)
   {
      delete JGss;
   }
   JGss = GenerateHypreParMatrixFromDiagonal(dofOffsetsx, one);

   // d / dy (G_t)_2 = - dF / dy
   if (JGsy)
   {
      delete JGsy;
   }
   JGsy = new HypreParMatrix(*(problem->DyF(X.GetBlock(0), X.GetBlock(2))));
   one = -1.0;
   JGsy->ScaleRows(one);
   one = 1.0;

   // d / dx (G_t)_3 = dQ / dx
   if (JGyx)
   {
      delete JGyx;
   }
   JGyx = new HypreParMatrix(*(problem->DxQ(X.GetBlock(0), X.GetBlock(2))));


   // d / dy (G_t)_3 = dQ / dy + (t * \gamma_y)^p
   if (JGyy)
   {
      delete JGyy;   
   }
   dQdy = problem->DyQ(X.GetBlock(0), X.GetBlock(2));
   Vector diagtgy(dimy); diagtgy = 0.0;
   for (int i = 0; i < dimy; i++)
   {
      diagtgy(i) = pow(theta * gammay(i), p);
   }
   HypreParMatrix * Dtgy = GenerateHypreParMatrixFromDiagonal(dofOffsetsy, diagtgy);
   JGyy = Add(1.0, *dQdy, 1.0, *Dtgy);
   delete Dtgy; 

   JacG.SetBlock(0, 0, JGxx); JacG.SetBlock(0, 1, JGxs);
   JacG.SetBlock(1, 0, JGsx); JacG.SetBlock(1, 1, JGss); JacG.SetBlock(1, 2, JGsy);
   JacG.SetBlock(2, 0, JGyx);                            JacG.SetBlock(2, 2, JGyy);
}




// solve J dX_N = - rk
void HomotopySolver::NewtonSolve(BlockOperator & JkOp, [[maybe_unused]] const BlockVector & rk, BlockVector & dXN)
{
   int num_row_blocks = JkOp.NumRowBlocks();
   int num_col_blocks = JkOp.NumColBlocks();
   
   
   // Direct solver (default)
   MFEM_VERIFY(linSolveOption == 0, "NewtonSolve only supports a direct solver (linSolveOption == 0)");
   if(linSolveOption == 0)
   {
      Array2D<const HypreParMatrix *> JkBlockMat(num_row_blocks, num_col_blocks);
      for(int i = 0; i < num_row_blocks; i++)
      {
         for(int j = 0; j < num_col_blocks; j++)
         {
            if(!JkOp.IsZeroBlock(i, j))
            {
               JkBlockMat(i, j) = dynamic_cast<HypreParMatrix *>(&(JkOp.GetBlock(i, j)));
               MFEM_VERIFY(JkBlockMat(i, j), "dynamic cast failure");
            }
            else
            {
               JkBlockMat(i, j) = nullptr;
            }
         }
      }
      
      HypreParMatrix * Jk = HypreParMatrixFromBlocks(JkBlockMat);
      Solver * JkSolver = nullptr;
      /* direct solve of the 3x3 IP-Newton linear system */
#ifdef MFEM_USE_MUMPS
      JkSolver = new MUMPSSolver(MPI_COMM_WORLD);
      auto Jksolver = dynamic_cast<MUMPSSolver *>(JkSolver);
      Jksolver->SetPrintLevel(0);
      Jksolver->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      Jksolver->SetOperator(*Jk);
#else 
#ifdef MFEM_USE_MKL_CPARDISO
      JkSolver = new CPardisoSolver(MPI_COMM_WORLD);
      JkSolver->SetOperator(*Jk);
#else
#ifdef MFEM_USE_STRUMPACK
      JkSolver = new STRUMPACKSolver(MPI_COMM_WORLD);
      auto Jksolver = dynamic_cast<STRUMPACKSolver*>(JkSolver);
      Jksolver->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
      Jksolver->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
      STRUMPACKRowLocMatrix *Jkstrumpack = new STRUMPACKRowLocMatrix(*Jk);
      Jksolver->SetOperator(*Jkstrumpack);
#else
      MFEM_VERIFY(false, "linSolveOption = 0 will not work unless compiled mfem is with MUMPS or MKL_CPARDISO");
#endif
#endif
#endif
      JkSolver->Mult(rk, dXN);
      dXN *= -1.0;
      delete Jk;
      delete JkSolver;
#ifdef MFEM_USE_STRUMPACK
      delete Jkstrumpack;
#endif
   }
}

void HomotopySolver::DogLeg(const BlockOperator & JkOp, const BlockVector & gk, const double delta, const BlockVector & dXN, BlockVector & dXtr)
{
   double dXN_norm = GlobalLpNorm(2, dXN.Norml2(), MPI_COMM_WORLD);
   if (dXN_norm <= delta)
   {
      dXtr.Set(1.0, dXN);
      if (iAmRoot)
      {
         cout << "dog-leg using Newton direction\n";
      }
   }
   else
   {
      BlockVector dXsd(block_offsets_xsy); dXsd = 0.0;
      BlockVector Jkgk(block_offsets_xsy); Jkgk = 0.0;
      JkOp.Mult(gk, Jkgk);
      double gk_norm = GlobalLpNorm(2, gk.Norml2(), MPI_COMM_WORLD);
      double Jkgk_norm = GlobalLpNorm(2, Jkgk.Norml2(), MPI_COMM_WORLD);
      dXsd.Set(-0.5 * pow(gk_norm, 2) / pow(Jkgk_norm, 2), gk);

      // || dXsd || = 0.5 * || gk ||^3 / || Jk gk||^2
      double dXsd_norm = 0.5 * pow(gk_norm, 3) / pow(Jkgk_norm, 2);
      if (dXsd_norm >= delta)
      {
         dXtr.Set(-delta / gk_norm, gk);
	 if (iAmRoot)
	 {
	    cout << "dog-leg using steepest descent direction\n";
	 }
      }
      else
      {
	 double t_star;
	 double a, b, c;
	 double dXsdTdXN = InnerProduct(MPI_COMM_WORLD, dXN, dXsd);
	 a = pow(dXN_norm, 2) - 2.0 * dXsdTdXN + pow(dXsd_norm, 2);
	 b = 2.0 * (dXsdTdXN - pow(dXsd_norm, 2));
	 c = pow(dXsd_norm, 2) - pow(delta, 2);
	 double discr = pow(b, 2) - 4.0 * a * c;
	 MFEM_VERIFY(discr >= 0. && a >= 0., "loss of accuracy: Gauss-Newton model not convex?!?");
	 if (b > 0.)
	 {
	    t_star = (-2. * c) / (b + sqrt(discr));
	 }
	 else
	 {
	    t_star = (-b + sqrt(discr)) / (2. * a);
	 }
         dXtr.Set(t_star, dXN);
	 dXtr.Add((1.0 - t_star), dXsd);
	 if (iAmRoot)
	 {
	    cout << "dog-leg using combination of Newton and SD directions\n";
	 }
      }
   }
}

bool HomotopySolver::FilterCheck(const Vector & r_comp_norm)
{
   // for each index j = 0, 1, 2 see if 
   // ||r_j(Xtrial)||_2 < alpha_j - gamma_f * || [alpha0; alpha1; alpha2;] ||_2
   // for all (alpha0, alpha1, alpha2) in the filter
   Array<bool> reductionJ;  reductionJ.SetSize(3);
   for (int j = 0; j < 3; j++)
   {
      reductionJ[j] = true;
   }

   for (int i = 0; i < filter.Size(); i++)
   {
      MFEM_VERIFY(filter[i]->Size() == 3, "each element of filter must be a 3-vector");
      for (int j = 0; j < 3; j++)
      {
         if (r_comp_norm(j) >= filter[i]->Elem(j) - gammaf * filter[i]->Norml2())
         {
            reductionJ[j] = false;
            continue;
         }
      }
   }
   
   // the region that the filter does not permit are those
   // points for which there does not exist and i such that
   // for each (alpha1, alpha2, alpha) in F that
   // reduction is achieved ||r_theta_i|| < alpha_i - gamma_f ||(alpha1, alpha2, alpha3)||
   // if reduction is achieved then we are not in the filtered region
   // and we say that the given point is not in F 
   bool inFilteredRegion = true;
   for (int j = 0; j < 3; j++)
   {
      if (reductionJ[j])
      {
         inFilteredRegion = false;
	 break;
      }
   }
   reductionJ.SetSize(0);
   return inFilteredRegion;
}

void HomotopySolver::UpdateFilter(const Vector & r_comp_norm)
{
   filter.Append(new Vector(3));
   for (int i = 0; i < 3; i++)
   {
      filter.Last()->Elem(i) = r_comp_norm(i);
   }
}

void HomotopySolver::ClearFilter()
{
   for (int i = 0; i < filter.Size(); i++)
   {
      delete filter[i];
   }
   filter.SetSize(0);
}


bool HomotopySolver::NeighborhoodCheck(const BlockVector & X, const BlockVector & r, const double theta, const double beta, double & betabar_)
{
   if (useNeighborhood1)
   {
      return NeighborhoodCheck_1(X, r, theta, beta, betabar_);
   }
   else
   {
      return NeighborhoodCheck_2(X, r, theta, beta, betabar_);
   }
}


bool HomotopySolver::NeighborhoodCheck_1([[maybe_unused]] const BlockVector & X, const BlockVector & r, const double theta, const double beta, double & betabar_)
{
   double r_inf_norm = GlobalLpNorm(infinity(), r.Normlinf(), MPI_COMM_WORLD);
   bool inNeighborhood = (r_inf_norm <= beta * theta);
   betabar_ = r_inf_norm / theta;
   return inNeighborhood;
}


// see isInNeigh_2 method of numerial experiments branch of mHICOp
bool HomotopySolver::NeighborhoodCheck_2(const BlockVector & X, const BlockVector & r, const double theta, const double beta, double & betabar_)
{
   double x_inf_norm = GlobalLpNorm(infinity(), X.GetBlock(0).Normlinf(), MPI_COMM_WORLD);
   double s_inf_norm = GlobalLpNorm(infinity(), X.GetBlock(1).Normlinf(), MPI_COMM_WORLD);
   double xs_inf_norm = max(x_inf_norm, s_inf_norm);


   Vector r_inf_norms(3); r_inf_norms = 0.0;
   for (int i = 0; i < 3; i++)
   {
      r_inf_norms(i) = GlobalLpNorm(infinity(), r.GetBlock(i).Normlinf(), MPI_COMM_WORLD);
   }


   bool inNeighborhood = ((r_inf_norms(0) <= beta * theta * xs_inf_norm) && (max(r_inf_norms(1), r_inf_norms(2)) <= beta * theta));
   betabar_ = max(r_inf_norms(0) / (theta * xs_inf_norm), max(r_inf_norms(1), r_inf_norms(2)) / theta);
   return inNeighborhood;
}





//void HomotopySolver::GradientCheck(const BlockVector & X0, const double theta)
//{
//   BlockVector Xhat(block_offsets_xsy); Xhat = 0.0; Xhat.Randomize();
//   BlockVector Xt(block_offsets_xsy);   Xt = 0.0;
//   BlockVector GX0(block_offsets_xsy);  GX0 = 0.0;
//   BlockOperator JGX0(block_offsets_xsy, block_offsets_xsy);   
//   G(X0, theta, GX0);
//   JacG(X0, theta, JGX0);
//
//
//   BlockVector GXt(block_offsets_xsy); GXt = 0.0;
//   BlockVector R(block_offsets_xsy); R = 0.0; // residual
//   BlockVector Temp(block_offsets_xsy); Temp = 0.0;
//
//   double eps = 1.0;
//   for (int i = 0; i < 30; i++)
//   {
//      cout << "---------------\n";
//      Xt.Set(1.0, X0);
//      Xt.Add(eps, Xhat);
//      G(Xt, theta, GXt);
//      // ||J * xhat - (G(x0 + eps * xhat) - G(x0)) / eps
//      JGX0.Mult(Xhat, R);
//      Temp.Set(1.0 / eps, GXt);
//      Temp.Add(-1.0 / eps, GX0);
//      R.Add(-1.0, Temp);
//      cout << "||J * Xhat - (G(x0 - eps * xhat) - G(x0)) / eps||_2 = " << R.Norml2() << ", eps = " << eps << endl;
//      eps *= 0.5;
//   }
//}



HomotopySolver::~HomotopySolver()
{
   block_offsets_xsy.DeleteAll();
   gammax.SetSize(0);
   gammay.SetSize(0);
   ax.SetSize(0);
   bx.SetSize(0);
   cx.SetSize(0);
   cy.SetSize(0);
   if (JGxx)
   {
      delete JGxx;
   }
   if (JGxs)
   {
      delete JGxs;
   }
   if (JGsx)
   {
      delete JGsx;
   }
   if (JGss)
   {
      delete JGss;
   }
   if (JGsy)
   {
      delete JGsy;
   }
   if (JGyx)
   {
      delete JGyx;
   }
   if (JGyy)
   {
      delete JGyy;
   }
   ClearFilter();
}
