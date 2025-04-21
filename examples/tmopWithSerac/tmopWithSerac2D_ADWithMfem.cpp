// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file without_input_file.cpp
 *
 * @brief A simple example of steady-state heat transfer that uses
 * the C++ API to configure the simulation
 */

#include "mfem.hpp"

#include <serac/infrastructure/terminator.hpp>
#include <serac/numerics/functional/functional.hpp>
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include <serac/physics/solid_mechanics.hpp>
#include <serac/physics/state/state_manager.hpp>
#include <serac/numerics/functional/domain.hpp>
#include <serac/numerics/stdfunction_operator.hpp>
#include "serac/mesh/mesh_utils.hpp"
#include <algorithm>
#include <cfenv>
#include <memory>
#include <numeric>
#include <functional>
#include <mfem/linalg/tensor.hpp>

//////////////////////////////////////////////
//////////////////////////////////////////////
struct CuboidLSF2D { 
  double x0;
  double y0; 
  double radius; 
  double exponent; 

  template < typename T >
  T SDF(const serac::tensor<T, 2> & x) const {
    using std::pow;
    return pow(pow(x[0]-x0, exponent) + pow(x[1]-y0, exponent), 1.0/exponent) - radius;
  }

  template < typename T >
  serac::tensor<T, 2> GRAD(const serac::tensor<T, 2> & x) const{
    using std::pow;
    auto dphi = 0.0*x;
    dphi[0] = (x[0] - x0)* pow( pow(x[0]-x0, exponent) + pow(x[1]-y0, exponent), 1.0/exponent-1);
    dphi[1] = (x[1] - y0)* pow( pow(x[0]-x0, exponent) + pow(x[1]-y0, exponent), 1.0/exponent-1);
    return dphi;
  }
};
//////////////////////////////////////////////
//////////////////////////////////////////////

// Define your level set function as a class
template <typename T1, typename T2, typename T3>
class LevelSetFunction
{
public:
    // Constructor to initialize parameters
    LevelSetFunction(T1 &cx, T1 &cy, T1 &r ) 
        : centerX(cx), centerY(cy), radius(r) {}

    // Override the value method to define your level set function
    void Eval(T2 &x, T1 &value)
    {
        auto phiVal = pow(pow(x[0]-centerX, 2.0) + pow(x[1]-centerY, 2.0), 0.5) - radius;
        value = phiVal; // Level set function value
    }

    // Method to calculate the gradient (derivative) at a given point
    T3 CalculateGradient(T1 &x)
    {
        auto dphi = 0.0*x; // dphidx*dxdu
        dphi[0] = (x[0] - centerX)* pow( pow(x[0]-centerX, 2.0) + pow(x[1]-centerY, 2.0), -0.5);
        dphi[1] = (x[1] - centerY)* pow( pow(x[0]-centerX, 2.0) + pow(x[1]-centerY, 2.0), -0.5);
        return dphi;
    }

private:
    T1 radius;      // Radius of the circle
    T1 centerX;    // X-coordinate of the center
    T1 centerY;    // Y-coordinate of the center
};
//////////////////////////////////////////////////
//////////////////////////////////////////////////
/// MFEM native AD-type for first derivatives
typedef ::mfem::internal::dual<::mfem::real_t, ::mfem::real_t> ADFType;
/// MFEM native AD-type for second derivatives
typedef ::mfem::internal::dual<ADFType, ADFType> ADSType;

::mfem::real_t ADVal_func( const ::mfem::Vector &p, std::function<ADFType(const std::vector<ADFType>&)> func)
{
  int numParams = p.Size();
  // int matsize = numParams;
   std::vector<ADFType> adinp(numParams);
   for (int i=0; i<numParams; i++) { adinp[i] = ADFType{p[i], 0.0}; }
 
   return func(adinp).value;
}

void ADGrad_func(const ::mfem::Vector &p, std::function<ADFType(const std::vector<ADFType>&)> func, ::mfem::Vector &grad)
{  
   int numParams = p.Size();
   std::vector<ADFType> adinp(numParams);
   for (int i=0; i<numParams; i++) { adinp[i] = ADFType{p[i], 0.0}; }
   for (int i=0; i<numParams; i++)
   {
      adinp[i] = ADFType{p[i], 1.0};
      ADFType rez = func(adinp);
      grad[i] = rez.gradient;
      adinp[i] = ADFType{p[i], 0.0};
   }
}

void ADHessian_func(const ::mfem::Vector &p, std::function<ADSType(const std::vector<ADSType>&)> func, ::mfem::DenseMatrix &H)
{
   int numParams = p.Size();
   //use forward-forward mode
   std::vector<ADSType> aduu(numParams);
   for (int ii = 0; ii < numParams; ii++)
   {
      aduu[ii].value = ADFType{p[ii], 0.0};
      aduu[ii].gradient = ADFType{0.0, 0.0};
   }

   for (int ii = 0; ii < numParams; ii++)
   {
      aduu[ii].value = ADFType{p[ii], 1.0};
      for (int jj = 0; jj < (ii + 1); jj++)
      {

        aduu[jj].gradient = ADFType{1.0, 0.0};
        ADSType rez = func(aduu);
        H(ii,jj) = rez.gradient.gradient;
        H(jj,ii) = rez.gradient.gradient;
        aduu[jj].gradient = ADFType{0.0, 0.0};
      }
      aduu[ii].value = ADFType{p[ii], 0.0};
   }
   return;
}

//example functions
template <typename type>
auto func_example(const std::vector<type>& x ) -> type {
    double x0=0.0;
    double y0=0.0;
    double exponent=2.0;
    return pow(pow(x[0]-x0, exponent) + pow(x[1]-y0, exponent), 1.0/exponent) - x[2];
};
//////////////////////////////////////////////////
//////////////////////////////////////////////////

// _main_init_start
int main(int argc, char* argv[])
{
  // Initialize Serac
  serac::initialize(argc, argv);
  ::axom::sidre::DataStore datastore;
  ::serac::StateManager::initialize(datastore, "sidreDataStore");
  
  // Define the spatial dimension of the problem and the type of finite elements used.
  static constexpr int ORDER {1};
  static constexpr int DIM {2};
  auto inputFilename = "../../data/meshes/circleTriMesh.g";
  int numElements = 285;

  auto mesh = serac::buildMeshFromFile(inputFilename);

  int num_ref = 0;
  for (int i=0; i<num_ref; i++) {mesh.UniformRefinement();}
  numElements *= static_cast<int>(std::pow(std::pow(DIM,num_ref),num_ref));

  auto pmesh = ::mfem::ParMesh(MPI_COMM_WORLD, mesh);
  pmesh.EnsureNodes();
  pmesh.ExchangeFaceNbrData();

  using shapeFES = serac::H1<ORDER, DIM>;

  // Create finite element space for design parameters, and register it with LiDO DataManager
  auto [shape_fes, shape_fec] = serac::generateParFiniteElementSpace<shapeFES>(&pmesh);
  mfem::HypreParVector node_disp_computed(shape_fes.get());
  std::unique_ptr<mfem::HypreParMatrix> dresidualdu;
  node_disp_computed = 0.0;

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = serac::H1<ORDER, DIM>;
  using trial_space = serac::H1<ORDER, DIM>; 

  // Construct the new functional object using the known test and trial spaces
  serac::Functional<test_space(trial_space)> residual(
                shape_fes.get(), {shape_fes.get()}); // shape, solution, and residual FESs

  residual.AddDomainIntegral( 
    serac::Dimension<DIM>{}, serac::DependsOn<0>{},
    [=](double /*t*/, auto position, auto nodeDisp) {
      // x = X + u
      auto [_, dXdxi] = position;
      auto du_dX = serac::get<1>(nodeDisp);
      
      // Jacobian from reference to the physical/current space (i.e., dx_dxi)
      auto Amat = dXdxi + serac::dot(du_dX, dXdxi); // (I + du/dX) * dX/dxi

      // auto mu = 0.5 * (serac:: = [ 1, -1/sqrt(3); 0, -2/sqrt(3)]
      serac::mat2 WInvMat = {{{1.00000000000000, -0.577350269189626}, {0, 1.15470053837925}}};
      // serac::mat2 WInvMat = {{{1.0, -1.0/std::sqrt(3.0)}, {0.0, -2.0/std::sqrt(3.0)}}};
      // serac::mat2 WInvMat = {{{0.0, 1.0}, {1.0, 0.0}}};
      // Need to compute dmu/dTmat : dTmat/dx, with mu = mu(Tmat)
      // Tmat = Amat * WInvMinner(Tmat, Tmat) / abs(serac::det(Tmat))) - 1.0;
      // triangular correctionat; WInvMat -> constant
      // dmu/dTmat is mu specific
      // dTmat/dx = dAmat/dx * WInvMat

      // Target matrix (updated Jacobian, Tmat or T)
      auto Tmat = serac::dot(Amat, WInvMat);

      // mu metric operations
      auto invTransTmat  = serac::inv(serac::transpose(Tmat));
      auto TmatInnerTmat = serac::inner(Tmat, Tmat);
      auto scale         = -1.0 / serac::det(Tmat);
      if (serac::det(Tmat) <= 0.0) {  scale = 0.0; }
      auto dmudTmat = scale * (0.5 * TmatInnerTmat * invTransTmat - Tmat);

      // compute flux contribution
      auto flux = (1.0/serac::det(dXdxi*WInvMat)) * serac::dot(dmudTmat, serac::transpose(dXdxi*WInvMat));
      auto source     = serac::zero{};
      return ::serac::tuple{source, flux};  /// N*source + DN*flux
    },
    pmesh
  );

  // Circle/cylinder geometry
  auto omega = 1.0e1;
  auto radius = 0.75;

  serac::Domain radial_boundary = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<DIM>(1));
  residual.AddBoundaryIntegral(
    serac::Dimension<DIM - 1>{}, serac::DependsOn<0>{},
    [=](double /*t*/, auto position, auto nodeDisp) {
      auto [X, dXdxi] = position;
      auto u = serac::get<0>(nodeDisp);
      auto x = X + u;
      
      ::mfem::Vector p(DIM+1);     // inputs
      ::mfem::Vector grad(DIM+1); 
      p = radius;

      // Overwrite first 2/3 parameters with coordinates to get shape derivatives
      for (int i = 0; i < DIM; i++) {
        if constexpr (serac::is_tensor_of_dual_number<decltype(x)>::value) {
          p[i]=x[i].value;
        } else {
          p[i]=x[i];
        }
      }
      auto phiVal = ADVal_func(p, func_example<ADFType>);
      ADGrad_func(p, func_example<ADFType>, grad); // grad = dphi/dp = [dphi/dx, dphi/dr]
      // Check if the dimensions match
      serac::tensor<double, DIM> dphi;
      // Populate the serac::tensor from the mfem::Vector
      for (int i = 0; i < DIM; i++) {
        dphi[i] = grad[i];
      }

      if constexpr (serac::is_tensor_of_dual_number<decltype(u)>::value) 
      {
          // Get d2phidXdp
          ::mfem::Vector d2phidpdXVal(DIM);
          ::mfem::DenseMatrix H;
          H.SetSize(DIM+1, DIM+1);
          ADHessian_func(p, func_example<ADSType>, H);

          // Extract components
          serac::tensor<double, DIM, DIM> d2phidx2;
          // d2phidx2.SetSize(DIM, DIM);
      
          for (int i = 0; i < DIM; i++){
              for (int j = 0; j < DIM; j++) { d2phidx2(i, j) = H(i, j); }
          }
          auto gradForDual = 2.0 * omega * ( serac::outer(dphi, dphi) + phiVal * d2phidx2 );

          // std::cout << "Type of phiVal: " << typeid(phiVal).name() << std::endl;
          // std::cout << "Type of dphi: " << typeid(dphi).name() << std::endl;
          // auto dual_dphi = serac::make_dual(dphi);
          // std::cout << "Type of dual_dphi: " << typeid(dual_dphi).name() << std::endl;
          // std::cout << "Type of gradForDual: " << typeid(gradForDual).name() << std::endl;
          // exit(0);
serac::tensor<serac::dual<serac::tensor<double, DIM>>, DIM> grad_for_dual;
// serac::tensor<serac::tuple<serac::tensor<double, DIM>, serac::tensor<double, DIM>>, DIM> grad_for_dual;
// serac::get<0>(grad_for_dual[0]) = 2.0 * omega * phiVal * dphi;
// serac::get<0>(grad_for_dual[1]) = 2.0 * omega * phiVal * dphi;
// serac::get<1>(grad_for_dual[0]) = gradForDual[0];
// serac::get<1>(grad_for_dual[1]) = gradForDual[1];
// serac::tuple<serac::tensor<double, DIM>, serac::tensor<double, DIM, DIM>> finalDualReturn{2.0 * omega * phiVal * dphi, gradForDual};
//           return serac::make_dual(finalDualReturn);
          // return gradForDual;
          return grad_for_dual;

        // auto dual_dphi = serac::make_dual(dphi); // Ensure dphi is compatible
        // return dual_dphi;
        // return 2.0 * omega * phiVal * dphi;
      }
      else
      {
        return 2.0 * omega * phiVal * dphi;
      }
    },
    radial_boundary // whole_boundary
  );

  int totNumDofs = shape_fes->TrueVSize();

  // Get dofs in z direction for all elements (pseudo 2D problem)
  mfem::Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[0] = 1;
  shape_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  mfem::Array<int> constrainedDofs(totNumDofs/DIM);
  int counter = 0;
  for(auto iDof=DIM-1; iDof<totNumDofs; iDof += DIM){
    constrainedDofs[counter] = ess_tdof_list[iDof];
    counter++;
  }

  // wrap residual and provide Jacobian
  serac::mfem_ext::StdFunctionOperator residual_opr(
    totNumDofs,
    // [&residual](const mfem::Vector& u, mfem::Vector& r) {
    [&constrainedDofs, &residual](const mfem::Vector& u, mfem::Vector& r) {
      double dummy_time = 1.0;
      const mfem::Vector res = residual(dummy_time, u);
      r = res;
    },
    // [&residual, &dresidualdu](const mfem::Vector& u) -> mfem::Operator& {      
    [&constrainedDofs, &residual, &dresidualdu](const mfem::Vector& u) -> mfem::Operator& {
      double dummy_time = 1.0;
      auto [val, dr_du] = residual(dummy_time, serac::differentiate_wrt(u));
      dresidualdu       = assemble(dr_du);
      return *dresidualdu;
    }
  );

  const serac::LinearSolverOptions lin_opts = {
                                        .linear_solver = ::serac::LinearSolver::CG,
                                        // .linear_solver  = serac::LinearSolver::Strumpack,
                                        .preconditioner = ::serac::Preconditioner::HypreJacobi,
                                        .relative_tol   = 1.0e-10,
                                        .absolute_tol   = 1.0e-12,
                                        .max_iterations = DIM * numElements,
                                        .print_level    = 0};

  const serac::NonlinearSolverOptions nonlin_opts = {
                                              // .nonlin_solver = ::serac::NonlinearSolver::Newton,
                                              .nonlin_solver  = serac::NonlinearSolver::TrustRegion,
                                              // .nonlin_solver  = serac::NonlinearSolver::NewtonLineSearch,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-10,
                                              .min_iterations = 1, 
                                              .max_iterations = 20, // 20, // 2000
                                              .max_line_search_iterations = 30, //0
                                              .print_level    = 1};

  serac::EquationSolver eq_solver(nonlin_opts, lin_opts, pmesh.GetComm());
  eq_solver.setOperator(residual_opr);
  eq_solver.solve(node_disp_computed);

  mfem::ParGridFunction nodeSolGF(shape_fes.get());
  nodeSolGF.SetFromTrueDofs(node_disp_computed);

  auto pd = mfem::ParaViewDataCollection("sol_mesh_morphing_serac_2D_ad_mfem", &pmesh);
  pd.RegisterField("solution_ad_mfem", &nodeSolGF);
  pd.SetCycle(1);
  pd.SetTime(1);
  pd.Save();
}