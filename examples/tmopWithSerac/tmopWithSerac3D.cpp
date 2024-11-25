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
struct CuboidLSF3D { 
  double x0;
  double y0; 
  // double z0; 
  double radius; 
  double exponent; 

  template < typename T >
  T SDF(const serac::tensor<T, 3> & x) const {
    using std::pow;
    // return pow(pow(x[0]-x0, exponent) + pow(x[1]-y0, exponent) + pow(x[2]-z0, exponent), 1.0/exponent) - radius;
    return pow(pow(x[0]-x0, exponent) + pow(x[1]-y0, exponent), 1.0/exponent) - radius;
  }

  template < typename T >
  serac::tensor<T, 3> GRAD(const serac::tensor<T, 3> & x) const{
    using std::pow;
    auto dphi = 0.0*x;

    // dphi[0] = (x[0] - x0)* pow( pow(x[0]-x0, exponent) + pow(x[1]-y0, exponent) + pow(x[2]-z0, exponent), 1.0/exponent-1);
    // dphi[1] = (x[1] - y0)* pow( pow(x[0]-x0, exponent) + pow(x[1]-y0, exponent) + pow(x[2]-z0, exponent), 1.0/exponent-1);
    // dphi[2] = (x[2] - z0)* pow( pow(x[0]-x0, exponent) + pow(x[1]-y0, exponent) + pow(x[2]-z0, exponent), 1.0/exponent-1);

    dphi[0] = (x[0] - x0)* pow(pow(x[0]-x0, exponent) + pow(x[1]-y0, exponent), 1.0/exponent-1);
    dphi[1] = (x[1] - y0)* pow(pow(x[0]-x0, exponent) + pow(x[1]-y0, exponent), 1.0/exponent-1);
    // dphi[2] = (x[2] - z0)* pow( pow(x[0]-x0, exponent) + pow(x[1]-y0, exponent), 1.0/exponent-1);
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
//////////////////////////////////////////////
//////////////////////////////////////////////

// _main_init_start
int main(int argc, char* argv[])
{
  // Initialize Serac
  serac::initialize(argc, argv);
  ::axom::sidre::DataStore datastore;
  ::serac::StateManager::initialize(datastore, "sidreDataStore");
  
  // Define the spatial dimension of the problem and the type of finite elements used.
  static constexpr int ORDER {1};
  static constexpr int DIM {3};
  // auto inputFilename = "../../data/meshes/cylOneElemThickness.g";
  // int numElements = 354;
  auto inputFilename = "../../data/meshes/cylOneElemThicknessTets.g";
  int numElements = 2485;// 9280;

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
      auto dudX = serac::get<1>(nodeDisp);
      
      // Jacobian from reference to the physical/current space (i.e., dx_dxi)
      auto Amat = dXdxi + serac::dot(dudX, dXdxi); // (I + du/dX) * dX/dxi

      // triangular correction
      serac::mat3 WInvMat = {{{1.00000, -0.577350, -0.408248}, {0, 1.15470, -0.408248}, {0, 0, 1.22474}}};
      // serac::mat3 WInvMat = {{{1.0, 0.0, 0.0}, {0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};

      // Target matrix (updated Jacobian, Tmat or T)
      auto Tmat = serac::dot(Amat, WInvMat);

      // auto mu = (serac::squared_norm(J) / (3 * pow(serac::det(J), 2.0 / 3.0))) - 1.0; // serac::dot(J, J)
      using std::pow;
      // auto J = dXdxi + serac::dot(dudX, dXdxi);
      // auto TmatdotTmat = serac::squared_norm(Tmat); // serac::dot(TmatdotTmat, TmatdotTmat)
      auto TmatInnerTmat = serac::inner(Tmat, Tmat);
      auto invTransTmat = serac::inv(serac::transpose(Tmat));
      auto scale = (2.0 / (3.0 * pow(serac::det(Tmat), 2.0 / 3.0) ));
      if (serac::det(Tmat) <= 0.0)
      {
        scale = 0.0;
// std::cout<<".......... warning ........... "<<std::endl;
      }

      // static constexpr auto I = serac::DenseIdentity<DIM>();
      // auto flux       = scale * (J - (JJ/3.0) * invTransTmat) * serac::det(I + dudX);
      auto dmudTmat = scale * (Tmat - (TmatInnerTmat/3.0) * invTransTmat);

      // compute flux contribution
      auto flux = (1.0/serac::det(dXdxi*WInvMat)) * serac::dot(dmudTmat, serac::transpose(dXdxi*WInvMat));

      auto source     = serac::zero{};
      return ::serac::tuple{source, flux};  /// N*source + DN*flux
    },
    pmesh
  );

  // Circle/cylinder geometry
  auto omega = 1.0e4;
  auto radius = 0.85;
  auto x0 = 0.0;
  auto y0 = 0.0;

  serac::Domain radial_boundary = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<DIM>(1));
  residual.AddBoundaryIntegral(
    serac::Dimension<DIM - 1>{}, serac::DependsOn<0>{},
    [=](double /*t*/, auto position, auto nodeDisp) {
      auto [X, dXdxi] = position;
      // auto u = serac::get<0>(nodeDisp);
      auto [u, dudX] = nodeDisp;
      auto x = X + u;
      // auto z0 = 0.0;
      auto phi_value = CuboidLSF3D{x0, y0, 1.0*radius, 2};
      auto phiVal = phi_value.SDF(x);
      auto dphi = phi_value.GRAD(x);

      return 2.0 * omega * phiVal * dphi;
      // auto dxdxi = dXdxi + dot(serac::transpose(dudX), dXdxi);
      // auto dA = norm(cross(dXdxi));
      // auto da = norm(cross(dxdxi));
      // auto area_correction = da / dA;
      // return 2.0 * omega * phiVal * dphi * area_correction;

      // serac::mat3 WInvMat = {{{1.00000, -0.577350, -0.408248}, {0, 1.15470, -0.408248}, {0, 0, 1.22474}}};
//       serac::mat2 WInvMat = {{{1.00000000000000, -0.577350269189626}, {0, 1.15470053837925}}};
// std::cout<<"... dXdxi = "<<dXdxi<<std::endl;
// std::cout<<"... dphi = "<<dphi<<std::endl;
// std::cout<<"... WInvMat = "<<WInvMat<<std::endl;
// std::cout<<"... serac::transpose(dXdxi*WInvMat) = "<<serac::transpose(dXdxi*WInvMat)<<std::endl;
// // std::cout<<"... serac::transpose(dXdxi*WInvMat)*dphi = "<<serac::transpose(dXdxi*WInvMat) * dphi<<std::endl;
// std::cout<<"... serac::transpose(dXdxi*WInvMat)*dphi = "<<serac::dot(serac::transpose(dXdxi*WInvMat), dphi)<<std::endl;
// // std::cout<<"... serac::dot(dphi, serac::transpose(WInvMat*dXdxi)) = "<<serac::dot(dphi, serac::transpose(WInvMat*dXdxi))<<std::endl;
// // std::cout<<"... serac::dot(dphi, serac::transpose(WInvMat*dXdxi)) = "<<serac::dot(serac::transpose(WInvMat*dXdxi), serac::transpose(dphi))<<std::endl;
// exit(0);
      // return 2.0 * omega * phiVal * serac::dot(dphi, serac::transpose(WInvMat*dXdxi));
      // return 2.0 * omega * phiVal * dphi;
    },
    radial_boundary // whole_boundary
  );

  int totNumDofs = shape_fes->TrueVSize();

  // Get dofs in z direction for all elements (pseudo 2D problem)
  mfem::Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
  ess_bdr = 0;
// ess_bdr[0] = 1;
  ess_bdr[1] = 1;
  ess_bdr[2] = 1;
  shape_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  // mfem::Array<int> ess_tdof_list_2, ess_bdr_2(mesh.bdr_attributes.Max());
  // ess_bdr_2[0] = 1;
  // shape_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list_2);
  // int totNumDofs2 = ess_tdof_list_2.Size();

  // mfem::Array<int> constrainedDofs(ess_tdof_list.Size() + ess_tdof_list_2.Size());
// std::cout<<"....................."<<std::endl;
  mfem::Array<int> constrainedDofs(ess_tdof_list.Size());
  int counter = 0;
  for(auto iDof=DIM-1; iDof<ess_tdof_list.Size(); iDof += DIM){
    constrainedDofs[counter] = ess_tdof_list[iDof];
    counter++;
  }
constrainedDofs.SetSize(counter);
// std::cout<<".......... 2 ..........."<<std::endl;
// std::cout<<".......... totNumDofs = "<<totNumDofs<<std::endl;
// std::cout<<".......... ess_tdof_list.Size() = "<<ess_tdof_list.Size()<<std::endl;
// std::cout<<".......... counter = "<<counter<<std::endl;
// exit(0);

  // for(int iDof=0; iDof<counter; iDof++){
// std::cout<<".......... 3a ..........."<<std::endl;
    // constrainedDofs[iDof] = ess_tdof_list[(DIM-1)+iDof*DIM];
// std::cout<<".......... iDof ..........."<< iDof <<std::endl;
// std::cout<<".......... (DIM-1)+iDof*DIM ..........."<< (DIM-1)+iDof*DIM <<std::endl;
  // }
// std::cout<<"... constrainedDofs.Size() = "<<constrainedDofs.Size()<<std::endl;
// std::cout<<"... ess_tdof_list.Size() = "<<ess_tdof_list.Size()<<std::endl;
// std::cout<<"... totNumDofs = "<<totNumDofs<<std::endl;
// std::cout<<"... counter = "<<counter<<std::endl;
// exit(0);
  // for(auto iDof=0; iDof<ess_tdof_list_2.Size(); iDof ++){
  //   constrainedDofs[counter] = ess_tdof_list_2[iDof];
  //   counter++;
  // }

  // wrap residual and provide Jacobian
  serac::mfem_ext::StdFunctionOperator residual_opr(
    totNumDofs,
    // [&residual](const mfem::Vector& u, mfem::Vector& r) {
    [&constrainedDofs, &residual](const mfem::Vector& u, mfem::Vector& r) {
      double dummy_time = 1.0;
      const mfem::Vector res = residual(dummy_time, u);
      r = res;
      r.SetSubVector(constrainedDofs, 0.0);
    },
    // [&residual, &dresidualdu](const mfem::Vector& u) -> mfem::Operator& {      
    [&constrainedDofs, &residual, &dresidualdu](const mfem::Vector& u) -> mfem::Operator& {
      double dummy_time = 1.0;
      auto [val, dr_du] = residual(dummy_time, serac::differentiate_wrt(u));
      dresidualdu       = assemble(dr_du);
      dresidualdu->EliminateBC(constrainedDofs, mfem::Operator::DiagonalPolicy::DIAG_ONE);
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
// nodeSolGF.Print();
  auto pd = mfem::ParaViewDataCollection("sol_mesh_morphing_serac_3D", &pmesh);
  pd.RegisterField("solution", &nodeSolGF);
  pd.SetCycle(1);
  pd.SetTime(1);
  pd.Save();
}