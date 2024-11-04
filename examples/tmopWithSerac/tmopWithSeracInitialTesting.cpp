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

#define TWO_DIM_SETUP
// #undef TWO_DIM_SETUP

// #define ONE_ELEM_TEST
#undef ONE_ELEM_TEST

// _main_init_start
int main(int argc, char* argv[])
{
  // Initialize Serac
  serac::initialize(argc, argv);
  ::axom::sidre::DataStore datastore;
  ::serac::StateManager::initialize(datastore, "sidreDataStore");
  
  // Define the spatial dimension of the problem and the type of finite elements used.
  static constexpr int ORDER {1};
#ifdef TWO_DIM_SETUP
  static constexpr int DIM {2};
  auto inputFilename = "../../data/meshes/circleTriMesh.g";
  int numElements = 285;
#ifdef ONE_ELEM_TEST  
  // inputFilename = "../../data/meshes/oneElemTriEquiMesh.g";
  inputFilename = "../../data/meshes/oneElemTriRectMesh.g";
  numElements = 1;
#endif  
#else
  static constexpr int DIM {3};
  auto inputFilename = "../../data/meshes/cylOneElemThickness.g";
  int numElements = 354;
#endif

  auto mesh = serac::buildMeshFromFile(inputFilename);
  auto pmesh = ::mfem::ParMesh(MPI_COMM_WORLD, mesh);
  pmesh.EnsureNodes();
  pmesh.ExchangeFaceNbrData();

  // ::mfem::FiniteElementCollection *fec = new ::mfem::H1_FECollection(1, DIM);
  // ::std::shared_ptr<::mfem::ParFiniteElementSpace> pfespace
  //     = ::std::make_shared<::mfem::ParFiniteElementSpace>(&pmesh, fec, DIM);
  // pmesh.SetNodalFESpace(pfespace.get());

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
#ifdef TWO_DIM_SETUP
      // auto mu = 0.5 * (serac::inner(Tmat, Tmat) / abs(serac::det(Tmat))) - 1.0;
      // triangular correction = [ 1, -1/sqrt(3); 0, -2/sqrt(3)]
      serac::mat2 WInvMat = {{{1.00000000000000, -0.577350269189626}, {0, 1.15470053837925}}};
      // serac::mat2 WInvMat = {{{0.0, 1.0}, {1.0, 0.0}}};

      // Need to compute dmu/dTmat : dTmat/dx, with mu = mu(Tmat)
      // Tmat = Amat * WInvMat; WInvMat -> constant
      // dmu/dTmat is mu specific
      // dTmat/dx = dAmat/dx * WInvMat

      // Jacobian from reference to the physical/current space (i.e., dx_dxi)
      auto Amat = dXdxi + serac::dot(du_dX, dXdxi); // (I + du/dX) * dX/dxi

      // Target matrix (updated Jacobian, Tmat or T)
      auto Tmat = serac::dot(Amat, WInvMat);

      // mu metric operations
      auto invTransTmat  = serac::inv(serac::transpose(Tmat));
      auto TmatInnerTmat = serac::inner(Tmat, Tmat);
      auto scale         = -1.0 / serac::det(Tmat);
      if (serac::det(Tmat) <= 0.0) {  scale = 0.0; }
      auto dmudTmat = scale * (0.5 * TmatInnerTmat * invTransTmat - Tmat);

      // compute dTmatdx
      // auto dTmatdx  = serac::dot(dmudTmat, WInvMat);

      // compute flux contribution
      // auto flux     = serac::dot(dmudTmat, WInvMat) * serac::det(serac::inv(WInvMat)); // dTmatdx * WInvMat;
      // auto flux     = dTmatdx * WInvMat; //  dmudTmat; 
      // auto flux     = dmudTmat * WInvMat * serac::det(serac::inv(WInvMat));
      auto flux     = (1.0/serac::det(dXdxi*WInvMat)) * serac::dot(dmudTmat, serac::transpose(dXdxi*WInvMat));
      
      ///// alternative mu (004, mu = serac::inner(Tmat, Tmat) - 2 * serac::det(Tmat); )
      // auto flux     = 2.0 * (Tmat - invTransTmat*serac::det(Tmat)) * serac::det(I + du_dX);
#else
      // auto mu = (serac::squared_norm(J) / (3 * pow(serac::det(J), 2.0 / 3.0))) - 1.0; // serac::dot(J, J)
      using std::pow;
      auto J = dXdxi + serac::dot(du_dX, dXdxi);
      auto JJ    = serac::squared_norm(J); // serac::dot(J, J)
      auto invJT = serac::inv(serac::transpose(J));
      auto scale = (2.0 / (3.0 * pow(serac::det(J), 2.0 / 3.0) ));
      if (serac::det(J) <= 0.0)
      {
        scale = 0.0;
      }
      // static constexpr auto I = serac::DenseIdentity<DIM>();
      // auto flux       = scale * (J - (JJ/3.0) * invJT) * serac::det(I + du_dX);
      auto flux       = scale * (J - (JJ/3.0) * invJT);
#endif
      auto source     = serac::zero{};
      return ::serac::tuple{source, flux};  /// N*source + DN*flux
    },
    pmesh
  );

  // Circle/cylinder geometry
  auto omega = 0.0e1;
  auto radius = 1.015;
  auto x0 = 0.0;
  auto y0 = 0.0;

  serac::Domain radial_boundary = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<DIM>(1));
  residual.AddBoundaryIntegral(
    serac::Dimension<DIM - 1>{}, serac::DependsOn<0>{},
    [=](double /*t*/, auto position, auto nodeDisp) {
      auto [X, dXdxi] = position;
      auto u = serac::get<0>(nodeDisp);
      auto x = X + u;
      auto phiVal = pow(pow(x[0]-x0, 2.0) + pow(x[1]-y0, 2.0), 0.5) - radius;
      // auto dphidXVal = (x[0] + x[1] - 2.0)* pow( pow(x[0]-x0, 2.0) + pow(x[1]-y0, 2.0), -0.5);
      auto dphi = 0.0*x; // dphidx*dxdu
      dphi[0] = (x[0] - x0)* pow( pow(x[0]-x0, 2.0) + pow(x[1]-y0, 2.0), -0.5);
      dphi[1] = (x[1] - y0)* pow( pow(x[0]-x0, 2.0) + pow(x[1]-y0, 2.0), -0.5);
      return 2.0 * omega * phiVal * dphi;
    },
    radial_boundary // whole_boundary
  );

  int totNumDofs = shape_fes->TrueVSize();

#ifdef TWO_DIM_SETUP
#ifdef ONE_ELEM_TEST
// Constrain half of the dofs in the one element triangular mesh setup
mfem::Array<int> constrainedDofs(4);
for(auto iDof=0; iDof<4; iDof ++){
  constrainedDofs[iDof] = iDof;
}
#else
  mfem::Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[0] = 1;
  mfem::Array<int> constrainedDofs;
  shape_fes->GetEssentialTrueDofs(ess_bdr, constrainedDofs);
#endif
#else
  // Get dofs in z direction for all elements (pseudo 2D problem)
  mfem::Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[1] = 1;
  ess_bdr[2] = 1;
  shape_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  mfem::Array<int> constrainedDofs(totNumDofs/DIM);
  int counter = 0;
  for(auto iDof=DIM-1; iDof<totNumDofs; iDof += DIM){
    constrainedDofs[counter] = ess_tdof_list[iDof];
    counter++;
  }
#endif

  // wrap residual and provide Jacobian
  serac::mfem_ext::StdFunctionOperator residual_opr(
    totNumDofs,
#ifdef ONE_ELEM_TEST
    [&constrainedDofs, &residual](const mfem::Vector& u, mfem::Vector& r) {
#else
    // [&residual](const mfem::Vector& u, mfem::Vector& r) {
    [&constrainedDofs, &residual](const mfem::Vector& u, mfem::Vector& r) {
#endif
      double dummy_time = 1.0;
      const mfem::Vector res = residual(dummy_time, u);
      r = res;
#ifdef ONE_ELEM_TEST
      r.SetSubVector(constrainedDofs, 0.0);
#endif
////////////////////////////////////////
r.SetSubVector(constrainedDofs, 0.0); 
////////////////////////////////////////
    },
#ifdef ONE_ELEM_TEST
    [&constrainedDofs, &residual, &dresidualdu](const mfem::Vector& u) -> mfem::Operator& {
#else
    // [&residual, &dresidualdu](const mfem::Vector& u) -> mfem::Operator& {      
    [&constrainedDofs, &residual, &dresidualdu](const mfem::Vector& u) -> mfem::Operator& {
#endif
      double dummy_time = 1.0;
      auto [val, dr_du] = residual(dummy_time, serac::differentiate_wrt(u));
      dresidualdu       = assemble(dr_du);
#ifdef ONE_ELEM_TEST
      dresidualdu->EliminateBC(constrainedDofs, mfem::Operator::DiagonalPolicy::DIAG_ONE);
#endif
////////////////////////////////////////
dresidualdu->EliminateBC(constrainedDofs, mfem::Operator::DiagonalPolicy::DIAG_ONE);  
////////////////////////////////////////  
      return *dresidualdu;
    }
  );

  const serac::LinearSolverOptions lin_opts = {
                                        // .linear_solver = ::serac::LinearSolver::CG,
                                        .linear_solver  = serac::LinearSolver::Strumpack,
                                        .preconditioner = ::serac::Preconditioner::HypreJacobi,
                                        .relative_tol   = 1.0e-10,
                                        .absolute_tol   = 1.0e-12,
                                        .max_iterations = DIM * numElements,
                                        .print_level    = 0};

  const serac::NonlinearSolverOptions nonlin_opts = {
                                              .nonlin_solver = ::serac::NonlinearSolver::Newton,
                                              // .nonlin_solver  = serac::NonlinearSolver::TrustRegion,
                                              // .nonlin_solver  = serac::NonlinearSolver::NewtonLineSearch,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-10,
                                              // .min_iterations = 1, 
                                              .max_iterations = 20, // 2000
                                              // .max_line_search_iterations = 20, //0
                                              .print_level    = 1};

  serac::EquationSolver eq_solver(nonlin_opts, lin_opts, pmesh.GetComm());
  eq_solver.setOperator(residual_opr);
  eq_solver.solve(node_disp_computed);

  mfem::ParGridFunction nodeSolGF(shape_fes.get());
  nodeSolGF.SetFromTrueDofs(node_disp_computed);
// nodeSolGF.Print();
#ifdef TWO_DIM_SETUP
  auto pd = mfem::ParaViewDataCollection("sol_mesh_morphing_serac_2D", &pmesh);
#else
  auto pd = mfem::ParaViewDataCollection("sol_mesh_morphing_serac_3D", &pmesh);
#endif
  pd.RegisterField("solution", &nodeSolGF);
  pd.SetCycle(1);
  pd.SetTime(1);
  pd.Save();

// delete fec;
}