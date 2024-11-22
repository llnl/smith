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

      // auto mu = 0.5 * (serac::inner(Tmat, Tmat) / abs(serac::det(Tmat))) - 1.0;
      // triangular correction = [ 1, -1/sqrt(3); 0, -2/sqrt(3)]
      serac::mat2 WInvMat = {{{1.00000000000000, -0.577350269189626}, {0, 1.15470053837925}}};
      // serac::mat2 WInvMat = {{{1.0, -1.0/std::sqrt(3.0)}, {0.0, -2.0/std::sqrt(3.0)}}};
      // serac::mat2 WInvMat = {{{0.0, 1.0}, {1.0, 0.0}}};
      // Need to compute dmu/dTmat : dTmat/dx, with mu = mu(Tmat)
      // Tmat = Amat * WInvMat; WInvMat -> constant
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
  auto x0 = 0.0;
  auto y0 = 0.0;

  serac::Domain radial_boundary = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<DIM>(1));
  residual.AddBoundaryIntegral(
    serac::Dimension<DIM - 1>{}, serac::DependsOn<0>{},
    [=](double /*t*/, auto position, auto nodeDisp) {
      auto [X, dXdxi] = position;
      auto u = serac::get<0>(nodeDisp);
      auto x = X + u;
      auto phi_value = CuboidLSF2D{x0, y0, 1.0*radius, 2};
      auto phiVal = phi_value.SDF(x);
      auto dphi = phi_value.GRAD(x);
      return 2.0 * omega * phiVal * dphi;
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

  auto pd = mfem::ParaViewDataCollection("sol_mesh_morphing_serac_2D", &pmesh);
  pd.RegisterField("solution", &nodeSolGF);
  pd.SetCycle(1);
  pd.SetTime(1);
  pd.Save();
}