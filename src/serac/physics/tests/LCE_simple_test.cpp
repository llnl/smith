// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

#include "axom/slic/core/SimpleLogger.hpp"
// #include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/materials/liquid_crystal_elastomer.hpp"
#include "serac/infrastructure/application_manager.hpp"

using namespace serac;

int main(int argc, char* argv[])
{

  // Initialize and automatically finalize MPI and other libraries
  serac::ApplicationManager applicationManager(argc, argv);

  constexpr int p                   = 1;
  constexpr int dim                 = 3;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "LCE_free_swelling_zhang");

  // Construct the appropriate dimension mesh and give it to the data store
  int nElem = 5;
  double lx = (15.8356e-3)/2, lz = (3.25e-3)/2, ly = (0.34e-3);
  ::mfem::Mesh cuboid =
      // mfem::Mesh(mfem::Mesh::MakeCartesian3D(20 * nElem, 10 * nElem, nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
      mfem::Mesh(mfem::Mesh::MakeCartesian3D(50 * nElem, 1 * nElem, 10 *nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));
  auto mesh = mesh::refineAndDistribute(std::move(cuboid), serial_refinement, parallel_refinement);
  std::string mesh_tag{"mesh"}; 
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Side set ordering for MFEM Cartesian mesh:
  // SS-1: XY bottom plane
  // SS-2: XZ left plane
  // SS-3: YZ front plane
  // SS-4: XZ right plane
  // SS-5: YZ back plane
  // SS-6: XY top plane

  // orient fibers in the beam like below:
  //
  // y
  //
  // ^                                             8
  // |                                             |
  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓-- 1
  // ┃ - - - - - - - - - - - - - - - - - - - - - - ┃
  // ┃ - - - - - - - - - - - - - - - - - - - - - - ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛--> x

  // LinearSolverOptions linear_options = {.linear_solver = LinearSolver::SuperLU};
  const LinearSolverOptions linear_options = {.linear_solver = LinearSolver::Strumpack, .print_level = 0};

  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::Newton,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-14,
                                              .max_iterations = 1000, // changed this from 1
                                              .print_level    = 1};
  SolidMechanics<p, dim, Parameters<L2<0>, L2<0>, L2<0> > > solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, 
      "lce_zhang_free_swelling", mesh_tag, {"orderParam", "gammaParam", "etaParam"});

  // -------------------
  // Material properties
  // -------------------

  double density         = 1.0;
  double shear_mod       = 4.4762e5; // 4476171.852e-6; // 3.113e4; //  young_modulus_ / 2.0 / (1.0 + poisson_ratio_);
  double ini_order_param = 0.535; // 0.28544; //   0.40;
  double min_order_param = 0.00001; // ini_order_param; //
  double omega_param     = 0.12531; // 0.1151; // 1.0e2;
  double bulk_mod        = 1.0e1*shear_mod; 
  // -------------------

  // Set material
  LiquidCrystalElastomerZhang lceMat(density, shear_mod, ini_order_param, omega_param, bulk_mod);

  // Parameter 1
  FiniteElementState orderParam(pmesh, L2<0>{}, "orderParam");
  orderParam = ini_order_param;

  // Parameter 2
  FiniteElementState gammaParam(pmesh, L2<0>{}, "gammaParam");

  auto gammaFunc         = [](const mfem::Vector&, double) -> double {
    return 0.0 * M_PI_2; // 0.0;  // 
  };

  mfem::FunctionCoefficient gammaCoef(gammaFunc);
  gammaParam.project(gammaCoef);

  // Paremetr 3
  FiniteElementState etaParam(pmesh, L2<0>{}, "etaParam");
  auto                      etaFunc = [](const mfem::Vector& /*x*/, double) -> double { return 0.0; };
  mfem::FunctionCoefficient etaCoef(etaFunc);
  etaParam.project(etaCoef);

  // Set parameters
  constexpr int ORDER_INDEX = 0;
  constexpr int GAMMA_INDEX = 1;
  constexpr int ETA_INDEX   = 2;

  solid_solver.setParameter(ORDER_INDEX, orderParam);
  solid_solver.setParameter(GAMMA_INDEX, gammaParam);
  solid_solver.setParameter(ETA_INDEX, etaParam);

  // this was adjusted to include the domain as an input
  serac::Domain whole_domain = serac::EntireDomain(pmesh);
  solid_solver.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat, whole_domain);

  // Boundary conditions domains, I added this to comply with new setDisplacementBC format
  auto BCBotDomain = ::serac::Domain::ofBoundaryElements(pmesh, ::serac::by_attr<dim>(1)); 
  auto BCLeftDomain = ::serac::Domain::ofBoundaryElements(pmesh, ::serac::by_attr<dim>(2)); 
  auto BCBackDomain = ::serac::Domain::ofBoundaryElements(pmesh, ::serac::by_attr<dim>(5)); 
  auto BCRightDomain = ::serac::Domain::ofBoundaryElements(pmesh, ::serac::by_attr<dim>(4)); 

  // Prescribe zero displacement at the supported end of the beam - updated this syntax
  solid_solver.setDisplacementBCs([=](auto, auto) { 
    tensor<double, dim> u{};
    u[1] = 0; 
    return u; }, BCBotDomain, serac::Component::Y);  // bottom face y-dir disp = 0
  solid_solver.setDisplacementBCs([=](auto, auto) { 
    tensor<double, dim> u{}; 
    u[0] = 0; 
    return u; }, BCLeftDomain, serac::Component::X);  // left face x-dir disp = 0
  solid_solver.setDisplacementBCs([=](auto, auto) { 
    tensor<double, dim> u{};
    u[2] = 0;  
    return u; }, BCBackDomain, serac::Component::Z);  // back face z-dir disp = 0

  // set global initial displacement as a field function - updated this syntax
  auto applied_displacement = [](tensor<double, dim> x, double) {
    tensor<double, dim> u{};
    u[0] = 0.0001 * x[0];
    u[1] = 0.0001 * x[1];
    u[2] = 0.0001 * x[2];
    return u;
  };

  solid_solver.setDisplacement(
    [applied_displacement](tensor<double, dim> X) { return applied_displacement(X, 0.0); });

  // set traction, updated this syntax - removing this does not fix the solver problem
  double loadVal = 0.0e5;
  solid_solver.setTraction(
    [&loadVal, &lx](auto x, auto, auto) {
      auto t = 0.0 * x;
      t[0] += loadVal * (x[0] > 0.975 * lx);
      return t;
    },
    BCRightDomain); 
 // solid_solver.setTraction([&loadVal, lx](auto x, auto /*n*/, auto /*t*/) {
 //   return tensor<double, 3>{loadVal * (x[0] > 0.975 * lx), 0.0, 0.0};
 // });

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  int num_steps = 10; if(loadVal>0.0){num_steps=1;}
  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;


  std::cout << "\n\n............................"
            << "\n... Entering time step: 0 "
            << "\n............................\n"
            << "\n... Using order parameter: " << ini_order_param
            << "\n... Using two gamma angles" << std::endl;

  solid_solver.advanceTimestep(dt);
  std::string outputFilename = "sol_tension_zhang_Dans_test_00_degrees_no_swelling_only_compresion";
  solid_solver.outputStateToDisk(outputFilename);

  for (int i = 0; i < num_steps; i++) {    

    t += dt;
    orderParam = min_order_param + (ini_order_param - min_order_param) * (tmax - t) / tmax;
    solid_solver.setParameter(ORDER_INDEX, orderParam);

    std::cout << "\n\n............................"
              << "\n... Entering time step: " << i + 1 << " (/" << num_steps << ")"
              << "\n............................\n"
              << "\n... Using order parameter: " << min_order_param + (ini_order_param - min_order_param) * (tmax - t) / tmax
              << "\n... Using two gamma angles" << std::endl;

    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk(outputFilename);

    // FiniteElementState& displacement = solid_solver.displacement();
    mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
    auto&               fes          = solid_solver.displacement().space();
    int                 numDofs      = fes.GetNDofs();
    mfem::Vector dispVecX(numDofs);
    dispVecX = 0.0;
    mfem::Vector dispVecY(numDofs);
    dispVecY = 0.0;
    mfem::Vector dispVecZ(numDofs);
    dispVecZ = 0.0;

    for (int k = 0; k < numDofs; k++) {
      dispVecX(k) = displacement_gf(0 * numDofs + k);
      dispVecY(k) = displacement_gf(1 * numDofs + k);
      dispVecZ(k) = displacement_gf(2 * numDofs + k);
    }

    double gblDispXmin, lclDispXmin = dispVecX.Min();
    double gblDispXmax, lclDispXmax = dispVecX.Max();
    double gblDispYmin, lclDispYmin = dispVecY.Min();
    double gblDispYmax, lclDispYmax = dispVecY.Max();
    double gblDispZmin, lclDispZmin = dispVecZ.Min();
    double gblDispZmax, lclDispZmax = dispVecZ.Max();

    MPI_Allreduce(&lclDispXmin, &gblDispXmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispXmax, &gblDispXmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispYmin, &gblDispYmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispYmax, &gblDispYmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispZmin, &gblDispZmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispZmax, &gblDispZmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


    std::cout << "\n... Entering time step: " << i + 1 << "\n... At time: " << t
              << "\n... Min X displacement: " << gblDispXmin
              << "\n... Max X displacement: " << gblDispXmax
              << "\n... Min Y displacement: " << gblDispYmin
              << "\n... Max Y displacement: " << gblDispYmax
              << "\n... Min Z displacement: " << gblDispZmin
              << "\n... Max Z displacement: " << gblDispZmax << std::endl;

    if (std::isnan(dispVecX.Max()) || std::isnan(-1 * dispVecX.Max())) {
      //if (rank == 0) {
        std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
     // }
      exit(1);
    }
  }


}
