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
  serac::StateManager::initialize(datastore, "LCE_ActiveInactiveTest");

  // Construct the appropriate dimension mesh and give it to the data store
  auto inputFileName = "../../tests/TubeLCE_Split1.g"; 
  auto meshInput = serac::buildMeshFromFile(inputFileName);
  std::string mesh_tag{"mesh"}; 
  auto mesh = mesh::refineAndDistribute(std::move(meshInput), serial_refinement, parallel_refinement);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);


  // orient fibers in the beam like below:
  //
  // x
  //
  // ^                                             8
  // |                                             |
  // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓-- 1
  // ┃ - - - - - - - - - - - - - - - - - - - - - - ┃
  // ┃ - - - - - - - - - - - - - - - - - - - - - - ┃
  // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛--> z

  // LinearSolverOptions linear_options = {.linear_solver = LinearSolver::SuperLU};
  const LinearSolverOptions linear_options = {.linear_solver = LinearSolver::CG,.preconditioner = Preconditioner::HypreAMG,.absolute_tol   = 1.0e-11,.max_iterations=1000, .print_level = 0};

  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::TrustRegion,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-14,
                                              .max_iterations = 1000, // changed this from 1
                                              .print_level    = 1,
                                              .trust_region_scaling = 0.1};
  bool use_warm_start = false;
  SolidMechanics<p, dim, Parameters<L2<0>, L2<0>, L2<0>, L2<0>, L2<0>, L2<0> > > solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, 
      "ActiveInactiveTest", mesh_tag, {"orderParam", "gammaParam", "etaParam","orderParam2","gammaParam2","etaParam2"}, 0, 0.0, false, use_warm_start);

  // -------------------
  // Material properties
  // -------------------

  // Active material properties - from Harvard
  double density         = 1.0;
  double shear_mod       = 9.9564e6;//4.4762e5; // 4476171.852e-6; // 3.113e4; //  young_modulus_ / 2.0*(1.0 + poisson_ratio_);
  double ini_order_param = 0.535; //0.535; // 0.28544; //   0.40;
  double min_order_param = 0.00001; // ini_order_param; //
  double omega_param     = 0.12531; // 0.1151; // 1.0e2;
  double bulk_mod        = 4.945e8;//1.0e1*shear_mod; // K = (E/3*(1-2v)), E = 29.67 MPa, v = 0.49 
  // -------------------

  // Set material
  LiquidCrystalElastomerZhang lceMat(density, shear_mod, ini_order_param, omega_param, bulk_mod);
  LiquidCrystalElastomerZhang regularMat(density, 1.9128e5, 0.0001, omega_param, 9.5e6);
  /*
  serac::solid_mechanics::NeoHookean regularMat;
  // inactive material properties - from Harvard
  regularMat.density = 1; // update
  regularMat.K = 9.5e6; // E = 0.57 MPa
  regularMat.G = 1.9128e5; //*/

  // Parameter 1
  FiniteElementState orderParam(pmesh, L2<0>{}, "orderParam");
  orderParam = ini_order_param;
  FiniteElementState orderParam2(pmesh, L2<0>{}, "orderParam2");
  orderParam2 = 0.0001;

  // Parameter 2
  FiniteElementState gammaParam(pmesh, L2<0>{}, "gammaParam");
  FiniteElementState gammaParam2(pmesh, L2<0>{}, "gammaParam2");

  auto gammaFunc         = [](const mfem::Vector& /*this is location*/, double) -> double {
    return 0.0 * M_PI_2; // 0.0;  // 
  };

  mfem::FunctionCoefficient gammaCoef(gammaFunc);
  gammaParam.project(gammaCoef);
  gammaParam2.project(gammaCoef);

  // Paremetr 3
  FiniteElementState etaParam(pmesh, L2<0>{}, "etaParam");
  auto                      etaFunc = [](const mfem::Vector& /*x*/, double) -> double { return 1.5708; }; //1.5708
  mfem::FunctionCoefficient etaCoef(etaFunc);
  FiniteElementState etaParam2(pmesh, L2<0>{}, "etaParam2");
  auto                      etaFunc2 = [](const mfem::Vector& /*x*/, double) -> double { return 0.0; }; //1.5708
  mfem::FunctionCoefficient etaCoef2(etaFunc2);
  etaParam.project(etaCoef);
  etaParam2.project(etaCoef2);

  // Set parameters
  constexpr int ORDER_INDEX = 0;
  constexpr int GAMMA_INDEX = 1;
  constexpr int ETA_INDEX   = 2;
  constexpr int ORDER_INDEX2 = 3;
  constexpr int GAMMA_INDEX2 = 4;
  constexpr int ETA_INDEX2 = 5;

  solid_solver.setParameter(ORDER_INDEX, orderParam);
  solid_solver.setParameter(ORDER_INDEX2, orderParam2);
  solid_solver.setParameter(GAMMA_INDEX, gammaParam);
  solid_solver.setParameter(ETA_INDEX, etaParam);
  solid_solver.setParameter(GAMMA_INDEX2, gammaParam2);
  solid_solver.setParameter(ETA_INDEX2, etaParam2);

  // Material domains and material setting
  serac::Domain whole_domain = serac::EntireDomain(pmesh);

  // Domains for the active and inactive material domains, noting that the current mesh has a slice at y = 0 that should make these a roughly even split
  auto ActiveMat = serac::Domain::ofElements(pmesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
    return average(vertices)[1] > 0.0;  
  }));
  auto NonActiveMat = serac::Domain::ofElements(pmesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
    return average(vertices)[1] < 0.0;  
  }));

  // Small areas on the end of the domain to fix in x and y, so that we constrain rigid modes while also allowing swelling on most of the end face
  auto FixedArea = serac::Domain::ofBoundaryElements(pmesh, std::function([](std::vector<vec3> vertices, int /*bdr_attr*/) {
    bool Outputs{};
    Outputs = false;
    if(average(vertices)[1] < -0.00045 && average(vertices)[2] < 0.0001){
      Outputs = true;
    };
    return Outputs;  
  }));


  solid_solver.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat, ActiveMat);
  //solid_solver.setMaterial(DependsOn<>{}, regularMat, NonActiveMat);
  solid_solver.setMaterial(DependsOn<ORDER_INDEX2, GAMMA_INDEX2, ETA_INDEX2>{}, regularMat, NonActiveMat); 

  // Boundary conditions domains
  auto BCOneEndDomain = ::serac::Domain::ofBoundaryElements(pmesh, ::serac::by_attr<dim>(1)); 
  auto BCOtherEndDomain = ::serac::Domain::ofBoundaryElements(pmesh, ::serac::by_attr<dim>(2)); 

  // Prescribe zero displacement at the supported end of the beam
  solid_solver.setDisplacementBCs([=](auto, auto) { 
    tensor<double, dim> u{};
    u[0] = 0; 
    u[1] = 0;
    return u; }, FixedArea,serac::Component::X + serac::Component::Y);
  solid_solver.setDisplacementBCs([=](auto, auto) { 
    tensor<double, dim> u{}; 
    u[2] = 0;
    return u; }, BCOneEndDomain, serac::Component::Z);

  // set traction - not currently using this, but leaving it as an option
  double loadVal = 0.0e5;
  
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
  std::string outputFilename = "ActiveInactiveFilamentTest";
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

/*
    std::cout << "\n... Entering time step: " << i + 1 << "\n... At time: " << t
              << "\n... Min X displacement: " << gblDispXmin
              << "\n... Max X displacement: " << gblDispXmax
              << "\n... Min Y displacement: " << gblDispYmin
              << "\n... Max Y displacement: " << gblDispYmax
              << "\n... Min Z displacement: " << gblDispZmin
              << "\n... Max Z displacement: " << gblDispZmax << std::endl;*/

    if (std::isnan(dispVecX.Max()) || std::isnan(-1 * dispVecX.Max())) {
        std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
      exit(1);
    }
  }


}
