// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "axom/fmt.hpp"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/liquid_crystal_elastomer.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/finite_element.hpp"  // for H1
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/finite_element_state.hpp"

using namespace smith;

TEST(LiquidCrystalElastomer, Brighenti)
{
  constexpr int p = 1;
  constexpr int dim = 3;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "lce_tensile_test_load");

  // Construct the appropriate dimension mesh
  int nElem = 2;
  double lx = 2.5e-3, ly = 30.0e-3, lz = 30.0e-3;

  std::string mesh_tag{"mesh"};
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh(mfem::Mesh::MakeCartesian3D(nElem, 2 * nElem, 2 * nElem, mfem::Element::HEXAHEDRON, lx, ly, lz)),
      mesh_tag);

  mesh->addDomainOfBoundaryElements("xmin_face", by_attr<dim>(5));
  mesh->addDomainOfBoundaryElements("ymin_face", by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("zmin_face", by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("xmax_face", by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("ymax_face", by_attr<dim>(4));
  mesh->addDomainOfBoundaryElements("zmax_face", by_attr<dim>(6));

  double initial_temperature = 25 + 273;
  double final_temperature = 430.0;
  FiniteElementState temperature(StateManager::newState(H1<p>{}, "temperature", mesh_tag));

  temperature = initial_temperature + 0.0 * final_temperature;

  FiniteElementState gamma(StateManager::newState(L2<p>{}, "gamma", mesh_tag));

  int lceArrangementTag = 1;
  auto gamma_func = [lceArrangementTag](const mfem::Vector& x, double) -> double {
    if (lceArrangementTag == 1) {
      return M_PI_2;
    } else if (lceArrangementTag == 2) {
      return (x[1] > 2.0) ? M_PI_2 : 0.0;
    } else if (lceArrangementTag == 3) {
      return ((x[0] - 2.0) * (x[1] - 2.0) > 0.0) ? 0.333 * M_PI_2 : 0.667 * M_PI_2;
    } else {
      double rad = 0.65;
      return (std::pow(x[0] - 3.0, 2) + std::pow(x[1] - 3.0, 2) - std::pow(rad, 2) < 0.0 ||
              std::pow(x[0] - 1.0, 2) + std::pow(x[1] - 3.0, 2) - std::pow(rad, 2) < 0.0 ||
              std::pow(x[0] - 3.0, 2) + std::pow(x[1] - 1.0, 2) - std::pow(rad, 2) < 0.0 ||
              std::pow(x[0] - 1.0, 2) + std::pow(x[1] - 1.0, 2) - std::pow(rad, 2) < 0.0)
                 ? 0.333 * M_PI_2
                 : 0.667 * M_PI_2;
    }
  };

  mfem::FunctionCoefficient coef(gamma_func);
  gamma.project(coef);

  // Construct a solid mechanics solver
  LinearSolverOptions linear_options = {
      .linear_solver = LinearSolver::GMRES,
      .preconditioner = Preconditioner::HypreAMG,
      .relative_tol = 1.0e-6,
      .absolute_tol = 1.0e-14,
      .max_iterations = 600,
      .print_level = 0,
  };

#ifdef SMITH_USE_SUNDIALS
  NonlinearSolverOptions nonlinear_options = {.nonlin_solver = smith::NonlinearSolver::KINBacktrackingLineSearch,
                                              .relative_tol = 1.0e-4,
                                              .absolute_tol = 1.0e-7,
                                              .max_iterations = 6,
                                              .print_level = 1};
#else
  NonlinearSolverOptions nonlinear_options = {.nonlin_solver = smith::NonlinearSolver::Newton,
                                              .relative_tol = 1.0e-4,
                                              .absolute_tol = 1.0e-7,
                                              .max_iterations = 6,
                                              .print_level = 1};
#endif

  SolidMechanics<p, dim, Parameters<H1<p>, L2<p> > > solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, "lce_solid_functional", mesh,
      {"temperature", "gamma"});

  constexpr int TEMPERATURE_INDEX = 0;
  constexpr int GAMMA_INDEX = 1;

  solid_solver.setParameter(TEMPERATURE_INDEX, temperature);
  solid_solver.setParameter(GAMMA_INDEX, gamma);

  double density = 1.0;
  double E = 7.0e7;
  double nu = 0.45;
  double shear_modulus = 0.5 * E / (1.0 + nu);
  double bulk_modulus = E / 3.0 / (1.0 - 2.0 * nu);
  double order_constant = 10;
  double order_parameter = 0.10;
  double transition_temperature = 348;
  double Nb2 = 1.0;

  LiquidCrystElastomerBrighenti mat(density, shear_modulus, bulk_modulus, order_constant, order_parameter,
                                    transition_temperature, Nb2);

  LiquidCrystElastomerBrighenti::State initial_state{};
  auto qdata = solid_solver.createQuadratureDataBuffer(initial_state);
  solid_solver.setMaterial(DependsOn<TEMPERATURE_INDEX, GAMMA_INDEX>{}, mat, mesh->entireBody(), qdata);

  // prescribe symmetry conditions
  solid_solver.setFixedBCs(mesh->domain("xmin_face"), Component::X);
  solid_solver.setFixedBCs(mesh->domain("ymin_face"), Component::Y);
  solid_solver.setFixedBCs(mesh->domain("zmin_face"), Component::Z);

  double iniLoadVal = 1.0e0;
  double maxLoadVal = 4 * 1.3e0 / lx / lz;
  double loadVal = iniLoadVal + 0.0 * maxLoadVal;
  solid_solver.setTraction([&loadVal](auto, auto n, auto) { return loadVal * n; }, mesh->domain("ymax_face"));

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform first quasi-static solve
  std::string output_filename = "sol_lce_brighenti_tensile";
  solid_solver.outputStateToDisk(output_filename);

  // QoI for output:
  Functional<double(H1<p, dim>)> avgYDispQoI({&solid_solver.displacement().space()});
  avgYDispQoI.AddSurfaceIntegral(
      DependsOn<0>{},
      [=](double /*t*/, auto position, auto displacement) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = displacement;
        auto n = normalize(cross(dX_dxi));
        return dot(u, n);
      },
      mesh->domain("ymax_face"));

  Functional<double(H1<p, dim>)> area({&solid_solver.displacement().space()});
  area.AddSurfaceIntegral(
      DependsOn<>{}, [=](double /*t*/, auto /*position*/) { return 1.0; }, mesh->domain("ymax_face"));

  double t = 0.0;
  double initial_area = area(t, solid_solver.displacement());
  SLIC_INFO_ROOT("... Initial Area of the top surface: " << initial_area);

  // initializations for quasi-static problem
  int num_steps = 3;
  double tmax = 1.0;
  double dt = tmax / num_steps;
  double gblDispYmax;
  bool outputDispInfo(true);

  // Perform remaining quasi-static solve
  for (int i = 0; i < (num_steps + 1); i++) {
    SLIC_INFO_ROOT(
        axom::fmt::format("\n\n............................"
                          "\n... Entering time step: {}"
                          "\n............................\n"
                          "\n... At time: {} \n... And with a tension load of: {} ( {} `%` of max)"
                          "\n... And with uniform temperature of: {}\n",
                          i + 1, t, loadVal, loadVal / maxLoadVal * 100, initial_temperature));

    // solve problem with current parameters
    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk(output_filename);

    // get QoI
    double current_qoi = avgYDispQoI(t, solid_solver.displacement());
    double current_area = area(t, solid_solver.displacement());

    // get displacement info
    if (outputDispInfo) {
      auto& fes = solid_solver.displacement().space();
      mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
      mfem::Vector dispVecY(fes.GetNDofs());
      dispVecY = 0.0;

      for (int k = 0; k < fes.GetNDofs(); k++) {
        dispVecY(k) = displacement_gf(3 * k + 1);
      }

      double lclDispYmax = dispVecY.Max();
      MPI_Allreduce(&lclDispYmax, &gblDispYmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      SLIC_INFO_ROOT(
          axom::fmt::format("\n... Max Y displacement: {}"
                            "\n... The QoIVal is: {}"
                            "\n... The top surface current area is: {}"
                            "\n... The vertical displacement integrated over the top surface is: {}",
                            gblDispYmax, current_qoi, current_area, current_qoi / current_area));
    }

    SLIC_ERROR_ROOT_IF(std::isnan(gblDispYmax), "... Solution blew up... Check boundary and initial conditions.");

    // update pseudotime-dependent information
    t += dt;
    loadVal = iniLoadVal + (maxLoadVal - iniLoadVal) * std::pow(t / tmax, 0.75);
  }

  // check output
  EXPECT_NEAR(gblDispYmax, 1.95036097e-05, 1.0e-8);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
