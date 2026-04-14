// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <set>
#include <string>
#include <cmath>
#include <memory>
#include <utility>

#include "axom/slic.hpp"
#include "axom/inlet.hpp"
#include "axom/CLI11.hpp"
#include "mfem.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/materials/viscoelastic.hpp"
#include "smith/smith.hpp"

using namespace smith;

/**
 * @brief Run buckling cylinder example
 *
 * @note Based on doi:10.1016/j.cma.2014.08.012
 */
int main(int argc, char* argv[])
{
  constexpr int dim = 3;
  constexpr int p = 1;

  // Command line arguments
  // Mesh options
  int serial_refinement = 0;
  int parallel_refinement = 0;
  double dt = 0.1;

  // Solver options
  NonlinearSolverOptions nonlinear_options = solid_mechanics::default_nonlinear_options;
  nonlinear_options.nonlin_solver = smith::NonlinearSolver::TrustRegion;
  nonlinear_options.relative_tol = 1e-6;
  nonlinear_options.absolute_tol = 1e-10;
  nonlinear_options.min_iterations = 1;
  nonlinear_options.max_iterations = 500;
  nonlinear_options.max_line_search_iterations = 20;
  nonlinear_options.print_level = 1;

  LinearSolverOptions linear_options = solid_mechanics::default_linear_options;
  linear_options.linear_solver = smith::LinearSolver::CG;
  linear_options.preconditioner = smith::Preconditioner::HypreAMG;
  linear_options.relative_tol = 1e-8;
  linear_options.absolute_tol = 1e-16;
  linear_options.max_iterations = 2000;

  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  // Handle command line arguments
  axom::CLI::App app{"Viscoelastic buckling of a curved shell"};
  // Mesh options
  app.add_option("--serial-refinement", serial_refinement, "Serial refinement steps")->check(axom::CLI::PositiveNumber);
  app.add_option("--parallel-refinement", parallel_refinement, "Parallel refinement steps")
      ->check(axom::CLI::PositiveNumber);
  // Solver options
  app.add_option("--nonlinear-solver", nonlinear_options.nonlin_solver,
                 "Nonlinear solver (Index of enum smith::NonlinearSolver)")
      ->expected(0, 10);
  app.add_option("--linear-solver", linear_options.linear_solver, "Linear solver (Index of enum smith::LinearSolver)")
      ->expected(0, 5);
  app.add_option("--preconditioner", linear_options.preconditioner,
                 "Preconditioner (Index of enum smith::NonlinearSolver)")
      ->expected(0, 7);
  app.add_option("--petsc-pc-type", linear_options.petsc_preconditioner,
                 "Petsc preconditioner (Index of enum smith::PetscPCType)")
      ->expected(0, 14);
  app.add_option("--dt", dt, "Size of pseudo-time step pre-contact")->check(axom::CLI::PositiveNumber);
  // Misc options
  app.set_help_flag("--help");

  // Need to allow extra arguments for PETSc support
  app.allow_extras();
  CLI11_PARSE(app, argc, argv);

    // Create DataStore
  std::string name = "viscoelastic_buckling";
  std::string mesh_tag = "mesh";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Create and refine mesh
  std::string filename = SMITH_REPO_DIR "/data/meshes/cap.g";
  auto mesh = std::make_shared<smith::Mesh>(filename, mesh_tag, serial_refinement, parallel_refinement);

  // Surfaces for boundary conditions
  mesh->addDomainOfBoundaryElements("zsymm", smith::by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("xsymm", smith::by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("bottom", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("top", smith::by_attr<dim>(4));

  // temperature parameter field
  smith::FiniteElementState temperature(mesh->mfemParMesh(), smith::L2<0>{}, "temperature");
  temperature = 300.0;

  // Create solver
  smith::SolidMechanics<p, dim, smith::Parameters<smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name,
      mesh, {"temperature"});
  
  solid_solver.setParameter(0, temperature);

  solid_solver.setTraction(
    [&](auto, auto, double t) { 
      return smith::vec3{0, -std::sin(M_PI*t), 0}; 
    },
     mesh->domain("top"));

  double G_inf = 1e3;
  double G_0 = 3*G_inf;
  double G_g = G_inf + G_0;
  double nu_g = 0.45;
  double K = 2.0/3.0*G_g*(1.0 + nu_g)/(1.0 - 2.0*nu_g);
  double tau_0 = 10.0;
  double eta_0 = G_0*tau_0;
  // solid_mechanics::NeoHookean mat{.density = 1.0, .K = (3 * lambda + 2 * G) / 3, .G = G};
  // solid_solver->setMaterial(mat, mesh->entireBody());
  solid_mechanics::ViscoelasticOldInterface mat(K, G_inf, 0.0, 300.0, G_0, eta_0, 300.0, 0.0, 50.0, 1.0);
  auto internal_states = solid_solver.createQuadratureDataBuffer(smith::solid_mechanics::Viscoelastic::State{}, mesh->entireBody());
  solid_solver.setRateDependentMaterial(smith::DependsOn<0>{}, mat, mesh->entireBody(), internal_states);

  // NOTE: somehow J2 material works fine
  // using Hardening = solid_mechanics::LinearHardening;
  // Hardening hardening{.sigma_y = 10.0,  .Hi= G_inf/100.0, .eta = 0.0};
  // solid_mechanics::J2SmallStrain<Hardening> mat2{.E = G_inf, .nu = 0.25, .hardening = hardening, .Hk = 0.0, .density = 1.0};
  // auto internal_states = solid_solver->createQuadratureDataBuffer(smith::solid_mechanics::J2SmallStrain<Hardening>::State{}, mesh->entireBody());
  // solid_solver->setRateDependentMaterial(mat2, mesh->entireBody(), internal_states);
  

  // Set up essential boundary conditions
  // Bottom of cylinder is fixed
  solid_solver.setFixedBCs(mesh->domain("bottom"));

#if 0
  // displacement control, for comparison
  auto compress = [&](const smith::tensor<double, dim>, double t) {
    smith::tensor<double, dim> u{};
    u[0] = u[2] = -1.35 / std::sqrt(2.0) * t;
    return u;
  };
  solid_solver.setDisplacementBCs(compress, mesh->domain("top"), Component::Y);
  solid_solver.setDisplacementBCs(compress, mesh->domain("top"));
#endif

  // Finalize the data structures
  solid_solver.completeSetup();

  // Save initial state
  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  SLIC_INFO_ROOT(axom::fmt::format("Snap-through of viscoelastic shell\n{} displacement dofs",
                                   solid_solver.displacement().GlobalSize()));
  SLIC_INFO_ROOT("Starting pseudo-timestepping.");
  smith::logger::flush();
  while (solid_solver.time() < 1.0 && std::abs(solid_solver.time() - 1) > DBL_EPSILON) {
    SLIC_INFO_ROOT("---------------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("start time = {}, dt = {}", solid_solver.time(), dt));
    smith::logger::flush();

    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }
  SLIC_INFO_ROOT(axom::fmt::format("final time = {}", solid_solver.time()));

  return 0;
}
