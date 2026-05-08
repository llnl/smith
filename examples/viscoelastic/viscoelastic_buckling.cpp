// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <cmath>
#include <fstream>
#include <memory>
#include <string>

// for debugging nans
#include <signal.h>
#include <fenv.h>
#include <cstdlib>
#include <cstdio>

#include "axom/slic.hpp"
#include "axom/inlet.hpp"
#include "axom/CLI11.hpp"
#include "mfem.hpp"

#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/geometry.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/materials/viscoelastic.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/smith.hpp"

static void fpe_signal_handler(int sig, siginfo_t *sip, void *scp)
{
    int fe_code = sip->si_code;

    printf("In signal handler : ");

    if (fe_code == ILL_ILLTRP)
        printf("Illegal trap detected\n");
    else
        printf("Code detected : %d\n",fe_code);

    abort();
}

#if 1
void enable_floating_point_exceptions()
{
    fenv_t env;
    fegetenv(&env);

    env.__fpcr = env.__fpcr | __fpcr_trap_invalid | __fpcr_trap_divbyzero;
    fesetenv(&env);

    struct sigaction act;
    act.sa_sigaction = fpe_signal_handler;
    sigemptyset (&act.sa_mask);
    act.sa_flags = SA_SIGINFO;
    sigaction(SIGILL, &act, NULL);
}
#endif

using namespace smith;

/**
 * @brief Run buckling cylinder example
 *
 * @note Based on doi:10.1016/j.cma.2014.08.012
 */
int main(int argc, char* argv[])
{
  enable_floating_point_exceptions();
  constexpr int dim = 3;
  constexpr int p = 2;

  // Command line arguments
  // Mesh options
  int serial_refinement = 0;
  int parallel_refinement = 0;

  // Solver options
  NonlinearSolverOptions nonlinear_options = solid_mechanics::default_nonlinear_options;
  nonlinear_options.nonlin_solver = smith::NonlinearSolver::TrustRegion;
  nonlinear_options.relative_tol = 1e-6;
  nonlinear_options.absolute_tol = 1e-10;
  nonlinear_options.min_iterations = 1;
  nonlinear_options.max_iterations = 500;
  nonlinear_options.max_line_search_iterations = 20;
  nonlinear_options.print_level = 2;

  LinearSolverOptions linear_options = solid_mechanics::default_linear_options;
  linear_options.linear_solver = smith::LinearSolver::CG;
  linear_options.preconditioner = smith::Preconditioner::HypreAMG;
  linear_options.relative_tol = 1e-8;
  linear_options.absolute_tol = 1e-16;
  linear_options.max_iterations = 2000;

  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  std::string meshfile{"cap_thin.g"};
  double load = 0.2;
  double max_time = 10.0;
  int steps = 20;

  // Handle command line arguments
  axom::CLI::App app{"Viscoelastic buckling of a curved shell"};
  // Mesh options
  app.add_option("--serial-refinement", serial_refinement, "Serial refinement steps")->check(axom::CLI::PositiveNumber);
  app.add_option("--parallel-refinement", parallel_refinement, "Parallel refinement steps")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--mesh", meshfile, "Mesh file");
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
  //app.add_option("--dt", dt, "Size of time step")->check(axom::CLI::PositiveNumber);
  app.add_option("--steps", steps, "Number of time steps")->check(axom::CLI::PositiveNumber);
  app.add_option("--load", load, "Total load")->check(axom::CLI::PositiveNumber);
  app.add_option("--time", max_time, "Total time")->check(axom::CLI::PositiveNumber);
  // Misc options
  app.set_help_flag("--help");

  // Need to allow extra arguments for PETSc support
  app.allow_extras();
  CLI11_PARSE(app, argc, argv);

  double dt = max_time/steps;

    // Create DataStore
  std::string name = "viscoelastic_buckling";
  std::string mesh_tag = "mesh";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Create and refine mesh
  auto mesh = std::make_shared<smith::Mesh>(meshfile, mesh_tag, serial_refinement, parallel_refinement);
  //mesh->mfemParMesh().Print();
  auto mesh_coord_order = mesh->mfemParMesh().GetNodes()->FESpace()->GetMaxElementOrder();
  SLIC_WARNING_ROOT_IF(p != mesh_coord_order, axom::fmt::format("Displacement order p does not match mesh coord order at runtime, p = {}, mesh.p = {}", p, mesh_coord_order));

  // Surfaces for boundary conditions
  mesh->addDomainOfBoundaryElements("zsymm", smith::by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("xsymm", smith::by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("bottom", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("top", smith::by_attr<dim>(4));

  // temperature parameter field
  smith::FiniteElementState temperature(mesh->mfemParMesh(), smith::L2<0>{}, "temperature");
  temperature = 300.0;

  std::cout << "Nonlinear solver = " << nonlinear_options.nonlin_solver << std::endl;

  // Create solver
  smith::SolidMechanics<p, dim, smith::Parameters<smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name,
      mesh, {"temperature"}, 0, 0.0, false, false);
  
  solid_solver.setParameter(0, temperature);

  // compute area of loading zone
  smith::Functional<double(smith::H1<p, dim>)> area_computer({&solid_solver.displacement().space()});
  area_computer.AddBoundaryIntegral(smith::Dimension<dim - 1>{},
                                    DependsOn<0>{}, 
                                    [](double, auto, auto) { return 1.0; },
                                    mesh->domain("top"));
  double area = area_computer(0.0, solid_solver.displacement());
  SLIC_INFO_ROOT(axom::fmt::format("Load patch area = {}", area));

  solid_solver.setTraction(
    [&](auto, auto, double t) {
      return smith::vec3{0, -load/area*std::sin(M_PI*t/max_time), 0};
    },
     mesh->domain("top"));

  double rho = 1.0;
  double G_inf = 1e3;
  double G_0 = 3*G_inf;
  double G_g = G_inf + G_0;
  double nu_g = 0.45;
  double K = 2.0/3.0*G_g*(1.0 + nu_g)/(1.0 - 2.0*nu_g);
  double tau_0 = 1.0;
  double eta_0 = G_0*tau_0;
  double alpha_inf = 0.0;
  double theta_r = 300.0;
  double theta_sf = theta_r;
  double C_1 = 0.0;
  double C_2 = 50.0;
  
  // VISCOELASTIC
  solid_mechanics::ViscoelasticOldInterface mat(rho, K, G_inf, alpha_inf, theta_sf, G_0, eta_0, theta_r, C_1, C_2);
  auto internal_states = solid_solver.createQuadratureDataBuffer(smith::solid_mechanics::Viscoelastic::State{}, mesh->entireBody());
  solid_solver.setRateDependentMaterial(smith::DependsOn<0>{}, mat, mesh->entireBody(), internal_states);

  // NOTE: somehow J2 material works fine
  // using Hardening = solid_mechanics::LinearHardening;
  // Hardening hardening{.sigma_y = 10.0,  .Hi= G_inf/100.0, .eta = 0.0};
  // solid_mechanics::J2SmallStrain<Hardening> mat2{.E = G_inf, .nu = 0.25, .hardening = hardening, .Hk = 0.0, .density = 1.0};
  // auto internal_states = solid_solver.createQuadratureDataBuffer(smith::solid_mechanics::J2SmallStrain<Hardening>::State{}, mesh->entireBody());
  // solid_solver.setRateDependentMaterial(mat2, mesh->entireBody(), internal_states);

  // NEOHOOKEAN
  // solid_mechanics::NeoHookean mat{.density = 1.0, .K = K, .G = G_inf};
  // SLIC_INFO_ROOT(axom::fmt::format("K = {}, G = {}\n", K, G_inf));
  // solid_solver.setMaterial(mat, mesh->entireBody());
  

  solid_solver.setFixedBCs(mesh->domain("bottom"));
  solid_solver.setFixedBCs(mesh->domain("xsymm"), smith::Component::X);
  solid_solver.setFixedBCs(mesh->domain("zsymm"), smith::Component::Z);

  //std::cout << "Num of elements in domain['bottom'] = " << mesh->domain("bottom").total_elements() << std::endl;

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

  // post-processing functions
  auto [ranks, rank] = smith::getMPIInfo();
  axom::fmt::print("I am rank {} of {}\n", rank, ranks);
  mfem::Array<int> force_dof_list = mesh->domain("bottom").dof_list(&solid_solver.displacement().space());
  solid_solver.displacement().space().DofsToVDofs(1, force_dof_list);
  auto compute_net_force = [&force_dof_list, ranks](const smith::FiniteElementDual& reaction) -> double {
    double R{};
    for (int i = 0; i < force_dof_list.Size(); i++) {
      R += reaction(force_dof_list[i]);
    }
    double total;
    MPI_Allreduce(&R, &total, ranks, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return total;
  };

  smith::Functional<double(H1<p, dim>)> applied_displacement({&solid_solver.displacement().space()});
  applied_displacement.AddBoundaryIntegral(smith::Dimension<dim - 1>{},
                                           DependsOn<0>{}, 
                                           [area](double, auto, auto u) { return get<0>(u)[1]/area; },
                                           mesh->domain("top"));
  
  std::ofstream file("force_displacement.csv");
  if (rank == 0) file << "# time displacement force\n";

  // Save initial state
  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Qusi-static timestepping
  SLIC_INFO_ROOT(axom::fmt::format("Snap-through of viscoelastic shell\n{} displacement dofs",
                                   solid_solver.displacement().GlobalSize()));
  smith::logger::flush();
  while (solid_solver.time() < max_time && std::abs(solid_solver.time() - max_time) > DBL_EPSILON) {
    SLIC_INFO_ROOT("---------------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("start time = {}, dt = {}", solid_solver.time(), dt));
    smith::logger::flush();

    solid_solver.advanceTimestep(dt);

    // Output
    solid_solver.outputStateToDisk(paraview_name);
    double u = applied_displacement(solid_solver.time(), solid_solver.displacement());
    double f = compute_net_force(solid_solver.dual("reactions"));
    if (rank == 0) {
      file << solid_solver.time() << " " << u << " " << f << std::endl;
    }
  }
  file.close();
  SLIC_INFO_ROOT(axom::fmt::format("final time = {}", solid_solver.time()));

  return 0;
}
