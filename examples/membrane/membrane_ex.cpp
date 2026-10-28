// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file membrane_ex.cpp
 *
 * @brief A soft solid with a stiff skin using membrane elements.
 *
 * @note Run:
 * @code{.sh}
 * ./build/examples/membrane_ex -options_file examples/membrane/options.yml
 * @endcode
 */

#include <cmath>
#include <format>
#include <memory>
#include <string>
#include <type_traits>

#include "axom/slic.hpp"
#include "axom/inlet.hpp"
#include "axom/CLI11.hpp"
#include "mfem.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/smith.hpp"

using namespace smith;

struct MembraneMaterial {

  MembraneMaterial(double E, double nu, double h) :
      mu(0.5*E/(1.0 + nu)), lambda(E*nu/(1.0 - nu*nu)), h0(h)
  {}

  SMITH_HOST_DEVICE auto operator()(double /* time */, auto position, auto displacement,
                                    auto /* acceleration */) const
  {
    auto dX_dxi = get<DERIVATIVE>(position);
    auto g0 = dot(transpose(dX_dxi), dX_dxi); // metric tensor in reference configuration
    auto du_dxi = get<DERIVATIVE>(displacement);
    // surface Green's strain (resolved in parametric coordinate system)
    auto epsilon = 0.5*(dot(transpose(dX_dxi), du_dxi) + dot(transpose(du_dxi), dX_dxi));
    auto g0_dual = inv(g0); // contravariant metric (dual metric)
    auto tr_epsilon = inner(epsilon, g0_dual);
    auto sigma = 2*mu*(g0_dual * epsilon * g0_dual) + lambda*tr_epsilon*g0_dual;
    return tuple{zero{}, h0*dX_dxi*sigma};
  }

  double mu;     /// shear modulus
  double lambda; /// Lame's second modulus
  double h0;     /// membrane thickness
};

int main(int argc, char* argv[])
{
  constexpr int dim = 3;
  constexpr int p = 2;

  // Command line arguments
  // Mesh options
  int serial_refinement = 0;
  int parallel_refinement = 0;
  double dt = 0.1;

  // Solver options
  NonlinearSolverOptions nonlinear_options = solid_mechanics::default_nonlinear_options;
  nonlinear_options.nonlin_solver = smith::NonlinearSolver::TrustRegion;
  nonlinear_options.relative_tol = 5e-5;
  nonlinear_options.absolute_tol = 0;
  nonlinear_options.min_iterations = 1;
  nonlinear_options.max_iterations = 500;
  nonlinear_options.subspace_option = SubSpaceOptions::ALWAYS;
  nonlinear_options.print_level = 2;

  LinearSolverOptions linear_options = solid_mechanics::default_linear_options;
  linear_options.linear_solver = smith::LinearSolver::CG;
  linear_options.preconditioner = smith::Preconditioner::HypreAMG;
  linear_options.relative_tol = 1e-9;
  linear_options.absolute_tol = 0;
  linear_options.max_iterations = 2000;

  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  // Handle command line arguments
  axom::CLI::App app{"Hollow cylinder buckling example"};
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

  // Need to allow extra arguments for PETSc support
  app.set_help_flag("--help");
  app.allow_extras();
  CLI11_PARSE(app, argc, argv);

  // Create DataStore
  std::string name = "membrane";
  std::string mesh_tag = "mesh";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Create and refine mesh
  mfem::Element::Type element_type = mfem::Element::Type::HEXAHEDRON;
  double x_length = 0.25, y_length = 0.1, z_length = 0.025;
  auto serial_mesh = mfem::Mesh::MakeCartesian3D(25, 10, 4, element_type, x_length, y_length, z_length);
  serial_mesh.SetCurvature(2, false, dim, mfem::Ordering::byVDIM);
  auto mesh = std::make_shared<smith::Mesh>(std::move(serial_mesh),
                                            "mesh", 
                                            serial_refinement, 
                                            parallel_refinement);
  mesh->mfemParMesh().Print();

  // Surfaces for boundary conditions
  constexpr int xmin_attr{5};
  constexpr int xmax_attr{3};
  constexpr int ymin_attr{2};
  constexpr int ymax_attr{4};
  constexpr int zmin_attr{1};
  constexpr int zmax_attr{6};
  mesh->addDomainOfBoundaryElements("xmin", smith::by_attr<dim>(xmin_attr));
  mesh->addDomainOfBoundaryElements("xmax", smith::by_attr<dim>(xmax_attr));
  mesh->addDomainOfBoundaryElements("ymin", smith::by_attr<dim>(ymin_attr));
  mesh->addDomainOfBoundaryElements("ymax", smith::by_attr<dim>(ymax_attr));
  mesh->addDomainOfBoundaryElements("zmin", smith::by_attr<dim>(zmin_attr));
  mesh->addDomainOfBoundaryElements("zmax", smith::by_attr<dim>(zmax_attr));

  mfem::out << "xmin has " << mesh->domain("xmin").total_elements() << std::endl;
  mfem::out << "xmax has " << mesh->domain("xmax").total_elements() << std::endl;
  mfem::out << "ymin has " << mesh->domain("ymin").total_elements() << std::endl;
  mfem::out << "ymax has " << mesh->domain("ymax").total_elements() << std::endl;
  mfem::out << "zmin has " << mesh->domain("zmin").total_elements() << std::endl;
  mfem::out << "zmax has " << mesh->domain("zmax").total_elements() << std::endl;

  // Create solver
  auto solid_solver = std::make_unique<smith::SolidMechanics<p, dim>>(
    nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh);

  // Define a Neo-Hookean material
  double E = 100e3;
  double nu = 0.0;
  solid_mechanics::NeoHookean mat{.density = 1.0, 
                                  .K = E / 3.0 / (1.0 - 2.0*nu), 
                                  .G = 0.5*E / (1.0 + nu)};
  solid_solver->setMaterial(mat, mesh->entireBody());

  // Add stiff membrane on zmax surface
  MembraneMaterial membrane_material(100e6, 0.45, 1e-4);
  solid_solver->addCustomBoundarySourceFluxIntegral(DependsOn<>{}, membrane_material, mesh->domain("zmax"));

  // Set up essential boundary conditions
  solid_solver->setFixedBCs(mesh->domain("xmin"));
  // solid_solver->setFixedBCs(mesh->domain("zmin"), Component::Z);
  // solid_solver->setFixedBCs(mesh->domain("ymin"), Component::Y);
  // solid_solver->setFixedBCs(mesh->domain("ymax"), Component::Y);

  auto compress = [&](const smith::tensor<double, dim>, double t) {
    smith::tensor<double, dim> u{};
    u[0] = t*2.5e-2;
    return u;
  };
  solid_solver->setDisplacementBCs(compress, mesh->domain("xmax"), Component::ALL);
  
  // Finalize the data structures
  solid_solver->completeSetup();

  // Save initial state
  std::string paraview_name = name + "_paraview";
  solid_solver->outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  SLIC_INFO_ROOT(std::format("Running membrane example with {} displacement dofs",
                             solid_solver->displacement().GlobalSize()));
  SLIC_INFO_ROOT("Starting pseudo-timestepping.");
  smith::logger::flush();
  while (solid_solver->time() < 1.0 && std::abs(solid_solver->time() - 1) > DBL_EPSILON) {
    SLIC_INFO_ROOT(std::format("----------------------"));
    SLIC_INFO_ROOT(std::format("time = {} (out of 1.0)", solid_solver->time()));
    smith::logger::flush();

    solid_solver->advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver->outputStateToDisk(paraview_name);
  }
  SLIC_INFO_ROOT(std::format("final time = {}", solid_solver->time()));

  return 0;
}
