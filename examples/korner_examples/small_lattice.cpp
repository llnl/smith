// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <string>
#include <fstream>

#include <memory>
#include "axom/slic.hpp"
#include "mfem.hpp"
#include "serac/physics/boundary_conditions/components.hpp"
#include "serac/physics/solid_mechanics.hpp"
#include "serac/serac.hpp"
constexpr int dim = 3;
constexpr int p = 1;
std::function<std::string(const std::string&)> petscPCTypeValidator = [](const std::string& in) -> std::string {
  return std::to_string(static_cast<int>(serac::mfem_ext::stringToPetscPCType(in)));
};

struct solve_options {
  std::string simulation_tag = "ring_pull";
  std::string mesh_location = "none";
  int serial_refinement = 0;
  int parallel_refinement = 0;
  double max_time = 1.0;
  int N_Steps = 400;
  double ground_stiffness = 1.0e-8;
  double strain_rate = 1.0e-0;
  bool enable_contact = true;
  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 0};
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = serac::NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-10,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 200,
                                                  .print_level = 1};
  serac::ContactOptions contact_options{.method = serac::ContactMethod::SingleMortar,
                                        .enforcement = serac::ContactEnforcement::Penalty,
                                        .type = serac::ContactType::Frictionless,
                                        .penalty = 1.0e-3};
};

void lattice_squish(const solve_options& so)
{


  // Creating DataStore
  const std::string& simulation_tag = so.simulation_tag;
  const std::string mesh_tag = simulation_tag + "mesh";

  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, simulation_tag + "_data");

  // Loading Mesh
  auto pmesh = std::make_shared<serac::Mesh>(serac::buildMeshFromFile(so.mesh_location), mesh_tag, so.serial_refinement, so.parallel_refinement);
  auto & whole_mesh = pmesh->entireBody();

  // Extracting boundary domains for boundary conditions
  pmesh->addDomainOfBoundaryElements("fix_bottom", serac::by_attr<dim>(1));
  pmesh->addDomainOfBoundaryElements("fix_top", serac::by_attr<dim>(2));
  constexpr int sideset1{3};
  constexpr int sideset2{4};
  constexpr int sideset3{5};
  constexpr int sideset4{6};
  // constexpr int bottom_contact1{3};
  // constexpr int bottom_contact2{4};
  // constexpr int top_contact1{5};
  // constexpr int top_contact2{6};

  // Setting up Solid Mechanics Problem


  serac::SolidMechanicsContact<p, dim> solid_solver(so.nonlinear_options, so.linear_options, serac::solid_mechanics::default_quasistatic_options, "name", pmesh, {});

  // Setting Ground Stiffness

  // Defining Material Properties
  auto lambda = 1.0;
  auto G = 0.1;
  serac::solid_mechanics::NeoHookean mat{.density = 1.0, .K = (3.0 * lambda + 2.0 * G) / 3.0, .G = G};

  // Defining Boundary Conditions

  solid_solver.setMaterial(mat, whole_mesh);
  solid_solver.setFixedBCs(pmesh->domain("fix_bottom"),serac::Component::Y);
  // solid_solver.setFixedBCs(fix_bottom, serac::Component::X);
  solid_solver.setFixedBCs(pmesh->domain("fix_bottom"),serac::Component::X);
  auto strain_rate = so.strain_rate;
  auto applied_displacement = [strain_rate](serac::vec3, double t) {
    return serac::vec3{0.0, strain_rate * t, 0.0};
  };


  solid_solver.setDisplacementBCs(applied_displacement, pmesh->domain("fix_top"));
  solid_solver.setFixedBCs(pmesh->entireBody(), serac::Component::Z);
  // Adding Contact Interactions
  if (so.enable_contact) {
    auto contact_interaction_id_1 = 0;
    solid_solver.addContactInteraction(contact_interaction_id_1, {sideset1}, {sideset2}, so.contact_options);

    auto contact_interaction_id_2 = 1;
    solid_solver.addContactInteraction(contact_interaction_id_2, {sideset2}, {sideset3}, so.contact_options);

    auto contact_interaction_id_3 = 2;
    solid_solver.addContactInteraction(contact_interaction_id_3, {sideset3}, {sideset4}, so.contact_options);

    auto contact_interaction_id_4 = 3;
    solid_solver.addContactInteraction(contact_interaction_id_4, {sideset4}, {sideset1}, so.contact_options);

    bool self_contact = true;
    if (self_contact){
	    auto self_contact_interaction_id_1 = 4;
    	solid_solver.addContactInteraction(self_contact_interaction_id_1, {sideset1}, {sideset1}, so.contact_options);

	    auto self_contact_interaction_id_2 = 5;
    	solid_solver.addContactInteraction(self_contact_interaction_id_2, {sideset2}, {sideset2}, so.contact_options);

	    auto self_contact_interaction_id_3 = 6;
    	solid_solver.addContactInteraction(self_contact_interaction_id_3, {sideset3}, {sideset3}, so.contact_options);

	    auto self_contact_interaction_id_4 = 7;
    	solid_solver.addContactInteraction(self_contact_interaction_id_4, {sideset4}, {sideset4}, so.contact_options);
	    }
  }

  // Completing Setup
  solid_solver.completeSetup();

  // Running Quasistatics
  double dt = so.max_time / (static_cast<double>(so.N_Steps - 1));

  // Save Initial State
  std::string paraview_tag = simulation_tag + "_paraview";
  solid_solver.outputStateToDisk(paraview_tag);


  std::ofstream reaction_log;
  if (mfem::Mpi::Root()){
    reaction_log.open("reaction_log.csv");
  }

  for (int i = 1; i < so.N_Steps; ++i) {
    SLIC_INFO_ROOT("------------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("TIME STEP {}", i));
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of {})", solid_solver.time() + dt, so.max_time));
    serac::logger::flush();
    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk(paraview_tag);


  }
  if (mfem::Mpi::Root()){
    reaction_log.close();
  }
}

int main(int argc, char* argv[])
{
  // serac::initialize(argc, argv);

  serac::ApplicationManager applicationManager(argc, argv);
  solve_options so;
  so.linear_options = serac::LinearSolverOptions{//.linear_solver  = serac::LinearSolver::Strumpack,
                                                 .linear_solver = serac::LinearSolver::CG,
                                                 // .linear_solver  = serac::LinearSolver::SuperLU,
                                                 // .linear_solver  = serac::LinearSolver::GMRES,
                                                 //.preconditioner = serac::Preconditioner::HypreJacobi,
                                                 .preconditioner = serac::Preconditioner::HypreAMG,
                                                 .relative_tol = 0.7 * 1.0e-8,
                                                 .absolute_tol = 0.7 * 1.0e-10,
                                                 .max_iterations = 5000,  // 3*(numElements),
                                                 .print_level = 0};
  so.nonlinear_options = serac::NonlinearSolverOptions{//.nonlin_solver  = serac::NonlinearSolver::Newton,
                                                       // .nonlin_solver  = serac::NonlinearSolver::NewtonLineSearch,
                                                       .nonlin_solver = serac::NonlinearSolver::TrustRegion,
                                                       .relative_tol = 1.0e-8,
                                                       .absolute_tol = 1.0e-9,
                                                       .min_iterations = 1,  // for trust region
                                                       .max_iterations = 75,
                                                       .max_line_search_iterations = 15,  // for trust region: 15,
                                                       .print_level = 1};
  so.contact_options = serac::ContactOptions{.method = serac::ContactMethod::SingleMortar,
    .enforcement = serac::ContactEnforcement::Penalty,
    .type = serac::ContactType::Frictionless,
    .penalty = 1.0e3,
    .jacobian = serac::ContactJacobian::Exact
  };

  so.mesh_location = SERAC_REPO_DIR "/data/meshes/small_lattice.g";
  so.simulation_tag = "small_lattice_squish";
  so.serial_refinement = 1;
  so.parallel_refinement = 1;
  so.strain_rate = -3.0e0;
  so.ground_stiffness = 0.0;
  so.enable_contact = true;
  lattice_squish(so);
  // serac::exitGracefully();

  return 0;
}

