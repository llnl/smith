
// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mpi.h>
#include "serac/infrastructure/terminator.hpp"
#include "serac/physics/solid_mechanics.hpp"

#include <functional>
#include <fstream>
#include <mfem/fem/pgridfunc.hpp>
#include <mfem/linalg/vector.hpp>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/serac_config.hpp"
#include "serac/physics/solid_mechanics_contact.hpp"
#include <vector>
#include <iostream>
#include <fstream>


namespace serac {

namespace custom_material {

struct StVenantKirchhoff {
  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a St. Venant Kirchhoff material model
   *
   * @tparam T Type of the displacement gradient components (number-like)
   *
   * @param[in] grad_u Displacement gradient
   *
   * @return The Cauchy stress
   */
  template <typename T, int dim, typename Pre_Strain_Type>
  auto operator()(State&, const tensor<T, dim, dim>& grad_u, Pre_Strain_Type pre_strain_tuple) const
  {
    auto                  estar = get<0>(pre_strain_tuple);
    static constexpr auto I     = Identity<dim>();
    auto                  F     = grad_u + I;
    const auto            E     = 0.5 * (dot(transpose(F), F) - I) - estar * I;
    // const auto            E = greenStrain(grad_u) - estar * I;

    // stress
    const auto S     = K * tr(E) * I + 2.0 * G * dev(E);
    const auto P     = dot(F, S);
    const auto sigma = dot(P, transpose(F)) / det(F);

    return sigma;
  }

  double density;  ///< density
  double K;        ///< Bulk modulus
  double G;        ///< Shear modulus
};

}  // namespace custom_material
struct running_parameters {
  bool        self_contact  = false;
  std::string save_location = "";
  size_t      N_Steps       = 100;
  double      T             = 2.0;
};

std::vector<mfem::Vector> run_beam_contact(const running_parameters & rp){
  std::vector<mfem::Vector> all_data;
  std::cout << rp.self_contact << std::endl;
  constexpr int p = 1;
  constexpr int dim = 3;
  // const bool    self_contact = rp.self_contact;

  std::string name;
  if (rp.self_contact){
    name = "contact_beam_example_self_contact";
  } else {
    name = "contact_beam_example_standard_contact";
  }
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex-with-contact-block.mesh";

  auto  mesh  = serac::mesh::refineAndDistribute(serac::buildMeshFromFile(filename), 2, 0);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), "beam_mesh");

  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 1};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = serac::NonlinearSolver::TrustRegion,
                                                  .relative_tol   = 1.0e-12,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 200,
                                                  .print_level    = 1};

  serac::ContactOptions contact_options{.method      = serac::ContactMethod::SingleMortar,
                                        .enforcement = serac::ContactEnforcement::Penalty,
                                        .type        = serac::ContactType::Frictionless,
                                        .penalty     = 1.0e3};

  serac::SolidMechanicsContact<p, dim, serac::Parameters<serac::L2<0>, serac::L2<0>>> solid_solver(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, name, "beam_mesh",
      {"bulk_mod", "shear_mod"});

  serac::FiniteElementState K_field(serac::StateManager::newState(serac::L2<0>{}, "bulk_mod", "beam_mesh"));
  // each vector value corresponds to a different element attribute:
  // [0] (element attribute 1) : the beam
  // [1] (element attribute 2) : indenter block
  mfem::Vector             K_values({10.0, 100.0});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  serac::FiniteElementState G_field(serac::StateManager::newState(serac::L2<0>{}, "shear_mod", "beam_mesh"));
  // each vector value corresponds to a different element attribute:
  // [0] (element attribute 1) : the beam
  // [1] (element attribute 2) : indenter block
  mfem::Vector             G_values({0.25, 2.5});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  serac::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  serac::Domain                                        whole_mesh = serac::EntireDomain(pmesh);
  solid_solver.setMaterial(serac::DependsOn<0, 1>{}, mat, whole_mesh);

  // Pass the BC information to the solver object
  solid_solver.setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& u) {
    u.SetSize(dim);
    u = 0.0;
  });
  solid_solver.setDisplacementBCs({6}, [](const mfem::Vector&, double t, mfem::Vector& u) {
    u.SetSize(dim);
    u    = 0.0;
    u[2] = -0.005 * t;
  });

  // Add the contact interaction
  auto          contact_interaction_id = 0;
  std::set<int> surface_1_boundary_attributes;
  std::set<int> surface_2_boundary_attributes;
  if (rp.self_contact){

    surface_1_boundary_attributes = std::set<int>({5, 7});
    surface_2_boundary_attributes = std::set<int>({5, 7});
  } else {

    surface_1_boundary_attributes = std::set<int>({7});
    surface_2_boundary_attributes = std::set<int>({5});
  }

  solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes,
                                     surface_2_boundary_attributes, contact_options);
  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;

  const int N_steps = 270;
  all_data.resize(N_steps);

  const mfem::ParFiniteElementSpace* solids_fespace = solid_solver.displacement().gridFunction().ParFESpace();
  mfem::Vector displacement_t;
  int displacement_t_size = solids_fespace->GetTrueVSize();
  displacement_t.SetSize(displacement_t_size);
  solids_fespace->GetRestrictionMatrix()->Mult(solid_solver.displacement().gridFunction(), displacement_t);
  all_data[0] = displacement_t;
  size_t cnt = 0;
  for (int i{0}; i < N_steps; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
    all_data[cnt] = displacement_t;
    solids_fespace->GetRestrictionMatrix()->Mult(solid_solver.displacement().gridFunction(), displacement_t);
    cnt++;
  }

  return all_data;
}

std::vector<mfem::Vector> run_test(const running_parameters& rp)
{

  std::vector<mfem::Vector> all_data;
  constexpr int p            = 1;
  constexpr int dim          = 3;
  const bool    self_contact = rp.self_contact;

  MPI_Barrier(MPI_COMM_WORLD);
  auto PRE_STRAIN_FESPACE = H1<p>{};

  // Create DataStore
  std::string            name = "active_contact_example";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  // std::string filename = SERAC_REPO_DIR "/data/meshes/beam-quad.mesh";

  // std::string filename =
  // "/p/lustre1/korner1/serac_test/SI_Project_Korner/cmake_project_template/meshes/contact_mesh.msh";
  std::string filename = "/usr/WS2/korner1/sentient_materials_korner/01_SI_Project_Korner/meshes/contact_mesh2.g";
  // if (self_contact) {
  //   // filename = "/usr/WS2/korner1/sentient_materials_korner/01_SI_Project_Korner/meshes/contact_mesh_self.g";
  //   filename = "/usr/WS2/korner1/sentient_materials_korner/01_SI_Project_Korner/meshes/contact_mesh2.g";
  // } else {
  //   filename = "/usr/WS2/korner1/sentient_materials_korner/01_SI_Project_Korner/meshes/contact_mesh2.g";
  // }

  auto  mesh  = mesh::refineAndDistribute(buildMeshFromFile(filename), 0, 0);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), name);
  std::cout << "Loaded mesh" << std::endl;

  // serac::LinearSolverOptions linear_options{.linear_solver = LinearSolver::Strumpack,
  //                                           .preconditioner = Preconditioner::HypreAMG,
  //                                           .relative_tol = 1.0e-6,
  //                                           .absolute_tol = 1.0e-8,
  //                                           .max_iterations = 500,
  //                                           .print_level = 1};
  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 1};

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = serac::NonlinearSolver::Newton,
    .relative_tol   = 1.0e-12,
    .absolute_tol   = 1.0e-16,
    .max_iterations = 200,
    .print_level    = 1};

  serac::ContactOptions                            contact_options{.method      = serac::ContactMethod::SingleMortar,
    .enforcement = serac::ContactEnforcement::Penalty,
    .type        = serac::ContactType::Frictionless,
    .penalty     = 1.0e3};
  SolidMechanicsContact<p, dim, Parameters<H1<p>>> solid_solver(nonlinear_options, linear_options,
                                                                solid_mechanics::default_quasistatic_options,
                                                                "solid_mechanics", name, {"pre_strain"});

  // SolidMechanics<p, dim, Parameters<L2<p>>> solid_solver(nonlinear_options, linear_options,
  // solid_mechanics::default_quasistatic_options,
  //                                                               GeometricNonlinearities::On, "solid_mechanics",
  //                                                               mesh_tag, {"pre_strain"});
  double K = 1.91666666666667;
  double G = 1.0;
  // custom_material::StVenantKirchhoff mat{1.0, K, G, 0.001};
  // solid_mechanics::NeoHookean mat{1.0, K, G};
  // solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Define a boundary attribute set and specify initial / boundary conditions
  std::set<int> ess_bdr = {1};
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  // Add the contact interaction
  auto contact_interaction_id = 0;
  if (self_contact) {
    std::set<int> surface_1_boundary_attributes({2, 3});
    std::set<int> surface_2_boundary_attributes({2, 3});
    solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes,
                                       surface_2_boundary_attributes, contact_options);
  } else {
    std::set<int> surface_1_boundary_attributes({2});
    std::set<int> surface_2_boundary_attributes({3});
    solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes,
                                       surface_2_boundary_attributes, contact_options);
  }

  // Finalize the data structures
  //
  size_t             N_steps       = rp.N_Steps;
  double             T             = rp.T;
  double             t             = 0.0;
  double             dt            = T / static_cast<double>(N_steps - 1);
  double             pre_strain    = 0.0;
  double             max_prestrain = 0.2;
  all_data.resize(N_steps);
  FiniteElementState pre_strain_Param(StateManager::newState(PRE_STRAIN_FESPACE, "pre_strain", name));
  FiniteElementState pre_strain_mask(StateManager::newState(PRE_STRAIN_FESPACE, "pre_strain_mask", name));
  auto               pre_strain_mask_func = [](const mfem::Vector& x, double) -> double {
    if (x[1] < 2) {
      return 0.0;
    } else {
      return 1.0;
    }
  };
  mfem::FunctionCoefficient maskCoef(pre_strain_mask_func);
  pre_strain_mask.project(maskCoef);
  pre_strain_Param = 0.0;

  constexpr int PRE_STRAIN_INDEX = 0;
  solid_solver.setParameter(PRE_STRAIN_INDEX, pre_strain_Param);
  custom_material::StVenantKirchhoff mat{1.0, K, G};
  serac::Domain                      whole_mesh = serac::EntireDomain(pmesh);
  solid_solver.setMaterial(DependsOn<PRE_STRAIN_INDEX>{}, mat, whole_mesh);

  solid_solver.completeSetup();
  solid_solver.advanceTimestep(0.0);
  solid_solver.outputStateToDisk(rp.save_location);
  // solid_solver.adjoint

  // custom_material::StVenantKirchhoff mat{1.0, K, G, 0.001};
  const mfem::ParFiniteElementSpace* solids_fespace = solid_solver.displacement().gridFunction().ParFESpace();
  mfem::Vector displacement_t;
  int displacement_t_size = solids_fespace->GetTrueVSize();
  displacement_t.SetSize(displacement_t_size);
  solids_fespace->GetRestrictionMatrix()->Mult(solid_solver.displacement().gridFunction(), displacement_t);
  // size_t displacement_t_size = displacement_t.Size();
  all_data[0] = displacement_t;
  for (size_t i = 1; i < N_steps; i++) {
    if (i != (N_steps - 1)) {
      t += dt;
    } else {
      dt = T - t;
      t  = T;
    }
    SLIC_INFO_ROOT(axom::fmt::format("Time Step: {}\n", t));
    pre_strain = max_prestrain * static_cast<double>(i) / (static_cast<double>(N_steps - 1));
    // pre_strain_Param = pre_strain * pre_strain_mask;
    pre_strain_Param.Set(pre_strain, pre_strain_mask);
    solid_solver.setParameter(PRE_STRAIN_INDEX, pre_strain_Param);

    // Perform the quasi-static solve
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(rp.save_location);
    solids_fespace->GetRestrictionMatrix()->Mult(solid_solver.displacement().gridFunction(), displacement_t);
    all_data[i] = displacement_t;
  }
  return all_data;
}

void test(const running_parameters& rp)
{
  constexpr int p            = 1;
  constexpr int dim          = 3;
  const bool    self_contact = rp.self_contact;

  MPI_Barrier(MPI_COMM_WORLD);
  auto PRE_STRAIN_FESPACE = H1<p>{};

  // Create DataStore
  std::string            name = "active_contact_example";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  // std::string filename = SERAC_REPO_DIR "/data/meshes/beam-quad.mesh";

  // std::string filename =
  // "/p/lustre1/korner1/serac_test/SI_Project_Korner/cmake_project_template/meshes/contact_mesh.msh";
  std::string filename;
  if (self_contact) {
    filename = "/usr/WS2/korner1/sentient_materials_korner/01_SI_Project_Korner/meshes/contact_mesh_self.g";
    filename = "/usr/WS2/korner1/sentient_materials_korner/01_SI_Project_Korner/meshes/contact_mesh2.g";
  } else {
    filename = "/usr/WS2/korner1/sentient_materials_korner/01_SI_Project_Korner/meshes/contact_mesh2.g";
  }

  auto  mesh  = mesh::refineAndDistribute(buildMeshFromFile(filename), 0, 0);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), name);
  std::cout << "Loaded mesh" << std::endl;

  // serac::LinearSolverOptions linear_options{.linear_solver = LinearSolver::Strumpack,
  //                                           .preconditioner = Preconditioner::HypreAMG,
  //                                           .relative_tol = 1.0e-6,
  //                                           .absolute_tol = 1.0e-8,
  //                                           .max_iterations = 500,
  //                                           .print_level = 1};
  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 1};

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = serac::NonlinearSolver::Newton,
    .relative_tol   = 1.0e-12,
    .absolute_tol   = 1.0e-16,
    .max_iterations = 200,
    .print_level    = 1};

  serac::ContactOptions                            contact_options{.method      = serac::ContactMethod::SingleMortar,
    .enforcement = serac::ContactEnforcement::Penalty,
    .type        = serac::ContactType::Frictionless,
    .penalty     = 1.0e3};
  SolidMechanicsContact<p, dim, Parameters<H1<p>>> solid_solver(nonlinear_options, linear_options,
                                                                solid_mechanics::default_quasistatic_options,
                                                                "solid_mechanics", name, {"pre_strain"});

  // SolidMechanics<p, dim, Parameters<L2<p>>> solid_solver(nonlinear_options, linear_options,
  // solid_mechanics::default_quasistatic_options,
  //                                                               GeometricNonlinearities::On, "solid_mechanics",
  //                                                               mesh_tag, {"pre_strain"});

  double K = 1.91666666666667;
  double G = 1.0;
  // custom_material::StVenantKirchhoff mat{1.0, K, G, 0.001};
  // solid_mechanics::NeoHookean mat{1.0, K, G};
  // solid_solver.setMaterial(mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Define a boundary attribute set and specify initial / boundary conditions
  std::set<int> ess_bdr = {1};
  solid_solver.setDisplacementBCs(ess_bdr, bc);
  solid_solver.setDisplacement(bc);

  // Add the contact interaction
  auto contact_interaction_id = 0;
  if (self_contact) {
    std::set<int> surface_1_boundary_attributes({2, 3});
    std::set<int> surface_2_boundary_attributes({2, 3});
    solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes,
                                       surface_2_boundary_attributes, contact_options);
  } else {
    std::set<int> surface_1_boundary_attributes({2});
    std::set<int> surface_2_boundary_attributes({3});
    solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes,
                                       surface_2_boundary_attributes, contact_options);
  }

  // Finalize the data structures
  //
  size_t             N_steps       = rp.N_Steps;
  double             T             = rp.T;
  double             t             = 0.0;
  double             dt            = T / static_cast<double>(N_steps - 1);
  double             pre_strain    = 0.0;
  double             max_prestrain = 0.2;
  FiniteElementState pre_strain_Param(StateManager::newState(PRE_STRAIN_FESPACE, "pre_strain", name));
  FiniteElementState pre_strain_mask(StateManager::newState(PRE_STRAIN_FESPACE, "pre_strain_mask", name));
  auto               pre_strain_mask_func = [](const mfem::Vector& x, double) -> double {
    if (x[1] < 2) {
      return 0.0;
    } else {
      return 1.0;
    }
  };
  mfem::FunctionCoefficient maskCoef(pre_strain_mask_func);
  pre_strain_mask.project(maskCoef);
  pre_strain_Param = 0.0;

  constexpr int PRE_STRAIN_INDEX = 0;
  solid_solver.setParameter(PRE_STRAIN_INDEX, pre_strain_Param);
  custom_material::StVenantKirchhoff mat{1.0, K, G};
  serac::Domain                      whole_mesh = serac::EntireDomain(pmesh);
  solid_solver.setMaterial(DependsOn<PRE_STRAIN_INDEX>{}, mat, whole_mesh);

  solid_solver.completeSetup();
  solid_solver.advanceTimestep(0.0);
  solid_solver.outputStateToDisk(rp.save_location);
  // solid_solver.adjoint

  // custom_material::StVenantKirchhoff mat{1.0, K, G, 0.001};
  for (size_t i = 1; i < N_steps; i++) {
    if (i != (N_steps - 1)) {
      t += dt;
    } else {
      dt = T - t;
      t  = T;
    }
    SLIC_INFO_ROOT(axom::fmt::format("Time Step: {}\n", t));
    pre_strain = max_prestrain * static_cast<double>(i) / (static_cast<double>(N_steps - 1));
    // pre_strain_Param = pre_strain * pre_strain_mask;
    pre_strain_Param.Set(pre_strain, pre_strain_mask);
    solid_solver.setParameter(PRE_STRAIN_INDEX, pre_strain_Param);

    // Perform the quasi-static solve
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(rp.save_location);
    auto disp  = solid_solver.displacement();
    auto disp2 = solid_solver.displacement();
    auto error = disp.gridFunction().GetTrueVector();
  }
}



}  // namespace serac

double norm_squared(const mfem::Vector & v){
  double val = 0.0;
  for (int i = 0; i < v.Size(); i++){
    val += v[i] * v[i];
  }
  return val;
}

int main(int argc, char* argv[])
{
  // MPI_Init(&argc, &argv);
  serac::initialize(argc, argv); 
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int result = 0;

  axom::slic::SimpleLogger  logger;
  serac::running_parameters rp;
  rp.save_location = "data_output/paraview_output_self_contact";
  std::vector<mfem::Vector> self_contact;
  std::vector<mfem::Vector> standard_contact;
  // 2nd Solve
  rp.self_contact  = false;
  rp.save_location = "data_output/paraview_output_standard_contact";
  standard_contact = serac::run_beam_contact(rp);
  
  // return 0;
  // 1st Solve
  rp.save_location = "data_output/paraview_output_self_contact";
  rp.self_contact = true;
  self_contact = serac::run_beam_contact(rp);

  std::ofstream file;
  if (rank == 0){
    std::string error_file_name = "errors.csv";
    file.open(error_file_name);
  }
  // Calculating the errors
  for (size_t i = 0; i < standard_contact.size(); i++){
    auto error_vector = standard_contact[i];
    error_vector.Add(-1.0, self_contact[i]);
    
    double local_error_2 = norm_squared(error_vector);
    double global_error_2 = 0.0;

    MPI_Allreduce(&local_error_2, &global_error_2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0){
      double error = sqrt(global_error_2) / standard_contact[i].Size();
      std::cout << error << std::endl;
      file << error << "\n";
    }
  }
  file.close();
  // MPI_Finalize();
  // 

  serac::exitGracefully();
  return result;
}
