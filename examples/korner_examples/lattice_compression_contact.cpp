
// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mpi.h>
#include "serac/infrastructure/terminator.hpp"
#include "serac/numerics/solver_config.hpp"
#include "serac/physics/contact/contact_config.hpp"
#include "serac/physics/materials/solid_material.hpp"
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

#define USING_CONTACT

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
  bool        constrain_3D = false;
  std::string save_location = "";
  const int   N_Steps       = 100;
  double      T             = 2.0;

};

void run_lattice_compression(const running_parameters& rp)
{
  constexpr int p    = 1;
  constexpr int dim  = 3;
  std::string   name = "lattice_compression";

  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store

  std::string filename = SERAC_REPO_DIR "/data/meshes/lattice_contact2.g";
  // std::string filename = "/p/lustre1/korner1/serac_test/meshes/korner_lattice/hexmesh3.g";

  auto  mesh  = serac::mesh::refineAndDistribute(serac::buildMeshFromFile(filename), 0, 0);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), "beam_mesh");

  // serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 1};
  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::CG,
                                              .preconditioner = serac::Preconditioner::HypreJacobi,
                                              .relative_tol = 1.0e-10,
                                              .absolute_tol = 1.0e-12,
                                              .max_iterations = dim * 10000,
                                              .print_level = 0
  };
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = serac::NonlinearSolver::TrustRegion,
                                                  .relative_tol   = 1.0e-8,
                                                  .absolute_tol   = 1.0e-10,
                                                  .max_iterations = 200,
                                                  .print_level    = 1};

#ifdef USING_CONTACT
  serac::ContactOptions contact_options{.method      = serac::ContactMethod::SingleMortar,
    .enforcement = serac::ContactEnforcement::Penalty,
    .type        = serac::ContactType::Frictionless,
    .penalty     = 1.0e6};
  serac::SolidMechanicsContact<p, dim, serac::Parameters<serac::L2<0>, serac::L2<0>>> solid_solver(
    nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, name, "beam_mesh",
    {"bulk_mod", "shear_mod"});
#else
  serac::SolidMechanics<p, dim, serac::Parameters<serac::L2<0>, serac::L2<0>>> solid_solver(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, name, "beam_mesh",
      {"bulk_mod", "shear_mod"});
#endif

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
  solid_solver.setDisplacementBCs({2}, [](const mfem::Vector&, double t, mfem::Vector& u) {
    u.SetSize(dim);
    u    = 0.0;
    u[1] = -0.01 * t;
  });


  if (rp.constrain_3D){
    solid_solver.setDisplacementBCs([](const mfem::Vector &){return true;}, [](const mfem::Vector &){return 0.0;}, 2);
  }
  // solid_solver.addBodyForce(BodyForceType body_force, Domain &domain)
  // solid_solver.setTraction(TractionType traction_function, Domain &domain)

  tensor<double, dim> constant_force{};
  constant_force[0] = 1.0e-3;
  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force, whole_mesh);

  // Add the contact interaction
  #ifdef USING_CONTACT
  auto          contact_interaction_id = 0;
  std::set<int> surface_1_boundary_attributes;
  std::set<int> surface_2_boundary_attributes;
    surface_1_boundary_attributes = std::set<int>({5});
    surface_2_boundary_attributes = std::set<int>({4});

  solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes,
                                     surface_2_boundary_attributes, contact_options);
  #endif
  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;

  const int N_steps = rp.N_Steps;

  size_t cnt = 0;
  for (int i{0}; i < N_steps; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
    cnt++;
  }
}

}  // namespace serac

double norm_squared(const mfem::Vector& v)
{
  double val = 0.0;
  for (int i = 0; i < v.Size(); i++) {
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
  rp.constrain_3D = false;
  serac::run_lattice_compression(rp);



  serac::exitGracefully();
  return result;
}
