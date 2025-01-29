
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
struct StVenantKirchhoff2 {
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
  template <typename T, int dim>
  auto operator()(State&, const tensor<T, dim, dim>& grad_u) const
  {
    static constexpr auto I     = Identity<dim>();
    auto                  F     = grad_u + I;
    const auto            E     = 0.5 * (dot(transpose(F), F) - I);
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

struct ParameterizedNeoHookeanSolid {
  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a NeoHookean material model
   *
   * @tparam dim The spatial dimension of the mesh
   * @tparam DispGradType Displacement gradient type
   * @tparam BulkType Bulk modulus type
   * @tparam ShearType Shear modulus type
   * @param du_dX Displacement gradient with respect to the reference configuration (displacement_grad)
   * @param DeltaK The bulk modulus offset
   * @param DeltaG The shear modulus offset
   * @return The calculated material response (Cauchy stress) for the material
   */
  template <int dim, typename DispGradType>
  SERAC_HOST_DEVICE auto operator()(State& /*state*/, const serac::tensor<DispGradType, dim, dim>& du_dX) const
  {
    using std::log1p;
    constexpr auto I         = Identity<dim>();
    auto           K         = K0;
    auto           G         = G0;
    auto           lambda    = K - (2.0 / dim) * G;
    auto           B_minus_I = du_dX * transpose(du_dX) + transpose(du_dX) + du_dX;
    auto           logJ      = log1p(detApIm1(du_dX));

    // Kirchoff stress, in form that avoids cancellation error when F is near I
    auto TK = lambda * logJ * I + G * B_minus_I;

    // Pull back to Piola
    auto F = du_dX + I;
    return dot(TK, inv(transpose(F)));
  }

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 2; }

  double density;  ///< mass density
  double K0;       ///< base bulk modulus
  double G0;       ///< base shear modulus
};

}  // namespace custom_material
struct running_parameters {
  bool        self_contact  = false;
  std::string save_location = "";
  std::string mesh_location = "";
  const int   N_Steps       = 200;
  double      T             = 2.0;
  int N_Serial_Refinement = 0;
  int N_Parallel_Refinement = 0;
  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::CG,
                                              .preconditioner = serac::Preconditioner::HypreJacobi,
                                              .relative_tol = 1.0e-10,
                                              .absolute_tol = 1.0e-12,
                                              .max_iterations = 10000,
                                              .print_level = 0
  };
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = serac::NonlinearSolver::TrustRegion,
                                                  .relative_tol   = 1.0e-8,
                                                  .absolute_tol   = 1.0e-9,
                                                  .max_iterations = 200,
                                                  .print_level    = 1};

  serac::ContactOptions contact_options{.method      = serac::ContactMethod::SingleMortar,
    .enforcement = serac::ContactEnforcement::Penalty,
    // .type        = serac::ContactType::TiedNormal,
    .type        = serac::ContactType::Frictionless,
    .penalty     = 1.0e6};

};

void floating_structure_test(const running_parameters& rp)
{
  constexpr int p    = 1;
  constexpr int dim  = 3;
  std::string   name = rp.save_location;

  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store

  std::string filename = rp.mesh_location;
  // std::string filename = "/p/lustre1/korner1/serac_test/meshes/korner_lattice/hexmesh3.g";

  auto  mesh  = serac::mesh::refineAndDistribute(serac::buildMeshFromFile(filename), rp.N_Serial_Refinement, rp.N_Parallel_Refinement);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), "mesh");

  // serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 1};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif


  serac::SolidMechanicsContact<p, dim> solid_solver(
      rp.nonlinear_options, rp.linear_options, serac::solid_mechanics::default_quasistatic_options, name, "mesh");

  custom_material::ParameterizedNeoHookeanSolid mat{1.0, 1.91666666666, 0.5};
  serac::Domain                                        whole_mesh = serac::EntireDomain(pmesh);
  solid_solver.setMaterial(mat, whole_mesh);

  // Pass the BC information to the solver object
  solid_solver.setDisplacementBCs({1}, [](const mfem::Vector&, double t, mfem::Vector& u) {
    u.SetSize(dim);
    u    = 0.0;
    auto t_max = 1.5;
    auto ramp = 0.5;
    if (t < t_max){
      u[1] = ramp * t;
    } else {
      u[1] = ramp * (2.0 * t_max - t);
    }
  });
  solid_solver.setDisplacementBCs({4}, [](const mfem::Vector&, mfem::Vector& u) {
    u.SetSize(dim);
    u    = 0.0;
  });

  tensor<double, dim> constant_force{};
  constant_force[0] = 0.0;
  constant_force[1] = -1.0e-12;
  constant_force[2] = 0.0;
  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force, whole_mesh);

  auto contact_interaction_id = 0;
  std::set<int> surface_1_boundary_attributes;
  std::set<int> surface_2_boundary_attributes;
  surface_1_boundary_attributes = std::set<int>({2});
  surface_2_boundary_attributes = std::set<int>({3});

  solid_solver.addContactInteraction(contact_interaction_id, surface_1_boundary_attributes, surface_2_boundary_attributes, rp.contact_options);


  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve

  const int N_steps = rp.N_Steps;
  const double dt = rp.T / (static_cast<double>(N_steps) - 1.0);

  size_t cnt = 0;
  for (int i{0}; i < N_steps; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
    cnt++;
  }
}

}  // namespace serac


int main(int argc, char* argv[])
{
  // MPI_Init(&argc, &argv);
  serac::initialize(argc, argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int result = 0;

  axom::slic::SimpleLogger  logger;
  serac::running_parameters rp;
  rp.save_location = "floating_structure_contact_test";
  rp.mesh_location = SERAC_REPO_DIR "/data/meshes/twobodies.g";
  rp.N_Serial_Refinement = 0;
  rp.N_Parallel_Refinement = 0;
  serac::floating_structure_test(rp);



  serac::exitGracefully();
  return result;
}
