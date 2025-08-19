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

namespace Zimmerman_EOS {
struct NeoHookean {
  // using State = Empty;  ///< this material has no internal variables
  struct State {
    double internal_variable;
  };

  /**
   * @brief stress calculation for a NeoHookean material model
   *
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   *
   * @tparam T Number-like type for the displacement gradient components
   * @tparam dim Dimensionality of space
   * @param du_dX displacement gradient with respect to the reference configuration (displacement_grad)
   * @return The first Piola stress
   */
  template <typename T, int dim>
  SERAC_HOST_DEVICE auto operator()(State& state, const serac::tensor<T, dim, dim>& du_dX) const
  {
    using std::log1p;

    double new_value = state.internal_variable;

    constexpr auto I = serac::Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G + new_value;
    auto B_minus_I = dot(du_dX, transpose(du_dX)) + transpose(du_dX) + du_dX;

    auto logJ = log1p(detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I
    auto TK = lambda * logJ * I + G * B_minus_I;

    // Pull back to Piola
    auto F = du_dX + I;

    state.internal_variable = serac::get_value(new_value);
    return dot(TK, inv(transpose(F)));
  }

  double density;  ///< mass density
  double K;        ///< bulk modulus
  double G;        ///< shear modulus
};

struct ThermalStiffening {
  // using State = Empty;  ///< this material has no internal variables
  struct State {
    double w_H = 0.0;
    double time = 0.0;
  };

  /**
   * @brief stress calculation for a NeoHookean material model
   *
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   *
   * @tparam T Number-like type for the displacement gradient components
   * @tparam dim Dimensionality of space
   * @param du_dX displacement gradient with respect to the reference configuration (displacement_grad)
   * @return The first Piola stress
   */

  //  template<typename DeformationType>
  // SERAC_HOST_DEVICE auto temperature_function(double time, const DeformationType & X) const { 
  //   double ymin = -3.;
  //   double ymax = 25.;
  //   // double lambda = (serac::get<0>(serac::get<0>(X))[1] - ymin) / (ymax - ymin);
  //   double lambda = (serac::get<0>(X)[1] - ymin) / (ymax - ymin);
  //   double time_final = 1.0;
  //   if (time <= time_final){
  //     double T_bottom = 120.0 * time + 353;
  //     double T_top = 353;
  //     return lambda * T_top + (1.0 - lambda) * T_bottom;
  //   } else {
  //     double T_bottom = 120.0 * time_final + 353;
  //     double T_top = 353;
  //     return lambda * T_top + (1.0 - lambda) * T_bottom;
  //   } 
  // }

  template <typename T, int dim, typename TemperatureType>
  SERAC_HOST_DEVICE auto operator()(State& state, const serac::tensor<T, dim, dim>& du_dX, const TemperatureType &Temp) const
  {
    auto whp = state.w_H;

    auto current_time = state.time;

    auto temperature = serac::get<0>(Temp);
    auto value = A * exp(-E_a / (R * temperature)) * dt;

    auto wh = whp + (1. - whp) * value / (1. + value);
    // std::cout << "wh: " << wh << "\n";

    constexpr auto I = serac::Identity<dim>();

    auto F = du_dX + I;

    auto B = dot(F, transpose(F));
    auto trB = tr(B);
    auto B_bar = B - (trB / 3.0) * I;
    auto J = det(F);

    auto T0 = 353.;
    auto N = 0.02;
    auto Gl_eff = Gl * exp(-N * (temperature - T0));
    auto Tl = Gl_eff * pow(J, -2. / 3.) * B_bar + J * Kl * (J - 1.) * I;
    auto Th = Gh * pow(J, -2. / 3.) * B_bar + J * Kh * (J - 1.) * I;

    auto scaling = 1.0e0;
    auto TK = scaling * (wh * Th + (1. - wh) * Tl);

    // Pull back to Piola

    state.w_H = serac::get_value(wh);
    state.time = serac::get_value(current_time + dt);
    return dot(TK, inv(transpose(F)));
  }

  double density;  ///< mass density
  double Kl;       ///< bulk modulus
  double Gl;       ///< shear modulus

  double Kh;  ///< bulk modulus
  double Gh;  ///< shear modulus

  double dt;  ///< fixed dt

  double A;
  double E_a;

  double R;
};
};  // namespace Zimmerman_EOS
double CalculateReaction(serac::FiniteElementDual& reactions, const int face, const int direction)
{
  auto fespace = reactions.space();

  mfem::ParGridFunction mask_gf(&fespace);
  mask_gf = 0.0;
  auto fhan = [direction](const mfem::Vector&, mfem::Vector& y) {
    y = 0.0;
    y[direction] = 1.0;
  };

  mfem::VectorFunctionCoefficient boundary_coeff(3, fhan);
  mfem::Array<int> ess_bdr(fespace.GetParMesh()->bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[face] = 1;
  mask_gf.ProjectBdrCoefficient(boundary_coeff, ess_bdr);

  mfem::Vector projections = reactions;
  mfem::Vector mask_t(fespace.GetTrueVSize());
  mask_t = 0.0;
  fespace.GetRestrictionMatrix()->Mult(mask_gf, mask_t);
  projections *= mask_t;

  double local_sum = projections.Sum();
  MPI_Comm mycomm = fespace.GetParMesh()->GetComm();
  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mycomm);

  return global_sum;
}

struct solve_options {
  std::string simulation_tag = "ring_pull";
  std::string mesh_location = "none";
  int serial_refinement = 0;
  int parallel_refinement = 0;
  double max_time = 1.0;
  int N_Steps = 1000;
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

  double dt = so.max_time / (static_cast<double>(so.N_Steps - 1));
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, simulation_tag + "_data");

  // Loading Mesh
  auto pmesh = std::make_shared<serac::Mesh>(serac::buildMeshFromFile(so.mesh_location), mesh_tag, so.serial_refinement,
                                             so.parallel_refinement);
  auto& whole_mesh = pmesh->entireBody();

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

  std::vector<std::string> fieldnames{"X0"};
  serac::FiniteElementState X0(serac::StateManager::mesh(mesh_tag), serac::H1<p, dim>{}, "X0");

  serac::FiniteElementState T_fes(serac::StateManager::mesh(mesh_tag),serac::L2<p>{}, "Temperature");
  T_fes = 353.0;

  using ParamT = serac::Parameters<serac::L2<p>>;

  //   std::unique_ptr<serac::SolidMechanics<p, dim, ParamT>> solid_solver =
  //       std::make_unique<serac::SolidMechanics<p, dim, ParamT>>(so.nonlinear_options, so.linear_options,
  //                                                               serac::solid_mechanics::default_quasistatic_options,
  //                                                               so.simulation_tag, mesh_tag, fieldnames);

  serac::SolidMechanicsContact<p, dim, ParamT> solid_solver(so.nonlinear_options, so.linear_options,
                                                            serac::solid_mechanics::default_quasistatic_options, "name",
                                                            pmesh, {"Temperature"});

  solid_solver.setParameter(0, T_fes);
  // Setting Ground Stiffness

  //   solid_solver->addCustomBoundaryIntegral(serac::DependsOn<0>{}, ground_force);


  // Defining Material Properties
  // auto lambda = 1.0;
  // auto G = 0.1;
  // serac::solid_mechanics::NeoHookean mat{.density = 1.0, .K = (3.0 * lambda + 2.0 * G) / 3.0, .G = G};

  // solid_solver.setMaterial( mat, whole_mesh);
  using Material = Zimmerman_EOS::ThermalStiffening;
  double Kl = 0.5;        ///< bulk modulus
  double Gl = 0.0074;        ///< shear modulus

  double Kh = 0.5;        ///< bulk modulus
  double Gh = 0.225;        ///< shear modulus


  double A = 1.5e18;
  double E_a = 1.5e5;

  double R = 8.314;
  Material mat{.density=1.0,.Kl=Kl, .Gl=Gl,.Kh=Kh,.Gh=Gh,.dt=dt,.A=A,.E_a=E_a,.R=R};
  auto internal_states = solid_solver.createQuadratureDataBuffer(Material::State{},whole_mesh); 
  solid_solver.setMaterial(serac::DependsOn<0>{}, mat, whole_mesh, internal_states);
  // Defining Boundary Conditions

  solid_solver.setFixedBCs(pmesh->domain("fix_bottom"), serac::Component::Y);
  // solid_solver.setFixedBCs(fix_bottom, serac::Component::X);
  solid_solver.setFixedBCs(pmesh->domain("fix_bottom"), serac::Component::X);
  auto strain_rate = so.strain_rate;
  auto applied_displacement = [strain_rate](serac::vec3, double t) { return serac::vec3{0.0, strain_rate * t, 0.0}; };

  solid_solver.setDisplacementBCs(applied_displacement, pmesh->domain("fix_top"),serac::Component::Y);
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

    bool self_contact = false;
    if (self_contact) {
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

  // auto contact_interaction_id_top = 1;
  // solid_solver->addContactInteraction(contact_interaction_id_top, {top_contact1}, {top_contact2},
  // so.contact_options);

  // Completing Setup
  solid_solver.completeSetup();

  // Running Quasistatics

  // Save Initial State
  std::string paraview_tag = simulation_tag + "_paraview";
  solid_solver.outputStateToDisk(paraview_tag);

  std::ofstream reaction_log;
  if (mfem::Mpi::Root()) {
    reaction_log.open("reaction_log.csv");
  }

  for (int i = 1; i < so.N_Steps; ++i) {
    SLIC_INFO_ROOT("------------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("TIME STEP {}", i));
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of {})", solid_solver.time() + dt, so.max_time));
    serac::logger::flush();

    double current_time = solid_solver.time();
  {
    // Evaluating the X0 function. Im sure there is a better way to do this with serac
    auto space_identity = [current_time](const mfem::Vector& X) {
      double ymin = -3.5;
      double ymax = 27.5;
      // double lambda = (serac::get<0>(serac::get<0>(X))[1] - ymin) / (ymax - ymin);
      double lambda = (X[1] - ymin) / (ymax - ymin);
      double time_final = 1.0;
      if (current_time <= time_final) {
        double T_bottom = 120.0 * current_time + 353;
        double T_top = 353;
        return lambda * T_top + (1.0 - lambda) * T_bottom;
      } else {
        double T_bottom = 120.0 * time_final + 353;
        double T_top = 353;
        return lambda * T_top + (1.0 - lambda) * T_bottom;
      }
    };
    mfem::FunctionCoefficient coeff(space_identity);
    mfem::ParGridFunction T_fes_gf(&T_fes.space());
    T_fes_gf = 0.0;
    T_fes_gf.ProjectCoefficient(coeff);
    T_fes.space().GetRestrictionMatrix()->Mult(T_fes_gf, T_fes);
    solid_solver.setParameter(0, T_fes);
  }
  solid_solver.advanceTimestep(dt);
  solid_solver.outputStateToDisk(paraview_tag);

  auto reactions = solid_solver.reactions();
  double val = CalculateReaction(reactions, 2, 1);
  if (mfem::Mpi::Root()) {
    std::cout << "---------------------------------" << std::endl;
    std::cout << "val: " << val << std::endl;
    std::cout << "---------------------------------" << std::endl;
    reaction_log << solid_solver.time() << "," << val << std::endl;
  }
  }
  if (mfem::Mpi::Root()) {
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
                                                //  .preconditioner = serac::Preconditioner::HypreJacobi,
                                                 .preconditioner = serac::Preconditioner::HypreAMG,
                                                 .relative_tol = 0.7 * 1.0e-8,
                                                 .absolute_tol = 0.7 * 1.0e-10,
                                                 .max_iterations = 5000,  // 3*(numElements),
                                                 .print_level = 0};
  so.nonlinear_options = serac::NonlinearSolverOptions{//.nonlin_solver  = serac::NonlinearSolver::Newton,
                                                      //  .nonlin_solver  = serac::NonlinearSolver::NewtonLineSearch,
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
                                             .jacobian = serac::ContactJacobian::Exact};

  // so.mesh_location = SERAC_REPO_DIR "/data/meshes/5x5lattice2.g";
  so.mesh_location = SERAC_REPO_DIR "/data/meshes/larger_lattice2.g";

  so.simulation_tag = "5x5lattice_squish";
  so.serial_refinement = 1;
  so.parallel_refinement = 0;
  so.max_time = 10.0;
  so.strain_rate = -2.0e0;
  so.ground_stiffness = 0.0;
  so.enable_contact = true;
  so.N_Steps = 1000;
  lattice_squish(so);
  // serac::exitGracefully();

  return 0;
}
