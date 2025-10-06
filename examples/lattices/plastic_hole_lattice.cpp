// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #define _GNU_SOURCE
#include <cfenv>   // C++ header for fenv
#include <fenv.h>  // for feenableexcept (GNU extension)
#include <mpi.h>
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <sstream>
#include <iomanip>
#include "axom/slic.hpp"
#include "mfem.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/physics/boundary_conditions/components.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/solid_mechanics.hpp"
#include "serac/serac.hpp"
#include <filesystem>
#include "../lua_loader/lua_loader.hpp"

constexpr int dim = 3;
constexpr int p = 1;
std::function<std::string(const std::string&)> petscPCTypeValidator = [](const std::string& in) -> std::string {
  return std::to_string(static_cast<int>(serac::mfem_ext::stringToPetscPCType(in)));
};

serac::FiniteElementState createReactionDirection(const serac::BasePhysics& solid_solver, int direction,
                                                  std::shared_ptr<serac::Mesh> mesh, std::string side_set_name)
{
  const serac::FiniteElementDual& reactions = solid_solver.dual("reactions");

  serac::FiniteElementState reactionDirections(reactions.space(), "reaction_directions");
  reactionDirections = 0.0;

  mfem::VectorFunctionCoefficient func(dim, [direction](const mfem::Vector& /*x*/, mfem::Vector& u) {
    u = 0.0;
    u[direction] = 1.0;
  });

  reactionDirections.project(func, mesh->domain(side_set_name));

  return reactionDirections;
}
namespace ViscousMaterials {

struct ViscousNeoHookean {
  static constexpr int dim = 3;  ///< spatial dimension

  double density;  ///< mass density
  double K;        ///< bulk modulus
  double G;        ///< shear modulus
  double mu;       ///< viscosity modulus
  double dt_;

  /// @brief variables required to characterize the hysteresis response
  struct State {
    // tensor<double, dim, dim> du_dx_old;
    serac::tensor<double, dim, dim> F_old = serac::DenseIdentity<dim>();
  };

  /** @brief calculate the first Piola stress, given the displacement gradient and previous material state */
  template <typename T>
  auto operator()(State& state, double dt, const serac::tensor<T, dim, dim>& du_dX) const
  {
    if (dt < 1.0e-10) {
      dt = dt_;
    }
    using std::log1p;
    constexpr double eps = 1.0e-12;
    constexpr auto I = serac::DenseIdentity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = dot(du_dX, transpose(du_dX)) + transpose(du_dX) + du_dX;

    auto logJ = log1p(detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I

    // Pull back to Piola
    auto F = du_dX + I;
    // auto F_old = state.du_dx_old + I;

    if (abs(dt) < eps) {
      // auto L = 0.0 * dot(state.F_old, inv(F)); // dF/dt =

      auto TK = lambda * logJ * I + G * B_minus_I;

      state.F_old = get_value(F);
      return dot(TK, inv(transpose(F)));
    } else {
      // auto L = sym(dot(F, inv(state.F_old))) / dt; // dF/dt =
      auto L = sym(F - state.F_old) / dt_;  // dF/dt =

      auto TK = lambda * logJ * I + G * B_minus_I + 0.5 * mu * L; //dot(L, inv(transpose(F)));

      state.F_old = get_value(F);
      return dot(TK, inv(transpose(F)));
    }
  }
};
};  // namespace ViscousMaterials

struct solve_options {
  std::string simulation_tag = "ring_pull";
  std::string mesh_location = "none";
  int serial_refinement = 0;
  int parallel_refinement = 0;
  double max_time = 1.0;
  int N_Steps = 200;
  double ground_stiffness = 0.0;
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
                                        .penalty = 1.0e3};
};

void lattice_squish(solve_options& so)
{
  bool lua_verbose = true;
  std::filesystem::path script_path = SERAC_REPO_DIR "/examples/lattices/plasticholelattice.lua";

  SLIC_INFO_ROOT("Loading Lua Parameters");

  // Loading Parameters/Functions From Lua Tables
  LuaLoader::LuaLoader lua_loader(script_path, lua_verbose);  // maintain this for the lifetime of the simulation

  std::string problem_type = lua_loader.ExtractLuaParameter<std::string>("problem_type", "nominal");

  so.simulation_tag = problem_type;
  if (problem_type == "nominal") {
    so.mesh_location = SERAC_REPO_DIR "/data/meshes/hole_array_nominal.g";
  } else if (problem_type == "optimized") {
    so.mesh_location = SERAC_REPO_DIR "/data/meshes/hole_array_optimized.g";
  } else if (problem_type == "synthetic") {
    so.mesh_location = SERAC_REPO_DIR "/data/meshes/circle_lattice.g";
  } else {
    MFEM_ABORT("Error wrong problem type specified");
  }

  so.enable_contact = lua_loader.ExtractLuaParameter<bool>("use_contact", false);

  std::function<serac::vec3(const serac::vec3&, const double)> default_func =
      [](const serac::vec3&, const double) -> serac::vec3 { return serac::vec3{0.0, 0.0, 0.0}; };
  std::function<double(const double)> scalar_default_func = [](const double) { return 0.0; };

  std::function<serac::vec3(const serac::vec3&, const double)> applied_displacement_func =
      lua_loader.ExtractLuaSpaceTimeFunction("applied_displacement", default_func);

  std::function<double(const double)> strain_func =
      lua_loader.ExtractLuaScalarFunction("strain_function", scalar_default_func);

  so.max_time = lua_loader.ExtractLuaParameter<double>("simulation_time", 1.0);

  so.N_Steps = lua_loader.ExtractLuaParameter<int>("simulation_steps", 100);

  so.serial_refinement = lua_loader.ExtractLuaParameter<int>("serial_refinement", 0);

  so.parallel_refinement = lua_loader.ExtractLuaParameter<int>("parallel_refinement", 0);

  std::vector<double> material_params =
      lua_loader.ExtractLuaTable<double>("material_parameters", "values", {1.0, 1.0, 1.0, 1.0});

  double density = material_params[0];
  double K = material_params[1];
  double G = material_params[2];
  double mu = material_params[3];
  std::ostringstream mu_stream;
  mu_stream << std::fixed << std::setprecision(10) << mu;
  std::string mu_str = mu_stream.str();
  bool use_custom_location = false;

  std::string custom_output = lua_loader.ExtractLuaParameter<std::string>("output_location", "");

  if (custom_output != "") {
    use_custom_location = true;
  }

  // Creating DataStore
  const std::string simulation_tag = so.simulation_tag + "_" + mu_str;
  const std::string mesh_tag = simulation_tag + "_mesh";

  axom::sidre::DataStore datastore;
  std::string output_directory;
  if (use_custom_location) {
    output_directory = custom_output + "/" + simulation_tag;
  } else {
    output_directory = simulation_tag;
  }

  SLIC_INFO_ROOT("Initializing System");
  serac::StateManager::initialize(datastore, output_directory + "_data");

  // Loading Mesh
  SLIC_INFO_ROOT("Loading Mesh");
  auto pmesh = std::make_shared<serac::Mesh>(serac::buildMeshFromFile(so.mesh_location), mesh_tag, so.serial_refinement,
                                             so.parallel_refinement);

  // Extracting boundary domains for boundary conditions
  pmesh->addDomainOfBoundaryElements("fix_bottom", serac::by_attr<dim>(2));
  pmesh->addDomainOfBoundaryElements("fix_top", serac::by_attr<dim>(3));

  // Setting up Solid Mechanics Problem

  SLIC_INFO_ROOT("Initializing Solid Mechanics");
  serac::SolidMechanicsContact<p, dim> solid_solver(so.nonlinear_options, so.linear_options,
                                                    serac::solid_mechanics::default_quasistatic_options, "name", pmesh,
                                                    {}, 0, 0.0, false, false);

  // Setting Ground Stiffness

  // Defining Material Properties
  // auto lambda = 1.0;
  // auto G = 0.1;
  // serac::solid_mechanics::LinearIsotropic mat{.density = 1.0, .K = (3.0 * lambda + 2.0 * G) / 3.0, .G = G};

  SLIC_INFO_ROOT("Initializing Material");

  double dt = so.max_time / (static_cast<double>(so.N_Steps - 1));
  using MyMat = ViscousMaterials::ViscousNeoHookean;
  MyMat mat{.density = density, .K = K, .G = G, .mu = mu, .dt_ = dt};
  // using MyMat = serac::solid_mechanics::NeoHookean;
  // // using MyMat = serac::solid_mechanics::LinearIsotropic;
  // MyMat mat{.density = material_params[0], .K = material_params[1], .G = material_params[2]};

  auto internal_states = solid_solver.createQuadratureDataBuffer(MyMat::State{}, pmesh->entireBody());

  solid_solver.setRateDependentMaterial(mat, pmesh->entireBody(), internal_states);
  // Defining Boundary Conditions
  SLIC_INFO_ROOT("Setting Boundary Conditions");
  solid_solver.setFixedBCs(pmesh->domain("fix_bottom"));
  // solid_solver.setFixedBCs(pmesh->domain("fix_bottom"), serac::Component::X);
  // auto strain_rate = so.strain_rate;
  // auto applied_displacement = [strain_rate](serac::vec3, double t) {

  //   return serac::vec3{0.0, strain_rate * t, 0.0};
  // };

  solid_solver.setDisplacementBCs(applied_displacement_func, pmesh->domain("fix_top"));
  // solid_solver.setFixedBCs(pmesh->entireBody(), serac::Component::Z);

  // Adding Contact Interactions
  if (so.enable_contact) {
    SLIC_INFO_ROOT("Initializing Contact Interactions");
    auto self_contact_interaction_id_1 = 0;
    solid_solver.addContactInteraction(self_contact_interaction_id_1, {1}, {1}, so.contact_options);
  }
  // Completing Setup
  SLIC_INFO_ROOT("Completing Setup");
  solid_solver.completeSetup();

  // Running Quasistatics

  // Save Initial State
  std::string paraview_tag = output_directory + "_paraview";
  solid_solver.outputStateToDisk(paraview_tag);
  std::ofstream reaction_log;
  if (mfem::Mpi::Root()) {
    std::string log_name;
    if (use_custom_location) {
      log_name = output_directory + "/reaction_log_" + mu_str + ".csv";
    } else {
      log_name = "reaction_log_" + mu_str + ".csv";
    }
    reaction_log.open(log_name);
  }

  auto reactiondirection = createReactionDirection(solid_solver, 1, pmesh, "fix_top");
  for (int i = 1; i < so.N_Steps; ++i) {
    SLIC_INFO_ROOT("------------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("TIME STEP {}", i));
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of {})", solid_solver.time() + dt, so.max_time));
    serac::logger::flush();
    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk(paraview_tag);

    auto reactions = solid_solver.reactions();
    double val = serac::innerProduct(reactions, reactiondirection);
    if (mfem::Mpi::Root()) {
      std::cout << "---------------------------------" << std::endl;
      std::cout << "val: " << val << std::endl;
      std::cout << "---------------------------------" << std::endl;
      reaction_log << strain_func(solid_solver.time()) << "," << val << std::endl;
    }
  }
  if (mfem::Mpi::Root()) {
    reaction_log.close();
  }
}

int main(int argc, char* argv[])
{
  feenableexcept(FE_INVALID | FE_DIVBYZERO);
  // serac::initialize(argc, argv);

  serac::ApplicationManager applicationManager(argc, argv);
  solve_options so;
  so.enable_contact = false;

  so.mesh_location = SERAC_REPO_DIR "/data/meshes/circle_lattice.g";
  so.simulation_tag = "circle_lattice";
  so.serial_refinement = 1;
  so.parallel_refinement = 0;

  so.strain_rate = -1.0e0;
  so.linear_options = serac::LinearSolverOptions{//.linear_solver = serac::LinearSolver::Strumpack,
                                                 .linear_solver = serac::LinearSolver::CG,
                                                 // .linear_solver  = serac::LinearSolver::SuperLU,
                                                 // .linear_solver = serac::LinearSolver::GMRES,
                                                 .preconditioner = serac::Preconditioner::HypreJacobi,
                                                 //  .preconditioner = serac::Preconditioner::HypreAMG,
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
                                                       .max_iterations = 750,
                                                       .max_line_search_iterations = 15,  // for trust region: 15,
                                                       .print_level = 1};
  so.contact_options = serac::ContactOptions{.method = serac::ContactMethod::SingleMortar,
                                             .enforcement = serac::ContactEnforcement::Penalty,
                                             .type = serac::ContactType::Frictionless,
                                             .penalty = 1.0e-1,
                                             .jacobian = serac::ContactJacobian::Exact};

  lattice_squish(so);

  return 0;
}
