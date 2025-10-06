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
#include "axom/slic.hpp"
#include "mfem.hpp"
#include "serac/physics/boundary_conditions/components.hpp"
#include "serac/physics/thermomechanics_monolithic.hpp"
#include "serac/serac.hpp"
#include <filesystem>
#include "../lua_loader/lua_loader.hpp"

constexpr int dim = 3;
constexpr int p = 1;
std::function<std::string(const std::string&)> petscPCTypeValidator = [](const std::string& in) -> std::string {
  return std::to_string(static_cast<int>(serac::mfem_ext::stringToPetscPCType(in)));
};


struct MyParameterizedThermoelasticMaterial {
  double density;    ///< density
  double E0;         ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha0;     ///< reference value of thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa0;     ///< thermal conductivity
  double dt;         ///< dt for rate terms

template <typename t, int dim>
auto greenstrain(const serac::tensor<t, dim, dim>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}
  /// internal variables for the material model
  struct State {
    double strain_trace;  ///< trace of Green-Saint Venant strain tensor
  };

  /**
   * @brief The number of parameters in the model
   *
   * @return The number of parameters in the model
   */
  static constexpr int numParameters() { return 3; }

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the temperature (number-like)
   * @tparam T3 Type of the temperature gradient components (number-like)
   * @tparam T4 Type of the coefficient of thermal expansion scale factor
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] theta Temperature
   * @param[in] grad_theta Temperature gradient
   * @param[in] DeltaE Parameterized Young's modulus offset
   * @param[in] DeltaKappa Parameterized thermal conductivity offset
   * @param[in] ScaleAlpha Parameterized thermal conductivity offset
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * First Piola stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */
  template <typename DispGradType, typename TempType, typename TempGradType, typename YoungsType, typename ConductType,
            typename CoupleType, int dim>
  auto operator()(State& state, const serac::tensor<DispGradType, dim, dim>& grad_u, TempType theta,
                  const serac::tensor<TempGradType, dim>& grad_theta, YoungsType DeltaE, ConductType DeltaKappa,
                  CoupleType ScaleAlpha) const
  {
    auto E = E0 * serac::get<0>(DeltaE);
    auto kappa = kappa0 + serac::get<0>(DeltaKappa);
    auto alpha = alpha0 * serac::get<0>(ScaleAlpha);

    auto K = E / (3.0 * (1.0 - 2.0 * nu));
    auto G = 0.5 * E / (1.0 + nu);
    static constexpr auto I = serac::Identity<dim>();
    auto F = grad_u + I;
    const auto Eg = 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
    const auto trEg = tr(Eg);

    // stress
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - 3.0 * alpha * (theta - theta_ref)) * I;
    const auto Piola = dot(F, S);

    // internal heat source
    const auto s0 = -3.0 * K * alpha * theta * (trEg - state.strain_trace) / dt;

    // heat flux
    const auto q0 = -kappa * grad_theta;

    state.strain_trace = serac::get_value(trEg);

    return serac::tuple{Piola, C_v, s0, q0};
  }
};

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
  std::filesystem::path script_path = SERAC_REPO_DIR "/examples/lattices/thermomech.lua";

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

  std::function<serac::vec3(const serac::vec3&, const double)> default_func =
      [](const serac::vec3&, const double) -> serac::vec3 { return serac::vec3{0.0, 0.0, 0.0}; };

  std::function<serac::vec3(const serac::vec3&, const double)> applied_displacement_func =
      lua_loader.ExtractLuaSpaceTimeFunction("applied_displacement", default_func);

  so.max_time = lua_loader.ExtractLuaParameter<double>("simulation_time", 1.0);

  so.N_Steps = lua_loader.ExtractLuaParameter<int>("simulation_steps", 100);

  so.serial_refinement = lua_loader.ExtractLuaParameter<int>("serial_refinement", 0);

  so.parallel_refinement = lua_loader.ExtractLuaParameter<int>("parallel_refinement", 0);

  std::vector<double> material_params =
      lua_loader.ExtractLuaTable<double>("material_parameters", "values", {1.0, 1.0, 1.0});

  bool use_custom_location = false;

  std::string custom_output = lua_loader.ExtractLuaParameter<std::string>("output_location", "");

  if (custom_output != "") {
    use_custom_location = true;
  }

  // Creating DataStore
  const std::string& simulation_tag = so.simulation_tag;
  const std::string mesh_tag = simulation_tag + "mesh";

  axom::sidre::DataStore datastore;
  std::string output_directory;
  if (use_custom_location) {
    output_directory = custom_output + "/" + simulation_tag;
  } else {
    output_directory = simulation_tag;
  }

  double dt = so.max_time / (static_cast<double>(so.N_Steps - 1));
  serac::StateManager::initialize(datastore, output_directory + "_data");

  // Loading Mesh
  auto pmesh = std::make_shared<serac::Mesh>(serac::buildMeshFromFile(so.mesh_location), mesh_tag, so.serial_refinement,
                                             so.parallel_refinement);

  // Extracting boundary domains for boundary conditions
  // constexpr int sideset1{1};
  constexpr int sideset_bottom{2};
  constexpr int sideset_top{3};
  constexpr int sideset_right{4};
  constexpr int sideset_left{5};
  pmesh->addDomainOfBoundaryElements("fix_bottom", serac::by_attr<dim>(sideset_bottom));
  pmesh->addDomainOfBoundaryElements("fix_top", serac::by_attr<dim>(sideset_top));
  pmesh->addDomainOfBoundaryElements("fix_right", serac::by_attr<dim>(sideset_right));
  pmesh->addDomainOfBoundaryElements("fix_left", serac::by_attr<dim>(sideset_left));
  serac::Domain bottomBoundary = serac::Domain::ofBoundaryElements(pmesh->mfemParMesh(), serac::by_attr<dim>(sideset_bottom));
  serac::Domain rightBoundary = serac::Domain::ofBoundaryElements(pmesh->mfemParMesh(), serac::by_attr<dim>(sideset_right));
  serac::Domain topBoundary = serac::Domain::ofBoundaryElements(pmesh->mfemParMesh(), serac::by_attr<dim>(sideset_top));
  serac::Domain leftBoundary = serac::Domain::ofBoundaryElements(pmesh->mfemParMesh(), serac::by_attr<dim>(sideset_left));
  serac::Domain entireDomain = serac::EntireDomain(pmesh->mfemParMesh());

  // Setting up Solid Mechanics Problem

  // serac::SolidMechanicsContact<p, dim> solid_solver(so.nonlinear_options, so.linear_options,
  //                                                   serac::solid_mechanics::default_quasistatic_options, "name", pmesh,
  //                                                   {}, 0, 0.0, false, false);
  using thermoMechType =
      serac::ThermomechanicsMonolithic<p, dim, serac::Parameters<serac::L2<0>, serac::L2<0>, serac::L2<0>>>;

  auto thermo_mech = thermoMechType(so.nonlinear_options, so.linear_options, std::string("thermomech"), pmesh,
                                    std::vector<std::string>{"E_bar", "Kappa_bar", "Alpha_bar"});
  // Defining Material Properties
  // Base material properties
  double rho = material_params[0];        // Density, kg/m³
  double E0 = material_params[1];         // Young's modulus, Pa
  double E = material_params[2];          // Young's modulus, Pa
  double nu = material_params[3];         // Poisson's ratio, dimensionless
  double c = material_params[4];          // Specific heat, J/(kg·K)
  double alpha0 = material_params[5];     // Thermal expansion, 1/K
  double alpha = material_params[6];      // Thermal expansion, 1/K
  double theta_ref = material_params[7];  // Reference temperature, K (20°C)
  double k0 = material_params[8];         // Thermal conductivity, W/(m·K)
  double k = material_params[9];          // Thermal conductivity, W/(m·K)
  using MyMat = MyParameterizedThermoelasticMaterial;
  MyMat mat{rho, E0, nu, c, alpha0, theta_ref, k0, dt};

  serac::FiniteElementState user_defined_youngs_modulus(
      serac::StateManager::newState(serac::L2<0>{}, "parameterized_youngs_modulus", mesh_tag));
  serac::FiniteElementState user_defined_thermal_conductivity(
      serac::StateManager::newState(serac::L2<0>{}, "parameterized_thermal_conductivity", mesh_tag));
  serac::FiniteElementState user_defined_coupling_coefficient(
      serac::StateManager::newState(serac::L2<0>{}, "parameterized_coupling_coefficient", mesh_tag));

  user_defined_youngs_modulus = E;
  user_defined_thermal_conductivity = k;
  user_defined_coupling_coefficient = alpha;

  thermo_mech.setParameter(0, user_defined_youngs_modulus);
  thermo_mech.setParameter(1, user_defined_thermal_conductivity);
  thermo_mech.setParameter(2, user_defined_coupling_coefficient);
  // Defining Boundary Conditions

  thermo_mech.setMaterial(serac::DependsOn<0,1,2>{},mat, pmesh->entireBody());
  thermo_mech.setFixedBCs(pmesh->domain("fix_bottom"));

  thermo_mech.setDisplacementBCs(applied_displacement_func, pmesh->domain("fix_top"));
  thermo_mech.setFixedBCs(pmesh->entireBody(), serac::Component::Z);

  thermo_mech.setTemperatureBCs({sideset_left}, [](const ::mfem::Vector &, double) { return 0.0; });
  thermo_mech.setTemperatureBCs({sideset_right}, [](const ::mfem::Vector &, double) { return 0.0; });

  thermo_mech.setSource(
    ::serac::DependsOn<> {},
    [=](auto, auto, auto, auto) { 
      double q0 = 1.0; // heat source in entire domain
      return q0;
    },
    entireDomain);

      // Set temperature gradients to be 0 along right and bottom boundaries
  auto flux_bc_function = [](auto /* X */, auto /* n */, auto /* time */, auto /* T */) { return 0.0; };
  thermo_mech.setFluxBCs(flux_bc_function, topBoundary);
  thermo_mech.setFluxBCs(flux_bc_function, bottomBoundary);
  // Completing Setup
  thermo_mech.completeSetup();

  // Running Quasistatics

  // Save Initial State
  std::string paraview_tag = output_directory + "_paraview";
  thermo_mech.outputStateToDisk(paraview_tag);

  for (int i = 1; i < so.N_Steps; ++i) {
    SLIC_INFO_ROOT("------------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("TIME STEP {}", i));
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of {})", thermo_mech.time() + dt, so.max_time));
    serac::logger::flush();
    thermo_mech.advanceTimestep(dt);
    thermo_mech.outputStateToDisk(paraview_tag);
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
  so.linear_options = serac::LinearSolverOptions{.linear_solver = serac::LinearSolver::Strumpack,
                                                 //  .linear_solver = serac::LinearSolver::CG,
                                                 // .linear_solver  = serac::LinearSolver::SuperLU,
                                                 //.linear_solver  = serac::LinearSolver::GMRES,
                                                 .preconditioner = serac::Preconditioner::HypreJacobi,
                                                 //  .preconditioner = serac::Preconditioner::HypreAMG,
                                                 .relative_tol = 0.7 * 1.0e-8,
                                                 .absolute_tol = 0.7 * 1.0e-10,
                                                 .max_iterations = 5000,  // 3*(numElements),
                                                 .print_level = 0};
  so.linear_options = serac::thermomechanics::direct_linear_options;
  so.linear_options.relative_tol = 1.0e-12;
  so.linear_options.absolute_tol = 1.0e-12;

  so.nonlinear_options = serac::NonlinearSolverOptions{
    //.nonlin_solver  = serac::NonlinearSolver::Newton,
    .nonlin_solver  = serac::NonlinearSolver::NewtonLineSearch,
    //.nonlin_solver  = serac::NonlinearSolver::TrustRegion,
    .relative_tol   = 1.0e-8,
    .absolute_tol   = 1.0e-9,
    .min_iterations = 1, // for trust region
    .max_iterations = 5000,
    .max_line_search_iterations = 30, // for trust region: 15,
    .print_level    = 1};

  lattice_squish(so);

  return 0;
}
