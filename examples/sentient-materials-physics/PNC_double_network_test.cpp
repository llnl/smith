// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <fstream>
#include <functional>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <type_traits>
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"


#include "smith/smith_config.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
// #include "smith/differentiable_numerics/solid_mechanics_state_advancer.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/tests/paraview_helper.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"

#include "smith/physics/materials/thermoelastic_material.hpp"

#include "../../korner_examples/quasistatic_thermomechanics.hpp"
#include "../../korner_examples/calculate_reactions.hpp"


#define SLIC_INFO_ROOT_FLUSH(...) do { SLIC_INFO_ROOT(__VA_ARGS__); smith::logger::flush(); } while (0)

using namespace smith;

int main(int argc, char* argv[])
{
  SMITH_MARK_FUNCTION;

  /// ==================================================================
  /// ======================= Initializations ==========================
  /// ==================================================================

  constexpr int order = 1;
  constexpr int dim = 3;

  using ShapeDispSpace = H1<1, dim>;
  using VectorSpace = H1<order, dim>;
  using ScalarSpace = H1<order, 1>;
  using ScalarParameterSpace = L2<0>;
  // hardcoded values that were in Kevin's lua files
  size_t N_Steps = 500;
  double Total_Time = 50.0;
  int serial_refinement = 0;
  int parallel_refinement = 0;
  std::string mesh_name = "2d_plate.g"; //50x50x1 unit plate
  //Km,Gm,betam,rhom0,etam,Ke,Ge,betae,rhoe0,etae,C_v,kappa,Af,E_af,Ar,E_ar,R,Tr,gw,wm
  /*
  double Km = 0.5;         ///< matrix bulk modulus, MPa
   double Gm = 0.0073976;//0.001759;    ///< matrix shear modulus, MPa
   double betam = 0;
   double cvm = 1.0;
   double rhom0 = 1.0;
   double eta = 0.;//0.002;//0.005;     ///< viscosity, MPa-s
 
   double Ke = 0.5;       ///< entanglement bulk modulus, MPa
   double Ge = 0.225075;//0.0006408;     ///< entanglement shear modulus, MPa
   double cvc = 4.0;
   double rhoc0 = 1.0;
 
   // E_a and R can be SI units since they cancel out in the exponent
   double Af = 2.5e15;      ///< forward (low-high) exponential prefactor, 1/s
   double E_af = 1.5e5;     ///< forward (low-high) activation energy, J/mol
   double Ar = 1.0e-21;//4.2e-24;      ///< reverse exponential prefactor, 1/s
   double E_ar = -1.55e5;//-1.5e5;    ///< reverse activation energy, J/mol
   double R = 8.314;        ///< universal gas constant, J/mol/K
   double Tr = 353;         ///< reference temperature, K

   double gw = 0.2;         ///< particle weight fraction

   double wm = 0.5;         ///< matrix mass fraction
  */
  std::vector<double> material_params = {0.5,0.0073976,0.,1.0,0.,0.5,0.225075,0.,1.0,0.,1.5,30.,2.5e15,1.5e5,1.0e-21,-1.55e5,8.314,353.,0.2,0.5};
  // size_t num_coupling_iters = 1;
  int heat_linear_print_level = 0;
  int solid_linear_print_level = 0;
  int solid_nonlinear_print_level = 0;
  double external_heat_source = 0.0;
  auto applied_displacement_func2 = [](const double t, const smith::tensor<double, dim> & x){
    auto output = 0.0 * x;
    output[1] = -5.0 * sin(M_PI * t / 2.0);
    return output;
  };
  auto fixed_displacement = [](const double t, const smith::tensor<double, dim> & x){
    auto output = 0.0 * x;
    output[0] = 0.0*t;
    return output;
  };
  auto applied_temp = [](const double t, const smith::tensor<double, dim>){
    return 120.0+0.0*t; //only the temperature differential here
  };

  /// ==================================================================
  /// ================== Physics solver setup ==========================
  /// ==================================================================

  SLIC_INFO_ROOT_FLUSH("Initializing ApplicationManager and DataStore");
  /// -------------------------------------------------------------------

  smith::ApplicationManager applicationManager(argc, argv);
  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;

  auto name = "does_not_matter_output";
  SLIC_INFO_ROOT_FLUSH("Initializing StateManager");
  smith::StateManager::initialize(datastore, name);

  SLIC_INFO_ROOT_FLUSH("Creating mesh");
  /// ----------------------------------

  std::string mesh_location = SMITH_REPO_DIR "/data/meshes/" + mesh_name;
  std::string mesh_tag = "optimal";
  mesh = std::make_shared<smith::Mesh>(smith::buildMeshFromFile(mesh_location), mesh_tag, serial_refinement,
                                       parallel_refinement);

  // Extracting boundary domains for boundary conditions
  mesh->addDomainOfBoundaryElements("fix_bottom", smith::by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("fix_top", smith::by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("fix_front", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("fix_back", smith::by_attr<dim>(4));
                                                     
  SLIC_INFO_ROOT_FLUSH("Building solvers");
  /// -------------------------------------

  smith::LinearSolverOptions heat_linear_options{.linear_solver = smith::LinearSolver::GMRES,
                                                 .relative_tol = 1e-8,
                                                 .absolute_tol = 1e-11,
                                                 .max_iterations = 50,
                                                 .print_level = heat_linear_print_level};

  smith::LinearSolverOptions solid_linear_options{.linear_solver = smith::LinearSolver::GMRES,
                                                  .preconditioner = smith::Preconditioner::HypreAMG,
                                                  .relative_tol = 1e-8,
                                                  .absolute_tol = 1e-11,
                                                  .max_iterations = 10000,
                                                  .print_level = solid_linear_print_level};

  smith::NonlinearSolverOptions solid_nonlinear_opts{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                     .relative_tol = 1.0e-8,
                                                     .absolute_tol = 1.0e-11,
                                                     .max_iterations = 500,
                                                     .print_level = solid_nonlinear_print_level};
                                                     
  std::shared_ptr<DifferentiableSolver> d_heat_linear_solver =
      buildDifferentiableLinearSolve(heat_linear_options, *mesh);

  std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver =
      buildDifferentiableNonlinearSolve(solid_nonlinear_opts, solid_linear_options, *mesh);

  SLIC_INFO_ROOT_FLUSH("Setting up time integration rules");
  /// -----------------------------------------------------

  smith::SecondOrderTimeIntegrationRule backward_euler_heat(1.0);
  smith::SecondOrderTimeIntegrationRule backward_euler_solid(1.0);

  SLIC_INFO_ROOT_FLUSH("Defining material properties");
  /// ------------------------------------------------

  double Km = material_params[0];       ///< matrix bulk modulus, MPa
  double Gm = material_params[1];       ///< matrix shear modulus, MPa
  double betam = material_params[2];    ///< matrix volumetric thermal expansion coefficient
  double rhom0 = material_params[3];    ///< matrix initial density
  double etam = material_params[4];     ///< matrix viscosity, MPa-s

  double Ke = material_params[5];       ///< entanglement bulk modulus, MPa
  double Ge = material_params[6];       ///< entanglement shear modulus, MPa
  double betae = material_params[7];    ///< entanglement volumetric thermal expansion coefficient
  double rhoe0 = material_params[8];    ///< entanglement (chain) initial density
  double etae = material_params[9];     ///< entanglement viscosity, MPa-s

  double Cv = material_params[10];      ///< net volumetric heat capacity (must account for matrix+chain+particle)
  double kappa = material_params[11];    ///< net thermal conductivity (must account for matrix+chain+particle)

  // E_a and R can be SI units since they cancel out in the exponent
  double Af = material_params[12];       ///< forward (low-high) exponential prefactor, 1/s
  double E_af = material_params[13];     ///< forward (low-high) activation energy, J/mol
  double Ar = material_params[14];       ///< reverse exponential prefactor, 1/s
  double E_ar = material_params[15];     ///< reverse activation energy, J/mol
  double R = material_params[16];        ///< universal gas constant, J/mol/K
  double Tr = material_params[17];       ///< reference temperature, K

  double gw = material_params[18];       ///< particle weight fraction
  double wm = material_params[19];       ///< matrix mass fraction (set to 0.5, not real for now)

  using Material = thermomechanics::ThermalStiffeningMaterial;
  Material material = Material{Km,Gm,betam,rhom0,etam,Ke,Ge,betae,rhoe0,etae,Cv,kappa,Af,E_af,Ar,E_ar,R,Tr,gw,wm};

  // warm-start.
  // implicit Newmark.
  SLIC_INFO_ROOT_FLUSH("Building initial states");
  /// --------------------------------------------

  std::string physics_name = "thermomech";
  auto [physics, solid_mechanics_weak_form, heat_transfer_weak_form, vector_bcs, scalar_bcs] =
      custom_physics::buildThermoMechanics<dim, ShapeDispSpace, VectorSpace, ScalarSpace, ScalarParameterSpace>(
          mesh, d_solid_nonlinear_solver, d_heat_linear_solver, backward_euler_solid, backward_euler_heat, physics_name, {"bulk"});

  SLIC_INFO_ROOT_FLUSH("Setting up boundary conditions");
  /// ---------------------------------------------------

  vector_bcs->setFixedVectorBCs<dim>(mesh->domain("fix_bottom"));
  vector_bcs->setVectorBCs<dim>(mesh->domain("fix_top"), applied_displacement_func2);
  scalar_bcs->setScalarBCs<dim>(mesh->domain("fix_bottom"), applied_temp);
  vector_bcs->setVectorBCs<dim>(mesh->domain("fix_front"), {2}, fixed_displacement);
  vector_bcs->setVectorBCs<dim>(mesh->domain("fix_back"), {2}, fixed_displacement);

  SLIC_INFO_ROOT_FLUSH("Setting up weak forms for solid mechanics and heat transfer");
  /// ---------------------------------------------------------------------------------

  solid_mechanics_weak_form->addBodyIntegral(
    smith::DependsOn<0, 1, 2, 3>{}, mesh->entireBodyName(),
    [material](const TimeInfo& time_info, auto /*X*/, auto u, auto v, auto /*a*/, auto theta, auto /*theta_dot*/,
                 auto /*theta_dot_dot*/, auto /*bulk*/){
      Material::State state;
      auto [pk, C_v, s0, q0] =
          material(time_info.dt(), state, get<DERIVATIVE>(u),
                  get<DERIVATIVE>(v), get<VALUE>(theta), get<DERIVATIVE>(theta));
      return smith::tuple{smith::zero{}, pk};
    });

  heat_transfer_weak_form->addBodyIntegral(
    smith::DependsOn<0, 1, 2, 3>{}, mesh->entireBodyName(),
    [material](const TimeInfo& t_info, auto /*X*/, auto theta, auto theta_dot, auto /*theta_dot_dot*/, auto u, auto v,
                 auto /*a*/, auto /*bulk*/)  {
      Material::State state;
      auto [pk, C_v, s0, q0] =
          material(t_info.dt(), state, get<DERIVATIVE>(u), get<DERIVATIVE>(v), get<VALUE>(theta), get<DERIVATIVE>(theta));
      auto dT_dt = get<VALUE>(theta_dot);
      return smith::tuple{C_v * dT_dt - s0, -q0};
    });

  heat_transfer_weak_form->addBodySource(
      smith::DependsOn<>(), mesh->entireBodyName(),
      [external_heat_source](auto /*t*/, auto /* x */) { return external_heat_source; });
  // heat_transfer_weak_form->addBodySource(smith::DependsOn<>(), mesh->entireBodyName(), external_heat_source);
  
  auto shape_disp = physics->getShapeDispFieldState();
  auto params = physics->getFieldParams();
  auto states = physics->getInitialFieldStates();

  physics->resetStates();

  /// ==================================================================
  /// =================== Solve forward problem ========================
  /// ==================================================================

  SLIC_INFO_ROOT_FLUSH("Creating Paraview writer and writing initial state");
  /// -----------------------------------------------------------------------

  auto pv_writer = smith::createParaviewOutput(*mesh, physics->getFieldStatesAndParamStates(), "paraview_heat_v2");
  pv_writer.write(0, physics->time(), physics->getFieldStatesAndParamStates());

  SLIC_INFO_ROOT_FLUSH("Starting time-stepping loop");
  /// ------------------------------------------------

  double time_increment = Total_Time / (static_cast<double>(N_Steps));
  for (size_t m = 0; m < N_Steps; ++m) {

    SLIC_INFO_ROOT_FLUSH(axom::fmt::format("\n... Solving Step = {}", m));
    physics->advanceTimestep(time_increment);

    TimeInfo time_info(physics->time() - time_increment, time_increment);
    auto reactions = physics->getStateAdvancer()->computeResultants(shape_disp, physics->getFieldStates(),
                                                                    physics->getFieldStatesOld(), params, time_info);
    double reaction = CalculateReaction(*reactions[0].get(), mesh, "fix_top", 1);
    SLIC_INFO_ROOT_FLUSH(axom::fmt::format("    Reaction = {}", reaction));

    // Compute reactions

    pv_writer.write(m + 1, physics->time(), physics->getFieldStatesAndParamStates());

    // // Get reaction force
    // const FiniteElementDual& all_reactions = solid_solver.dual("reactions");
    
    // auto X_Dir = createReactionDirection(solid_solver, 0, mesh, dim);
    // auto Y_Dir = createReactionDirection(solid_solver, 1, mesh, dim);
    // auto Z_Dir = createReactionDirection(solid_solver, 2, mesh, dim);

    // auto contact_reaction_X = innerProduct(all_reactions, X_Dir);
    // auto contact_reaction_Y = innerProduct(all_reactions, Y_Dir);
    // auto contact_reaction_Z = innerProduct(all_reactions, Z_Dir);

    // auto total_reaction = std::pow(std::pow(contact_reaction_X, 2.0) + std::pow(contact_reaction_Y, 2.0) + std::pow(contact_reaction_Z, 2.0), 0.5);

    // // Compute applied displacement at this time (Y component)
    // auto current_time = solid_solver.time();
    // auto applied_dip = contact_displacement_per_step * current_time;

    // // Write to file (only on rank 0)
    // if (myid == 0) {
    //     std::cout << "... i = " << i 
    //     << ", time = " << current_time 
    //     << ", applied_dip = " << applied_dip 
    //     << ", total_reaction = " << total_reaction 
    //     << std::endl;

    //     outputFile << i << ", " << current_time << ", " << applied_dip << ", " << total_reaction << std::endl;
    // }
  }

  /// ==================================================================
  /// ================== Quantities of Interest ========================
  /// ==================================================================

  // // smith::FunctionalObjective<dim, Parameters<ScalarSpace>> reaction_force("reaction_force", mesh, spaces({states[ThermoMechAdvancer::TEMPERATURE]}));
  // SLIC_INFO_ROOT_FLUSH("Setting up and evaluating objective function");
  // /// -----------------------------------------------------------------
  // smith::FunctionalObjective<dim, Parameters<ScalarSpace>> objective("integrated_squared_temperature", mesh,
  //                                                                    spaces({states[ThermoMechAdvancer::TEMPERATURE]}));
  // objective.addBodyIntegral(smith::DependsOn<0>(), mesh->entireBodyName(), [](auto /*t*/, auto /*X*/, auto T) {
  //   auto temperature = get<VALUE>(T);
  //   return temperature * temperature;
  // });

  // auto final_states = physics->getAllFieldStates();
  // DoubleState temperature_squared = smith::evaluate_objective(
  //     TimeInfo(0.0, 0.0, 0), shape_disp, {final_states[ThermoMechAdvancer::TEMPERATURE]}, &objective);
  // gretl::set_as_objective(temperature_squared);

  SLIC_INFO_ROOT_FLUSH("Main function completed successfully");
  return 0;
}