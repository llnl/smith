
#include <set>
#include <string>
#include <cfenv>

#include "axom/slic/core/SimpleLogger.hpp"
#include "axom/inlet.hpp"

#include "mfem.hpp"

#include "serac/numerics/functional/domain.hpp"
#include "serac/physics/boundary_conditions/components.hpp"
#include "serac/physics/solid_mechanics_contact.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/serac_config.hpp"

using namespace serac;

std::function<std::string(const std::string&)> petscPCTypeValidator = [](const std::string& in) -> std::string {
  return std::to_string(static_cast<int>(mfem_ext::stringToPetscPCType(in)));
};

namespace NonViscousMaterials{

struct NeoHookean{
  static constexpr int dim = 3;         ///< spatial dimension

  double density;  ///< mass density
  double K;        ///< bulk modulus
  double G;        ///< shear modulus

  /// @brief variables required to characterize the hysteresis response
  // struct State {
  //   // tensor<double, dim, dim> du_dx_old;
  // };
  using State = Empty;

  /** @brief calculate the first Piola stress, given the displacement gradient and previous material state */
  template <typename T>
  auto operator()(State&, const tensor<T, dim, dim>& du_dX) const
  {
    using std::log1p;
    constexpr auto I = Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = dot(du_dX, transpose(du_dX)) + transpose(du_dX) + du_dX;

    auto logJ = log1p(detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I

    // Pull back to Piola
    auto F = du_dX + I;
    // auto F_old = state.du_dx_old + I;

    // state.du_dx_old = get_value(du_dX);
    // auto L = sym(dot(F, inv(F_old))) / dt; // dF/dt =  
    // auto L = dot(F_old, sym(F))/ dt; // dF/dt =  

    // auto TK = lambda * logJ * I + G * B_minus_I + 0.5 * mu * L;
    // auto trL = tr(L);
    auto TK = lambda * logJ * I + G * B_minus_I ;

    return dot(TK, inv(transpose(F)));

    // state.accumulated_plastic_strain += get_value(delta_eqps);
    // state.plastic_strain += get_value(delta_eqps) * get_value(Np);
  }

};
};
namespace ViscousMaterials{

struct ViscousNeoHookean{
  static constexpr int dim = 3;         ///< spatial dimension

  double density;  ///< mass density
  double K;        ///< bulk modulus
  double G;        ///< shear modulus
  double mu;      ///< viscosity modulus

  /// @brief variables required to characterize the hysteresis response
  struct State {
    // tensor<double, dim, dim> du_dx_old;
    tensor<double, dim, dim> F_old = DenseIdentity<3>();
  };

  /** @brief calculate the first Piola stress, given the displacement gradient and previous material state */
  template <typename T>
  auto operator()(State& state, double dt, const tensor<T, dim, dim>& du_dX) const
  {
    using std::log1p;
    constexpr double eps = 1.0e-12;
    constexpr auto I = Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = dot(du_dX, transpose(du_dX)) + transpose(du_dX) + du_dX;

    auto logJ = log1p(detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I

    // Pull back to Piola
    auto F = du_dX + I;
    // auto F_old = state.du_dx_old + I;

    state.F_old = get_value(F);
    if (abs(dt) < eps){
       
      // auto L = 0.0 * dot(state.F_old, inv(F)); // dF/dt =  
      auto L = 0.0 * DenseIdentity<3>();

      auto TK = lambda * logJ * I + G * B_minus_I + 0.5 * mu * L;

      return dot(TK, inv(transpose(F)));
    } else {

      // auto L = sym(dot(F, inv(state.F_old))) / dt; // dF/dt =  
      auto L = sym(F - state.F_old) / dt; // dF/dt =  

      auto TK = lambda * logJ * I + G * B_minus_I + 0.5 * mu * dot(L, inv(transpose(F)));

      return dot(TK, inv(transpose(F)));
    }
  }

};
};


template <class Physics>
void output(double u, double f, const Physics& solid, const std::string& paraview_tag, std::ofstream& file)
{
  solid.outputStateToDisk(paraview_tag);
  file << solid.time() << " " << u << " " << f << std::endl;
}
int ViscousTest(int argc, char* argv[]){


  constexpr int p = 2;
  constexpr int dim = 3;
  constexpr double x_length = 1.0;
  constexpr double y_length = 0.1;
  constexpr double z_length = 0.1;
  constexpr int elements_in_x = 10;
  constexpr int elements_in_y = 1;
  constexpr int elements_in_z = 1;

  int serial_refinements = 1;
  int parallel_refinements = 1;
  int time_steps = 200;
  double strain_rate = 1e-2;

  constexpr double E = 1.0;
  // constexpr double nu = 0.25;
  // constexpr double density = 1.0;

  constexpr double sigma_y = 0.001;
  // constexpr double sigma_sat = 3.0 * sigma_y;
  constexpr double strain_constant = 10 * sigma_y / E;
  // constexpr double eta = 1e-2;

  constexpr double max_strain = 3 * strain_constant;
  std::string output_filename = "viscous_uniaxial_fd.txt";

  // Handle command line arguments
  axom::CLI::App app{"Plane strain uniaxial extension of a bar."};
  // Mesh options
  app.add_option("--serial-refinements", serial_refinements, "Serial refinement steps", true);
  app.add_option("--parallel-refinements", parallel_refinements, "Parallel refinement steps", true);
  app.add_option("--time-steps", time_steps, "Number of time steps to divide simulation", true);
  app.add_option("--strain-rate", strain_rate, "Nominal strain rate", true);
  app.add_option("--output-file", output_filename, "Name for force-displacement output file", true);
  app.set_help_flag("--help");

  CLI11_PARSE(app, argc, argv);

  SLIC_INFO_ROOT(axom::fmt::format("strain rate: {}", strain_rate));
  SLIC_INFO_ROOT(axom::fmt::format("time_steps: {}", time_steps));

  double max_time = max_strain / strain_rate;

  // Create DataStore
  const std::string simulation_tag = "viscous_uniaxial";
  const std::string mesh_tag = simulation_tag + "mesh";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, simulation_tag + "_data");

  auto mesh = serac::mesh::refineAndDistribute(
      serac::buildCuboidMesh(elements_in_x, elements_in_y, elements_in_z, x_length, y_length, z_length),
      serial_refinements, parallel_refinements);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // create boundary domains for boundary conditions
  auto fix_x = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(5));
  auto fix_y = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(2));
  auto fix_z = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(1));
  auto apply_displacement = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(3));
  serac::Domain whole_mesh = serac::EntireDomain(pmesh);
 serac::LinearSolverOptions linearOptions = {
    //.linear_solver  = serac::LinearSolver::Strumpack, 
     .linear_solver  = serac::LinearSolver::CG,
    // .linear_solver  = serac::LinearSolver::SuperLU,
    // .linear_solver  = serac::LinearSolver::GMRES,
    //.preconditioner = serac::Preconditioner::HypreJacobi,
    .preconditioner = serac::Preconditioner::HypreAMG,
    .relative_tol   = 0.7*1.0e-8,
    .absolute_tol   = 0.7*1.0e-10,
    .max_iterations = 5000, //3*(numElements),
    .print_level    = 0
  };
  serac::NonlinearSolverOptions nonlinearOptions = {
    //.nonlin_solver  = serac::NonlinearSolver::Newton,
    // .nonlin_solver  = serac::NonlinearSolver::NewtonLineSearch,
     .nonlin_solver  = serac::NonlinearSolver::TrustRegion,
    .relative_tol   = 1.0e-8,
    .absolute_tol   = 1.0e-9,
     .min_iterations = 1, // for trust region
    .max_iterations = 75,
     .max_line_search_iterations = 15, // for trust region: 15,
    .print_level    = 1
  };
  // serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::, .print_level = 0};
  //
  // serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = serac::NonlinearSolver::Newton,
  //                                                 .relative_tol = 1.0e-6,
  //                                                 .absolute_tol = 1.0e-8,
  //                                                 .max_iterations = 200,
  //                                                 .print_level = 1};

  serac::SolidMechanics<p, dim> solid_solver(
      nonlinearOptions , linearOptions , serac::solid_mechanics::default_quasistatic_options, simulation_tag, mesh_tag);

  using MyMaterial = ViscousMaterials::ViscousNeoHookean;
  MyMaterial mymaterial{1.0, 10.0, .250, 1.0e-1};

  auto internal_states = solid_solver.createQuadratureDataBuffer(MyMaterial::State{}, whole_mesh);

  solid_solver.setRateDependentMaterial(mymaterial, whole_mesh, internal_states);

  solid_solver.setFixedBCs(fix_x, serac::Component::X);
  // solid_solver.setFixedBCs(fix_y, serac::Component::Y);
  // solid_solver.setFixedBCs(fix_z, serac::Component::Z);
  auto applied_displacement = [strain_rate](serac::vec3, double t) {
    return serac::vec3{strain_rate * x_length * t, 0., 0.};
  };
  solid_solver.setDisplacementBCs(applied_displacement, apply_displacement, serac::Component::ALL);

  solid_solver.completeSetup();

  double dt = max_time / (time_steps - 1);

  // get nodes and dofs to compute total force
  mfem::Array<int> dof_list = apply_displacement.dof_list(&solid_solver.displacement().space());
  solid_solver.displacement().space().DofsToVDofs(0, dof_list);

  auto compute_net_force = [&dof_list](const serac::FiniteElementDual& reaction) -> double {
    double R{};
    for (int i = 0; i < dof_list.Size(); i++) {
      R += reaction(dof_list[i]);
    }
    return R;
  };

  std::string paraview_tag = simulation_tag + "_paraview";
  std::ofstream file(output_filename);
  file << "# time displacement force" << std::endl;
  {
    // double u = applied_displacement(serac::vec3{}, solid_solver.time())[0];
    // double f = compute_net_force(solid_solver.dual("reactions"));
    // output(u, f, solid_solver, paraview_tag, file);
  }

  for (int i = 1; i < time_steps; ++i) {
    SLIC_INFO_ROOT("------------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("TIME STEP {}", i));
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of {})", solid_solver.time() + dt, max_time));
    serac::logger::flush();

    solid_solver.advanceTimestep(dt);

    double u = applied_displacement(serac::vec3{}, solid_solver.time())[0];
    double f = compute_net_force(solid_solver.dual("reactions"));
    output(u, f, solid_solver, paraview_tag, file);
  }

  file.close();
  return 0;
}

int NonViscousTest(int argc, char* argv[]){


  constexpr int p = 2;
  constexpr int dim = 3;
  constexpr double x_length = 1.0;
  constexpr double y_length = 0.1;
  constexpr double z_length = 0.1;
  constexpr int elements_in_x = 10;
  constexpr int elements_in_y = 1;
  constexpr int elements_in_z = 1;

  int serial_refinements = 1;
  int parallel_refinements = 1;
  int time_steps = 200;
  double strain_rate = 1e-2;

  constexpr double E = 1.0;
  // constexpr double nu = 0.25;
  // constexpr double density = 1.0;

  constexpr double sigma_y = 0.001;
  // constexpr double sigma_sat = 3.0 * sigma_y;
  constexpr double strain_constant = 10 * sigma_y / E;
  // constexpr double eta = 1e-2;

  constexpr double max_strain = 3 * strain_constant;
  std::string output_filename = "uniaxial_fd.txt";

  // Handle command line arguments
  axom::CLI::App app{"Plane strain uniaxial extension of a bar."};
  // Mesh options
  app.add_option("--serial-refinements", serial_refinements, "Serial refinement steps", true);
  app.add_option("--parallel-refinements", parallel_refinements, "Parallel refinement steps", true);
  app.add_option("--time-steps", time_steps, "Number of time steps to divide simulation", true);
  app.add_option("--strain-rate", strain_rate, "Nominal strain rate", true);
  app.add_option("--output-file", output_filename, "Name for force-displacement output file", true);
  app.set_help_flag("--help");

  CLI11_PARSE(app, argc, argv);

  SLIC_INFO_ROOT(axom::fmt::format("strain rate: {}", strain_rate));
  SLIC_INFO_ROOT(axom::fmt::format("time_steps: {}", time_steps));

  double max_time = max_strain / strain_rate;

  // Create DataStore
  const std::string simulation_tag = "uniaxial";
  const std::string mesh_tag = simulation_tag + "mesh";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, simulation_tag + "_data");

  auto mesh = serac::mesh::refineAndDistribute(
      serac::buildCuboidMesh(elements_in_x, elements_in_y, elements_in_z, x_length, y_length, z_length),
      serial_refinements, parallel_refinements);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // create boundary domains for boundary conditions
  auto fix_x = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(5));
  auto fix_y = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(2));
  auto fix_z = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(1));
  auto apply_displacement = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(3));
  serac::Domain whole_mesh = serac::EntireDomain(pmesh);

  serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::Strumpack, .print_level = 0};

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = serac::NonlinearSolver::TrustRegion,
                                                  .relative_tol = 1.0e-10,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 200,
                                                  .print_level = 1};

  serac::SolidMechanics<p, dim> solid_solver(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, simulation_tag, mesh_tag);

  // using Hardening = serac::solid_mechanics::VoceHardening;
  // using Material = serac::solid_mechanics::J2<Hardening>;
  // using MyMaterial = ViscousMaterials::ViscousNeoHookean;
  using MyNonViscousMaterial = NonViscousMaterials::NeoHookean;

  // Hardening hardening{sigma_y, sigma_sat, strain_constant, eta};
  // Material material{E, nu, hardening, density};
  // MyMaterial mymaterial{1.0, 10.0, .250, 0.0};
  MyNonViscousMaterial mymaterial{1.0, 10.0, .250};

  // auto internal_states = solid_solver.createQuadratureDataBuffer(MyMaterial::State{}, whole_mesh);

  // solid_solver.setRateDependentMaterial(mymaterial, whole_mesh, internal_states);
  solid_solver.setMaterial(mymaterial, whole_mesh);

  solid_solver.setFixedBCs(fix_x, serac::Component::X);
  // solid_solver.setFixedBCs(fix_y, serac::Component::Y);
  // solid_solver.setFixedBCs(fix_z, serac::Component::Z);
  auto applied_displacement = [strain_rate](serac::vec3, double t) {
    return serac::vec3{strain_rate * x_length * t, 0., 0.};
  };
  solid_solver.setDisplacementBCs(applied_displacement, apply_displacement, serac::Component::ALL);

  solid_solver.completeSetup();

  double dt = max_time / (time_steps - 1);

  // get nodes and dofs to compute total force
  mfem::Array<int> dof_list = apply_displacement.dof_list(&solid_solver.displacement().space());
  solid_solver.displacement().space().DofsToVDofs(0, dof_list);

  auto compute_net_force = [&dof_list](const serac::FiniteElementDual& reaction) -> double {
    double R{};
    for (int i = 0; i < dof_list.Size(); i++) {
      R += reaction(dof_list[i]);
    }
    return R;
  };

  std::string paraview_tag = simulation_tag + "_paraview";
  std::ofstream file(output_filename);
  file << "# time displacement force" << std::endl;
  {
    // double u = applied_displacement(serac::vec3{}, solid_solver.time())[0];
    // double f = compute_net_force(solid_solver.dual("reactions"));
    // output(u, f, solid_solver, paraview_tag, file);
  }

  for (int i = 1; i < time_steps; ++i) {
    SLIC_INFO_ROOT("------------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("TIME STEP {}", i));
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of {})", solid_solver.time() + dt, max_time));
    serac::logger::flush();

    solid_solver.advanceTimestep(dt);

    double u = applied_displacement(serac::vec3{}, solid_solver.time())[0];
    double f = compute_net_force(solid_solver.dual("reactions"));
    output(u, f, solid_solver, paraview_tag, file);
  }

  file.close();
  return 0;
}

int main(int argc, char* argv[])
{

  serac::initialize(argc, argv);
  // NonViscousTest(argc, argv);
  ViscousTest(argc, argv);
  serac::exitGracefully();

  return 0;
}
