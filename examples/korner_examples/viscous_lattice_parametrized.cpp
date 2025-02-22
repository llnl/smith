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

int main(int argc, char* argv[])
{
  feenableexcept(FE_INVALID | FE_DIVBYZERO);

  serac::initialize(argc, argv);
  std::string params_file;
  std::string base_dir;
  mfem::OptionsParser argument_parser(argc, argv);
  argument_parser.AddOption(&params_file, "-p", "--params", "Location of Parameter File");
  argument_parser.AddOption(&base_dir, "-o", "--output", "Output directory for solutions");
  argument_parser.Parse();
  if (!argument_parser.Good()) {

    if (mfem::Mpi::Root()) {
      argument_parser.PrintUsage(std::cout);
    }
  }
  if (mfem::Mpi::Root()) {
    argument_parser.PrintOptions(std::cout);
  }
  std::ifstream inputFile(params_file);
  if (!inputFile) {
      std::cerr << "Error opening file!" << std::endl;
  }

  double value;
  std::vector<double> problem_params;
  while (inputFile >> value) { // Read values from file
      problem_params.push_back(value);
  }

  inputFile.close(); // Close the file
  // MFEM_ABORT("Testing this out");

  constexpr int dim = 3;
  constexpr int p = 1;

  // Mesh Options
  int serial_refinement = 0;
  int parallel_refinement = 2;
  //   double dt = 0.05;

  // Solver Options
  NonlinearSolverOptions nonlinear_options = solid_mechanics::default_nonlinear_options;

  LinearSolverOptions linear_options = solid_mechanics::default_linear_options;

  nonlinear_options.nonlin_solver = serac::NonlinearSolver::TrustRegion;

  nonlinear_options.relative_tol = 1e-6;
  nonlinear_options.absolute_tol = 1e-10;
  nonlinear_options.min_iterations = 1;
  nonlinear_options.max_iterations = 500;
  nonlinear_options.max_line_search_iterations = 20;
  nonlinear_options.print_level = 1;
#ifdef SERAC_USE_PETSC
  linear_options.linear_solver = serac::LinearSolver::GMRES;
  linear_options.preconditioner = serac::Preconditioner::HypreAMG;
  linear_options.relative_tol = 1e-8;
  linear_options.absolute_tol = 1e-16;
  linear_options.max_iterations = 2000;
#endif


  nonlinear_options.force_monolithic = linear_options.preconditioner != Preconditioner::Petsc;

  std::string name = "viscous_lattice";
  std::string mesh_tag = "mesh";
  axom::sidre::DataStore datastore;
  std::string output_data = base_dir + "/data";
  serac::StateManager::initialize(datastore, output_data);

  // Create and Refine Mesh
  std::string filename = SERAC_REPO_DIR "/data/meshes/larger_lattice.g";
  auto mesh = serac::buildMeshFromFile(filename);
  auto refined_mesh = mesh::refineAndDistribute(std::move(mesh), serial_refinement, parallel_refinement);
  auto& pmesh = serac::StateManager::setMesh(std::move(refined_mesh), mesh_tag);
  // MFEM_ABORT("Got here");

  // Surfaces for boundary conditions
  constexpr int bottom_attr{1};
  constexpr int top_attr{2};

  auto bottom = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(bottom_attr));
  auto top = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(top_attr));
  //    auto top = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(top_attr));

  Domain whole_mesh = EntireDomain(pmesh);

  std::vector<std::string> fieldnames{"disp_old"};
  FiniteElementState disp_old(StateManager::mesh(mesh_tag), serac::H1<p, dim>{});
  disp_old = 0.0;

  using ParamT = serac::Parameters<serac::H1<p, dim>>;

  // Creating Solver
  std::unique_ptr<SolidMechanics<p, dim, ParamT>> solid_solver =
      std::make_unique<serac::SolidMechanics<p, dim, ParamT>>(nonlinear_options, linear_options,
                                                              serac::solid_mechanics::default_quasistatic_options, name,
                                                              mesh_tag, fieldnames);
  solid_solver->setParameter(0, disp_old);

  // double ground_stiffness = 1.0e-1;
  double ground_stiffness = problem_params[0];
  solid_solver->addCustomDomainIntegral(
      DependsOn<0>{},
      [ground_stiffness](double /* t */, auto /*position*/, [[maybe_unused]] auto displacement, auto /*acceleration*/,
                         [[maybe_unused]] auto displacement_old) {
        return ground_stiffness * (displacement - displacement_old);
      },
      whole_mesh);

  tensor<double, dim> constant_force{};
  double force_val = -1.0e-4;
  // constant_force[1] = force_val;
  constant_force[0] = force_val;
  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
  solid_solver->addBodyForce(force, whole_mesh);
  // MFEM_ABORT("testing here");
  // Define a Neo-Hookean material
  auto lambda = 1.0;
  auto G = 0.1;
  solid_mechanics::NeoHookean mat{.density = 1.0, .K = (3 * lambda + 2 * G) / 3, .G = G};
  solid_solver->setMaterial(mat, whole_mesh);

  // Set up essential boundary conditions
  const double ramp = -0.3;
  auto top_compress = [&](const serac::tensor<double, dim>, double t) {
    serac::tensor<double, dim> u{};
    u[0] = u[2] = 0.0;
    u[1] = 0.5 * ramp * t;
    return u;
  };
  auto bottom_compress = [&](const serac::tensor<double, dim>, double t) {
    serac::tensor<double, dim> u{};
    u[0] = u[2] = 0.0;
    u[1] = -0.5 * ramp * t;
    return u;
  };
  solid_solver->setDisplacementBCs(top_compress, top, Component::ALL);
  solid_solver->setDisplacementBCs(bottom_compress, bottom, Component::ALL);
  // Finalize the data structures
  solid_solver->completeSetup();

  // Save initial state
  std::string paraview_name = base_dir + "/" + name + "_paraview";
  solid_solver->outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  SLIC_INFO_ROOT(axom::fmt::format("Running viscous lattice example with {} displacement dofs",
                                   solid_solver->displacement().GlobalSize()));
  SLIC_INFO_ROOT("Starting pseudo-timestepping.");
  serac::logger::flush();
  constexpr double T = 10.0;
  while (solid_solver->time() < T && std::abs(solid_solver->time() - T) > DBL_EPSILON) {
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of 10.0)", solid_solver->time()));
    serac::logger::flush();

    // Refine dt as contact starts
    // auto next_dt = solid_solver->time() < 0.65 ? dt : dt * 0.1;
    auto next_dt = 0.05;

    disp_old = solid_solver->state("displacement");
    solid_solver->setParameter(0, disp_old);

    solid_solver->advanceTimestep(next_dt);

    // Output the sidre-based plot files
    solid_solver->outputStateToDisk(paraview_name);
  }
  SLIC_INFO_ROOT(axom::fmt::format("final time = {}", solid_solver->time()));

  // Exit without error
  serac::exitGracefully();
}
