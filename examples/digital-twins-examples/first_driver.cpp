// Driver for Digital Twins SI to run generated meshes though a solid mechanics solver.
// Later, we can add Functional-based QOIs to define acceptance scores.

#include "serac/physics/solid_mechanics.hpp"

#include <functional>
#include <fstream>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "serac/infrastructure/about.hpp"
#include "serac/infrastructure/cli.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/serac_config.hpp"

// template <typename lambda>
// struct ParameterizedBodyForce {
//   template <int dim, typename T1, typename T2>
//   auto operator()(const tensor<T1, dim> x, double /*t*/, T2 density) const
//   {
//     return get<0>(density) * acceleration(x);
//   }
//   lambda acceleration;
// };
//
// template <typename T>
// ParameterizedBodyForce(T) -> ParameterizedBodyForce<T>;

int main(int argc, char *argv[])
{
  using namespace serac;

  // Initialize problem: init MPI; start logger; create data store; and init StateManager
  MPI_Init(&argc, &argv);
  axom::slic::SimpleLogger logger;
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "digital_twins_solid_mech_driver");

  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // Set problem parameters. Only the input/output paths can be overwritten by the command line parser, below.
  const int order = 1; // displacement FE space polynomial order
  const int dim = 3;   // spatial dimension of the problem
  const int serial_refinement = 0;
  const int parallel_refinement = 0;
  //std::string mesh_file = "./mesh-explorer.mesh";
  // std::string mesh_file = "../../twoLayerHemiPunchout.vtu";
  std::string mesh_file = "/g/g90/barrera/codes/serac-digital-twins/serac/data/meshes/twoLayerHemiPunchout.vtu";
  std::string output_directory = "./";

  // CLI IS STRICT ABOUT INPUT FILE EXISTING, WHICH IS COUNTER TO USAGE FOR PRE-PARTITIONED CASE. SINCE I WAS NOT
  // CHANGING THE COMMAND LINE ARGUMENTS ANYWAY, IT'S EASIEST TO JUST COMMENT OUT THIS WHOLE BLOCK.

  // // Read command line. Provide information and exit if requested. Otherwise check and overwrite problem parameters.
  // std::unordered_map<std::string, std::string> cli_opts = serac::cli::defineAndParse(argc, argv, "This is Serac");
  // bool print_version = cli_opts.find("version") != cli_opts.end();
  // if (print_version) {
  //   SLIC_INFO(serac::about());
  //   serac::exitGracefully();
  // }
  // auto search = cli_opts.find("input-file"); // hijacking this argument since Serac won't let you add your own willy-nilly
  // if (search != cli_opts.end()) {
  //   mesh_file = search->second;
  // }
  // search = cli_opts.find("output-directory");
  // if (search != cli_opts.end()) {
  //   output_directory = search->second;
  // }
  // axom::utilities::filesystem::makeDirsForPath(output_directory);
  //
  // // Output helpful run information
  // serac::printRunInfo();
  // serac::cli::printGiven(cli_opts);

  if (0 == myid) {
    std::cout << "Made it past initialization and CLI" << std::endl;
    std::cout << "Mesh file name is '" << mesh_file << "'" << std::endl;
  }

  // Load mesh from generated file; give it a name; pass to StateManager
  // UPDATE: handle the pre-partitioned mesh file
  const std::string mesh_tag = "mesh";
  bool doesNotEndWithDotVTU = (mesh_file.size() < 5) || (mesh_file.compare(mesh_file.size() - 4, 4, ".vtu") != 0);
  bool isHemisphere = doesNotEndWithDotVTU;
  if (doesNotEndWithDotVTU) { // presumed to be parallel partioned file

    if (0 == myid) { std::cout << "Mesh file does not end in '.vtu' so assume whole-hemi files" << std::endl; }

    std::stringstream suffix;
    suffix << "." << std::setfill('0') << std::setw(6) << myid;
    mesh_file += suffix.str();                          // every rank now has its own target file to read
    std::filebuf fb;
    fb.open(mesh_file.c_str(), std::ios::in);
    MFEM_ASSERT(&fb, "Could not open file '" + mesh_file + "'");
    std::istream is(&fb);
    auto mesh = std::make_unique<mfem::ParMesh>(mfem::ParMesh(MPI_COMM_WORLD, is, /* refine */ false));
                                           /* in current mfem/main: 0,      // generate edges */
                                           /*                       false); // fix orientation */
    mesh->Finalize(/* refine */ false, /* fix orientation */ true);
    serac::StateManager::setMesh(std::move(mesh), mesh_tag);
    fb.close();
  }
  else { // presumed to be the single punchout file

    if (0 == myid) { std::cout << "Mesh file ends in '.vtu' so assume serial file like punchout" << std::endl; }

    auto mesh = mesh::refineAndDistribute(buildMeshFromFile(mesh_file), serial_refinement, parallel_refinement);
    serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  }
  auto &pmesh = serac::StateManager::mesh(mesh_tag);

  // Create domain of entire mesh
  ::serac::Domain entireDomain = ::serac::EntireDomain(pmesh);

  if (0 == myid) { std::cout << "ParMesh formed and passed to serac::StateManager." << std::endl; }

  // Create the solid mechanics solver
  auto linearOptions = serac::LinearSolverOptions {
      .linear_solver = serac::LinearSolver::GMRES,
      .preconditioner = serac::Preconditioner::HypreAMG, // HypreJacobi
      .relative_tol = 4.0e-4,
      .absolute_tol = 4.0e-4,
      .max_iterations = 4000,
      .print_level = 1};
  auto nonlinearOptions = serac::NonlinearSolverOptions {
      .nonlin_solver  = serac::NonlinearSolver::Newton,
      .relative_tol   = 5.0e-4,
      .absolute_tol   = 5.0e-4,
      .max_iterations = 1,
      .print_level    = 1};
  SolidMechanics<order, dim, Parameters<H1<1>, H1<1>>> solid_solver(
      nonlinearOptions, linearOptions, serac::solid_mechanics::default_quasistatic_options,
      // serac::GeometricNonlinearities::Off, // TODO
      "parameterized_solid", mesh_tag, {"shear", "bulk"}); // TODO add density

  if (0 == myid) { std::cout << "Completed creation of SolidMechanics object; moving on to define loads, etc." << std::endl; }

  // Create user-defied material property fields. We can compute derivatives w.r.t. these terms. TODO add shape
  FiniteElementState user_defined_shear_modulus(pmesh, H1<1>{}, "parameterized_shear"); // TODO could project coefficient
  user_defined_shear_modulus = 2.69; // e6;
  FiniteElementState user_defined_bulk_modulus(pmesh, H1<1>{}, "parameterized_bulk"); // TODO could use attribute or field value
  user_defined_bulk_modulus = 4.39; // e6; // 43.9e6;

  // Define the material property fields as parameters for the solver.
  solid_solver.setParameter(0, user_defined_bulk_modulus);
  solid_solver.setParameter(1, user_defined_shear_modulus);

  // Create a material model for the solver. Material property fields override the values defined at initialization
  solid_mechanics::ParameterizedLinearIsotropicSolid mat{1.0, 0.0, 0.0}; // TODO density, delta bulk, delta shear
  solid_solver.setMaterial(DependsOn<0, 1>{}, mat, entireDomain);

  // Set essential BCs
  // auto zero_vector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };
  // This is appropriate for hemisphere
  // auto is_on_bottom = [](const mfem::Vector& x) {
  //   if (x(2) < 0.00) { // TODO
  //     return true;
  //   }
  //   return false;
  // };
  // This is appropriate for punchout
  // auto is_on_sides = [](const mfem::Vector& x) {
  //   if ( x(0) < 10.125801 || x(1) < 10.169919 || x(0) > 19.703215 || x(1) > 19.733761 ) {
  //     return true;
  //   }
  //   return false;
  // };

  auto is_on_bottom = [](std::vector<vec3> vertices, int /* attr */) {
    return std::all_of(vertices.begin(), vertices.end(), [](vec3 X) { 
      if (X(2) < 0.00) { // TODO
        return true;
      }
      return false;
     });
  };
  auto is_on_sides = [](std::vector<vec3> vertices, int /* attr */) {
    return std::all_of(vertices.begin(), vertices.end(), [](vec3 X) { 
      if ( X(0) < 10.125801 || X(1) < 10.169919 || X(0) > 19.703215 || X(1) > 19.733761 ) {
        return true;
      }
      return false;
     });
  };
  Domain bottomBoundary = Domain::ofBoundaryElements(pmesh, is_on_bottom);
  Domain sideBoundary = Domain::ofBoundaryElements(pmesh, is_on_sides);

  if (isHemisphere) {
    // solid_solver.setDisplacementBCs(is_on_bottom, zero_vector);
    solid_solver.setFixedBCs(bottomBoundary);
      
  }
  else {
    solid_solver.setFixedBCs(sideBoundary);
  }

  // Set body force (self weight) // TODO made dependent on density field
  tensor<double, dim> constant_force;
  constant_force[0] = 0.0;
  constant_force[1] = 0.0;
  constant_force[2] = -5.0e0; // TODO - set appropriately
  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force, entireDomain);

  // TODO
  //solid_solver.addBodyForce(DependsOn<1>{}, ParameterizedBodyForce{[](const auto& x) { return 0.0 * x; }});

  // Set a zero initial guess for the displacement solution
  FiniteElementState zero_state = solid_solver.displacement(); // (pmesh, H1<1>{}, "zero");
  zero_state = 0.0;
  solid_solver.setDisplacement(zero_state);

  // Finalize the data structures
  solid_solver.completeSetup();

  if (0 == myid) { std::cout << "Completed solid_solver setup. Calling solve...." << std::endl; }

  // Perform the quasi-static solve
  solid_solver.advanceTimestep(1.0);

  if (0 == myid) { std::cout << "Solve completed. Saving output...." << std::endl; }

  // Save problem state for later visualization
  std::string outname = "paraview_first_driver";
  solid_solver.outputStateToDisk(outname);

  if (0 == myid) { std::cout << "Complete." << std::endl; }

  // TODO compute some QOIs

  // Close out problem.
  MPI_Finalize();

  return 0;
}
