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

  int myid, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  // Set problem parameters. Only the input/output paths can be overwritten by the command line parser, below.
  const int order = 2; // displacement FE space polynomial order
  const int dim = 3;   // spatial dimension of the problem
  // const int serial_refinement = 0;
  // const int parallel_refinement = 0;

  // Read command line. Provide information and exit if requested. Otherwise check and overwrite problem parameters.
  axom::CLI::App app{"A driver for curing study analyses"};

  std::string mesh_prefix;
  app.add_option("-m, --mesh-prefix", mesh_prefix, "name prefix for input mesh files")->required();

  uint32_t num_parts = 4;
  app.add_option("-n, --num-parts", num_parts, "number of partitions on which to solve")->required();

  std::string output_directory = "./";
  app.add_option("-o, --output-directory", output_directory, "directory for writing any output files");

  app.parse(argc, argv);

  // NOTE
  // All spatial units are in mm.
  // Thus, loads in Newtons yields stresses and stiffnesses in units of MPa
  //       mass in kilograms yields densities in Tg/m^3
  //       density of kg/m^3 is equivalent to mg/mm^3
  const double shear_modulus_value = 43.9; // MPa [between 0.47 and 0.495]
  const double poisson_ratio_value = 0.48; // MPa [between 0.47 and 0.495]
  const double bulk_modulus_value = (2.0*shear_modulus_value*(1.0+poisson_ratio_value)) / (3.0*(1.0 - 2.0*poisson_ratio_value)); // MPa  
  const double mass_density_value = 1.20; // mg/mm^3; = 1.20 kg/m^3
  const double body_force_value = 11.76e-9; // N/mm^3; = (1.2 kg/m^3)*(9.8 m/s^2) = 11.76 N/m^3

  if (0 == myid) {
    std::cout << "Made it past initialization and CLI" << std::endl;
    std::cout << "    Mesh file prefix is '" << mesh_prefix << "'" << std::endl;
    std::cout << "    Expecting mesh to be split over " << num_parts << " partitions " << std::endl;
    std::cout << "    Output files will be written to '" << output_directory << "'" << std::endl;
    std::cout << "    Spatial dimension " << dim << " polynomial order " << order << std::endl;
  }
  MFEM_ASSERT(num_parts == comm_size, "Driver must be run with one rank per mesh part");

  // Load mesh from generated file; give it a name; pass to StateManager
  serac::StateManager::initialize(datastore, output_directory);
  const std::string mesh_tag = "mesh";
  std::stringstream mesh_name_stream;
  mesh_name_stream << mesh_prefix << "." << std::setfill('0') << std::setw(6) << myid;
  std::string mesh_name = mesh_name_stream.str();
  std::filebuf fb;
  fb.open(mesh_name.c_str(), std::ios::in);
  MFEM_ASSERT(&fb, "Could not open file '" + mesh_name + "'");
  std::istream is(&fb);
  auto mesh = std::make_unique<mfem::ParMesh>(mfem::ParMesh(MPI_COMM_WORLD, is, /* refine */ false));
  mesh->Finalize(/* refine */ false, /* fix orientation */ true);
  serac::StateManager::setMesh(std::move(mesh), mesh_tag);
  fb.close();
  auto &pmesh = serac::StateManager::mesh(mesh_tag);

  // Create domain of entire mesh
  ::serac::Domain entireDomain = ::serac::EntireDomain(pmesh);

  if (0 == myid) { std::cout << "ParMesh formed and passed to serac::StateManager." << std::endl; }

  double x_min(1e6), x_max(-1e6), y_min(1e6), y_max(-1e6), z_min(1e6), z_max(-1e6), xy_min(1e6), xy_max(-1e6);
  for (int i = 0; i < pmesh.GetNV(); ++i) {
    mfem::Vertex vertex(pmesh.GetVertex(i), dim);
    x_min  = std::min(x_min, vertex(0));            x_max  = std::max(x_max, vertex(0));
    y_min  = std::min(y_min, vertex(1));            y_max  = std::max(y_max, vertex(1));
    z_min  = std::min(z_min, vertex(2));            z_max  = std::max(z_max, vertex(2));
    xy_min = std::min(xy_min, vertex(0)+vertex(1)); xy_max = std::max(xy_max, vertex(0)+vertex(1));
  }
  double global_x_min, global_x_max, global_y_min, global_y_max, global_z_min, global_z_max, global_xy_min, global_xy_max;
  MPI_Allreduce(&x_min,  &global_x_min,  1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&x_max,  &global_x_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&y_min,  &global_y_min,  1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&y_max,  &global_y_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&z_min,  &global_z_min,  1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&z_max,  &global_z_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&xy_min, &global_xy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&xy_max, &global_xy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double z_range = global_z_max - global_z_min;
  double z_bottom_cutoff_value = global_z_min + 0.01 * z_range;
  double z_top_cutoff_value = global_z_min + 0.99 * z_range;
  double xy_range = global_xy_max - global_xy_min;
  double xy_cutoff_value = global_xy_min + 0.25 * xy_range;

  if (0 == myid) {
    std::cout << "Mesh vertex bounding box:"<< std::endl;
    std::cout << "    [" << global_x_min << ", " << global_x_max << "] x [" << global_y_min << ", " << global_y_max
              << "] x [" << global_z_min << ", " << global_z_max << "]" << std::endl;
    std::cout << "    Z bottom cutoff height: " << z_bottom_cutoff_value << std::endl;
  }

  // Create the solid mechanics solver
  auto linearOptions = serac::LinearSolverOptions {
      .linear_solver = serac::LinearSolver::CG,
      .preconditioner = serac::Preconditioner::HypreAMG, // HypreJacobi
      .relative_tol = 0.7*1.0e-9,
      .absolute_tol = 0.7*1.0e-9,
      .max_iterations = 5000,
      .print_level = 1};
  auto nonlinearOptions = serac::NonlinearSolverOptions {
      // .nonlin_solver  = serac::NonlinearSolver::Newton,
      .nonlin_solver  = serac::NonlinearSolver::TrustRegion,
      .relative_tol   = 1.0e-9,
      .absolute_tol   = 1.0e-9,
      .min_iterations = 1, 
      .max_iterations = 75,
      .max_line_search_iterations = 15,
      .print_level    = 1};
  SolidMechanics<order, dim, Parameters<H1<1>, H1<1>>> solid_solver(
      nonlinearOptions, linearOptions, serac::solid_mechanics::default_quasistatic_options,
      // serac::GeometricNonlinearities::Off, // TODO
      "curing_study_solid_neoHookean", mesh_tag, {"bulk_mod", "shear_mod"},
      0, // initial cycle index
      0.0, // initial time
      false, // checkpoint to disk
      false); // warmstart

  if (0 == myid) { std::cout << "Completed creation of SolidMechanics object; moving on to define loads, etc." << std::endl; }

  // Create user-defied material property fields. We can compute derivatives w.r.t. these terms. TODO add shape
  FiniteElementState user_defined_bulk_modulus(pmesh, H1<1>{}, "parameterized_bulk_modulus"); // TODO could use attribute or field value
  user_defined_bulk_modulus = bulk_modulus_value;
  FiniteElementState user_defined_shear_modulus(pmesh, H1<1>{}, "parameterized_shear_modulus"); // TODO could project coefficient
  user_defined_shear_modulus = shear_modulus_value;
  // FiniteElementState user_defined_mass_density(pmesh, H1<1>{}, "parameterized_density"); // TODO could use attribute or coeff
  // user_defined_mass_density = mass_density_value;

  // Define the material property fields as parameters for the solver.
  solid_solver.setParameter(0, user_defined_bulk_modulus);
  solid_solver.setParameter(1, user_defined_shear_modulus);

  // Create a material model for the solver. Material property fields override the values defined at initialization
  solid_mechanics::ParameterizedNeoHookeanSolid mat{mass_density_value, 0.0, 0.0}; // density, delta bulk, delta shear
  solid_solver.setMaterial(DependsOn<0, 1>{}, mat, entireDomain);

  // Set essential BCs
  int local_bc_count(0), global_bc_count;
  // auto zero_vector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };
  auto is_on_angled_top_or_bottom_patch = [&](std::vector<vec3> vertices, int /* attr */) {
    // if ((x(0) + x(1)) <= xy_cutoff_value && (x(2) <= z_bottom_cutoff_value || x(2) >= z_top_cutoff_value)) {
    //   ++local_bc_count;
    //   return true;
    // }
    // return false;
    return std::all_of(vertices.begin(), vertices.end(), [&](vec3 x) { 
      if ((x(0) + x(1)) <= xy_cutoff_value && (x(2) <= z_bottom_cutoff_value || x(2) >= z_top_cutoff_value)) {
        ++local_bc_count;
        return true;
      }
      return false;
     });
  };

  Domain angled_top_or_bottom_bundary_patch = Domain::ofBoundaryElements(pmesh, is_on_angled_top_or_bottom_patch);

  // solid_solver.setDisplacementBCs(is_on_angled_top_or_bottom_patch, zero_vector);
  solid_solver.setFixedBCs(angled_top_or_bottom_bundary_patch);
  MPI_Reduce(&local_bc_count, &global_bc_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (myid == 0) { std::cout << "Fixed BCs for a total of " << global_bc_count << " nodes across all ranks" << std::endl; }

  // Set body force (self weight)
  tensor<double, dim> constant_force;
  constant_force[0] = 0.0;
  constant_force[1] = 0.0;
  constant_force[2] = -body_force_value;
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
  std::string outname = "paraview_curing_study_driver";
  solid_solver.outputStateToDisk(outname);

  if (0 == myid) { std::cout << "Complete." << std::endl; }

  // TODO compute some QOIs

  // Close out problem.
  MPI_Finalize();

  return 0;
}
