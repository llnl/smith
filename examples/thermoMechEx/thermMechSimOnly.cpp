// Copyright (c) 2015-2024, Lawrence Livermore National Security, LLC.
// All rights reserved.  LLNL-CODE-728517

// OFFICIAL USE ONLY This work was produced at the Lawrence Livermore
// National Laboratory (LLNL) under contract no. DE-AC52-07NA27344
// (Contract 44) between the U.S. Department of Energy (DOE) and
// Lawrence Livermore National Security, LLC (LLNS) for the operation
// of LLNL.  See license for disclaimers, notice of U.S. Government
// Rights and license terms and conditions.

#include "Lido.hpp"
#include "mfem.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/input.hpp"
#include "meshes/lido_meshes.hpp"
#include "serac/physics/heat_transfer.hpp"
#include "serac/physics/materials/thermal_material.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/physics/materials/parameterized_thermal_material.hpp"
#include "serac/physics/materials/parameterized_thermoelastic_material.hpp"

#include "serac/numerics/functional/tensor.hpp"
#include "serac/physics/thermomechanics_monolithic.hpp"
#include "serac/physics/materials/thermal_material.hpp"
#include "serac/physics/materials/solid_material.hpp"
//#include "serac/physics/materials/green_saint_venant_thermoelastic.hpp"

#include <fstream>


#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/infrastructure/application_manager.hpp"

#include <cfenv>
#include <cmath>

int main(int argc, char *argv[])
{
  feenableexcept(FE_INVALID | FE_DIVBYZERO);
  
  // Get MPI info. Initialize MPI, SLIC logging to standard out and Caliper profiling
  int numProcs, managerRank = 0, myRank;
  MPI_Comm comm = MPI_COMM_WORLD;
  ::std::shared_ptr<lido::LidoFinalize> lidoFinalize;
  ::std::tie(lidoFinalize, myRank, numProcs) = lido::initialize(argc, argv, comm);

  double Lx = 1.0; // mesh dimension in the x direction
  double Ly = 1.0; // mesh dimension in the y direction
  int Nx = 200; // number of elements in the x direction
  int Ny = 200; // number of elements in the y direction
  int numSerRef = 0;
  int numParRef = 0;

  // static constexpr int ORDER {1};
  static constexpr int DIM {2};

  // Base material properties
  double rho        = 7850;      // Density, kg/m³
  double E0         = 1;     // Young's modulus, Pa
  double E          = 210e9;    // Young's modulus, Pa
  double nu         = 0.3;       // Poisson's ratio, dimensionless
  double c          = 500;       // Specific heat, J/(kg·K)
  double alpha0     = 1;    // Thermal expansion, 1/K
  double alpha      = 1.2e-5;    // Thermal expansion, 1/K
  double theta_ref  = 0;       // Reference temperature, K (20°C)
  double k0         = 1e-03;        // Thermal conductivity, W/(m·K)
  double k          = 45;        // Thermal conductivity, W/(m·K)

  // Time integration options.
  int num_steps = 5;
  double dt = 0.1;

  // Build inlet object to parse cla
  ::axom::CLI::App app {"Inputs"};
  app.add_option("--s-ref", numSerRef, "Number of (serial) mesh refinements");
  app.add_option("--p-ref", numParRef, "Number of (parallel) mesh refinements");
  app.allow_extras();
  CLI11_PARSE(app, argc, argv);
  if (myRank == managerRank) {
    ::std::cout << "Finished parsing inputs." << ::std::endl;
  }

  // print command line arguments
  if (myRank == managerRank) {
    std::cout << app.config_to_str(true, true);
  }

  // initialize serac
  ::axom::sidre::DataStore datastore;
  ::serac::StateManager::initialize(datastore, "sidreDataStore");
  
  //s::mfem::Mesh mesh = ::mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, ::mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  ::mfem::Mesh mesh = ::mfem::Mesh::MakeCartesian2D(Nx, Ny, mfem::Element::QUADRILATERAL, Lx, Ly);
  ::std::cout << "Successfully loaded mesh" << ::std::endl;

  // build mesh and refine
  // ++ ::mfem::Mesh mesh(meshFileName.c_str());
  assert(mesh.SpaceDimension() == DIM);
  for (int i = 0; i < numSerRef; i++) {
    mesh.UniformRefinement();
  }
  auto pmesh = ::std::make_unique<::mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  pmesh->EnsureNodes();
  pmesh->ExchangeFaceNbrData();
  for (int i = 0; i < numParRef; i++) {
    pmesh->UniformRefinement();
  }
  if (myRank == managerRank) {
    ::std::cout << "Successfully loaded and refined mesh." << ::std::endl;
  }

  // lido tags
  ::std::string meshTag = "pmesh";

  // register mesh with serac and lido
  ::mfem::ParMesh *meshPtr = &::serac::StateManager::setMesh(::std::move(pmesh), meshTag);
  lido::DataManager::getInstance().addMesh(meshTag, meshPtr);
  
  const ::std::string densFecTag("dens_fec");
  const ::std::string densFesTag("dens_fes");

  // define the boundary conditions attributes
  ::serac::Domain bottomBoundary = ::serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(1));
  ::serac::Domain rightBoundary = ::serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(4));
  ::serac::Domain topBoundary = ::serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(3));
  ::serac::Domain leftBoundary = ::serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(2));

   // Create domain of entire mesh
  ::serac::Domain entireDomain = ::serac::EntireDomain(*meshPtr);
  ::serac::Domain boundary = ::serac::EntireBoundary(*meshPtr);
  
  ::serac::FiniteElementState user_defined_youngs_modulus(serac::StateManager::newState(serac::L2<0>{}, "parameterized_youngs_modulus", meshTag));
  ::serac::FiniteElementState user_defined_thermal_conductivity(serac::StateManager::newState(serac::L2<0>{}, "parameterized_thermal_conductivity", meshTag));                                                    
  ::serac::FiniteElementState user_defined_coupling_coefficient(serac::StateManager::newState(serac::L2<0>{}, "parameterized_coupling_coefficient", meshTag));                                                    
 
  user_defined_youngs_modulus = E;
  user_defined_thermal_conductivity = k;
  user_defined_coupling_coefficient = alpha;

  serac::NonlinearSolverOptions nonlinearOptions = {
    //.nonlin_solver  = serac::NonlinearSolver::Newton,
    .nonlin_solver  = serac::NonlinearSolver::NewtonLineSearch,
    //.nonlin_solver  = serac::NonlinearSolver::TrustRegion,
    .relative_tol   = 1.0e-8,
    .absolute_tol   = 1.0e-9,
    .min_iterations = 1, // for trust region
    .max_iterations = 75,
    .max_line_search_iterations = 30, // for trust region: 15,
    .print_level    = 1
  };


  // Define solver options
  auto linear_opts = ::serac::thermomechanics::direct_linear_options;
  linear_opts.relative_tol = 1e-12;
  linear_opts.absolute_tol = 1e-12;
  
  // Instantitate the thermo-mechanics solver 

  // The output requested fields are "temperature", "displacement", .....
  using thermoMechType = serac::ThermomechanicsMonolithic<1, DIM, serac::Parameters< serac::L2<0>, serac::L2<0>, serac::L2<0> >>;
  auto thermo_mech = std::make_unique<thermoMechType>(nonlinearOptions, linear_opts, std::string("thermomech_topopt"), meshTag, std::vector<std::string>{"E_bar", "Kappa_bar", "Alpha_bar"});

  thermo_mech->setParameter(0, user_defined_youngs_modulus);
  thermo_mech->setParameter(1, user_defined_thermal_conductivity);
  thermo_mech->setParameter(2, user_defined_coupling_coefficient);
  
  // Instantiate the thermo-mech class with the base material properties
  ::serac::thermomechanics::ParameterizedThermoelasticMaterial material{rho, E0, nu, c, alpha0, theta_ref, k0};
  thermo_mech->setMaterial(::serac::DependsOn< 0, 1, 2>{}, material, entireDomain);
  
  /// Thermal part
  /// ------------

  // Temperature is 0 at left and top boundaries
  thermo_mech->setTemperatureBCs({2}, [](const ::mfem::Vector &, double) { return 0.0; });
  thermo_mech->setTemperatureBCs({3}, [](const ::mfem::Vector &, double) { return 0.0; });

  thermo_mech->setSource(
    ::serac::DependsOn<> {},
    [=](auto, auto, auto, auto) { 
      double q0 = 1.0; // heat source in entire domain
      return q0;
    },
    entireDomain);
  
  // Set temperature gradients to be 0 along right and bottom boundaries
  auto flux_bc_function = [](auto /* X */, auto /* n */, auto /* time */, auto /* T */) { return 0.0; };
  thermo_mech->setFluxBCs(flux_bc_function, rightBoundary);
  thermo_mech->setFluxBCs(flux_bc_function, bottomBoundary);
 

  /// Mechanical part
  /// ---------------

  // Fix the left boundary
  thermo_mech->setFixedBCs(leftBoundary);

  // set load on the right edge
  // Set natural / Neumann / traction boundary conditions
  double Py = 0.02; // magnitude of the distributed load applied to the end of the cantilever
  double Py_Ly = 0.05; // area of influence of load
  ::serac::Domain tractionDomain
      = ::serac::Domain::ofBoundaryElements(*meshPtr, [&Ly, &Py_Ly](::std::vector<::serac::tensor<double, DIM>> X, int bdr_attr) {
          if (3 != bdr_attr) {
            return false;
          }
          for (auto &x: X) {
            if (::std::fabs(x[1] - Ly / 2.0) < Py_Ly / 2.0) {
              return true;
            }
          }
          return false;
        });
  thermo_mech->setTraction(
      [&Py](auto x, auto, auto) {
        auto t = 0.0 * x;
        t[1] -= Py;
        return t;
      },
      tractionDomain);

  // Finalize
  thermo_mech->completeSetup();

      // Perform the transient solve.
  for (int it = 0; it < num_steps; it++) {
    // int const cycle = it + 1;
    // double const time = cycle * dt; // Compute the current simulation time.

    // Advance the solution: solve for velocity and pressure at the next time step.
    thermo_mech->advanceTimestep(dt);

    // Write state fields (e.g., velocity, pressure) and dual functionals (e.g. reaction force) to disk.
    thermo_mech->outputStateToDisk("sol_thermo_mech");
  }

  return 0;
}