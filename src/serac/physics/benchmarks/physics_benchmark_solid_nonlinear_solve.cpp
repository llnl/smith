// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/solid_mechanics_contact.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <ostream>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "serac/physics/materials/liquid_crystal_elastomer.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/infrastructure/terminator.hpp"

using namespace serac;

enum class Prec
{
  JACOBI,
  STRUMPACK,
  CHOLESKI,
  LU,
  MULTIGRID,
  PETSC_MULTIGRID,
  NONE
};

enum class NonlinSolve
{
  NEWTON,
  LINESEARCH,
  CRITICALPOINT,
  TRUSTREGION,
  NONE
};

enum class ProblemSize
{
  SMALL,
  LARGE,
};

std::string mesh_path = ".";
// string->value matching for optionally entering options as string in command line
std::map<std::string, Prec> precMap = {
    {"jacobi", Prec::JACOBI}, {"strumpack", Prec::STRUMPACK}, {"choleski", Prec::CHOLESKI},
    {"lu", Prec::LU},         {"multigrid", Prec::MULTIGRID}, {"petsc_multigrid", Prec::PETSC_MULTIGRID},
    {"none", Prec::NONE},
};
std::map<std::string, NonlinSolve> nonlinSolveMap = {
    {"newton", NonlinSolve::NEWTON},
    {"linesearch", NonlinSolve::LINESEARCH},
    {"critialpoint", NonlinSolve::CRITICALPOINT},
    {"trustregion", NonlinSolve::TRUSTREGION},
    {"none", NonlinSolve::NONE},
};
std::map<std::string, ProblemSize> problemSizeMap = {
    {"small", ProblemSize::SMALL},
    {"large", ProblemSize::LARGE},
};

const std::string precToString(Prec prec)
{
  for (const auto& p : precMap) {
    if (p.second == prec) {
      return p.first;
    }
  }
  return "unknown";
}

const std::string nonlinSolveToString(NonlinSolve nonlinSolve)
{
  for (const auto& n : nonlinSolveMap) {
    if (n.second == nonlinSolve) {
      return n.first;
    }
  }
  return "unknown";
}

auto get_opts(NonlinSolve nonlinSolve, Prec prec, int max_iters, double abs_tol = 1e-9)
{
  serac::NonlinearSolverOptions nonlinear_options{
      .nonlin_solver = NonlinearSolver::TrustRegion,
      //.nonlin_solver = NonlinearSolver::NewtonLineSearch, //LineSearch,
      //.nonlin_solver = NonlinearSolver::PetscNewton,  //LineSearch,
      //.nonlin_solver = NonlinearSolver::PetscNewtonCriticalPoint, // breaks for snap_cell
      .relative_tol               = abs_tol,
      .absolute_tol               = abs_tol,
      .min_iterations             = 1,
      .max_iterations             = 2000,
      .max_line_search_iterations = 20,
      .print_level                = 1};

  // best for critical point newton: ls = PetscGMRES, petsc_preconditioner = PetscPCType::LU;
  serac::LinearSolverOptions linear_options = {.linear_solver = LinearSolver::CG,
                                               //.linear_solver  = LinearSolver::PetscGMRES,
                                               //.linear_solver  = LinearSolver::PetscCG,
                                               .preconditioner = Preconditioner::HypreJacobi,
                                               //.preconditioner = Preconditioner::Petsc,
                                               //.petsc_preconditioner = PetscPCType::JACOBI,
                                               //.petsc_preconditioner = PetscPCType::JACOBI_ROWMAX,
                                               //.petsc_preconditioner = PetscPCType::GAMG,
                                               //.petsc_preconditioner = PetscPCType::HMG, // Zach's prefered
                                               //.petsc_preconditioner = PetscPCType::LU,
                                               //.petsc_preconditioner = PetscPCType::CHOLESKY,
                                               .relative_tol   = 0.7 * abs_tol,
                                               .absolute_tol   = 0.7 * abs_tol,
                                               .max_iterations = max_iters,
                                               .print_level    = 1};

  switch (nonlinSolve) {
    case NonlinSolve::NEWTON: {
      SLIC_INFO_ROOT("using newton solver");
      nonlinear_options.min_iterations             = 0;
      nonlinear_options.max_line_search_iterations = 0;
      nonlinear_options.nonlin_solver              = NonlinearSolver::Newton;
      break;
    }
    case NonlinSolve::LINESEARCH: {
      SLIC_INFO_ROOT("using newton linesearch solver");
      nonlinear_options.min_iterations = 0;
      nonlinear_options.nonlin_solver  = NonlinearSolver::PetscNewtonBacktracking;
      // nonlinear_options.nonlin_solver = NonlinearSolver::NewtonLineSearch;
      break;
    }
    case NonlinSolve::CRITICALPOINT: {
      SLIC_INFO_ROOT("using newton critical point solver");
      nonlinear_options.min_iterations = 0;
      nonlinear_options.nonlin_solver  = NonlinearSolver::PetscNewtonCriticalPoint;
      break;
    }
    case NonlinSolve::TRUSTREGION: {
      SLIC_INFO_ROOT("using trust region solver");
      nonlinear_options.nonlin_solver = NonlinearSolver::TrustRegion;
      break;
    }
    case NonlinSolve::NONE:
    default: {
      SLIC_ERROR_ROOT("invalid nonlinear solver specified");
    }
  }

  switch (prec) {
    case Prec::JACOBI: {
      SLIC_INFO_ROOT("using jacobi");
      linear_options.linear_solver  = LinearSolver::CG;
      linear_options.preconditioner = Preconditioner::HypreJacobi;
      break;
    }
    case Prec::STRUMPACK: {
      SLIC_INFO_ROOT("using strumpack");
      linear_options.linear_solver = LinearSolver::Strumpack;
      break;
    }
    case Prec::CHOLESKI: {
      SLIC_INFO_ROOT("using choleski");
      linear_options.linear_solver        = LinearSolver::CG;
      linear_options.preconditioner       = Preconditioner::Petsc;
      linear_options.petsc_preconditioner = PetscPCType::CHOLESKY;
      break;
    }
    case Prec::LU: {
      SLIC_INFO_ROOT("using lu");
      linear_options.linear_solver        = LinearSolver::GMRES;
      linear_options.preconditioner       = Preconditioner::Petsc;
      linear_options.petsc_preconditioner = PetscPCType::LU;
      break;
    }
    case Prec::MULTIGRID: {
      SLIC_INFO_ROOT("using multigrid");
      linear_options.linear_solver  = LinearSolver::CG;
      linear_options.preconditioner = Preconditioner::HypreAMG;
      break;
    }
    case Prec::PETSC_MULTIGRID: {
      SLIC_INFO_ROOT("using petsc multigrid");
      linear_options.linear_solver        = LinearSolver::CG;
      linear_options.preconditioner       = Preconditioner::Petsc;
      linear_options.petsc_preconditioner = PetscPCType::HMG;
      break;
    }
    case Prec::NONE:
    default: {
      SLIC_ERROR_ROOT("invalid preconditioner specified");
    }
  }

  return std::make_pair(nonlinear_options, linear_options);
}

void functional_solid_test_euler(NonlinSolve nonlinSolve, Prec prec)
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "eulerStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  int Nx = 4;
  int Ny = 7;
  int Nz = 10 * 5;

  double Lx = 1.0;
  double Ly = 1.2;
  double Lz = 30.0;

  double density  = 1.0;
  double E        = 10.0;
  double v        = 0.33;
  double bulkMod  = E / (3. * (1. - 2. * v));
  double shearMod = E / (2. * (1. + v));
  double load     = 0.002;  // 0.004

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(nonlinSolve, prec, 3 * Nx * Ny * Nz, 1e-9);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  serac::Domain backSurface =
      serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(3));  // 4,5 with traction makes a twist
  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));

  int    num_time_steps = 2;
  double total_time     = 1.0;
  double dt             = total_time / num_time_steps;

  seracSolid->setTraction([&](auto, auto n, auto t) { return -load * t * n; }, topSurface);
  seracSolid->setTraction([&](auto, auto n, auto) { return 1e-5 * n; }, backSurface);

  // displacement on bottom surface
  seracSolid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  seracSolid->completeSetup();

  seracSolid->outputStateToDisk("paraview_euler");
  for (int i = 0; i < num_time_steps; ++i) {
    seracSolid->advanceTimestep(dt);
    seracSolid->outputStateToDisk("paraview_euler");
  }
}

void functional_solid_test_nonlinear_buckle(NonlinSolve nonlinSolve, Prec prec, ProblemSize problemSize)
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "buckleStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  int Nx, Ny, Nz;
  switch (problemSize) {
    case ProblemSize::SMALL:
      Nx = 500;
      Ny = 6;
      Nz = 5;
      break;
    default:
    case ProblemSize::LARGE:
      Nx = 1000;
      Ny = 60;
      Nz = 50;
      break;
  }

  double Lx = Nx * 0.1;
  double Ly = Ny * 0.03;
  double Lz = Nz * 0.06;

  double loadMagnitude = 5e-10;

  double density  = 1.0;
  double E        = 1.0;
  double v        = 0.33;
  double bulkMod  = E / (3. * (1. - 2. * v));
  double shearMod = E / (2. * (1. + v));

  SERAC_MARK_FUNCTION;

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(nonlinSolve, prec, 3 * Nx * Ny * Nz, 1e-11);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  // fix displacement on side surface
  seracSolid->setDisplacementBCs({2, 3, 4, 5}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });
  // seracSolid->setDisplacementBCs({3}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));
  // seracSolid->setTraction([&](auto, auto n, auto) { return -loadMagnitude * n; }, topSurface);
  seracSolid->setPressure([&](auto, auto) { return loadMagnitude; }, topSurface);
  seracSolid->completeSetup();
  seracSolid->advanceTimestep(1.0);

  seracSolid->outputStateToDisk("paraview_buckle_easy");
}

int main(int argc, char* argv[])
{
  serac::initialize(argc, argv);

  SERAC_MARK_FUNCTION;

  NonlinSolve nonlinSolve = NonlinSolve::NONE;
  Prec        prec        = Prec::NONE;
  ProblemSize problemSize = ProblemSize::LARGE;

  axom::CLI::App app{"Solid Nonlinear Solve Benchmark"};
  // app.add_option("-m,--mesh", mesh_path, "Path to mesh files")->check(axom::CLI::ExistingDirectory);
  app.add_option("-n,--nonlinear-solver", nonlinSolve, "Nonlinear Solver")
      ->transform(axom::CLI::CheckedTransformer(nonlinSolveMap, axom::CLI::ignore_case));
  app.add_option("-p,--preconditioner", prec, "Preconditioner")
      ->transform(axom::CLI::CheckedTransformer(precMap, axom::CLI::ignore_case));
  app.add_option("-s,--problem-size", problemSize, "Problem Size")
      ->transform(axom::CLI::CheckedTransformer(problemSizeMap, axom::CLI::ignore_case));

  // Parse the arguments and check if they are good
  try {
    CLI11_PARSE(app, argc, argv);
  } catch (const axom::CLI::ParseError& e) {
    serac::logger::flush();
    if (e.get_name() == "CallForHelp") {
      auto msg = app.help();
      SLIC_INFO_ROOT(msg);
      serac::exitGracefully();
    } else {
      auto err_msg = axom::CLI::FailureMessage::simple(&app, e);
      SLIC_ERROR_ROOT(err_msg);
    }
  }

  SERAC_SET_METADATA("test", "solid_nonlinear_solve");

  // If you do not specify preconditioner and nonlinear solver, run the following pre-selected options
  if (nonlinSolve == NonlinSolve::NONE && prec == Prec::NONE) {
    SERAC_MARK_BEGIN("Jacobi Preconditioner");
    functional_solid_test_nonlinear_buckle(NonlinSolve::NEWTON, Prec::JACOBI, problemSize);
    SERAC_MARK_END("Jacobi Preconditioner");

    SERAC_MARK_BEGIN("Multigrid Preconditioner");
    functional_solid_test_nonlinear_buckle(NonlinSolve::NEWTON, Prec::MULTIGRID, problemSize);
    SERAC_MARK_END("Multigrid Preconditioner");

#ifdef SERAC_USE_PETSC
    SERAC_MARK_BEGIN("Petsc Multigrid Preconditioner");
    functional_solid_test_nonlinear_buckle(NonlinSolve::NEWTON, Prec::PETSC_MULTIGRID, problemSize);
    SERAC_MARK_END("Petsc Multigrid Preconditioner");
#endif
  } else {
    SERAC_SET_METADATA("nonlinear solver", nonlinSolveToString(nonlinSolve));
    SERAC_SET_METADATA("preconditioner", precToString(prec));

    SERAC_MARK_BEGIN("Custom Preconditioner");
    functional_solid_test_nonlinear_buckle(nonlinSolve, prec, problemSize);
    SERAC_MARK_END("Custom Preconditioner");
  }

  // functional_solid_test_euler();

  serac::exitGracefully(0);
}
