// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <fstream>
#include <functional>
#include <ostream>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "serac/infrastructure/application_manager.hpp"
#include "serac/mesh_utils/mesh_utils.hpp"
#include "serac/numerics/solver_config.hpp"
#include "serac/physics/materials/liquid_crystal_elastomer.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/solid_mechanics_contact.hpp"
#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"

enum class ProblemSize
{
  Small,
  Large,
};

std::string problemSizeName(const ProblemSize& ps)
{
  switch (ps) {
    case ProblemSize::Small:
      return "Small";
    case ProblemSize::Large:
      return "Large";
  }
  // This cannot happen, but GCC doesn't know that
  return "UNKNOWN";
}

std::map<std::string, ProblemSize> problemSizeMap = {
    {"Small", ProblemSize::Small},
    {"Large", ProblemSize::Large},
};

auto get_opts(serac::NonlinearSolver nonlinearSolver, serac::LinearSolver linearSolver,
              serac::Preconditioner preconditioner, int max_iters, double abs_tol = 1e-9)
{
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = nonlinearSolver,
                                                  .relative_tol = abs_tol,
                                                  .absolute_tol = abs_tol,
                                                  .min_iterations = 1,
                                                  .max_iterations = 2000,
                                                  .max_line_search_iterations = 20,
                                                  .print_level = 1};

  // best for critical point newton: ls = PetscGMRES, petsc_preconditioner = PetscPCType::LU;
  serac::LinearSolverOptions linear_options = {.linear_solver = linearSolver,
                                               .preconditioner = preconditioner,
                                               .relative_tol = 0.7 * abs_tol,
                                               .absolute_tol = 0.7 * abs_tol,
                                               .max_iterations = max_iters,
                                               .print_level = 1};

  SLIC_INFO_ROOT("================================================================================");
  SLIC_INFO_ROOT(axom::fmt::format("Nonlinear Solver = {}", serac::nonlinearName(nonlinearSolver)));
  SLIC_INFO_ROOT(axom::fmt::format("Linear Solver    = {}", serac::linearName(linearSolver)));
  SLIC_INFO_ROOT(axom::fmt::format("Preconditioner   = {}", serac::preconditionerName(preconditioner)));
  SLIC_INFO_ROOT("================================================================================");

  switch (nonlinearSolver) {
    case serac::NonlinearSolver::Newton: {
      nonlinear_options.min_iterations = 0;
      nonlinear_options.max_line_search_iterations = 0;
      break;
    }
    case serac::NonlinearSolver::NewtonLineSearch: {
      nonlinear_options.min_iterations = 0;
      break;
    }
    case serac::NonlinearSolver::PetscNewtonCriticalPoint: {
      nonlinear_options.min_iterations = 0;
      break;
    }
    default:
      break;
  }

  return std::make_pair(nonlinear_options, linear_options);
}

void functional_solid_test_nonlinear_buckle(serac::NonlinearSolver nonlinearSolver, serac::LinearSolver linearSolver,
                                            serac::Preconditioner preconditioner, ProblemSize problemSize)
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "buckleStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  int Nx, Ny, Nz;
  switch (problemSize) {
    case ProblemSize::Small:
      Nx = 500;
      Ny = 6;
      Nz = 5;
      break;
    default:
    case ProblemSize::Large:
      Nx = 1000;
      Ny = 60;
      Nz = 50;
      break;
  }

  double Lx = Nx * 0.1;
  double Ly = Ny * 0.03;
  double Lz = Nz * 0.06;

  double loadMagnitude = 5e-10;

  double density = 1.0;
  double E = 1.0;
  double v = 0.33;
  double bulkMod = E / (3. * (1. - 2. * v));
  double shearMod = E / (2. * (1. + v));

  SERAC_MARK_FUNCTION;

  std::string meshTag = "mesh";
  auto pmesh = std::make_shared<serac::Mesh>(
      mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz), meshTag, 0, 0);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] =
      get_opts(nonlinearSolver, linearSolver, preconditioner, 3 * Nx * Ny * Nz, 1e-11);

  auto seracSolid = std::make_unique<seracSolidType>(nonlinear_options, linear_options,
                                                     serac::solid_mechanics::default_quasistatic_options, "serac_solid",
                                                     pmesh, std::vector<std::string>{});

  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  seracSolid->setMaterial(material, pmesh->entireBody());

  // fix displacement on side surface
  std::string sideSurfaceName = "side_surface";
  pmesh->addDomainOfBoundaryElements(sideSurfaceName, serac::by_attr<DIM>({2, 3, 4, 5}));
  seracSolid->setFixedBCs(pmesh->domain(sideSurfaceName));

  std::string topSurfaceName = "top_surface";
  pmesh->addDomainOfBoundaryElements(topSurfaceName, serac::by_attr<DIM>(6));
  seracSolid->setPressure([&](auto, auto) { return loadMagnitude; }, pmesh->domain(topSurfaceName));
  seracSolid->completeSetup();
  seracSolid->advanceTimestep(1.0);

  seracSolid->outputStateToDisk("paraview_buckle_easy");
}

int main(int argc, char* argv[])
{
  serac::ApplicationManager applicationManager(argc, argv);

  SERAC_MARK_FUNCTION;

  ProblemSize problemSize = ProblemSize::Small;
  serac::NonlinearSolver nonlinearSolver = serac::NonlinearSolver::Newton;
  serac::LinearSolver linearSolver = serac::LinearSolver::CG;
  serac::Preconditioner preconditioner = serac::Preconditioner::HypreJacobi;

  axom::CLI::App app{"Solid Nonlinear Solve Benchmark"};
  auto* nonlinOpt = app.add_option("-n,--nonlinear-solver", nonlinearSolver, "Nonlinear Solver")
                        ->transform(axom::CLI::CheckedTransformer(serac::nonlinearSolverMap, axom::CLI::ignore_case));
  auto* linOpt = app.add_option("-l,--linear-solver", linearSolver, "Linear Solver")
                     ->transform(axom::CLI::CheckedTransformer(serac::linearSolverMap, axom::CLI::ignore_case));
  auto* precOpt = app.add_option("-p,--preconditioner", preconditioner, "Preconditioner")
                      ->transform(axom::CLI::CheckedTransformer(serac::preconditionerMap, axom::CLI::ignore_case));
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
      return 0;
    } else {
      auto err_msg = axom::CLI::FailureMessage::simple(&app, e);
      SLIC_ERROR_ROOT(err_msg);
    }
  }

  SERAC_SET_METADATA("test", "solid_nonlinear_solve");
  SERAC_SET_METADATA("Problem Size", problemSizeName(problemSize));

  // If you do not specify solver options, run the following pre-selected options
  if (nonlinOpt->count() == 0 && linOpt->count() == 0 && precOpt->count() == 0) {
    SERAC_MARK_BEGIN("Jacobi Preconditioner");
    functional_solid_test_nonlinear_buckle(serac::NonlinearSolver::Newton, serac::LinearSolver::CG,
                                           serac::Preconditioner::HypreJacobi, problemSize);
    SERAC_MARK_END("Jacobi Preconditioner");

    SERAC_MARK_BEGIN("Multigrid Preconditioner");
    functional_solid_test_nonlinear_buckle(serac::NonlinearSolver::Newton, serac::LinearSolver::CG,
                                           serac::Preconditioner::HypreAMG, problemSize);
    SERAC_MARK_END("Multigrid Preconditioner");

#ifdef SERAC_USE_PETSC
    SERAC_MARK_BEGIN("Petsc Multigrid Preconditioner");
    // newton, cg, petsc, petsc-HMG
    // NOTE: not supporting different petsc types atm, since petsc less prio
    functional_solid_test_nonlinear_buckle(serac::NonlinearSolver::Newton, serac::LinearSolver::CG,
                                           serac::Preconditioner::Petsc, problemSize);
    SERAC_MARK_END("Petsc Multigrid Preconditioner");
#endif
  } else {
    SERAC_SET_METADATA("Nonlinear Solver", serac::nonlinearName(nonlinearSolver));
    SERAC_SET_METADATA("Linear Solver", serac::linearName(linearSolver));
    SERAC_SET_METADATA("Preconditioner", serac::preconditionerName(preconditioner));

    SERAC_MARK_BEGIN("Custom Preconditioner");
    functional_solid_test_nonlinear_buckle(nonlinearSolver, linearSolver, preconditioner, problemSize);
    SERAC_MARK_END("Custom Preconditioner");
  }

  return 0;
}
