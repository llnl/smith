// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/infrastructure/application_manager.hpp"

#ifdef WIN32
#include <windows.h>
#include <tchar.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

#include <string.h>
#include <csignal>
#include <cstdlib>

#include "mfem.hpp"

#include "smith/smith_config.hpp"

#ifdef SMITH_USE_PETSC
#include "petsc.h"  // for PetscPopSignalHandler
#endif

#include "smith/infrastructure/accelerator.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/infrastructure/profiling.hpp"
#include "smith/infrastructure/about.hpp"

namespace smith {
/**
 * @brief Destroy MPI, signal handling, logging, profiling, hypre, sundials, petsc, and slepc. Note this should not be
 * called by or exposed to users.
 */
void finalizer();
}  // namespace smith

namespace {
void signalHandler(int signal)
{
  std::cerr << "[SIGNAL]: Received signal " << signal << " (" << strsignal(signal) << "), exiting" << std::endl;
  smith::finalizer();
  exit(1);
}
}  // namespace

namespace smith {

void finalizer()
{
  if (axom::slic::isInitialized()) {
    smith::logger::flush();
    smith::logger::finalize();
  }

#ifdef SMITH_USE_PETSC
#ifdef SMITH_USE_SLEPC
  mfem::MFEMFinalizeSlepc();
#else
  mfem::MFEMFinalizePetsc();
#endif
#endif

  profiling::finalize();

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  int mpi_finalized = 0;
  MPI_Finalized(&mpi_finalized);
  if (mpi_initialized && !mpi_finalized) {
    MPI_Finalize();
  }

  accelerator::terminateDevice();
}

ApplicationManager::ApplicationManager(int argc, char* argv[], MPI_Comm comm, bool doesPrintRunInfo,
                                       ExecutionSpace exec_space)
    : comm_(comm)
{
  // Initialize MPI
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "Failed to initialize MPI" << std::endl;
    exit(1);
  }

  // Initialize SLIC logger
  if (!logger::initialize(comm_)) {
    std::cerr << "Failed to initialize SLIC logger" << std::endl;
    exit(1);
  }

  if (doesPrintRunInfo) {
    printRunInfo();
  }

  // Start the profiler (no-op if not enabled)
  profiling::initialize(comm_);

  mfem::Hypre::Init();

#ifdef SMITH_USE_SUNDIALS
  mfem::Sundials::Init();
#endif

#ifdef SMITH_USE_PETSC
  // PETSc tries to parse all command line options, but Smith applications
  // may have others intended for MPI or the application itself.
  // Silence the PETSc warning that there are leftover options it doesn't
  // know.
  PetscOptionsSetValue(NULL, "-options_left", "no");
#ifdef SMITH_USE_SLEPC
  mfem::MFEMInitializeSlepc(&argc, &argv);
#else
  mfem::MFEMInitializePetsc(&argc, &argv);
#endif
  PetscPopSignalHandler();
#endif

  // Initialize GPU (no-op if not enabled/available)
  accelerator::initializeDevice(exec_space);

  // Register signal handlers
  std::signal(SIGABRT, signalHandler);
  std::signal(SIGINT, signalHandler);
  std::signal(SIGSEGV, signalHandler);
  std::signal(SIGTERM, signalHandler);
}

ApplicationManager::~ApplicationManager() { smith::finalizer(); }

}  // namespace smith
