// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/application_manager.hpp"

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

#include "serac/serac_config.hpp"

#ifdef SERAC_USE_PETSC
#include "petsc.h"  // for PetscPopSignalHandler
#endif

#include "serac/infrastructure/accelerator.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/infrastructure/about.hpp"

namespace serac {
/**
 * @brief Destroy MPI, signal handling, logging, profiling, hypre, sundials, petsc, and slepc. Note this should not be
 * called by or exposed to users.
 */
void finalizer();
}  // namespace serac

namespace {
void signalHandler(int signal)
{
  std::cerr << "[SIGNAL]: Received signal " << signal << " (" << strsignal(signal) << "), exiting" << std::endl;
  serac::finalizer();
  exit(1);
}
}  // namespace

namespace serac {

void finalizer()
{
  if (axom::slic::isInitialized()) {
    serac::logger::flush();
    serac::logger::finalize();
  }

#ifdef SERAC_USE_PETSC
#ifdef SERAC_USE_SLEPC
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

ApplicationManager::ApplicationManager(int argc, char* argv[], MPI_Comm comm) : comm_(comm)
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

  printRunInfo();

  // Start the profiler (no-op if not enabled)
  profiling::initialize(comm_);

  mfem::Hypre::Init();

#ifdef SERAC_USE_SUNDIALS
  mfem::Sundials::Init();
#endif

#ifdef SERAC_USE_PETSC
#ifdef SERAC_USE_SLEPC
  mfem::MFEMInitializeSlepc(&argc, &argv);
#else
  mfem::MFEMInitializePetsc(&argc, &argv);
#endif
  PetscPopSignalHandler();
#endif

  // Initialize GPU (no-op if not enabled/available)
  accelerator::initializeDevice();

  // Register signal handlers
  std::signal(SIGABRT, signalHandler);
  std::signal(SIGINT, signalHandler);
  std::signal(SIGSEGV, signalHandler);
  std::signal(SIGTERM, signalHandler);
}

ApplicationManager::~ApplicationManager() { serac::finalizer(); }

}  // namespace serac
