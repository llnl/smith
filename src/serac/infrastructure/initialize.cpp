// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/initialize.hpp"

#ifdef WIN32
#include <windows.h>
#include <tchar.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

#include "serac/infrastructure/accelerator.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/infrastructure/terminator.hpp"

#include "mfem.hpp"

#ifdef SERAC_USE_PETSC
#include "petsc.h"
#endif

namespace serac {

std::pair<int, int> getMPIInfo(MPI_Comm comm)
{
  int num_procs = 0;
  int rank = 0;
  if (MPI_Comm_size(comm, &num_procs) != MPI_SUCCESS) {
    SLIC_ERROR("Failed to determine number of MPI processes");
  }

  if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS) {
    SLIC_ERROR("Failed to determine MPI rank");
  }
  return {num_procs, rank};
}

std::pair<int, int> initialize(int argc, char* argv[], MPI_Comm comm)
{
  // Initialize MPI.
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "Failed to initialize MPI" << std::endl;
    serac::exitGracefully(true);
  }

  // Initialize the signal handler
  terminator::registerSignals();

  // Initialize SLIC logger
  if (!logger::initialize(comm)) {
    serac::exitGracefully(true);
  }

  // Start the profiler (no-op if not enabled)
  profiling::initialize(comm);

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
  // TODO for some reason this causes errors on Lassen. We need to look into this ASAP.
  // accelerator::initializeDevice();

  return getMPIInfo(comm);
}

}  // namespace serac
