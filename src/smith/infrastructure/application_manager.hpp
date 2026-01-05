// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <string>
#include <utility>
#include <iostream>

#include "mpi.h"
#include "smith/infrastructure/accelerator.hpp"

namespace smith {

/**
 * @brief RAII Application Manager class. Initializes MPI and other important libraries as
 * well as automatically finalizes them upon going out of scope.
 */
class ApplicationManager {
 public:
  /**
   * @brief Initialize MPI, signal handling, logging, profiling, hypre, sundials, petsc, and slepc.
   *
   * @param argc The number of command-line arguments
   * @param argv The command-line arguments, as C-strings
   * @param comm The MPI communicator to initialize with
   * @param doesPrintRunInfo Whether or not to print build information
   * @param exec_space The desired execution space of device-capable lambda functions
   */
  ApplicationManager(int argc, char* argv[], MPI_Comm comm = MPI_COMM_WORLD, bool doesPrintRunInfo = true,
                     ExecutionSpace exec_space = ExecutionSpace::CPU);

  /**
   * @brief Calls smith::finalizer
   */
  ~ApplicationManager();

  ApplicationManager(ApplicationManager const&) = delete;
  ApplicationManager& operator=(ApplicationManager const&) = delete;

 private:
  MPI_Comm comm_;
};

}  // namespace smith
