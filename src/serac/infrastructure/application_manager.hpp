// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <string>
#include <utility>
#include <iostream>

#include "mpi.h"

namespace serac {

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
   */
  ApplicationManager(int argc, char* argv[], MPI_Comm comm = MPI_COMM_WORLD);

  /**
   * @brief Calls serac::finalizer
   */
  ~ApplicationManager();

  ApplicationManager(ApplicationManager const&) = delete;
  ApplicationManager& operator=(ApplicationManager const&) = delete;

 private:
  MPI_Comm comm_;
};

}  // namespace serac
