// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file smith_example.cpp
 *
 * @brief Basic Smith example
 *
 * Intended to verify that external projects can include Smith
 */

#include "smith/infrastructure/about.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "mfem.hpp"
#include "axom/core.hpp"

int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  SLIC_INFO_ROOT(smith::about());

  SLIC_INFO_ROOT("Testing important TPLs...");
  SLIC_INFO_ROOT("MFEM version: " << mfem::GetVersionStr());
  axom::about();

  SLIC_INFO_ROOT("\nSmith loaded successfully.");

  return 0;
}
