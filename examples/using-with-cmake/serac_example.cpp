// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file serac.cpp
 *
 * @brief Basic serac example
 *
 * Intended to verify that external projects can include Serac
 */

#include "serac/infrastructure/about.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "mfem.hpp"
#include "axom/core.hpp"

int main(int argc, char* argv[])
{
  serac::ApplicationManager applicationManager(argc, argv);

  SLIC_INFO_ROOT(serac::about());

  SLIC_INFO_ROOT("Testing important TPLs...");
  SLIC_INFO_ROOT("MFEM version: " << mfem::GetVersionStr());
  axom::about();

  SLIC_INFO_ROOT("\nSerac loaded successfully.");

  return 0;
}
