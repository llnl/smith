// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file about.hpp
 *
 * @brief This file contains the interface used for retrieving information
 * about how the driver is configured.
 */

#pragma once

#include <string>
#include <utility>

#include "mpi.h"

namespace smith {

/**
 * @brief Returns a string about the configuration of Smith
 *
 * @return string containing various configuration information about Smith
 */
std::string about();

/**
 * @brief Returns a string for the Git SHA when the driver was built
 *
 * Note: This will not update unless you reconfigure CMake after a commit.
 *
 * @return string value of the Git SHA if built in a Git repo, empty if not
 */
std::string gitSHA();

/**
 * @brief Outputs basic run information to the screen
 *
 * Note: Command line options are handled in `infrastructure/cli.cpp`
 */
void printRunInfo();

/**
 * @brief Returns a string for the version of Smith
 *
 * @param[in] add_SHA boolean for whether to add the Git SHA to the version if available
 *
 * @return string value of the version of Smith
 */
std::string version(bool add_SHA = true);

/**
 * @brief Returns a string for the current compiler name and version
 *
 * @return string value of the current compiler name and version
 */
std::string compiler();

/**
 * @brief Returns a string for the current CMake build type (e.g. Debug, Release)
 *
 * @return string value of the build type
 */
std::string buildType();

/**
 * @brief Get MPI Info
 *
 * @return std::pair<int, int> Pair containing the number of MPI processes and ranks
 */
std::pair<int, int> getMPIInfo(MPI_Comm comm = MPI_COMM_WORLD);

}  // namespace smith
