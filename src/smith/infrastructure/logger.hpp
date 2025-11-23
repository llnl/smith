// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file logger.hpp
 *
 * @brief This file contains the all the necessary functions and macros required
 *        for logging as well as a helper function to exit the program gracefully.
 */

#pragma once

#include "axom/slic.hpp"
#include "axom/fmt.hpp"
#include "mpi.h"

// Logger functionality
namespace smith::logger {
/**
 * @brief Initializes and setups the logger.
 *
 * Setups and tailors the SLIC logger for Smith.  Sets the SLIC loggings streams
 * and tells SLIC how to format the messages.  This function also creates different
 * logging streams if you are running serial or parallel.
 *
 * @param[in] comm MPI communicator that the logger will use
 */
bool initialize(MPI_Comm comm);

/**
 * @brief Finalizes the logger.
 *
 * Closes and finalizes the SLIC logger.
 */
void finalize();

/**
 * @brief Flushes messages currently held by the logger.
 *
 * If running in parallel, SLIC doesn't output messages immediately.  This flushes
 * all messages currently held by SLIC.  This is a collective operation because
 * messages can be spread across MPI ranks.
 */
void flush();

}  // namespace smith::logger
