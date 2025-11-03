// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file cli.hpp
 *
 * @brief This file contains the all the necessary functions and macros required
 *        for interacting with the command line interface.
 */

#pragma once

#include <string>
#include <unordered_map>

/**
 * @brief Command line functionality
 */
namespace smith::cli {

/**
 * @brief Defines command line options and parses the found values.
 *
 * @param[in] argc Argument count
 * @param[in] argv Argument vector
 * @param[in] app_description Description of application to be printed by usage
 * @return map of all given command line options
 */
std::unordered_map<std::string, std::string> defineAndParse(int argc, char* argv[], std::string app_description);

/**
 * @brief Prints all given command line options to the screen.
 *
 * @param[in] cli_opts Given command line options to be printed
 */
void printGiven(std::unordered_map<std::string, std::string>& cli_opts);

}  // namespace smith::cli
