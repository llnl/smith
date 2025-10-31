// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file output.hpp
 *
 * @brief This file contains the all the necessary functions for outputting simulation data
 */

#pragma once

#include <string>

#include "axom/sidre.hpp"

/**
 * @brief The output related helper functions and objects
 *
 */
namespace smith::output {

/**
 * @brief The output file formats supported
 */
enum class FileFormat
{
  JSON,
  YAML
};

/**
 * @brief Outputs simulation summary data from the datastore to the given file
 * only on rank 0
 *
 * @param[in] datastore Root of the Sidre datastore
 * @param[in] output_directory Directory to write output file into
 * @param[in] file_format The output file format
 */
void outputSummary(const axom::sidre::DataStore& datastore, const std::string& output_directory,
                   const FileFormat file_format = FileFormat::JSON);

}  // namespace smith::output
