// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/output.hpp"

#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

#include "axom/core.hpp"
#include "axom/sidre.hpp"

#include "mpi.h"

namespace serac::output {

namespace detail {
std::string file_format_string(const FileFormat file_format)
{
  std::string value = "";
  if (file_format == FileFormat::JSON) {
    value = "json";
  } else if (file_format == FileFormat::YAML) {
    value = "yaml";
  }
  return value;
}
}  // namespace detail

void outputSummary(const axom::sidre::DataStore& datastore, const std::string& output_directory,
                   const FileFormat file_format)
{
  auto [_, rank] = getMPIInfo();
  if (rank != 0) {
    return;
  }

  std::string file_format_string = detail::file_format_string(file_format);

  const std::string file_name = axom::fmt::format("summary.{0}", file_format_string);
  const std::string path = axom::utilities::filesystem::joinPath(output_directory, file_name);
  datastore.getRoot()->getGroup("serac_summary")->save(path, file_format_string);
}

}  // namespace serac::output
