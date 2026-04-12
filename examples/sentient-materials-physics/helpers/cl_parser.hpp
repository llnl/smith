// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace smith {
namespace cl_parser {

static bool parseIntFlag(const std::string& arg, const std::string& name, int& value)
{
  const auto prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }
  value = std::stoi(arg.substr(prefix.size()));
  return true;
}

static bool parseDoubleFlag(const std::string& arg, const std::string& name, double& value)
{
  const auto prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }
  value = std::stod(arg.substr(prefix.size()));
  return true;
}

static bool parseStringFlag(const std::string& arg, const std::string& name, std::string& value)
{
  const auto prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }
  value = arg.substr(prefix.size());
  return true;
}

};  // namespace cl_parser
};  // namespace smith