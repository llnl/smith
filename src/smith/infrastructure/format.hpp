// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file format.hpp
 *
 * @brief Shared helpers for C++ standard formatting support in Smith
 */

#pragma once

#include <format>
#include <sstream>
#include <string>
#include <string_view>

namespace smith::format {

/// @brief Converts a value to a string using its stream insertion operator.
template <typename T>
std::string streamed(const T& value)
{
  std::ostringstream stream;
  stream << value;
  return stream.str();
}

/// @brief Formatter that renders values through `operator<<`.
struct OstreamFormatter {
  /// @brief Accepts the default formatting syntax.
  constexpr auto parse(std::format_parse_context& ctx) const { return ctx.begin(); }

  /// @brief Formats a value by streaming it to a temporary string.
  template <typename T>
  auto format(const T& value, std::format_context& ctx) const
  {
    return std::formatter<std::string_view>{}.format(streamed(value), ctx);
  }
};

}  // namespace smith::format
