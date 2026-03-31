// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <utility>

#include "smith/differentiable_numerics/paraview_writer.hpp"

namespace smith {

template <typename... Args>
auto createParaviewOutput(Args&&... args)
{
  return createParaviewWriter(std::forward<Args>(args)...);
}

}  // namespace smith
