// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrature_data.cpp
 *
 * @brief This file exists to instantiate some global QuadratureData objects
 */

#include "smith/numerics/functional/quadrature_data.hpp"

namespace smith {

/// a single instance of a QuadratureData container of `Nothing`s, since they are all interchangeable
std::shared_ptr<QuadratureData<Nothing>> NoQData = ::std::make_shared<QuadratureData<Nothing>>();

/// a single instance of a QuadratureData container of `Empty`s, since they are all interchangeable
std::shared_ptr<QuadratureData<Empty>> EmptyQData = ::std::make_shared<QuadratureData<Empty>>();

}  // namespace smith
