// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file tuple.hpp
 *
 * @brief Smith tuple compatibility layer over mfem::future::tuple
 */
#pragma once

#include "mfem.hpp"

namespace smith {

/// @brief Expose MFEM tuple in the Smith namespace.
template <typename... T>
using tuple = mfem::future::tuple<T...>;

/// @brief Expose MFEM tuple application in the Smith namespace.
using mfem::future::apply;
/// @brief Expose MFEM tuple element access in the Smith namespace.
using mfem::future::get;
/// @brief Expose MFEM tuple construction in the Smith namespace.
using mfem::future::make_tuple;
/// @brief Expose MFEM tuple type selection in the Smith namespace.
using mfem::future::type;

/// @brief Alias for the MFEM tuple size trait.
template <class... Types>
using tuple_size = mfem::future::tuple_size<Types...>;

/// @brief Alias for the MFEM tuple element trait.
template <size_t I, class T>
using tuple_element = mfem::future::tuple_element<I, T>;

/// @brief Alias for the MFEM tuple detection trait.
template <typename T>
using is_tuple = mfem::future::is_tuple<T>;

/// @brief Alias for the MFEM nested tuple detection trait.
template <typename T>
using is_tuple_of_tuples = mfem::future::is_tuple_of_tuples<T>;

}  // namespace smith

#include "smith/numerics/functional/tuple_tensor_dual_functions.hpp"
