// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once
#include <camp/camp.hpp>
#include "smith/numerics/functional/finite_element.hpp"

template <typename T>
struct FunctionSignature;

/**
 * @brief a type that encodes information about a function signature (return type, input types)
 * @tparam output_type the function signature's return type
 * @tparam ...input_types the function signature's input types
 */
template <typename output_type, typename... input_types>
struct FunctionSignature<output_type(input_types...)> {
  using return_type = output_type;                     ///< the type returned by the function
  using parameter_types = std::tuple<input_types...>;  ///< the types of the function arguments

  /// the number of input arguments in the function signature
  static constexpr int num_args = sizeof...(input_types);

  /// integer sequence used to make iterating over arguments easier
  static constexpr auto index_seq = camp::make_int_seq<int, num_args>{};
};

/**
 * @brief This helper function template expands the variadic parameter trials and uses
 * the pack to construct a tuple of finite elements.
 *
 * Expansion of the template parameter trials must occur in a separate template
 * function from evaluation_kernel_impl.  The __host__ __device__ marker on the
 * lambda declared in evaluation_kernel_impl prevents use of multiple variadic
 * template parameters in evaluation_kernel_impl.  See section 14.7.3 of the
 * CUDA programming guide, item 9.
 */
template <mfem::Geometry::Type geom, typename test, typename... trials>
auto trial_elements_tuple(FunctionSignature<test(trials...)>)
{
  return smith::tuple<smith::finite_element<geom, trials>...>{};
}

/**
 * @brief This helper function template returns a finite element based on the output
 * type of FunctionSignature.
 *
 * See the comments for trial_elements_tuple to understand why this is needed in
 * domain_integral_kernels::evaluation_kernel and boundary_integral_kernels::evaluation_kernel.
 */
template <mfem::Geometry::Type geom, typename test, typename... trials>
auto get_test_element(FunctionSignature<test(trials...)>)
{
  return smith::finite_element<geom, test>{};
}
