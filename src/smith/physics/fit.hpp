// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <fstream>
#include <iostream>
#include <array>
#include <utility>

#include "mpi.h"
#include "mfem.hpp"

#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/numerics/functional/differentiate_wrt.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/function_signature.hpp"
#include "smith/numerics/functional/geometry.hpp"
#include "smith/numerics/functional/tuple.hpp"

namespace smith {

namespace detail {

/// @overload
template <int dim, typename signature, int... i, typename func, typename... T>
FiniteElementState fit(std::integer_sequence<int, i...>, func f, mfem::ParMesh& pmesh, const T&... solution_fields)
{
  // signature looks like return_type(arg0_type, arg1_type);
  // so this unpacks the return type
  using output_space = typename FunctionSignature<signature>::return_type;

  FiniteElementState fitted_field(pmesh, output_space{});
  fitted_field = 0.0;

  // mass term
  Domain whole_domain = EntireDomain(pmesh);
  smith::Functional<output_space(output_space)> phi_phi(&fitted_field.space(), {&fitted_field.space()});
  phi_phi.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{}, [](double /*t*/, auto /*x*/, auto u) { return tuple{get<0>(u), zero{}}; },
      whole_domain);
  auto M = get<1>(phi_phi(DifferentiateWRT<0>{}, 0.0 /* t */, fitted_field));

  // rhs
  std::array<const mfem::ParFiniteElementSpace*, sizeof...(T)> trial_spaces = {&solution_fields.space()...};
  smith::Functional<signature> phi_f(&fitted_field.space(), trial_spaces);
  phi_f.AddDomainIntegral(Dimension<dim>{}, DependsOn<i...>{}, f, whole_domain);
  mfem::Vector b = phi_f(0.0, solution_fields...);

  mfem::CGSolver cg(MPI_COMM_WORLD);
  cg.SetOperator(M);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(500);
  cg.SetPrintLevel(2);
  cg.Mult(b, fitted_field);

  return fitted_field;
}

}  // namespace detail

/**
 * @brief determine field parameters to approximate the output of a user-provided q-function
 * @param[in] f the user-provided function to approximate
 * @param[in] pmesh the region over which to approximate the function f
 * @param[in] solution_fields [optional] any auxiliary field quantities needed to evaluate f
 *
 * @note: mesh is passed by non-const ref because mfem mutates the mesh when creating ParGridFunctions
 */
template <int dim, typename signature, int... n, typename func, typename... T>
FiniteElementState fit(func f, mfem::ParMesh& pmesh, const T&... solution_fields)
{
  auto iseq = std::make_integer_sequence<int, sizeof...(T)>{};
  return detail::fit<dim, signature>(iseq, f, pmesh, solution_fields...);
}

}  // namespace smith
