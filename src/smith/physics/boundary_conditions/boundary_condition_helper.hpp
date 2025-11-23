// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file boundary_condition_helper.hpp
 *
 * @brief This file contains the declarations of helper methods for boundary conditions
 */

#pragma once

#include "mfem.hpp"

namespace smith::mfem_ext {

/**
 * @brief Get a list of essential true dofs, @p ess_tdof_list , corresponding to the element attributes marked in the
 *array @p elem_attr_is_ess
 *
 * @param[in] fespace   The parallel finite element space
 * @param[in] elem_attr_is_ess An mfem::Array which indicates whether each element attribute is essential (the length
 *must equal the number of attributes defined on the mesh)
 * @param[out] ess_tdof_list An mfem::Array containing all the true dofs
 * @param[in] component An optional argument denoting which component (of a vector space) is essential, -1 implies all
 *components
 *
 **/
void GetEssentialTrueDofsFromElementAttribute(const mfem::ParFiniteElementSpace& fespace,
                                              const mfem::Array<int>& elem_attr_is_ess, mfem::Array<int>& ess_tdof_list,
                                              int component = -1);

/**
 * @brief A method to mark degrees of freedom associated with elements with the attributes specified in @p
 *element_attr_is_ess
 *
 * @param[in] fespace   The parallel finite element space
 * @param[in] elem_attr_is_ess An mfem::Array which indicates whether each element attribute is essential (the length
 *must equal the number of attributes defined on the mesh)
 * @param[out] ess_vdofs An mfem::Array marking the degrees of freedom
 * @param[in] component An optional argument denoting which component (of a vector space) is essential, -1 implies all
 *components
 *
 **/
void GetEssentialVDofsFromElementAttribute(const mfem::ParFiniteElementSpace& fespace,
                                           const mfem::Array<int>& elem_attr_is_ess, mfem::Array<int>& ess_vdofs,
                                           int component = -1);

}  // namespace smith::mfem_ext
