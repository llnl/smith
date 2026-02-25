// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file external_interface.hpp
 *
 * @brief C++ abstract interface and C/Fortran-callable API for computing
 *        per-element stiffness matrices using Smith's Functional framework.
 *
 * Typical workflow
 * ================
 * C/C++ caller:
 * @code{.cpp}
 *   void* handle = smith_create_hex3d_interface(
 *       num_verts, num_elems, vertices, connectivity,
 *       polynomial_order, lambda, mu);
 *
 *   int dpe = smith_get_dofs_per_element(handle);
 *   int ne  = smith_get_num_elements(handle);
 *
 *   std::vector<double> K(ne * dpe * dpe);
 *   std::vector<int>    row_map(ne * dpe), col_map(ne * dpe);
 *   smith_compute_element_stiffness(handle, K.data(), row_map.data(), col_map.data());
 *
 *   smith_destroy_interface(handle);
 * @endcode
 *
 * Fortran caller (ISO_C_BINDING):
 * @code{.f90}
 *   use iso_c_binding
 *   type(c_ptr) :: handle
 *   real(c_double), allocatable :: K(:)
 *   integer(c_int), allocatable :: row_map(:), col_map(:)
 *   integer(c_int) :: dpe, ne
 *
 *   handle = smith_create_hex3d_interface( &
 *       num_verts, num_elems, vertices, connectivity, poly_order, lam, mu)
 *
 *   dpe = smith_get_dofs_per_element(handle)
 *   ne  = smith_get_num_elements(handle)
 *   allocate(K(ne*dpe*dpe), row_map(ne*dpe), col_map(ne*dpe))
 *
 *   call smith_compute_element_stiffness(handle, K, row_map, col_map)
 *   ! row_map and col_map are 0-based; add 1 for Fortran array indices
 *
 *   call smith_destroy_interface(handle)
 * @endcode
 *
 * Element stiffness matrix storage
 * =================================
 * For element e with D = dofs_per_element degrees of freedom:
 *
 *   K_elem[e * D*D + col * D + row]  =  K_e(trial_col, test_row)
 *
 * This matches Smith's internal convention where the first element index is
 * the trial (column) DOF and the second is the test (row) DOF.  For
 * symmetric material models (e.g., linear isotropic elasticity) the matrix
 * is symmetric, so the distinction does not matter.
 *
 * DOF maps (0-based)
 * ==================
 *   row_dof_map[e * D + j]  = global row (test)  DOF index for local DOF j of element e
 *   col_dof_map[e * D + i]  = global col (trial) DOF index for local DOF i of element e
 */

#pragma once

#ifdef __cplusplus

#include <memory>

namespace smith {
namespace external_interface {

/**
 * @brief Abstract base class for the hex-mesh FEM interface.
 *
 * Concrete instantiations are created via @p create_hex3d_interface().
 * Users interact with the object through the C API below, but may also
 * use this interface directly from C++.
 */
class FEMInterfaceBase {
public:
  virtual ~FEMInterfaceBase() = default;

  /// @brief Total number of globally-numbered true DOFs on this rank.
  virtual int total_dofs() const = 0;

  /// @brief Number of hexahedral elements (CUBE geometry) in the mesh.
  virtual int num_elements() const = 0;

  /**
   * @brief Number of DOFs per element.
   *
   * For H1<1,3> (trilinear hex): 8 nodes × 3 components = 24.
   * For H1<2,3> (serendipity hex): 27 nodes × 3 components = 81.
   */
  virtual int dofs_per_element() const = 0;

  /**
   * @brief Compute element stiffness matrices.
   *
   * Internally this
   *   1. evaluates the Functional at zero displacement (with AD enabled) to
   *      populate the quadrature-point derivative buffers, and then
   *   2. calls @p assemble_element_matrices() to extract the per-element data.
   *
   * @param[out] K_elem     Flat buffer of size num_elements * D * D (D = dofs_per_element).
   *                        Layout: K_elem[e*D*D + col*D + row] = K_e(trial_col, test_row).
   * @param[out] row_dof_map Flat buffer of size num_elements * D.
   *                        row_dof_map[e*D + j] = global (0-based) row DOF index of local DOF j.
   * @param[out] col_dof_map Flat buffer of size num_elements * D.
   *                        col_dof_map[e*D + i] = global (0-based) col DOF index of local DOF i.
   */
  virtual void compute_element_stiffness(double* K_elem, int* row_dof_map, int* col_dof_map) = 0;
};

/**
 * @brief Create a 3-D hexahedral-mesh FEM interface using linear isotropic elasticity.
 *
 * An @p mfem::ParMesh is constructed on MPI_COMM_WORLD from the provided raw
 * vertex coordinates and element-to-vertex connectivity.  An H1 finite element
 * space of the requested polynomial order (1 or 2) is built on this mesh, and a
 * @p Functional with the linear-elastic q-function is initialised.
 *
 * @param num_verts        Number of mesh vertices.
 * @param num_elems        Number of hexahedral elements.
 * @param vertices         Flat array [num_verts * 3]: (x0,y0,z0, x1,y1,z1, ...).
 * @param connectivity     Flat array [num_elems * 8]: element→vertex connectivity (0-based).
 *                         The 8 vertex indices must follow mfem's hexahedron ordering.
 * @param polynomial_order Polynomial order of the H1 space (1 or 2).
 * @param lambda           Lamé first parameter (Pa or consistent units).
 * @param mu               Lamé second parameter / shear modulus.
 * @return Owning pointer to a @p FEMInterfaceBase concrete object.
 */
std::unique_ptr<FEMInterfaceBase> create_hex3d_interface(int num_verts, int num_elems, const double* vertices,
                                                         const int* connectivity, int polynomial_order, double lambda,
                                                         double mu);

}  // namespace external_interface
}  // namespace smith

extern "C" {
#endif  // __cplusplus

/* =========================================================================
 * C / Fortran-callable API
 * =========================================================================
 *
 * All functions use a void* opaque handle that wraps the C++ object.
 * Fortran callers should declare the handle as TYPE(C_PTR) and bind
 * with BIND(C, NAME='smith_...').
 */

/**
 * @brief Initialise Smith's logging and MPI infrastructure if not yet done.
 *
 * Safe to call multiple times; subsequent calls are no-ops.
 * If MPI has already been initialised by the calling application (as is
 * typical in HPC codes), Smith will detect this and not re-initialise it.
 */
void smith_initialize();

/**
 * @brief Create a 3-D hexahedral FEM interface (linear isotropic elasticity).
 *
 * @param[in] num_verts        Number of vertices.
 * @param[in] num_elems        Number of hexahedral elements.
 * @param[in] vertices         Flat array [num_verts * 3]: vertex coordinates.
 * @param[in] connectivity     Flat array [num_elems * 8]: element→vertex (0-based).
 * @param[in] polynomial_order 1 or 2.
 * @param[in] lambda           Lamé first parameter.
 * @param[in] mu               Lamé second parameter.
 * @return Opaque handle; must be freed with smith_destroy_interface().
 */
void* smith_create_hex3d_interface(int num_verts, int num_elems, const double* vertices, const int* connectivity,
                                   int polynomial_order, double lambda, double mu);

/**
 * @brief Destroy an interface object created by smith_create_hex3d_interface().
 * @param[in] handle Opaque handle returned by smith_create_hex3d_interface().
 */
void smith_destroy_interface(void* handle);

/**
 * @brief Query the total number of true DOFs managed by the interface.
 * @param[in] handle Opaque handle.
 * @return Number of true DOFs.
 */
int smith_get_total_dofs(void* handle);

/**
 * @brief Query the number of elements in the mesh.
 * @param[in] handle Opaque handle.
 * @return Number of hexahedral elements.
 */
int smith_get_num_elements(void* handle);

/**
 * @brief Query the number of DOFs per element.
 *
 * This determines the size of the element stiffness matrix (D×D) and
 * the length of the per-element DOF-map arrays.
 *
 * @param[in] handle Opaque handle.
 * @return DOFs per element (e.g. 24 for H1<1,3>, 81 for H1<2,3>).
 */
int smith_get_dofs_per_element(void* handle);

/**
 * @brief Compute element stiffness matrices and DOF mappings.
 *
 * @param[in]  handle      Opaque handle.
 * @param[out] K_elem      Caller-allocated buffer of size num_elements * D * D.
 *                         Layout: K_elem[e*D*D + col*D + row].
 * @param[out] row_dof_map Caller-allocated buffer of size num_elements * D.
 *                         0-based global row DOF indices.
 * @param[out] col_dof_map Caller-allocated buffer of size num_elements * D.
 *                         0-based global col DOF indices.
 */
void smith_compute_element_stiffness(void* handle, double* K_elem, int* row_dof_map, int* col_dof_map);

#ifdef __cplusplus
}  // extern "C"
#endif
