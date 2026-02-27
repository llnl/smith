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
 * Workflow
 * ========
 * The interface is split into a one-time setup phase and a per-step computation
 * phase, so that the expensive FEM infrastructure (mesh partitioning, global DOF
 * numbering, domain construction) is paid only once.
 *
 * Step 1 – Setup (once)
 * ---------------------
 * Create the interface from the mesh topology and material parameters.
 * The initial vertex coordinates are used only to seed the first geometry;
 * they can be the reference (beginning-of-step) positions.
 *
 * @code{.cpp}
 *   void* h = smith_create_hex3d_interface(
 *       num_verts, num_elems, initial_vertices, connectivity,
 *       polynomial_order, lambda, mu);
 *
 *   int D  = smith_get_dofs_per_element(h);
 *   int NE = smith_get_num_elements(h);
 * @endcode
 *
 * Step 2 – DOF maps (once)
 * ------------------------
 * Element DOF maps are topology-dependent only: they do not change when
 * nodal coordinates are updated.  Extract them once and reuse.
 *
 * @code{.cpp}
 *   std::vector<int> row_map(NE * D), col_map(NE * D);
 *   smith_get_dof_maps(h, row_map.data(), col_map.data());
 *   // row_map / col_map are 0-based; Fortran callers add 1.
 * @endcode
 *
 * Step 3 – Per-step stiffness computation (repeated)
 * ---------------------------------------------------
 * Pass the updated (end-of-step) vertex coordinates on each call.  The
 * caller may supply either the new absolute positions or compute them from
 * beginning-of-step coordinates + an increment before calling.
 * Passing @p nullptr for @p new_vertices reuses the coordinates from the
 * previous call (or the initial coordinates on the very first call).
 *
 * @code{.cpp}
 *   std::vector<double> K(NE * D * D);
 *   for (int step = 0; step < n_steps; ++step) {
 *     update_positions(current_vertices);   // caller's update
 *     smith_compute_element_stiffness(h, current_vertices, K.data());
 *     apply_boundary_conditions(K, ...);   // caller modifies K_e
 *     assemble_into_global_sparse(K, row_map, col_map);
 *   }
 *
 *   smith_destroy_interface(h);
 * @endcode
 *
 * Fortran example
 * ---------------
 * @code{.f90}
 *   use iso_c_binding
 *   type(c_ptr) :: h
 *   real(c_double), allocatable :: K(:), verts(:)
 *   integer(c_int), allocatable :: row_map(:), col_map(:)
 *   integer(c_int) :: D, NE
 *
 *   h  = smith_create_hex3d_interface(nv, ne, init_verts, conn, p, lam, mu)
 *   D  = smith_get_dofs_per_element(h)
 *   NE = smith_get_num_elements(h)
 *   allocate(row_map(NE*D), col_map(NE*D), K(NE*D*D))
 *   call smith_get_dof_maps(h, row_map, col_map)
 *   ! row_map / col_map are 0-based; add 1 for Fortran indices.
 *
 *   do step = 1, n_steps
 *     ! update verts to end-of-step positions ...
 *     call smith_compute_element_stiffness(h, verts, K)
 *     ! assemble K using row_map / col_map ...
 *   end do
 *   call smith_destroy_interface(h)
 * @endcode
 *
 * Element stiffness matrix storage
 * =================================
 * For element e with D = dofs_per_element degrees of freedom:
 *
 *   K_elem[e * D*D + col * D + row]  =  K_e(trial_col, test_row)
 *
 * DOF maps (0-based)
 * ==================
 *   row_dof_map[e * D + j]  = global row (test)  DOF index for local DOF j
 *   col_dof_map[e * D + i]  = global col (trial) DOF index for local DOF i
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
 * Users may interact through the C API below or directly from C++.
 */
class FEMInterfaceBase {
public:
  virtual ~FEMInterfaceBase() = default;

  /// @brief Total number of globally-numbered true DOFs.
  virtual int total_dofs() const = 0;

  /// @brief Number of hexahedral elements in the mesh.
  virtual int num_elements() const = 0;

  /**
   * @brief Number of DOFs per element.
   *
   * For H1<1,3> (trilinear hex): 8 nodes × 3 components = 24.
   * For H1<2,3> (serendipity hex): 27 nodes × 3 components = 81.
   */
  virtual int dofs_per_element() const = 0;

  /**
   * @brief Extract per-element DOF maps (call once after construction).
   *
   * Element DOF maps are topology-dependent only and do not change when
   * nodal coordinates are updated.  The maps are cached internally on the
   * first call; subsequent calls simply copy the cached data.
   *
   * @param[out] row_dof_map  Flat buffer, size num_elements * D.
   *                          row_dof_map[e*D + j] = global (0-based) row DOF
   *                          index for local DOF j of element e.
   * @param[out] col_dof_map  Flat buffer, size num_elements * D.
   *                          col_dof_map[e*D + i] = global (0-based) col DOF
   *                          index for local DOF i of element e.
   */
  virtual void get_dof_maps(int* row_dof_map, int* col_dof_map) = 0;

  /**
   * @brief Compute element stiffness matrices at the supplied nodal positions.
   *
   * Internally this
   *   1. Updates the mesh node positions to @p new_vertices (unless nullptr).
   *   2. Rebuilds the Smith Functional so the geometric factors reflect the
   *      new positions (O(n_elem * n_qpts), same order as the evaluation).
   *   3. Evaluates the Functional at zero displacement with AD enabled to
   *      populate quadrature-point derivative buffers.
   *   4. Calls assemble_element_matrices() to extract the per-element data.
   *
   * For linear isotropic elasticity the stiffness is displacement-independent,
   * so zero is always a valid linearisation point.
   *
   * @param[in]  new_vertices  Updated vertex coordinates, flat array
   *                           [num_verts * 3] in interleaved order
   *                           (x0,y0,z0, x1,y1,z1, ...).
   *                           Pass @p nullptr to reuse the coordinates from
   *                           the previous call (no geometry update).
   * @param[out] K_elem        Flat buffer, size num_elements * D * D.
   *                           Layout: K_elem[e*D*D + col*D + row].
   */
  virtual void compute_element_stiffness(const double* new_vertices, double* K_elem) = 0;
};

/**
 * @brief Create a 3-D hexahedral-mesh FEM interface using linear isotropic elasticity.
 *
 * @param num_verts        Number of mesh vertices.
 * @param num_elems        Number of hexahedral elements.
 * @param vertices         Initial vertex coordinates, flat [num_verts * 3]:
 *                         (x0,y0,z0, x1,y1,z1, ...).
 * @param connectivity     Element-to-vertex connectivity, flat [num_elems * 8],
 *                         0-based.  The 8 vertex indices follow mfem's hex ordering.
 * @param polynomial_order Polynomial order of the H1 space (1 or 2).
 * @param lambda           Lamé first parameter.
 * @param mu               Lamé second parameter / shear modulus.
 * @return Owning pointer to a concrete @p FEMInterfaceBase object.
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
 * All functions use a void* opaque handle wrapping the C++ object.
 * Fortran callers should declare the handle as TYPE(C_PTR) and bind
 * with BIND(C, NAME='smith_...').
 */

/**
 * @brief Initialise Smith's MPI and logging infrastructure if not yet done.
 * Safe to call multiple times; subsequent calls are no-ops.
 */
void smith_initialize();

/**
 * @brief Create a 3-D hexahedral FEM interface (linear isotropic elasticity).
 *
 * @param[in] num_verts        Number of vertices.
 * @param[in] num_elems        Number of hexahedral elements.
 * @param[in] vertices         Initial flat vertex coords [num_verts * 3].
 * @param[in] connectivity     Element→vertex connectivity [num_elems * 8], 0-based.
 * @param[in] polynomial_order 1 or 2.
 * @param[in] lambda           Lamé first parameter.
 * @param[in] mu               Lamé second parameter.
 * @return Opaque handle; must be freed with smith_destroy_interface().
 */
void* smith_create_hex3d_interface(int num_verts, int num_elems, const double* vertices, const int* connectivity,
                                   int polynomial_order, double lambda, double mu);

/**
 * @brief Destroy an interface object created by smith_create_hex3d_interface().
 */
void smith_destroy_interface(void* handle);

/** @brief Total number of true DOFs. */
int smith_get_total_dofs(void* handle);

/** @brief Number of hexahedral elements in the mesh. */
int smith_get_num_elements(void* handle);

/**
 * @brief DOFs per element (e.g. 24 for H1<1,3>, 81 for H1<2,3>).
 * Determines element matrix size D×D and length of per-element DOF-map arrays.
 */
int smith_get_dofs_per_element(void* handle);

/**
 * @brief Extract per-element DOF maps (call once; results are cached).
 *
 * @param[in]  handle      Opaque handle.
 * @param[out] row_dof_map Buffer of size num_elements * D; 0-based global row DOF indices.
 * @param[out] col_dof_map Buffer of size num_elements * D; 0-based global col DOF indices.
 */
void smith_get_dof_maps(void* handle, int* row_dof_map, int* col_dof_map);

/**
 * @brief Compute element stiffness matrices at updated nodal positions.
 *
 * @param[in]  handle        Opaque handle.
 * @param[in]  new_vertices  Updated vertex coords [num_verts * 3], interleaved
 *                           (x0,y0,z0, x1,y1,z1, ...).  Pass NULL to reuse
 *                           coordinates from the previous call.
 * @param[out] K_elem        Buffer of size num_elements * D * D.
 *                           Layout: K_elem[e*D*D + col*D + row].
 */
void smith_compute_element_stiffness(void* handle, const double* new_vertices, double* K_elem);

#ifdef __cplusplus
}  // extern "C"
#endif
