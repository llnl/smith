// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file external_interface.cpp
 *
 * @brief Implementation of the C++ FEM interface and C/Fortran-callable API
 *        defined in external_interface.hpp.
 *
 * Template instantiations are provided for polynomial orders p=1 and p=2 on
 * 3-D hexahedral meshes (H1<p,3> spaces, linear isotropic elasticity).
 *
 * Coordinate update strategy
 * ==========================
 * The expensive FEM setup (mesh partitioning, global DOF numbering, domain
 * construction) is done once in the constructor.  Per-step coordinate updates
 * work as follows:
 *
 *   1. The new vertex positions are written directly into the mesh's nodal
 *      GridFunction (stored in `mfem::Mesh::Nodes`).  This avoids rebuilding
 *      the ParMesh or the finite element space.
 *
 *   2. The `smith::Functional` is then rebuilt for that integral.  This is
 *      cheap: it constructs `GeometricFactors` (positions and Jacobians at
 *      each quadrature point, O(n_elem * n_qpts)) and sets up lightweight
 *      closure objects.
 *
 *   3. The Functional is evaluated at zero displacement with AD enabled to
 *      populate quadrature-point derivative buffers.
 *
 *   4. `assemble_element_matrices()` extracts K_e and DOF maps.  DOF maps are
 *      topology-dependent (constant across steps) and are cached after the
 *      first extraction.
 *
 * Node coordinate layout
 * ======================
 * MFEM stores mesh node coordinates in a GridFunction whose DOF ordering
 * depends on the ordering type used when `EnsureNodes()` was called.  This
 * implementation queries `node_fes->GetOrdering()` at runtime so that it
 * works regardless of the MFEM build configuration.  For a p=1 hex mesh
 * (one node per vertex) the DOF index for vertex v, spatial component d is:
 *
 *   byNODES : dof = v + d * n_scalar_dofs
 *   byVDIM  : dof = v * spaceDim + d
 *
 * This mapping is computed once during construction and stored in
 * `node_dof_of_vertex_`, avoiding repeated runtime branches on the hot path.
 */

#include "external_interface.hpp"

#include <cstring>
#include <stdexcept>

#include "mpi.h"

#include "axom/slic.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/isotropic_tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/numerics/functional/differentiate_wrt.hpp"

namespace smith {
namespace external_interface {

// ---------------------------------------------------------------------------
// Linear isotropic elasticity q-function
// ---------------------------------------------------------------------------

/**
 * @brief Q-function for 3-D linear isotropic elasticity.
 *
 * Returns (source=0, flux=stress) so that the weak-form contribution is
 *   ∫ σ : ∇v dΩ
 * where σ = λ tr(ε) I + 2μ ε  and  ε = sym(∇u).
 *
 * The functor form (rather than a capturing lambda) ensures compatibility
 * with GPU execution back-ends.
 */
struct LinearElasticQFunction {
  double lambda_;
  double mu_;

  template <typename PositionType, typename DisplacementType>
  SMITH_HOST_DEVICE auto operator()(double /*time*/, PositionType /*position*/,
                                    DisplacementType displacement) const
  {
    auto [u, grad_u] = displacement;
    auto strain      = 0.5 * (grad_u + transpose(grad_u));
    auto stress      = lambda_ * tr(strain) * Identity<3>() + 2.0 * mu_ * strain;
    return smith::tuple{zero{}, stress};
  }
};

// ---------------------------------------------------------------------------
// Templated C++ implementation (one per polynomial order)
// ---------------------------------------------------------------------------

/**
 * @brief Concrete FEM interface for 3-D hexahedral meshes.
 *
 * @tparam p Polynomial order of the H1 finite element space (1 or 2).
 */
template <int p>
class FEMInterface : public FEMInterfaceBase {
public:
  using test_space     = H1<p, 3>;
  using trial_space    = H1<p, 3>;
  using FunctionalType = Functional<test_space(trial_space)>;

  // -------------------------------------------------------------------
  // Constructor: build mesh, FES, domain; initial functional
  // -------------------------------------------------------------------

  /**
   * @param num_verts    Total number of mesh vertices.
   * @param num_elems    Number of hexahedral elements.
   * @param vertices     Initial vertex coords, flat [num_verts * 3].
   * @param connectivity Element→vertex connectivity, flat [num_elems * 8], 0-based.
   * @param lambda       Lamé λ parameter.
   * @param mu           Lamé μ (shear modulus).
   */
  FEMInterface(int num_verts, int num_elems, const double* vertices, const int* connectivity, double lambda, double mu)
      : num_verts_(num_verts), lambda_(lambda), mu_(mu)
  {
    // ------------------------------------------------------------------
    // 1. Build serial mfem::Mesh from raw vertex / connectivity data.
    // ------------------------------------------------------------------
    mfem::Mesh serial_mesh(3, num_verts, num_elems, /*nbdr=*/0, /*sdim=*/3);

    for (int v = 0; v < num_verts; ++v) {
      serial_mesh.AddVertex(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2]);
    }
    for (int e = 0; e < num_elems; ++e) {
      const int* v = connectivity + 8 * e;
      serial_mesh.AddHex(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], /*attr=*/1);
    }
    serial_mesh.FinalizeHexMesh(/*generate_edges=*/1, /*refine=*/0, /*fix_orientation=*/true);

    // Smith's Functional requires the mesh to have a nodal GridFunction.
    // EnsureNodes() creates a p=1 H1 GF matching the geometric order of
    // the mesh (here p=1, so nodes coincide with vertices).
    serial_mesh.EnsureNodes();

    // ------------------------------------------------------------------
    // 2. Wrap in a ParMesh.
    //    The ParMesh inherits the nodal GF from the serial mesh.
    // ------------------------------------------------------------------
    mesh_ = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);

    // Required for parallel DOF exchange (face-neighbor data).
    mesh_->ExchangeFaceNbrData();

    // ------------------------------------------------------------------
    // 3. Build the vertex→node-DOF index map.
    //
    //    The mesh's nodal GF uses whatever ordering MFEM chose in
    //    EnsureNodes().  We query this once and store the DOF index for
    //    each (vertex, component) pair so coordinate updates are fast.
    // ------------------------------------------------------------------
    build_node_dof_map();

    // ------------------------------------------------------------------
    // 4. Build the H1 finite element space for the solution.
    // ------------------------------------------------------------------
    fec_ = std::make_unique<mfem::H1_FECollection>(p, /*dim=*/3);
    fes_ = std::make_unique<mfem::ParFiniteElementSpace>(mesh_.get(), fec_.get(), /*vdim=*/3, smith::ordering);

    // ------------------------------------------------------------------
    // 5. Build the domain (all hexahedral elements).
    // ------------------------------------------------------------------
    domain_ = std::make_unique<Domain>(EntireDomain(*mesh_));

    // ------------------------------------------------------------------
    // 6. Build the Functional for the initial coordinates.
    // ------------------------------------------------------------------
    build_functional();
  }

  // -------------------------------------------------------------------
  // FEMInterfaceBase overrides
  // -------------------------------------------------------------------

  int total_dofs()       const override { return fes_->GetTrueVSize(); }
  int num_elements()     const override { return mesh_->GetNE();        }
  int dofs_per_element() const override
  {
    // For H1<p,3>: nodes_per_element * 3 components.
    // p=1 → 8 * 3 = 24 ; p=2 → 27 * 3 = 81.
    return fes_->GetFE(0)->GetDof() * 3;
  }

  /**
   * @brief Extract per-element DOF maps; results are cached after the first call.
   */
  void get_dof_maps(int* row_dof_map, int* col_dof_map) override
  {
    ensure_dof_maps_cached();

    const int D  = dofs_per_element();
    const int NE = num_elements();
    std::memcpy(row_dof_map, row_dofs_cache_.data(), sizeof(int) * NE * D);
    std::memcpy(col_dof_map, col_dofs_cache_.data(), sizeof(int) * NE * D);
  }

  /**
   * @brief Update mesh node coordinates (if supplied) and compute K_e.
   *
   * @param new_vertices  Updated vertex coordinates [num_verts * 3], interleaved
   *                      (x0,y0,z0, x1,y1,z1, ...).  Pass nullptr to reuse the
   *                      coordinates set on the previous call.
   * @param K_elem        Output buffer [num_elements * D * D].
   */
  void compute_element_stiffness(const double* new_vertices, double* K_elem) override
  {
    if (new_vertices != nullptr) {
      update_mesh_nodes(new_vertices);
      build_functional();  // recompute GeometricFactors for the new positions
    }

    ElementMatrices em = evaluate_and_extract();

    // Cache DOF maps on the first call (they are geometry-independent).
    if (!dof_maps_computed_) {
      cache_dof_maps(em);
    }

    // Copy element stiffness data to caller's buffer.
    const auto geom = mfem::Geometry::CUBE;
    if (em.K.find(geom) == em.K.end()) {
      throw std::runtime_error("smith external_interface: no CUBE elements found in assemble_element_matrices result");
    }
    const int D  = em.rows_per_elem.at(geom);
    const int NE = em.num_elements.at(geom);
    std::memcpy(K_elem, em.K.at(geom).data(), sizeof(double) * NE * D * D);
  }

private:
  // -------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------

  /**
   * @brief Build the vertex→node-DOF mapping for the mesh's nodal GridFunction.
   *
   * For a p=1 mesh with n_verts vertices, the node GF has n_verts * 3 DOFs
   * (one per vertex per spatial component).  The DOF index for vertex v and
   * component d is:
   *   byNODES : v + d * n_scalar_dofs
   *   byVDIM  : v * 3 + d
   *
   * The map is stored as node_dof_of_vertex_[v * 3 + d] = dof_index.
   */
  void build_node_dof_map()
  {
    mfem::GridFunction* nodes = mesh_->GetNodes();
    mfem::FiniteElementSpace* node_fes = nodes->FESpace();
    const int n_scalar = node_fes->GetNDofs();
    const auto ord     = node_fes->GetOrdering();

    node_dof_of_vertex_.resize(num_verts_ * 3);
    for (int v = 0; v < num_verts_; ++v) {
      for (int d = 0; d < 3; ++d) {
        int dof_idx = (ord == mfem::Ordering::byNODES) ? v + d * n_scalar : v * 3 + d;
        node_dof_of_vertex_[v * 3 + d] = dof_idx;
      }
    }
  }

  /**
   * @brief Write @p new_vertices into the mesh's nodal GridFunction.
   *
   * Uses the precomputed vertex→DOF map so the hot path has no branches.
   *
   * @note This is designed for the case where all mesh vertices are owned by
   *       a single MPI rank (or equivalently, the caller provides all vertices
   *       local to this rank in the same numbering used at construction).
   */
  void update_mesh_nodes(const double* new_vertices)
  {
    mfem::GridFunction* nodes = mesh_->GetNodes();
    for (int v = 0; v < num_verts_; ++v) {
      for (int d = 0; d < 3; ++d) {
        (*nodes)[node_dof_of_vertex_[v * 3 + d]] = new_vertices[3 * v + d];
      }
    }
  }

  /**
   * @brief (Re)build the Functional using the current mesh node positions.
   *
   * Creates a new `smith::Functional` and calls `AddVolumeIntegral`, which
   * triggers recomputation of the GeometricFactors (quadrature-point
   * positions and Jacobians) from the current mesh nodes.  The cost is
   * O(n_elem * n_qpts), the same order as the subsequent evaluation.
   */
  void build_functional()
  {
    functional_ = std::make_unique<FunctionalType>(
        fes_.get(), std::array<const mfem::ParFiniteElementSpace*, 1>{fes_.get()});
    functional_->AddVolumeIntegral(DependsOn<0>{}, LinearElasticQFunction{lambda_, mu_}, *domain_);
  }

  /**
   * @brief Evaluate the Functional at zero displacement (seeds AD derivatives)
   *        and extract per-element stiffness matrices plus DOF maps.
   */
  ElementMatrices evaluate_and_extract()
  {
    // For linear elasticity, K_e is displacement-independent.
    // Evaluating at zero displacement is always valid.
    mfem::Vector U_zero(fes_->GetTrueVSize());
    U_zero = 0.0;

    auto [residual, dR_dU] = (*functional_)(0.0, smith::differentiate_wrt(U_zero));
    return assemble_element_matrices(dR_dU);
  }

  /**
   * @brief Ensure DOF maps have been extracted and cached.
   *
   * If not yet cached, performs one functional evaluation (at the current
   * node positions) to obtain the maps.
   */
  void ensure_dof_maps_cached()
  {
    if (!dof_maps_computed_) {
      ElementMatrices em = evaluate_and_extract();
      cache_dof_maps(em);
    }
  }

  /**
   * @brief Store DOF maps from an ElementMatrices result into the cache.
   */
  void cache_dof_maps(const ElementMatrices& em)
  {
    const auto geom = mfem::Geometry::CUBE;
    if (em.test_dofs.find(geom) == em.test_dofs.end()) {
      throw std::runtime_error("smith external_interface: no CUBE geometry in element matrices");
    }
    row_dofs_cache_ = em.test_dofs.at(geom);
    col_dofs_cache_ = em.trial_dofs.at(geom);
    dof_maps_computed_ = true;
  }

  // -------------------------------------------------------------------
  // Data members
  // -------------------------------------------------------------------

  /// Number of vertices (matches the count at construction).
  int num_verts_;

  /// Material parameters (stored for Functional rebuild on coordinate update).
  double lambda_;
  double mu_;

  /// Precomputed vertex→node-DOF index map: node_dof_of_vertex_[v*3+d] = dof.
  std::vector<int> node_dof_of_vertex_;

  /// Mesh, FES, and domain — allocated once, never rebuilt.
  std::unique_ptr<mfem::ParMesh>                 mesh_;
  std::unique_ptr<mfem::FiniteElementCollection> fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace>   fes_;
  std::unique_ptr<Domain>                        domain_;

  /// Functional — rebuilt whenever coordinates change.
  std::unique_ptr<FunctionalType> functional_;

  /// DOF map cache (topology-dependent, computed once and reused).
  bool             dof_maps_computed_ = false;
  std::vector<int> row_dofs_cache_;
  std::vector<int> col_dofs_cache_;
};

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

std::unique_ptr<FEMInterfaceBase> create_hex3d_interface(int num_verts, int num_elems, const double* vertices,
                                                         const int* connectivity, int polynomial_order, double lambda,
                                                         double mu)
{
  switch (polynomial_order) {
    case 1:
      return std::make_unique<FEMInterface<1>>(num_verts, num_elems, vertices, connectivity, lambda, mu);
    case 2:
      return std::make_unique<FEMInterface<2>>(num_verts, num_elems, vertices, connectivity, lambda, mu);
    default:
      throw std::invalid_argument("smith external_interface: polynomial_order must be 1 or 2");
  }
}

}  // namespace external_interface
}  // namespace smith

// =============================================================================
// C / Fortran-callable API implementation
// =============================================================================

using namespace smith::external_interface;

static bool g_smith_initialized = false;

extern "C" {

void smith_initialize()
{
  if (g_smith_initialized) return;

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    MPI_Init(nullptr, nullptr);
  }

  smith::logger::initialize(MPI_COMM_WORLD);
  mfem::Device device("cpu");

  g_smith_initialized = true;
}

void* smith_create_hex3d_interface(int num_verts, int num_elems, const double* vertices, const int* connectivity,
                                   int polynomial_order, double lambda, double mu)
{
  smith_initialize();
  auto* ptr = create_hex3d_interface(num_verts, num_elems, vertices, connectivity, polynomial_order, lambda, mu)
                  .release();
  return static_cast<void*>(ptr);
}

void smith_destroy_interface(void* handle) { delete static_cast<FEMInterfaceBase*>(handle); }

int smith_get_total_dofs(void* handle) { return static_cast<FEMInterfaceBase*>(handle)->total_dofs(); }

int smith_get_num_elements(void* handle) { return static_cast<FEMInterfaceBase*>(handle)->num_elements(); }

int smith_get_dofs_per_element(void* handle) { return static_cast<FEMInterfaceBase*>(handle)->dofs_per_element(); }

void smith_get_dof_maps(void* handle, int* row_dof_map, int* col_dof_map)
{
  static_cast<FEMInterfaceBase*>(handle)->get_dof_maps(row_dof_map, col_dof_map);
}

void smith_compute_element_stiffness(void* handle, const double* new_vertices, double* K_elem)
{
  static_cast<FEMInterfaceBase*>(handle)->compute_element_stiffness(new_vertices, K_elem);
}

}  // extern "C"
