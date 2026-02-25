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
 * Returns (source=0, flux=stress) so that the weak form contribution is
 *   ∫ σ : ∇v dΩ
 * where σ = λ tr(ε) I + 2μ ε and ε = sym(∇u).
 *
 * The functor form (instead of a lambda) is required so that the runtime
 * values of lambda_ and mu_ can be captured without a non-literal lambda
 * causing issues with some CUDA/HIP back-ends.
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
  using test_space  = H1<p, 3>;
  using trial_space = H1<p, 3>;
  using FunctionalType = Functional<test_space(trial_space)>;

  // -------------------------------------------------------------------
  // Constructor: build mesh, FES, domain and functional
  // -------------------------------------------------------------------

  /**
   * @param num_verts    Number of vertices.
   * @param num_elems    Number of hexahedral elements.
   * @param vertices     Flat array [num_verts * 3].
   * @param connectivity Flat array [num_elems * 8], 0-based.
   * @param lambda       Lamé λ parameter.
   * @param mu           Lamé μ (shear modulus).
   */
  FEMInterface(int num_verts, int num_elems, const double* vertices, const int* connectivity, double lambda, double mu)
  {
    // ------------------------------------------------------------------
    // 1. Build serial mfem::Mesh from raw vertex / connectivity data.
    //
    //    mfem::Mesh(dim, nv, ne, nbdr, sdim) where:
    //      dim  = 3 (element dimension)
    //      nv   = num_verts
    //      ne   = num_elems
    //      nbdr = 0 (boundary elements added later if needed)
    //      sdim = 3 (space dimension)
    // ------------------------------------------------------------------
    mfem::Mesh serial_mesh(3, num_verts, num_elems, /*nbdr=*/0, /*sdim=*/3);

    for (int v = 0; v < num_verts; ++v) {
      serial_mesh.AddVertex(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2]);
    }

    for (int e = 0; e < num_elems; ++e) {
      const int* v = connectivity + 8 * e;
      // mfem::Mesh::AddHex expects 8 vertex indices (0-based).
      // Attribute 1 for all elements.
      serial_mesh.AddHex(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], /*attr=*/1);
    }

    serial_mesh.FinalizeHexMesh(/*generate_edges=*/1, /*refine=*/0, /*fix_orientation=*/true);

    // ------------------------------------------------------------------
    // 2. Wrap in a ParMesh on MPI_COMM_WORLD.
    //    For a serial (single-rank) run this is a trivial wrapper.
    // ------------------------------------------------------------------
    mesh_ = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);

    // ------------------------------------------------------------------
    // 3. Build the H1 finite element space.
    // ------------------------------------------------------------------
    fec_ = std::make_unique<mfem::H1_FECollection>(p, /*dim=*/3);
    fes_ = std::make_unique<mfem::ParFiniteElementSpace>(mesh_.get(), fec_.get(), /*vdim=*/3, smith::ordering);

    // ------------------------------------------------------------------
    // 4. Build the domain (all elements).
    // ------------------------------------------------------------------
    domain_ = std::make_unique<Domain>(EntireDomain(*mesh_));

    // ------------------------------------------------------------------
    // 5. Build the Functional and add the linear-elastic domain integral.
    // ------------------------------------------------------------------
    functional_ = std::make_unique<FunctionalType>(fes_.get(), std::array<const mfem::ParFiniteElementSpace*, 1>{fes_.get()});

    functional_->AddVolumeIntegral(DependsOn<0>{}, LinearElasticQFunction{lambda, mu}, *domain_);
  }

  // -------------------------------------------------------------------
  // FEMInterfaceBase overrides
  // -------------------------------------------------------------------

  int total_dofs()      const override { return fes_->GetTrueVSize();           }
  int num_elements()    const override { return mesh_->GetNE();                  }
  int dofs_per_element()const override
  {
    // For H1<p,3> on hexahedra:
    //   p=1 → 8 nodes × 3 components = 24
    //   p=2 → 27 nodes × 3 components = 81
    // We query this from the FES so the answer is always correct.
    return fes_->GetFE(0)->GetDof() * 3;
  }

  /**
   * @brief Seed the AD buffers and extract per-element stiffness matrices.
   *
   * Step 1: evaluate the Functional at zero displacement with differentiation
   *         enabled; this populates the quadrature-point derivative buffers.
   * Step 2: call assemble_element_matrices() to extract K and DOF maps.
   *
   * For linear elasticity the stiffness matrix is independent of the
   * displacement field, so zero is a valid linearisation point.
   */
  void compute_element_stiffness(double* K_elem, int* row_dof_map, int* col_dof_map) override
  {
    // Zero-displacement field used to seed the AD derivatives.
    mfem::Vector U_zero(fes_->GetTrueVSize());
    U_zero = 0.0;

    // Evaluate residual + record derivatives w.r.t. argument 0.
    auto [residual, dR_dU] = (*functional_)(0.0, smith::differentiate_wrt(U_zero));

    // Extract element matrices and DOF maps.
    ElementMatrices em = assemble_element_matrices(dR_dU);

    // We only expect hexahedral (CUBE) elements in this interface.
    const auto geom = mfem::Geometry::CUBE;
    if (em.K.find(geom) == em.K.end()) {
      throw std::runtime_error("smith_compute_element_stiffness: no CUBE elements found");
    }

    const int D  = em.rows_per_elem.at(geom);  // = cols_per_elem for symmetric space
    const int NE = em.num_elements.at(geom);

    // Copy K data.
    std::memcpy(K_elem, em.K.at(geom).data(), sizeof(double) * NE * D * D);

    // Copy DOF maps.
    std::memcpy(row_dof_map, em.test_dofs.at(geom).data(),  sizeof(int) * NE * D);
    std::memcpy(col_dof_map, em.trial_dofs.at(geom).data(), sizeof(int) * NE * D);
  }

private:
  std::unique_ptr<mfem::ParMesh>                 mesh_;
  std::unique_ptr<mfem::FiniteElementCollection> fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace>   fes_;
  std::unique_ptr<Domain>                        domain_;
  std::unique_ptr<FunctionalType>                functional_;
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

// Internal flag to guard against double-initialisation.
static bool g_smith_initialized = false;

extern "C" {

void smith_initialize()
{
  if (g_smith_initialized) return;

  // Initialise MPI if the calling application has not already done so.
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    MPI_Init(nullptr, nullptr);
  }

  // Initialise Smith/Axom SLIC logging.
  smith::logger::initialize(MPI_COMM_WORLD);

  // Initialise MFEM device (CPU mode by default).
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

void smith_destroy_interface(void* handle)
{
  delete static_cast<FEMInterfaceBase*>(handle);
}

int smith_get_total_dofs(void* handle)
{
  return static_cast<FEMInterfaceBase*>(handle)->total_dofs();
}

int smith_get_num_elements(void* handle)
{
  return static_cast<FEMInterfaceBase*>(handle)->num_elements();
}

int smith_get_dofs_per_element(void* handle)
{
  return static_cast<FEMInterfaceBase*>(handle)->dofs_per_element();
}

void smith_compute_element_stiffness(void* handle, double* K_elem, int* row_dof_map, int* col_dof_map)
{
  static_cast<FEMInterfaceBase*>(handle)->compute_element_stiffness(K_elem, row_dof_map, col_dof_map);
}

}  // extern "C"
