// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/functional/geometric_factors.hpp"

#include <iostream>
#include <vector>

#include "smith/numerics/functional/finite_element.hpp"

namespace smith {

/**
 * @brief a kernel to compute the positions and jacobians at each quadrature point (mfem calls this "geometric factors")
 * @tparam Q a parameter controlling the number of quadrature points an element
 * @tparam function_space the polynomial order and kind of function space used to interpolate
 * @tparam geom the element geometry
 * @param positions_q (output) the positions for each quadrature point
 * @param jacobians_q (output) the jacobians for each quadrature point
 * @param positions_e (input) the "e-vector" of position data
 * @param elements (input) the list of element indices that are part of this domain
 */
template <int Q, mfem::Geometry::Type geom, typename function_space>
void compute_geometric_factors(mfem::Vector& positions_q, mfem::Vector& jacobians_q, const mfem::Vector& positions_e,
                               const std::vector<int>& elements)
{
  static constexpr TensorProductQuadratureRule<Q> rule{};

  constexpr int spatial_dim = function_space::components;
  constexpr int geometry_dim = dimension_of(geom);
  constexpr int qpts_per_elem = num_quadrature_points(geom, Q);

  using element_type = finite_element<geom, function_space>;
  using position_type = tensor<double, spatial_dim, qpts_per_elem>;
  using jacobian_type = tensor<double, geometry_dim, spatial_dim, qpts_per_elem>;

  auto X_q = reinterpret_cast<position_type*>(positions_q.ReadWrite());
  auto J_q = reinterpret_cast<jacobian_type*>(jacobians_q.ReadWrite());
  auto X = reinterpret_cast<const typename element_type::dof_type*>(positions_e.Read());

  std::size_t num_elements = elements.size();

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements; e++) {
    // load the positions for the nodes in this element
    auto X_e = X[e];

    // calculate the values and derivatives (w.r.t. xi) of X at each quadrature point
    auto quadrature_values = element_type::interpolate(X_e, rule);

    // mfem wants to store this data in a different layout, so we have to transpose it
    for (int q = 0; q < qpts_per_elem; q++) {
      auto [value, gradient] = quadrature_values[q];
      for (int i = 0; i < spatial_dim; i++) {
        X_q[e](i, q) = value[i];
        if constexpr (std::is_same_v<decltype(value), decltype(gradient)>) {
          J_q[e](0, i, q) = gradient[i];
        }
        if constexpr (!std::is_same_v<decltype(value), decltype(gradient)>) {
          for (int j = 0; j < geometry_dim; j++) {
            J_q[e](j, i, q) = gradient(i, j);
          }
        }
      }
    }
  }
}

GeometricFactors::GeometricFactors(const Domain& domain, int q, mfem::Geometry::Type geom)
{
  const mfem::ParGridFunction* nodes = static_cast<const mfem::ParGridFunction*>(domain.mesh_.GetNodes());
  mfem::ParFiniteElementSpace* fes = nodes->ParFESpace();
  // const mfem::GridFunction* nodes = domain.mesh_.GetNodes();
  // const mfem::FiniteElementSpace* fes = nodes->FESpace();

  const std::vector<int>& element_ids = domain.get_mfem_ids(geom);

  auto restriction = smith::ElementRestriction(fes, geom, element_ids);
  mfem::Vector X_e(int(restriction.ESize()));
  restriction.Gather(*nodes, X_e);

  // assumes all elements are the same order
  int p = fes->GetElementOrder(0);

  int spatial_dim = domain.mesh_.SpaceDimension();
  int geometry_dim = dimension_of(geom);
  int qpts_per_elem = num_quadrature_points(geom, q);

  num_elements = element_ids.size();

  X = mfem::Vector(int(num_elements) * qpts_per_elem * spatial_dim);
  J = mfem::Vector(int(num_elements) * qpts_per_elem * spatial_dim * geometry_dim);

#define DISPATCH_KERNEL(GEOM, P, Q, BDR)                                                                               \
  if (geom == mfem::Geometry::GEOM && p == P && q == Q && (spatial_dim - geometry_dim) == BDR) {                       \
    compute_geometric_factors<Q, mfem::Geometry::GEOM, H1<P, dimension_of(mfem::Geometry::GEOM) + BDR> >(X, J, X_e,    \
                                                                                                         element_ids); \
    return;                                                                                                            \
  }

  DISPATCH_KERNEL(SEGMENT, 1, 1, 1);
  DISPATCH_KERNEL(SEGMENT, 1, 2, 1);
  DISPATCH_KERNEL(SEGMENT, 1, 3, 1);
  DISPATCH_KERNEL(SEGMENT, 1, 4, 1);

  DISPATCH_KERNEL(SEGMENT, 2, 1, 1);
  DISPATCH_KERNEL(SEGMENT, 2, 2, 1);
  DISPATCH_KERNEL(SEGMENT, 2, 3, 1);
  DISPATCH_KERNEL(SEGMENT, 2, 4, 1);

  DISPATCH_KERNEL(SEGMENT, 3, 1, 1);
  DISPATCH_KERNEL(SEGMENT, 3, 2, 1);
  DISPATCH_KERNEL(SEGMENT, 3, 3, 1);
  DISPATCH_KERNEL(SEGMENT, 3, 4, 1);

  ///////////////////////////////////////

  DISPATCH_KERNEL(TRIANGLE, 1, 1, 0);
  DISPATCH_KERNEL(TRIANGLE, 1, 2, 0);
  DISPATCH_KERNEL(TRIANGLE, 1, 3, 0);
  DISPATCH_KERNEL(TRIANGLE, 1, 4, 0);

  DISPATCH_KERNEL(TRIANGLE, 2, 1, 0);
  DISPATCH_KERNEL(TRIANGLE, 2, 2, 0);
  DISPATCH_KERNEL(TRIANGLE, 2, 3, 0);
  DISPATCH_KERNEL(TRIANGLE, 2, 4, 0);

  DISPATCH_KERNEL(TRIANGLE, 3, 1, 0);
  DISPATCH_KERNEL(TRIANGLE, 3, 2, 0);
  DISPATCH_KERNEL(TRIANGLE, 3, 3, 0);
  DISPATCH_KERNEL(TRIANGLE, 3, 4, 0);

  DISPATCH_KERNEL(TRIANGLE, 1, 1, 1);
  DISPATCH_KERNEL(TRIANGLE, 1, 2, 1);
  DISPATCH_KERNEL(TRIANGLE, 1, 3, 1);
  DISPATCH_KERNEL(TRIANGLE, 1, 4, 1);

  DISPATCH_KERNEL(TRIANGLE, 2, 1, 1);
  DISPATCH_KERNEL(TRIANGLE, 2, 2, 1);
  DISPATCH_KERNEL(TRIANGLE, 2, 3, 1);
  DISPATCH_KERNEL(TRIANGLE, 2, 4, 1);

  DISPATCH_KERNEL(TRIANGLE, 3, 1, 1);
  DISPATCH_KERNEL(TRIANGLE, 3, 2, 1);
  DISPATCH_KERNEL(TRIANGLE, 3, 3, 1);
  DISPATCH_KERNEL(TRIANGLE, 3, 4, 1);

  ///////////////////////////////////////

  DISPATCH_KERNEL(SQUARE, 1, 1, 0);
  DISPATCH_KERNEL(SQUARE, 1, 2, 0);
  DISPATCH_KERNEL(SQUARE, 1, 3, 0);
  DISPATCH_KERNEL(SQUARE, 1, 4, 0);

  DISPATCH_KERNEL(SQUARE, 2, 1, 0);
  DISPATCH_KERNEL(SQUARE, 2, 2, 0);
  DISPATCH_KERNEL(SQUARE, 2, 3, 0);
  DISPATCH_KERNEL(SQUARE, 2, 4, 0);

  DISPATCH_KERNEL(SQUARE, 3, 1, 0);
  DISPATCH_KERNEL(SQUARE, 3, 2, 0);
  DISPATCH_KERNEL(SQUARE, 3, 3, 0);
  DISPATCH_KERNEL(SQUARE, 3, 4, 0);

  DISPATCH_KERNEL(SQUARE, 1, 1, 1);
  DISPATCH_KERNEL(SQUARE, 1, 2, 1);
  DISPATCH_KERNEL(SQUARE, 1, 3, 1);
  DISPATCH_KERNEL(SQUARE, 1, 4, 1);

  DISPATCH_KERNEL(SQUARE, 2, 1, 1);
  DISPATCH_KERNEL(SQUARE, 2, 2, 1);
  DISPATCH_KERNEL(SQUARE, 2, 3, 1);
  DISPATCH_KERNEL(SQUARE, 2, 4, 1);

  DISPATCH_KERNEL(SQUARE, 3, 1, 1);
  DISPATCH_KERNEL(SQUARE, 3, 2, 1);
  DISPATCH_KERNEL(SQUARE, 3, 3, 1);
  DISPATCH_KERNEL(SQUARE, 3, 4, 1);

  ///////////////////////////////////////

  DISPATCH_KERNEL(TETRAHEDRON, 1, 1, 0);
  DISPATCH_KERNEL(TETRAHEDRON, 1, 2, 0);
  DISPATCH_KERNEL(TETRAHEDRON, 1, 3, 0);
  DISPATCH_KERNEL(TETRAHEDRON, 1, 4, 0);

  DISPATCH_KERNEL(TETRAHEDRON, 2, 1, 0);
  DISPATCH_KERNEL(TETRAHEDRON, 2, 2, 0);
  DISPATCH_KERNEL(TETRAHEDRON, 2, 3, 0);
  DISPATCH_KERNEL(TETRAHEDRON, 2, 4, 0);

  DISPATCH_KERNEL(TETRAHEDRON, 3, 1, 0);
  DISPATCH_KERNEL(TETRAHEDRON, 3, 2, 0);
  DISPATCH_KERNEL(TETRAHEDRON, 3, 3, 0);
  DISPATCH_KERNEL(TETRAHEDRON, 3, 4, 0);

  ///////////////////////////////////////

  DISPATCH_KERNEL(CUBE, 1, 1, 0);
  DISPATCH_KERNEL(CUBE, 1, 2, 0);
  DISPATCH_KERNEL(CUBE, 1, 3, 0);
  DISPATCH_KERNEL(CUBE, 1, 4, 0);

  DISPATCH_KERNEL(CUBE, 2, 1, 0);
  DISPATCH_KERNEL(CUBE, 2, 2, 0);
  DISPATCH_KERNEL(CUBE, 2, 3, 0);
  DISPATCH_KERNEL(CUBE, 2, 4, 0);

  DISPATCH_KERNEL(CUBE, 3, 1, 0);
  DISPATCH_KERNEL(CUBE, 3, 2, 0);
  DISPATCH_KERNEL(CUBE, 3, 3, 0);
  DISPATCH_KERNEL(CUBE, 3, 4, 0);

#undef DISPATCH_KERNEL

  std::cout << "should never be reached " << std::endl;
}

}  // namespace smith
