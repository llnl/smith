#pragma once

#include "refactor/interpolation.hpp"

namespace refactor {

template <>
struct FiniteElement < Geometry::Edge, Family::Hcurl >{

  using source_type = vec1;
  using flux_type = vec1;

  static constexpr uint32_t dim = 1;

  __host__ __device__ uint32_t num_nodes() const { return p; }

  void nodes(nd::view< double, 2 > xi) const {
    if (p == 1) {
      xi(0, 0) = GaussLegendreNode01<1, 0>();
    }

    if (p == 2) {
      xi(0, 0) = GaussLegendreNode01<2, 0>();
      xi(1, 0) = GaussLegendreNode01<2, 1>();
    }

    if (p == 3) {
      xi(0, 0) = GaussLegendreNode01<3, 0>();
      xi(1, 0) = GaussLegendreNode01<3, 1>();
      xi(2, 0) = GaussLegendreNode01<3, 2>();
    }
  }

  void directions(nd::view< double, 2 > xi) {
    for (int i = 0; i < p; i++) { xi(i, 0) = 1.0; }
  }

  __host__ __device__ uint32_t num_interior_nodes() { return p; }

  void interior_nodes(nd::view< double, 2 > xi) {
    nodes(xi); // all the nodes in these elements are interior
  }

  void interior_directions(nd::view< double, 2 > xi) {
    directions(xi); // all the nodes in these elements are interior
  }

  __host__ __device__ void indices(const GeometryInfo & offsets, const Connection * edge, uint32_t * ids) {

    uint32_t edge_id = edge[Edge::cell_offset].index;

    if (p == 1) {
      ids[0] = offsets.edge + edge_id;
      return;
    }

    if (p == 2) {
      ids[0] = offsets.edge + 2 * edge_id + 0;
      ids[1] = offsets.edge + 2 * edge_id + 1;
    }

    if (p == 3) {
      ids[0] = offsets.edge + 3 * edge_id + 0;
      ids[1] = offsets.edge + 3 * edge_id + 1;
      ids[2] = offsets.edge + 3 * edge_id + 2;
      if (flip(edge[Edge::cell_offset])) { fm::swap(ids[0], ids[2]); }
      return;
    }
    
  }

  template < typename T >
  __host__ __device__ void reorient(const TransformationType type, const Connection * edge, T * values) {

    // TODO
    switch (type) {
      case TransformationType::PhysicalToParent:
      case TransformationType::TransposePhysicalToParent:
        break;
    }

  }

  __host__ __device__ void reorient(const TransformationType type, const Connection * edge, int8_t * transformation) {

    for (int i = 0; i < p; i++) {
      transformation[i] = 0;
    }

  }

  constexpr vec<1> shape_function(vec<1> xi, uint32_t i) const {
    if (p == 1 && i == 0) { return GaussLegendreInterpolation01<1, 0>(xi[0]); }

    if (p == 2 && i == 0) { return GaussLegendreInterpolation01<2, 0>(xi[0]); }
    if (p == 2 && i == 1) { return GaussLegendreInterpolation01<2, 1>(xi[0]); }

    if (p == 3 && i == 0) { return GaussLegendreInterpolation01<3, 0>(xi[0]); }
    if (p == 3 && i == 1) { return GaussLegendreInterpolation01<3, 1>(xi[0]); }
    if (p == 3 && i == 2) { return GaussLegendreInterpolation01<3, 2>(xi[0]); }

    return 1000.0f;
  }

  constexpr vec1 reoriented_shape_function(vec1 xi, uint32_t i, int8_t transformation) const {
    if (transformation == -1) {
      return -shape_function(xi, i);
    } else {
      return  shape_function(xi, i);
    }
  }

  // the other Nedelec elements define this function,
  // but what is the appropriate interpretation of "curl" (if any) for 1D? 
  //
  // the implementation below is just the derivative
  constexpr vec<1> shape_function_curl(vec<1> xi, uint32_t i) const {
    if (p == 1 && i == 0) { return GaussLegendreInterpolationDerivative01<1, 0>(xi); }

    if (p == 2 && i == 0) { return GaussLegendreInterpolationDerivative01<2, 0>(xi); }
    if (p == 2 && i == 1) { return GaussLegendreInterpolationDerivative01<2, 1>(xi); }

    if (p == 3 && i == 0) { return GaussLegendreInterpolationDerivative01<3, 0>(xi); }
    if (p == 3 && i == 1) { return GaussLegendreInterpolationDerivative01<3, 1>(xi); }
    if (p == 3 && i == 2) { return GaussLegendreInterpolationDerivative01<3, 2>(xi); }

    return 1000.0f;
  }

  constexpr vec1 reoriented_shape_function_curl(vec1 xi, uint32_t i, int8_t transformation) const {
    if (transformation == -1) {
      return -shape_function_curl(xi, i);
    } else {
      return  shape_function_curl(xi, i);
    }
  }

  constexpr vec<1> shape_function_derivative(vec<1> xi, uint32_t i) {
    return shape_function_curl(xi, i);
  }

  __host__ __device__ uint32_t batch_interpolation_scratch_space(nd::view<const double,2> xi) const {
    return 0;
  }

  nd::array< double, 2 > evaluate_shape_functions(nd::view<const double, 2> xi) {
    uint32_t nnodes = num_nodes();
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fns({q, num_nodes()});
    for (uint32_t i = 0; i < q; i++) {
      vec1 xi_i = xi(i, 0);
      for (uint32_t j = 0; j < nnodes; j++) {
        shape_fns(i, j) = shape_function(xi_i, j);
      }
    }
    return shape_fns;
  }

  void interpolate(nd::view<source_type> values_q, nd::view<const double> values_e, nd::view<const double, 2> shape_fns, double * /*buffer*/) {
    int nnodes = num_nodes();
    int nqpts = values_q.shape[0];

    for (int q = 0; q < nqpts; q++) {
      double sum = 0.0;
      for (int i = 0; i < nnodes; i++) {
        sum += shape_fns(q, i) * values_e(i);
      }
      values_q(q) = sum;
    }
  }

  nd::array< double > evaluate_shape_function_curls(nd::view<const double,2> xi) {
    uint32_t q = xi.shape[0];
    nd::array<double> buffer({q * p});
    for (int i = 0; i < q; i++) {
      GaussLegendreInterpolation(xi(i, 0), p, &buffer(p * i));
    }
    return buffer;
  }

  void curl(nd::view<flux_type> values_q, nd::view<const double> values_e, nd::view<double> buffer, double * /*buffer*/) {
  #if 0
    uint32_t n = p + 1;
    uint32_t q = sqrt(values_q.shape[0]);

    // 1D shape function evaluations
    nd::view<double, 2> B1(buffer.data(), {q, p}); // legendre shape functions
    nd::view<double, 2> B2(B1.end(),      {q, n}); // lobatto  shape functions
    nd::view<double, 2> G2(B2.end(),      {q, n}); // lobatto  shape function derivatives
    nd::view<double, 2> A1(G2.end(),      {q, n}); // storage for intermediates

    nd::view<const double, 2> ue(values_e.data(), {n, p});
    nd::view<double, 2> uq(values_q.data(), {q, q}, {2*q, 2u});
    
    _contract(A1, B1, ue); //  A1(qx, iy) = sum_{ix} B1(qx, ix) * ue(iy, ix)  
    _contract(uq, B2, A1); //  uq(qy, qx) = sum_{iy} B2(qy, iy) * A1(qx, iy)

    ue = nd::view< const double, 2 >(values_e.data() + (n * p), {n, p});

    // note: column-major strides here, since quadrature points 
    // are still enumerated lexicographically as {y, x} but y-component nodes are {x, y}
    uq = nd::view<double, 2>(values_q.data() + 1, {q, q}, {2u, 2*q}); 
    
    _contract(A1, B1, ue); //  A1(qx, iy) = sum_{ix} B1(qx, ix) * ue(iy, ix)  
    _contract(uq, B2, A1); //  uq(qy, qx) = sum_{iy} B2(qy, iy) * A1(qx, iy)
  #endif
  }

  nd::array< double, 2 > evaluate_weighted_shape_functions(nd::view<const double, 2> xi, nd::view<const double, 1> weights) {
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fns({q, num_nodes()});
    for (int i = 0; i < q; i++) {
      for (int j = 0; j < num_nodes(); j++) {
        shape_fns(i, j) = shape_function(xi(i, 0), j) * weights(i);
      }
    }
    return shape_fns;
  }

  __host__ __device__ void integrate_source(nd::view<double> residual_e, nd::view<const source_type> source_q, nd::view<const double, 2> shape_fn, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = source_q.shape[0];

    for (int i = 0; i < nnodes; i++) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn(q, i) * source_q(q);
      }
      residual_e(i) = sum;
    }
  }

  nd::array< double, 2 > evaluate_weighted_shape_function_curls(nd::view<const double, 2> xi, nd::view<const double, 1> weights) {
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fns({q, num_nodes()});
    for (int i = 0; i < q; i++) {
      for (int j = 0; j < num_nodes(); j++) {
        shape_fns(i, j) = shape_function_curl(xi(i, 0), j) * weights(i);
      }
    }
    return shape_fns;
  }

  __host__ __device__ void integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<const double, 2> shape_fn, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];

    for (int i = 0; i < nnodes; i++) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn(q, i) * flux_q(q);
      }
      residual_e(i) = sum;
    }
  }

  #ifdef __CUDACC__
  __device__ void cuda_integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<const double, 2> shape_fn_curl, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];

    for (int i = threadIdx.x; i < nnodes; i += blockDim.x) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn_curl(q, i) * flux_q(q);
      }
      residual_e(i) = sum;
    }
  }
  #endif

  uint32_t p;

};

} // namespace refactor
