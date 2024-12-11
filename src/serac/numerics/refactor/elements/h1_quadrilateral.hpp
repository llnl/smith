#pragma once

#include "refactor/connection.hpp" 
#include "refactor/tensor_contractions.hpp"

namespace refactor {

template <>
struct FiniteElement<Geometry::Quadrilateral, Family::H1> {

  using source_type = vec1;
  using flux_type = vec2;

  using value_type = vec1;
  using grad_type = vec2;

  __host__ __device__ constexpr uint32_t num_nodes() const { return (p + 1) * (p + 1); }

  void nodes(nd::view< double, 2 > xi) const {
    double lobatto_nodes[4];
    GaussLobattoNodes(p + 1, lobatto_nodes);

    uint32_t count = 0;
    for (uint32_t i = 0; i < (p+1); i++) {
      for (uint32_t j = 0; j < (p+1); j++) {
        xi(count, 0) = lobatto_nodes[j];
        xi(count, 1) = lobatto_nodes[i];
        count++;
      }
    }
  }

  __host__ __device__ uint32_t num_interior_nodes() const { return (p < 2) ? 0 : (p - 1) * (p - 1); }

  void interior_nodes(nd::view< double, 2 > xi) const {
    if (p > 1) {
      double lobatto_nodes[4];
      GaussLobattoNodes(p + 1, lobatto_nodes);

      uint32_t count = 0;
      for (uint32_t i = 1; i < p; i++) {
        for (uint32_t j = 1; j < p; j++) {
          xi(count, 0) = lobatto_nodes[j];
          xi(count, 1) = lobatto_nodes[i];
          count++;
        }
      }
    }
  }

  __host__ __device__ void indices(const GeometryInfo & offsets, const Connection * quad, uint32_t * indices) const {
 
    const Connection * vertex = quad;
    const Connection * edge = quad + 4;
    const Connection cell = *(quad + 8);

    if (p == 1) {
      //  2 3
      //  0 1
      indices[0] = offsets.vert + vertex[0].index;
      indices[1] = offsets.vert + vertex[1].index;
      indices[2] = offsets.vert + vertex[3].index;
      indices[3] = offsets.vert + vertex[2].index;
      return;
    }

    if (p == 2) {
      //  6 7 8
      //  3 4 5
      //  0 1 2
      indices[0] = offsets.vert + quad[0].index;
      indices[1] = offsets.edge + quad[4].index;
      indices[2] = offsets.vert + quad[1].index;
      indices[3] = offsets.edge + quad[7].index;
      indices[4] = offsets.quad + quad[8].index;
      indices[5] = offsets.edge + quad[5].index;
      indices[6] = offsets.vert + quad[3].index;
      indices[7] = offsets.edge + quad[6].index;
      indices[8] = offsets.vert + quad[2].index;
    }

    if (p == 3) {
      //  12 13 14 15
      //   8  9 10 11
      //   4  5  6  7
      //   0  1  2  3
      indices[ 0] = offsets.vert +     vertex[0].index;
      indices[ 1] = offsets.edge + 2 *   edge[0].index + 0;
      indices[ 2] = offsets.edge + 2 *   edge[0].index + 1;
      indices[ 3] = offsets.vert +     vertex[1].index;
      indices[ 4] = offsets.edge + 2 *   edge[3].index + 1;
      indices[ 5] = offsets.quad + 4 *      cell.index + 0;
      indices[ 6] = offsets.quad + 4 *      cell.index + 1;
      indices[ 7] = offsets.edge + 2 *   edge[1].index + 0;
      indices[ 8] = offsets.edge + 2 *   edge[3].index + 0;
      indices[ 9] = offsets.quad + 4 *   cell   .index + 2;
      indices[10] = offsets.quad + 4 *   cell   .index + 3;
      indices[11] = offsets.edge + 2 *   edge[1].index + 1;
      indices[12] = offsets.vert +     vertex[3].index;
      indices[13] = offsets.edge + 2 *   edge[2].index + 1;
      indices[14] = offsets.edge + 2 *   edge[2].index + 0;
      indices[15] = offsets.vert +     vertex[2].index;

      if (flip(edge[0])) { fm::swap(indices[ 1], indices[ 2]); }
      if (flip(edge[1])) { fm::swap(indices[ 7], indices[11]); }
      if (flip(edge[2])) { fm::swap(indices[14], indices[13]); }
      if (flip(edge[3])) { fm::swap(indices[ 8], indices[ 4]); }
      return;
    }   

  }

  constexpr double phi_1D(double xi, uint32_t i) const {
    return GaussLobattoInterpolation(xi, p + 1, i); 
  }

  constexpr double dphi_1D(double xi, uint32_t i) const {
    return GaussLobattoInterpolationDerivative(xi, p + 1, i); 
  }

  constexpr double shape_function(vec2 xi, uint32_t i) const {
    uint32_t ix = i % (p + 1);
    uint32_t iy = i / (p + 1);
    return phi_1D(xi[0], ix) * phi_1D(xi[1], iy);
  }

  vec2 shape_function_gradient(vec2 xi, uint32_t i) const {
    uint32_t ix = i % (p + 1);
    uint32_t iy = i / (p + 1);
    return {
      dphi_1D(xi[0], ix) *  phi_1D(xi[1], iy),
       phi_1D(xi[0], ix) * dphi_1D(xi[1], iy)
    };
  }

  vec2 shape_function_derivative(vec2 xi, uint32_t i) const {
    return shape_function_gradient(xi, i);
  }

  double interpolate(vec2 xi, double * values) const {
    double phi[4];
    GaussLobattoInterpolation(xi[0], p+1, phi); 

    double interpolated_in_xi[4]{};
    for (int i = 0; i < (p+1); i++) {
      for (int j = 0; j < (p+1); j++) {
        interpolated_in_xi[i] += phi[j] * values[i*(p+1)+j];
      }
    }

    GaussLobattoInterpolation(xi[1], p+1, phi); 
    double result{};
    for (int i = 0; i < (p+1); i++) {
      result += phi[i] * interpolated_in_xi[i];
    }
    return result;
  }

  vec2 gradient(vec2 xi, double * values) const {
    vec2 output{};

    double phi[4];
    double partially_interpolated[4]{};

    GaussLobattoInterpolationDerivative(xi[0], p+1, phi); 
    for (int i = 0; i < (p+1); i++) {
      for (int j = 0; j < (p+1); j++) {
        partially_interpolated[i] += phi[j] * values[i*(p+1)+j];
      }
    }

    GaussLobattoInterpolation(xi[1], p+1, phi); 
    for (int i = 0; i < (p+1); i++) {
      output[0] += phi[i] * partially_interpolated[i];
    }

    //--------------------------------------------------

    GaussLobattoInterpolation(xi[0], p+1, phi); 
    for (int i = 0; i < (p+1); i++) {
      for (int j = 0; j < (p+1); j++) {
        partially_interpolated[i] += phi[j] * values[i*(p+1)+j];
      }
    }

    GaussLobattoInterpolationDerivative(xi[1], p+1, phi); 
    for (int i = 0; i < (p+1); i++) {
      output[1] += phi[i] * partially_interpolated[i];
    }

    return output;
  }

  __host__ __device__ uint32_t batch_interpolation_scratch_space(nd::view<const double,2> xi) const {
    uint32_t q = xi.shape[0];
    return q * (p + 1);
  }

  // note: the return value here includes the shape function evaluations as well as
  // a buffer space to hold intermediate values in the `interpolation` function
  nd::array< double, 2 > evaluate_shape_functions(nd::view<const double, 2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fns({q, p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p + 1, &shape_fns(i, 0));
    }
    return shape_fns;
  }

  void interpolate(nd::view< value_type > values_q, nd::view<const double, 1> values_e, nd::view<double, 2> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[0];

    nd::view<double, 2> A(buffer, {n, q});

    nd::view<const double, 2> values_e_2D = nd::reshape<2>(values_e, {n, n});
    nd::view<double, 2> values_q_2D((double*)&values_q[0], {q, q}, {1, q});
    contract<1>(values_e_2D, shape_fns, A);
    contract<0>(A, shape_fns, values_q_2D);
  }

  // note: the return value here includes the shape function evaluations as well as
  // a buffer space to hold intermediate values in the `interpolation` function
  nd::array< double, 3 > evaluate_shape_function_gradients(nd::view<const double,2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 3> shape_fn_grads({2, q, p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i,0), p+1, &shape_fn_grads(0, i, 0));
      GaussLobattoInterpolationDerivative(xi(i,0), p+1, &shape_fn_grads(1, i, 0));
    }
    return shape_fn_grads;
  }

  void gradient(nd::view<grad_type> gradients_q, nd::view<const double, 1> values_e, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[1];

    nd::view<double, 2> B = shape_fns(0);
    nd::view<double, 2> G = shape_fns(1);
    nd::view<double, 2> A(buffer, {n, q});

    nd::range All{0u, gradients_q.shape[0]};
    nd::view<const double, 2> values_e_2D = nd::reshape<2>(values_e, {n, n});
    nd::view<double, 2> gradients_q_2D((double*)&gradients_q[0], {q, q}, {2, 2*q});

    // du_dxi
    contract<1>(values_e_2D, G, A);
    contract<0>(          A, B, gradients_q_2D);
    gradients_q_2D.values++;

    // du_deta
    contract<1>(values_e_2D, B, A);
    contract<0>(          A, G, gradients_q_2D);
    gradients_q_2D.values++;
  }

  // note: the return value here includes the shape function evaluations as well as
  // a buffer space to hold intermediate values in the `interpolation` function
  nd::array< double, 2 > evaluate_weighted_shape_functions(nd::view<const double,2> xi,
                                                           nd::view<const double,1> weights) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 2> buffer({p+1, q});
    nd::array<double, 1> tmp({p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p+1, &tmp(0));
      for (int j = 0; j < p+1; j++) {
        buffer(j, i) = tmp(j) * weights(i);
      }
    }
    return buffer;
  }

  __host__ __device__ void integrate_source(nd::view<double, 1> residual_e, nd::view<const source_type> source_q, nd::view<double, 2> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[1];

    nd::view<double, 2> BT = shape_fns;
    nd::view<double, 2> A(buffer, {q, n});

    nd::view<const double, 2> s2D((double*)&source_q[0], {q, q}, {1, q});
    nd::view<double, 2> r2D = nd::reshape<2>(residual_e, {n, n});

    contract<1>(s2D, BT, A);
    contract<0>(A, BT, r2D);
  }

  // note: the return value here includes the shape function evaluations as well as
  // a buffer space to hold intermediate values in the `interpolation` function
  nd::array< double, 3 > evaluate_weighted_shape_function_gradients(nd::view<const double,2> xi,
                                                                  nd::view<const double,1> weights) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 3> buffer({2, p+1, q});
    nd::array<double, 1> tmp({p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p+1, &tmp(0));
      for (int j = 0; j < p+1; j++) {
        buffer(0, j, i) = tmp(j) * weights(i);
      }

      GaussLobattoInterpolationDerivative(xi(i, 0), p+1, &tmp(0));
      for (int j = 0; j < p+1; j++) {
        buffer(1, j, i) = tmp(j) * weights(i);
      }
    }
    return buffer;
  }

  void integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[2];

    nd::view<double, 2> BT = shape_fns(0);
    nd::view<double, 2> GT = shape_fns(1);
    nd::view<double, 2> A(buffer, {q, n});

    uint32_t s = 2 * flux_q.stride[0];
    nd::range All{0u, flux_q.shape[0]};
    nd::view<const double, 2> f2D(&flux_q(0)[0], {q, q}, {s, s*q}); // ?
    nd::view<double, 2> r2D = nd::reshape<2>(residual_e, {n, n});

    contract<1>(f2D, BT, A);
    contract<0>(  A, GT, r2D);

    f2D.values = &(flux_q(0)[1]);
    contract<1>(f2D, GT, A);
    contract<0>(  A, BT, r2D, true /* accumulate */);
  }

  #ifdef __CUDACC__
  __device__ void cuda_gradient(nd::view<grad_type> gradients_q, nd::view<double, 1> values_e, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[1];

    nd::view<double, 2> B = shape_fns(0);
    nd::view<double, 2> G = shape_fns(1);
    nd::view<double, 2> A(buffer, {n, q});

    nd::range All{0u, gradients_q.shape[0]};
    nd::view<double, 2> values_e_2D = nd::reshape<2>(values_e, {n, n});
    nd::view<double, 2> gradients_q_2D((double*)&gradients_q[0], {q, q}, {2, 2*q});

    threadid tid; 
    tid.x = threadIdx.x % q;
    tid.y = threadIdx.x / q;
    tid.stride = q;

    // du_dxi
    cuda_contract<1>(tid, values_e_2D, G, A);
    cuda_contract<0>(tid,           A, B, gradients_q_2D);
    gradients_q_2D.values++;

    // du_deta
    cuda_contract<1>(tid, values_e_2D, B, A);
    cuda_contract<0>(tid,           A, G, gradients_q_2D);
    gradients_q_2D.values++;
  }

  __device__ void cuda_integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[2];

    nd::view<double, 2> BT = shape_fns(0);
    nd::view<double, 2> GT = shape_fns(1);
    nd::view<double, 2> A(buffer, {q, n});

    uint32_t s = 2 * flux_q.stride[0];
    nd::range All{0u, flux_q.shape[0]};
    nd::view<double, 2> f2D(&flux_q(0)[0], {q, q}, {s, s*q}); // ?
    nd::view<double, 2> r2D = nd::reshape<2>(residual_e, {n, n});

    threadid tid; 
    tid.x = threadIdx.x % n;
    tid.y = threadIdx.x / n;
    tid.stride = n;

    cuda_contract<1>(tid, f2D, BT, A);
    cuda_contract<0>(tid,   A, GT, r2D);

    f2D.values = &(flux_q(0)[1]);
    cuda_contract<1>(tid, f2D, GT, A);
    cuda_contract<0>(tid,   A, BT, r2D, true /* accumulate */);
  }
  #endif



  uint32_t p;

};

} // namespace refactor
