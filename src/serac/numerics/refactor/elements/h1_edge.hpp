#pragma once

#include "fm/macros.hpp"
#include "refactor/connection.hpp" 
#include "refactor/interpolation.hpp"

namespace refactor {

using namespace fm;

template <>
struct FiniteElement < Geometry::Edge, Family::H1 >{

  using source_type = vec1;
  using flux_type = vec1;

  using value_type = vec1;
  using grad_type = vec1;

  __host__ __device__ constexpr uint32_t num_nodes() const { return p + 1; }

  constexpr double shape_function(vec<1> xi, uint32_t i) const {
    if (p == 1 && i == 0) { return 1.0 - xi[0]; }
    if (p == 1 && i == 1) { return xi[0]; }

    if (p == 2 && i == 0) { return (-1.0 + xi[0]) * (-1.0 + 2.0 * xi[0]); }
    if (p == 2 && i == 1) { return -4.0 * (-1.0 + xi[0]) * xi[0]; }
    if (p == 2 && i == 2) { return xi[0] * (-1.0 + 2.0 * xi[0]); }

    constexpr double sqrt5 = 2.23606797749978981;
    if (p == 3 && i == 0) { return -(-1.0 + xi[0]) * (1.0 + 5.0 * (-1.0 + xi[0]) * xi[0]); }
    if (p == 3 && i == 1) { return -0.5 * sqrt5 * (5.0 + sqrt5 - 10.0 * xi[0]) * (-1.0 + xi[0]) * xi[0]; }
    if (p == 3 && i == 2) { return -0.5 * sqrt5 * (-1.0 + xi[0]) * xi[0] * (-5.0 + sqrt5 + 10.0 * xi[0]); }
    if (p == 3 && i == 3) { return xi[0] * (1.0 + 5.0 * (-1.0 + xi[0]) * xi[0]); }

    return 7777.77;
  }

  constexpr vec<1> shape_function_gradient(vec<1> xi, uint32_t i) const {
    if (p == 1 && i == 0) { return -1.0; }
    if (p == 1 && i == 1) { return +1.0; }

    if (p == 2 && i == 0) return  -3.0 + 4.0 * xi[0];
    if (p == 2 && i == 1) return  4.0 - 8.0 * xi[0];
    if (p == 2 && i == 2) return  -1.0 + 4.0 * xi[0];

    constexpr double sqrt5 = 2.23606797749978981;
    if (p == 3 && i == 0) return -6.0 + 5.0 * (4.0 - 3.0 * xi[0]) * xi[0];
    if (p == 3 && i == 1) return  2.5 * (1.0 + sqrt5 + 2.0 * xi[0] * (-1.0 - 3.0 * sqrt5 + 3.0 * sqrt5 * xi[0]));
    if (p == 3 && i == 2) return -2.5 * (-1.0 + sqrt5 + 2.0 * xi[0] * (1.0 - 3.0 * sqrt5 + 3.0 * sqrt5 * xi[0]));
    if (p == 3 && i == 3) return  1.0 + 5.0 * xi[0] * (-2.0 + 3.0 * xi[0]);

    return 7777.77;
  }

  vec<1> shape_function_derivative(vec<1> xi, uint32_t i) const {
    return shape_function_gradient(xi, i);
  }

  __host__ __device__ uint32_t batch_interpolation_scratch_space(nd::view<const double,2> xi) const {
    return 0;
  }

  nd::array< double, 2 > evaluate_shape_functions(nd::view<const double, 2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fns({q, p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p + 1, &shape_fns(i, 0));
    }
    return shape_fns;
  }

  nd::array< double, 2 > evaluate_shape_function_gradients(nd::view<const double, 2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fn_grads({q, num_nodes()});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolationDerivative(xi(i, 0), p + 1, &shape_fn_grads(i, 0));
    }
    return shape_fn_grads;
  }

  __host__ __device__ void interpolate(nd::view<value_type> values_q, nd::view<const double> values_e, nd::view<const double, 2> shape_fns, double * /*buffer*/) const {
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

  __host__ __device__ void gradient(nd::view<grad_type> gradients_q, nd::view<const double, 1> values_e, nd::view<const double, 2> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = gradients_q.shape[0];
    for (int j = 0; j < nqpts; j++) {
      double sum = 0.0;
      for (int i = 0; i < nnodes; i++) {
        sum += values_e(i) * shape_fn_grads(j, i);
      }
      gradients_q(j) = sum;
    }
  }

  nd::array< double, 2 > evaluate_weighted_shape_functions(nd::view<const double,2> xi,
                                                           nd::view<const double,1> weights) const {
    uint32_t nnodes = num_nodes();
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fns({q, nnodes});
    for (uint32_t i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p+1, &shape_fns(i, 0));
      for (uint32_t j = 0; j < nnodes; j++) {
        shape_fns(i, j) = shape_fns(i, j) * weights(i);
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

  nd::array< double, 2 > evaluate_weighted_shape_function_gradients(nd::view<const double, 2> xi,
                                                                    nd::view<const double, 1> weights) const {
    uint32_t nnodes = num_nodes();
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fn_grads({q, num_nodes()});
    for (uint32_t i = 0; i < q; i++) {
      GaussLobattoInterpolationDerivative(xi(i, 0), p + 1, &shape_fn_grads(i, 0));
      for (uint32_t j = 0; j < nnodes; j++) {
        shape_fn_grads(i, j) = shape_fn_grads(i, j) * weights(i);
      }
    }
    return shape_fn_grads;
  }

  __host__ __device__ void integrate_flux(nd::view<double> output_e, nd::view<const flux_type> flux_q, nd::view<const double, 2> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];
    for (int i = 0; i < nnodes; i++) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn_grads(q, i) * flux_q(q);
      }
      output_e(i) = sum;
    }
  }

  #ifdef __CUDACC__
  __device__ void cuda_gradient(nd::view<grad_type, 1> gradients_q, nd::view<const double, 1> values_e, nd::view<const double, 2> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = gradients_q.shape[0];
    for (int j = threadIdx.x; j < nqpts; j += blockDim.x) {
      double sum = 0.0;
      for (int i = 0; i < nnodes; i++) {
        sum += values_e(i) * shape_fn_grads(j, i);
      }
      gradients_q(j) = sum;
    }
  }

  __device__ void cuda_integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<const double, 2> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];

    for (int i = threadIdx.x; i < nnodes; i += blockDim.x) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn_grads(q, i) * flux_q(q);
      }
      residual_e(i) = sum;
    }
  }
  #endif

  uint32_t p;

};

} // namespace refactor
