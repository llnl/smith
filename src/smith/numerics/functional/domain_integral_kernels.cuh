// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include <cstring>

#include "smith/infrastructure/accelerator.hpp"
#include "smith/numerics/functional/quadrature_data.hpp"

namespace smith {

namespace detail {

/**
 * @brief Defines whether to loop by elements or quadrature points
 */
enum ThreadParallelizationStrategy
{
  THREAD_PER_QUADRATURE_POINT,
  THREAD_PER_ELEMENT
};

/**
 * @brief Contains the GPU launch configuration
 */
struct GPULaunchConfiguration {
  int blocksize;
};

}  // namespace detail

namespace domain_integral {

/**
 * @brief derivatives_ptr access
 *
 * Templating this will allow us to change the stride-access patterns more consistently
 * By default derivatives_ptr is accessed using row_major ordering derivatives_ptr(element, quadrature).
 *
 * @tparam derivatives_type The type of the derivatives
 * @tparam rule_type The type of the quadrature rule
 * @tparam row_major A boolean to choose to use row major access patterns or column major
 * @param[in] derivatives_ptr pointer to derivatives
 * @param[in] e element number
 * @param[in] q qaudrature number
 * @param[in] rule quadrature rule
 * @param[in] num_elements number of finite elements
 */
template <typename derivatives_type, typename rule_type, bool row_major = true>
SMITH_HOST_DEVICE constexpr derivatives_type& AccessDerivatives(derivatives_type* derivatives_ptr, int e, int q,
                                                                [[maybe_unused]] rule_type& rule,
                                                                [[maybe_unused]] int num_elements)
{
  if constexpr (row_major) {
    return derivatives_ptr[e * static_cast<int>(rule.size()) + q];
  } else {
    return derivatives_ptr[q * num_elements + e];
  }
}

/**
 * @brief The GPU kernel template used to create different finite element calculation routines.
 *
 * This GPU kernel proccess one element per thread.
 *
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function (see below) w.r.t. its input arguments
 * @tparam lambda The actual quadrature-function (either lambda function or functor object) to
 * be evaluated at each quadrature point.
 * @see https://libceed.readthedocs.io/en/latest/libCEEDapi/#theoretical-framework for additional
 * information on the idea behind a quadrature function and its inputs/outputs
 *
 * // I don't know why these template parameters are here
 * @tparam solution_type element solution
 * @tparam residual_type element residual
 * @tparam jacobian_type quadrature point jacobian
 * @tparam position_type quadrature point position
 *
 * @param[in] u The element DOF values (primary input)
 * @param[inout] r The element residuals (primary output)
 * @param[out] derivatives_ptr The address at which derivatives of @a lambda with
 * respect to its arguments will be stored
 * @param[in] J The Jacobians of the element transformation at all quadrature points
 * @param[in] X The actual (not reference) coordinates of all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 * @param[in] qf The actual quadrature function, see @p lambda
 * @param[inout] data The data for each quadrature point
 */

template <mfem::Geometry::Type g, typename test, typename trial, int Q, typename derivatives_type, typename lambda,
          typename solution_type, typename residual_type, typename jacobian_type, typename position_type,
          typename qpt_data_type = void>
__global__ void eval_cuda_element(const solution_type u, residual_type r, derivatives_type* derivatives_ptr,
                                  jacobian_type J, position_type X, int num_elements, lambda qf,
                                  QuadratureData<qpt_data_type> data)
{
  using test_element = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;
  using element_residual_type = typename trial_element::residual_type;
  static constexpr auto rule = GaussQuadratureRule<g, Q>();
  static constexpr int dim = dimension_of(g);

  // for each element in the domain
  const int grid_stride = blockDim.x * gridDim.x;
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_elements; e += grid_stride) {
    // get the DOF values for this particular element
    auto u_elem = detail::Load<trial_element>(u, e);

    // this is where we will accumulate the element residual tensor
    element_residual_type r_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      auto xi = rule.points[q];
      auto dxi = rule.weights[q];
      auto x_q = make_tensor<dim>([&](int i) { return X(q, i, e); });  // Physical coords of qpt
      auto J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx = det(J_q) * dxi;

      // evaluate the value/derivatives needed for the q-function at this quadrature point
      auto arg = Preprocess<trial_element>(u_elem, xi, J_q);

      // evaluate the user-specified constitutive model
      //
      // note: make_dual(arg) promotes those arguments to dual number types
      // so that qf_output will contain values and derivatives
      auto qf_output = detail::apply_qf(qf, x_q, make_dual(arg), data(e, q));

      // integrate qf_output against test space shape functions / gradients
      // to get element residual contributions
      r_elem += Postprocess<test_element>(get_value(qf_output), xi, J_q) * dx;

      // here, we store the derivative of the q-function w.r.t. its input arguments
      //
      // this will be used by other kernels to evaluate gradients / adjoints / directional derivatives

      // Note: This pattern may result in non-coalesced access depend on how it executed.
      detail::AccessDerivatives(derivatives_ptr, e, q, rule, num_elements) = get_gradient(qf_output);
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(r, r_elem, e);
  }  // element loop
}

/**
 * @brief The GPU kernel template used to create different finite element calculation routines.
 *
 * This GPU kernel proccess one quadrature point per thread.
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p DomainIntegral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function (see below) w.r.t. its input arguments
 * @tparam lambda The actual quadrature-function (either lambda function or functor object) to
 * be evaluated at each quadrature point.
 * @see https://libceed.readthedocs.io/en/latest/libCEEDapi/#theoretical-framework for additional
 * information on the idea behind a quadrature function and its inputs/outputs
 * @tparam qpt_data_type The type of the data to store for each quadrature point
 *
 * // I don't know why these template parameters are here
 * @tparam solution_type element solution
 * @tparam residual_type element residual
 * @tparam jacobian_type quadrature point jacobian
 * @tparam position_type quadrature point position
 *
 * @param[in] u The element DOF values (primary input)
 * @param[inout] r The element residuals (primary output)
 * @param[out] derivatives_ptr The address at which derivatives of @a lambda with
 * respect to its arguments will be stored
 * @param[in] J The Jacobians of the element transformation at all quadrature points
 * @param[in] X The actual (not reference) coordinates of all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 * @param[in] qf The actual quadrature function, see @p lambda
 * @param[inout] data The data for each quadrature point
 */
template <mfem::Geometry::Type g, typename test, typename trial, int Q, typename derivatives_type, typename lambda,
          typename solution_type, typename residual_type, typename jacobian_type, typename position_type,
          typename qpt_data_type = void>
__global__ void eval_cuda_quadrature(const solution_type u, residual_type r,
                                     GPUArrayView<derivatives_type, 2> qf_derivatives, jacobian_type J, position_type X,
                                     int num_elements, lambda qf, QuadratureData<qpt_data_type> data)
{
  using test_element = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;
  using element_residual_type = typename test_element::residual_type;
  static constexpr auto rule = GaussQuadratureRule<g, Q>();
  static constexpr int dim = dimension_of(g);

  const int grid_stride = blockDim.x * gridDim.x;
  // launch a thread for each quadrature x element point
  for (int qe = blockIdx.x * blockDim.x + threadIdx.x; qe < num_elements * rule.size(); qe += grid_stride) {
    // warps won't fetch that many elements ... not great.. but not horrible
    int e = qe / rule.size();
    int q = qe % rule.size();

    // get the DOF values for this particular element
    auto u_elem = detail::Load<trial_element>(u, e);

    // this is where we will accumulate the element residual tensor
    element_residual_type r_elem{};

    // for each quadrature point in the element
    auto xi = rule.points[q];
    auto dxi = rule.weights[q];
    auto x_q = make_tensor<dim>([&](int i) { return X(q, i, e); });  // Physical coords of qpt
    auto J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
    double dx = det(J_q) * dxi;

    // evaluate the value/derivatives needed for the q-function at this quadrature point
    auto arg = Preprocess<trial_element>(u_elem, xi, J_q);

    // evaluate the user-specified constitutive model
    //
    // note: make_dual(arg) promotes those arguments to dual number types
    // so that qf_output will contain values and derivatives
    auto qf_output = detail::apply_qf(qf, x_q, make_dual(arg), data(e, q));

    // integrate qf_output against test space shape functions / gradients
    // to get element residual contributions
    r_elem += Postprocess<test_element>(get_value(qf_output), xi, J_q) * dx;

    // here, we store the derivative of the q-function w.r.t. its input arguments
    //
    // this will be used by other kernels to evaluate gradients / adjoints / directional derivatives
    qf_derivatives(e, q) = get_gradient(qf_output);

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(r, r_elem, e);
  }
}

/**
 * @brief The GPU base template used to create different finite element calculation routines
 *
 * This function is used to select which GPU kernel implementation to invoke.
 *
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function (see below) w.r.t. its input arguments
 * @tparam lambda The actual quadrature-function (either lambda function or functor object) to
 * be evaluated at each quadrature point.
 * @see https://libceed.readthedocs.io/en/latest/libCEEDapi/#theoretical-framework for additional
 * information on the idea behind a quadrature function and its inputs/outputs
 * @tparam qpt_data_type The type of the data to store for each quadrature point
 *
 * @param[in] config Execution configuration for the GPU kernel
 * @param[in] U The full set of per-element DOF values (primary input)
 * @param[inout] R The full set of per-element residuals (primary output)
 * @param[out] derivatives_ptr The address at which derivatives of @a lambda with
 * respect to its arguments will be stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @param[in] X_ The actual (not reference) coordinates of all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 * @param[in] qf The actual quadrature function, see @p lambda
 * @param[inout] data The data for each quadrature point
 */

template <mfem::Geometry::Type g, typename test, typename trial, int Q,
          smith::detail::ThreadParallelizationStrategy policy, typename derivatives_type, typename lambda,
          typename qpt_data_type = void>
void evaluation_kernel_cuda(smith::detail::GPULaunchConfiguration config, const mfem::Vector& U, mfem::Vector& R,
                            GPUArrayView<derivatives_type, 2> qf_derivatives, const mfem::Vector& J_,
                            const mfem::Vector& X_, int num_elements, lambda qf,
                            std::shared_ptr<QuadratureData<qpt_data_type>> data = NoQData)
{
  using test_element = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;
  using element_residual_type = typename test_element::residual_type;
  static constexpr int test_ndof = test_element::ndof;
  static constexpr int trial_ndof = trial_element::ndof;
  static constexpr auto rule = GaussQuadratureRule<g, Q>();
  static constexpr int dim = dimension_of(g);

  // Note: Since we cannot call Reshape (__host__) within a kernel we pass in the resulting mfem::DeviceTensors which
  // should be pointing to Device pointers via .Read() and .ReadWrite()

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto X = mfem::Reshape(X_.Read(), rule.size(), dim, num_elements);
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto u = detail::Reshape<trial>(U.Read(), trial_ndof, num_elements);
  auto r = detail::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

  cudaDeviceSynchronize();
  smith::accelerator::displayLastCUDAMessage();

  if constexpr (policy == smith::detail::ThreadParallelizationStrategy::THREAD_PER_QUADRATURE_POINT) {
    int blocks_quadrature_element = (num_elements * rule.size() + config.blocksize - 1) / config.blocksize;
    eval_cuda_quadrature<g, test, trial, Q>
        <<<blocks_quadrature_element, config.blocksize>>>(u, r, qf_derivatives, J, X, num_elements, qf, *data);

  } else if constexpr (policy == smith::detail::ThreadParallelizationStrategy::THREAD_PER_ELEMENT) {
    int blocks_element = (num_elements + config.blocksize - 1) / config.blocksize;
    eval_cuda_element<g, test, trial, Q>
        <<<blocks_element, config.blocksize>>>(u, r, qf_derivatives, J, X, num_elements, qf, *data);
  }

  cudaDeviceSynchronize();
  smith::accelerator::displayLastCUDAMessage();
}

/**
 * @brief The GPU kernel template used to create create custom directional derivative
 * kernels associated with finite element calculations
 *
 * This kernel processes the gradient of one element per thread
 *
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function w.r.t. its input arguments
 *
 * @note lambda does not appear as a template argument, as the directional derivative is
 * inherently just a linear transformation
 *
 * @tparam dsolution_type element solution
 * @tparam dresidual_type element residual
 *
 * @param[in] dU The element DOF values (primary input)
 * @param[inout] dR The element residuals (primary output)
 * @param[in] derivatives_ptr The address at which derivatives of the q-function with
 * respect to its arguments are stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 */

template <mfem::Geometry::Type g, typename test, typename trial, int Q, typename derivatives_type,
          typename dsolution_type, typename dresidual_type>
__global__ void gradient_cuda_element(const dsolution_type du, dresidual_type dr,
                                      GPUArrayView<derivatives_type, 2> qf_derivatives,
                                      const mfem::DeviceTensor<4, const double> J, int num_elements)
{
  using test_element = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;
  using element_residual_type = typename trial_element::residual_type;
  static constexpr auto rule = GaussQuadratureRule<g, Q>();
  static constexpr int dim = dimension_of(g);

  const int grid_stride = blockDim.x * gridDim.x;
#pragma unroll
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_elements; e += grid_stride) {
    // get the (change in) values for this particular element
    tensor du_elem = detail::Load<trial_element>(du, e);

    // this is where we will accumulate the (change in) element residual tensor
    element_residual_type dr_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent and physical space,
      // and calculate the measure of that point in physical space.
      auto xi = rule.points[q];
      auto dxi = rule.weights[q];
      auto J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx = det(J_q) * dxi;

      // evaluate the (change in) value/derivatives at this quadrature point
      auto darg = Preprocess<trial_element>(du_elem, xi, J_q);

      // recall the derivative of the q-function w.r.t. its arguments at this quadrature point
      auto dq_darg = qf_derivatives(e, q);

      // use the chain rule to compute the first-order change in the q-function output
      auto dq = chain_rule(dq_darg, darg);

      // integrate dq against test space shape functions / gradients
      // to get the (change in) element residual contributions
      dr_elem += Postprocess<test_element>(dq, xi, J_q) * dx;
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(dr, dr_elem, e);
  }
}

/**
 * @brief The GPU kernel template used to create create custom directional derivative
 * kernels associated with finite element calculations
 *
 * This kernel processes the gradient of one quadrature point per thread
 *
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function w.r.t. its input arguments
 *
 * @note lambda does not appear as a template argument, as the directional derivative is
 * inherently just a linear transformation
 *
 * @tparam dsolution_type element solution
 * @tparam dresidual_type element residual
 *
 * @param[in] dU The element DOF values (primary input)
 * @param[inout] dR The element residuals (primary output)
 * @param[in] derivatives_ptr The address at which derivatives of the q-function with
 * respect to its arguments are stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 */

template <mfem::Geometry::Type g, typename test, typename trial, int Q, typename derivatives_type,
          typename dsolution_type, typename dresidual_type>
__global__ void gradient_cuda_quadrature(const dsolution_type du, dresidual_type dr,
                                         GPUArrayView<derivatives_type, 2> qf_derivatives,
                                         const mfem::DeviceTensor<4, const double> J, int num_elements)
{
  using test_element = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;
  using element_residual_type = typename trial_element::residual_type;
  static constexpr auto rule = GaussQuadratureRule<g, Q>();
  static constexpr int dim = dimension_of(g);

  const int grid_stride = blockDim.x * gridDim.x;
  auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  auto num_quadrature_points = num_elements * rule.size();
#pragma unroll
  for (int qe = thread_id; qe < num_quadrature_points; qe += grid_stride) {
    int e = qe / rule.size();
    int q = qe % rule.size();
    // get the (change in) values for this particular element
    tensor du_elem = detail::Load<trial_element>(du, e);

    // this is where we will accumulate the (change in) element residual tensor
    element_residual_type dr_elem{};

    // get the position of this quadrature point in the parent and physical space,
    // and calculate the measure of that point in physical space.
    auto xi = rule.points[q];
    auto dxi = rule.weights[q];
    auto J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
    double dx = det(J_q) * dxi;

    // evaluate the (change in) value/derivatives at this quadrature point
    auto darg = Preprocess<trial_element>(du_elem, xi, J_q);

    // recall the derivative of the q-function w.r.t. its arguments at this quadrature point
    auto dq_darg = qf_derivatives(e, q);

    // use the chain rule to compute the first-order change in the q-function output
    auto dq = chain_rule(dq_darg, darg);

    // integrate dq against test space shape functions / gradients
    // to get the (change in) element residual contributions
    dr_elem += Postprocess<test_element>(dq, xi, J_q) * dx;

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(dr, dr_elem, e);
  }
}

/**
 * @brief The base template used to create create custom directional derivative
 * kernels associated with finite element calculations
 *
 * This function is used to invoke the GPU kernels
 *
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function w.r.t. its input arguments
 *
 * @note lambda does not appear as a template argument, as the directional derivative is
 * inherently just a linear transformation
 *
 * @param[in] config Execution configuration for the GPU kernel
 * @param[in] dU The full set of per-element DOF values (primary input)
 * @param[inout] dR The full set of per-element residuals (primary output)
 * @param[in] derivatives_ptr The address at which derivatives of the q-function with
 * respect to its arguments are stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 */

template <mfem::Geometry::Type g, typename test, typename trial, int Q,
          smith::detail::ThreadParallelizationStrategy policy, typename derivatives_type>
void action_of_gradient_kernel(smith::detail::GPULaunchConfiguration config, const mfem::Vector& dU, mfem::Vector& dR,
                               GPUArrayView<derivatives_type, 2> qf_derivatives, const mfem::Vector& J_,
                               int num_elements)
{
  using test_element = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;
  using element_residual_type = typename trial_element::residual_type;
  static constexpr int test_ndof = test_element::ndof;
  static constexpr int trial_ndof = trial_element::ndof;
  static constexpr auto rule = GaussQuadratureRule<g, Q>();
  static constexpr int dim = dimension_of(g);

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto du = detail::Reshape<trial>(dU.Read(), trial_ndof, num_elements);
  auto dr = detail::Reshape<test>(dR.ReadWrite(), test_ndof, num_elements);

  cudaDeviceSynchronize();
  smith::accelerator::displayLastCUDAMessage();

  // call gradient_cuda
  if constexpr (policy == smith::detail::ThreadParallelizationStrategy::THREAD_PER_QUADRATURE_POINT) {
    int blocks_quadrature_element = (num_elements * rule.size() + config.blocksize - 1) / config.blocksize;
    gradient_cuda_quadrature<g, test, trial, Q, derivatives_type>
        <<<blocks_quadrature_element, config.blocksize>>>(du, dr, qf_derivatives, J, num_elements);

  } else if constexpr (policy == smith::detail::ThreadParallelizationStrategy::THREAD_PER_ELEMENT) {
    int blocks_element = (num_elements + config.blocksize - 1) / config.blocksize;
    gradient_cuda_element<g, test, trial, Q, derivatives_type>
        <<<blocks_element, config.blocksize>>>(du, dr, qf_derivatives, J, num_elements);
  }

  cudaDeviceSynchronize();
  smith::accelerator::displayLastCUDAMessage();
  dR.HostRead();
}

}  // namespace domain_integral

}  // namespace smith
