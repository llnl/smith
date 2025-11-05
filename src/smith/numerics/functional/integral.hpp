// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <array>
#include <memory>

#include "mfem.hpp"

#include "smith/infrastructure/accelerator.hpp"
#include "smith/numerics/functional/geometric_factors.hpp"
#include "smith/numerics/functional/function_signature.hpp"
#include "smith/numerics/functional/domain_integral_kernels.hpp"
#include "smith/numerics/functional/boundary_integral_kernels.hpp"
#include "smith/numerics/functional/interior_face_integral_kernels.hpp"
#include "smith/numerics/functional/differentiate_wrt.hpp"

namespace smith {

/// @brief a class for representing a Integral calculations and their derivatives
struct Integral {
  /// @brief the number of different kinds of integration domains
  static constexpr std::size_t num_types = 3;

  /**
   * @brief Construct an "empty" Integral object, whose kernels are to be initialized later
   * in one of the Make****Integral free functions below.
   *
   * @note It is not intended that users construct these objects manually
   *
   * @param d the domain of integration
   * @param trial_space_indices a list of which trial spaces are used in the integrand
   */
  Integral(const Domain& d, std::vector<uint32_t> trial_space_indices)
      : domain_(d), active_trial_spaces_(trial_space_indices)
  {
    std::size_t num_trial_spaces = trial_space_indices.size();
    evaluation_with_AD_.resize(num_trial_spaces);
    jvp_.resize(num_trial_spaces);
    element_gradient_.resize(num_trial_spaces);

    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      functional_to_integral_index_[active_trial_spaces_[i]] = i;
    }
  }

  /**
   * @brief evaluate the integral, optionally storing q-function derivatives with respect to
   *        a specific trial space.
   *
   * @param t the time
   * @param input_E a collection (one for each trial space) of block vectors (block index corresponds to the element
   * geometry) containing input values for each element.
   * @param output_E a block vector (block index corresponds to the element geometry) of the output values for each
   * element.
   * @param differentiation_index a non-negative value indicates differentiation with respect to the trial space with
   * that index. A value of -1 indicates no differentiation will occur.
   * @param update_state whether or not to store the updated state values computed in the q-function. For plasticity and
   * other path-dependent materials, this flag should only be set to `true` once a solution to the nonlinear system has
   * been found.
   */
  void Mult(double t, const std::vector<mfem::BlockVector>& input_E, mfem::BlockVector& output_E,
            uint32_t differentiation_index, bool update_state) const
  {
    output_E = 0.0;

    bool with_AD =
        (functional_to_integral_index_.count(differentiation_index) > 0 && differentiation_index != NO_DIFFERENTIATION);
    auto& kernels =
        (with_AD) ? evaluation_with_AD_[functional_to_integral_index_.at(differentiation_index)] : evaluation_;
    for (auto& [geometry, func] : kernels) {
      std::vector<const double*> inputs(active_trial_spaces_.size());
      for (std::size_t i = 0; i < active_trial_spaces_.size(); i++) {
        inputs[i] = input_E[uint32_t(active_trial_spaces_[i])].GetBlock(geometry).Read();
      }
      func(t, inputs, output_E.GetBlock(geometry).ReadWrite(), update_state);
    }
  }

  /**
   * @brief evaluate the jacobian(with respect to some trial space)-vector product of this integral
   *
   * @param dinput_E a block vector (block index corresponds to the element geometry) of a specific trial space element
   * values
   * @param doutput_E a block vector (block index corresponds to the element geometry) of the output values for each
   * element.
   * @param differentiation_index a non-negative value indicates directional derivative with respect to the trial space
   * with that index.
   */
  void GradientMult(const mfem::BlockVector& dinput_E, mfem::BlockVector& doutput_E,
                    uint32_t differentiation_index) const
  {
    doutput_E = 0.0;

    // if this integral actually depends on the specified variable
    if (functional_to_integral_index_.count(differentiation_index) > 0) {
      for (auto& [geometry, func] : jvp_[functional_to_integral_index_.at(differentiation_index)]) {
        func(dinput_E.GetBlock(geometry).Read(), doutput_E.GetBlock(geometry).ReadWrite());
      }
    }
  }

  /**
   * @brief evaluate the jacobian (with respect to some trial space) of this integral
   *
   * @param K_e a collection (one for each element type) of element jacobians (num_elements x trial_dofs_per_elem x
   * test_dofs_per_elem)
   * @param differentiation_index the index of the trial space being differentiated
   */
  void ComputeElementGradients(std::map<mfem::Geometry::Type, ExecArray<double, 3, ExecutionSpace::CPU> >& K_e,
                               uint32_t differentiation_index) const
  {
    // if this integral actually depends on the specified variable
    if (functional_to_integral_index_.count(differentiation_index) > 0) {
      for (auto& [geometry, func] : element_gradient_[functional_to_integral_index_.at(differentiation_index)]) {
        func(view(K_e[geometry]));
      }
    }
  }

  /**
   * @brief returns whether or not this integral depends on argument `which`
   *
   * @param which an argument index
   * @return true when if this Integral object was created with a DependsOn<...> statement that includes `i`
   */
  bool DependsOn(uint32_t which) const
  {
    for (uint32_t i : active_trial_spaces_) {
      if (which == i) return true;
    }
    return false;
  }

  /// @brief information about which elements to integrate over
  Domain domain_;

  /// @brief signature of integral evaluation kernel
  using eval_func = std::function<void(double, const std::vector<const double*>&, double*, bool)>;

  /// @brief kernels for integral evaluation over each type of element
  std::map<mfem::Geometry::Type, eval_func> evaluation_;

  /// @brief kernels for integral evaluation + derivative w.r.t. specified argument over each type of element
  std::vector<std::map<mfem::Geometry::Type, eval_func> > evaluation_with_AD_;

  /// @brief signature of element jvp kernel
  using jacobian_vector_product_func = std::function<void(const double*, double*)>;

  /// @brief kernels for jacobian-vector product of integral calculation
  std::vector<std::map<mfem::Geometry::Type, jacobian_vector_product_func> > jvp_;

  /// @brief signature of element gradient kernel
  using grad_func = std::function<void(ExecArrayView<double, 3, ExecutionSpace::CPU>)>;

  /// @brief kernels for calculation of element jacobians
  std::vector<std::map<mfem::Geometry::Type, grad_func> > element_gradient_;

  /// @brief a list of the trial spaces that take part in this integrand
  std::vector<uint32_t> active_trial_spaces_;

  /**
   * @brief a way of translating between the indices used by `Functional` and `Integral` to refer to the same
   *        trial space.
   *
   * e.g. A `Functional` may have 4 trial spaces {A, B, C, D}, but an `Integral` may only
   *        depend on a subset, say {B, C}. From `Functional`'s perspective, trial spaces {B, C} have (zero-based)
   *        indices of {1, 2}, but from `Integral`'s perspective, those are trial spaces {0, 1}.
   *
   * So, in this example functional_to_integral_index_ would have the values:
   * @code{.cpp}
   * std::map<int,int> functional_to_integral = {{1, 0}, {2, 1}};
   * @endcode
   *
   */
  std::map<uint32_t, uint32_t> functional_to_integral_index_;

  /// @brief the spatial positions and jacobians (dx_dxi) for each element type and quadrature point
  std::map<mfem::Geometry::Type, GeometricFactors> geometric_factors_;
};

/**
 * @brief function to generate kernels held by an `Integral` object of type "Domain", with a specific element type
 *
 * @tparam geom the element geometry
 * @tparam Q a parameter that controls the number of quadrature points
 * @tparam test the kind of test functions used in the integral
 * @tparam trials the trial space(s) of the integral's inputs
 * @tparam lambda_type a callable object that implements the q-function concept
 * @tparam qpt_data_type any quadrature point data needed by the material model
 * @param s an object used to pass around test/trial information
 * @param integral the Integral object to initialize
 * @param qf the quadrature function
 * @param qdata the values of any quadrature point data for the material
 */
template <mfem::Geometry::Type geom, int Q, typename test, typename... trials, typename lambda_type,
          typename qpt_data_type>
void generate_kernels(FunctionSignature<test(trials...)> s, Integral& integral, const lambda_type& qf,
                      std::shared_ptr<QuadratureData<qpt_data_type> > qdata)
{
  integral.geometric_factors_[geom] = GeometricFactors(integral.domain_, Q, geom);
  GeometricFactors& gf = integral.geometric_factors_[geom];
  if (gf.num_elements == 0) return;

  const double* positions = gf.X.Read();
  const double* jacobians = gf.J.Read();
  const uint32_t num_elements = uint32_t(gf.num_elements);
  const uint32_t qpts_per_element = num_quadrature_points(geom, Q);

  std::shared_ptr<zero> dummy_derivatives;
  integral.evaluation_[geom] = domain_integral::evaluation_kernel<NO_DIFFERENTIATION, Q, geom>(
      s, qf, positions, jacobians, qdata, dummy_derivatives, num_elements);

  constexpr std::size_t num_args = s.num_args;
  [[maybe_unused]] static constexpr int dim = dimension_of(geom);
  for_constexpr<num_args>([&](auto index) {
    // allocate memory for the derivatives of the q-function at each quadrature point
    //
    // Note: ptrs' lifetime is managed in an unusual way! It is captured by-value in the
    // action_of_gradient functor below to augment the reference count, and extend its lifetime to match
    // that of the DomainIntegral that allocated it.
    using derivative_type = decltype(domain_integral::get_derivative_type<index, dim, trials...>(qf, qpt_data_type{}));
    auto ptr = accelerator::make_shared_array<ExecutionSpace::CPU, derivative_type>(num_elements * qpts_per_element);

    integral.evaluation_with_AD_[index][geom] =
        domain_integral::evaluation_kernel<index, Q, geom>(s, qf, positions, jacobians, qdata, ptr, num_elements);

    integral.jvp_[index][geom] = domain_integral::jacobian_vector_product_kernel<index, Q, geom>(s, ptr, num_elements);
    integral.element_gradient_[index][geom] =
        domain_integral::element_gradient_kernel<index, Q, geom>(s, ptr, num_elements);
  });
}

/**
 * @brief function to generate kernels held by an `Integral` object of type "Domain", for all element types
 *
 * @tparam s a function signature type containing test/trial space informationa type containing a function signature
 * @tparam Q a parameter that controls the number of quadrature points
 * @tparam dim the dimension of the domain
 * @tparam lambda_type a callable object that implements the q-function concept
 * @tparam qpt_data_type any quadrature point data needed by the material model
 * @param domain the domain of integration
 * @param qf the quadrature function
 * @param qdata the values of any quadrature point data for the material
 * @param argument_indices the indices of trial space arguments used in the Integral
 * @return Integral the initialized `Integral` object
 */
template <typename s, int Q, int dim, typename lambda_type, typename qpt_data_type>
Integral MakeDomainIntegral(const Domain& domain, const lambda_type& qf,
                            std::shared_ptr<QuadratureData<qpt_data_type> > qdata,
                            std::vector<uint32_t> argument_indices)
{
  FunctionSignature<s> signature;

  SLIC_ERROR_IF(domain.type_ != Domain::Type::Elements, "Error: trying to evaluate a domain integral over a boundary");

  Integral integral(domain, argument_indices);

  if constexpr (dim == 2) {
    generate_kernels<mfem::Geometry::TRIANGLE, Q>(signature, integral, qf, qdata);
    generate_kernels<mfem::Geometry::SQUARE, Q>(signature, integral, qf, qdata);
  }

  if constexpr (dim == 3) {
    generate_kernels<mfem::Geometry::TETRAHEDRON, Q>(signature, integral, qf, qdata);
    generate_kernels<mfem::Geometry::CUBE, Q>(signature, integral, qf, qdata);
  }

  return integral;
}

/**
 * @brief function to generate kernels held by an `Integral` object of type "BoundaryDomain", with a specific element
 * type
 *
 * @tparam geom the element geometry
 * @tparam Q a parameter that controls the number of quadrature points
 * @tparam test the kind of test functions used in the integral
 * @tparam trials the trial space(s) of the integral's inputs
 * @tparam lambda_type a callable object that implements the q-function concept
 * @param s an object used to pass around test/trial information
 * @param integral the Integral object to initialize
 * @param qf the quadrature function
 */
template <mfem::Geometry::Type geom, int Q, typename test, typename... trials, typename lambda_type>
void generate_bdr_kernels(FunctionSignature<test(trials...)> s, Integral& integral, const lambda_type& qf)
{
  integral.geometric_factors_[geom] = GeometricFactors(integral.domain_, Q, geom);
  GeometricFactors& gf = integral.geometric_factors_[geom];
  if (gf.num_elements == 0) return;

  const double* positions = gf.X.Read();
  const double* jacobians = gf.J.Read();
  const uint32_t num_elements = uint32_t(gf.num_elements);
  const uint32_t qpts_per_element = num_quadrature_points(geom, Q);

  std::shared_ptr<zero> dummy_derivatives;
  integral.evaluation_[geom] = boundary_integral::evaluation_kernel<NO_DIFFERENTIATION, Q, geom>(
      s, qf, positions, jacobians, dummy_derivatives, num_elements);

  constexpr std::size_t num_args = s.num_args;
  [[maybe_unused]] static constexpr int dim = dimension_of(geom);
  for_constexpr<num_args>([&](auto index) {
    // allocate memory for the derivatives of the q-function at each quadrature point
    //
    // Note: ptrs' lifetime is managed in an unusual way! It is captured by-value in the
    // action_of_gradient functor below to augment the reference count, and extend its lifetime to match
    // that of the boundaryIntegral that allocated it.
    using derivative_type = decltype(boundary_integral::get_derivative_type<index, dim, trials...>(qf));
    auto ptr = accelerator::make_shared_array<ExecutionSpace::CPU, derivative_type>(num_elements * qpts_per_element);

    integral.evaluation_with_AD_[index][geom] =
        boundary_integral::evaluation_kernel<index, Q, geom>(s, qf, positions, jacobians, ptr, num_elements);

    integral.jvp_[index][geom] =
        boundary_integral::jacobian_vector_product_kernel<index, Q, geom>(s, ptr, num_elements);
    integral.element_gradient_[index][geom] =
        boundary_integral::element_gradient_kernel<index, Q, geom>(s, ptr, num_elements);
  });
}

/**
 * @brief function to generate kernels held by an `Integral` object of type "Boundary", for all element types
 *
 * @tparam s a function signature type containing test/trial space informationa type containing a function signature
 * @tparam Q a parameter that controls the number of quadrature points
 * @tparam dim the dimension of the domain
 * @tparam lambda_type a callable object that implements the q-function concept
 * @param domain the domain of integration
 * @param qf the quadrature function
 * @param argument_indices the indices of trial space arguments used in the Integral
 * @return Integral the initialized `Integral` object
 *
 * @note this function is not meant to be called by users
 */
template <typename s, int Q, int dim, typename lambda_type>
Integral MakeBoundaryIntegral(const Domain& domain, const lambda_type& qf, std::vector<uint32_t> argument_indices)
{
  FunctionSignature<s> signature;

  SLIC_ERROR_IF(domain.type_ != Domain::Type::BoundaryElements,
                "Error: trying to evaluate a boundary integral over a non-boundary domain of integration");

  Integral integral(domain, argument_indices);

  if constexpr (dim == 1) {
    generate_bdr_kernels<mfem::Geometry::SEGMENT, Q>(signature, integral, qf);
  }

  if constexpr (dim == 2) {
    generate_bdr_kernels<mfem::Geometry::TRIANGLE, Q>(signature, integral, qf);
    generate_bdr_kernels<mfem::Geometry::SQUARE, Q>(signature, integral, qf);
  }

  return integral;
}

/**
 * @brief function to generate kernels held by an `Integral` object of type "InteriorFaceDomain", with a specific
 * element type
 *
 * @tparam geom the element geometry
 * @tparam Q a parameter that controls the number of quadrature points
 * @tparam test the kind of test functions used in the integral
 * @tparam trials the trial space(s) of the integral's inputs
 * @tparam lambda_type a callable object that implements the q-function concept
 * @param s an object used to pass around test/trial information
 * @param integral the Integral object to initialize
 * @param qf the quadrature function
 */
template <mfem::Geometry::Type geom, int Q, typename test, typename... trials, typename lambda_type>
void generate_interior_face_kernels(FunctionSignature<test(trials...)> s, Integral& integral, const lambda_type& qf)
{
  integral.geometric_factors_[geom] = GeometricFactors(integral.domain_, Q, geom);
  GeometricFactors& gf = integral.geometric_factors_[geom];
  if (gf.num_elements == 0) return;

  const double* positions = gf.X.Read();
  const double* jacobians = gf.J.Read();
  const uint32_t num_elements = uint32_t(gf.num_elements);
  const uint32_t qpts_per_element = num_quadrature_points(geom, Q);

  std::shared_ptr<zero> dummy_derivatives;
  integral.evaluation_[geom] = interior_face_integral::evaluation_kernel<NO_DIFFERENTIATION, Q, geom>(
      s, qf, positions, jacobians, dummy_derivatives, num_elements);

  constexpr std::size_t num_args = s.num_args;
  [[maybe_unused]] static constexpr int dim = dimension_of(geom);
  for_constexpr<num_args>([&](auto index) {
    // allocate memory for the derivatives of the q-function at each quadrature point
    //
    // Note: ptrs' lifetime is managed in an unusual way! It is captured by-value in the
    // action_of_gradient functor below to augment the reference count, and extend its lifetime to match
    // that of the boundaryIntegral that allocated it.
    using derivative_type = decltype(interior_face_integral::get_derivative_type<index, dim, trials...>(qf));
    auto ptr = accelerator::make_shared_array<ExecutionSpace::CPU, derivative_type>(num_elements * qpts_per_element);

    integral.evaluation_with_AD_[index][geom] =
        interior_face_integral::evaluation_kernel<index, Q, geom>(s, qf, positions, jacobians, ptr, num_elements);

    integral.jvp_[index][geom] =
        interior_face_integral::jacobian_vector_product_kernel<index, Q, geom>(s, ptr, num_elements);
    integral.element_gradient_[index][geom] =
        interior_face_integral::element_gradient_kernel<index, Q, geom>(s, ptr, num_elements);
  });
}

/**
 * @brief function to generate kernels held by an `Integral` object of type "Boundary", for all element types
 *
 * @tparam s a function signature type containing test/trial space informationa type containing a function signature
 * @tparam Q a parameter that controls the number of quadrature points
 * @tparam dim the dimension of the domain
 * @tparam lambda_type a callable object that implements the q-function concept
 * @param domain the domain of integration
 * @param qf the quadrature function
 * @param argument_indices the indices of trial space arguments used in the Integral
 * @return Integral the initialized `Integral` object
 *
 * @note this function is not meant to be called by users
 */
template <typename s, int Q, int dim, typename lambda_type>
Integral MakeInteriorFaceIntegral(const Domain& domain, const lambda_type& qf, std::vector<uint32_t> argument_indices)
{
  FunctionSignature<s> signature;

  SLIC_ERROR_IF(domain.type_ != Domain::Type::InteriorFaces,
                "Error: trying to evaluate a boundary integral over a non-boundary domain of integration");

  Integral integral(domain, argument_indices);

  if constexpr (dim == 1) {
    generate_interior_face_kernels<mfem::Geometry::SEGMENT, Q>(signature, integral, qf);
  }

  if constexpr (dim == 2) {
    generate_interior_face_kernels<mfem::Geometry::TRIANGLE, Q>(signature, integral, qf);
    generate_interior_face_kernels<mfem::Geometry::SQUARE, Q>(signature, integral, qf);
  }

  return integral;
}

}  // namespace smith
