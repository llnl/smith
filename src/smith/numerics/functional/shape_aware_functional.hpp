// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file shape_aware_functional.hpp
 *
 * @brief Wrapper of smith::Functional for evaluating integrals and derivatives of quantities with shape displacement
 * fields
 */

#pragma once

#include "smith/infrastructure/accelerator.hpp"
#include "smith/numerics/functional/functional.hpp"

namespace smith {

namespace detail {

/**
 * @brief A helper struct that contains the appropriate parent-to-physical and physical-to-parent
 * transformations for an applied shape field
 *
 * @tparam dim The dimension of the input position
 * @tparam shape_type The type of the shape field (may be dual)
 */
template <int dim, typename shape_type>
struct ShapeCorrection {
 public:
  /**
   * @brief Construct a new Shape Correction object with the appropriate transformations for a shape field
   *
   * @param p The current shape displacement field at the underlying quadrature point
   */
  SMITH_HOST_DEVICE ShapeCorrection(Dimension<dim>, shape_type p)
      : J_(Identity<dim>() + get<DERIVATIVE>(p)), detJ_(det(J_)), inv_J_(inv(J_)), inv_JT_(transpose(inv_J_))
  {
  }

  /**
   * @brief Modify the trial argument using the correct physical to reference to shape-adjusted transformation for the
   * underlying trial function space
   *
   * @tparam trial_type The trial function type (may be dual)
   * @tparam space_type The trial function finite element space (e.g. H1, L2, Hcurl
   *
   * @param u The input trial function at a quadrature point
   *
   * @return The modified trial function adjusted for the underlying shape displacement field
   */
  template <typename trial_type, typename space_type>
  SMITH_HOST_DEVICE auto modify_trial_argument(space_type /* space */, const trial_type& u) const
  {
    if constexpr (space_type{}.family == Family::H1 || space_type{}.family == Family::L2) {
      // Our updated spatial coordinate is x' = x + p. We want to calculate
      // du/dx' = du/dx * dx/dx'
      //        = du/dx * (dx'/dx)^-1
      //        = du_dx * (J)^-1
      auto trial_derivative = dot(get<DERIVATIVE>(u), inv_J_);

      return smith::tuple{get<VALUE>(u), trial_derivative};
    }

    if constexpr (space_type{}.family == Family::HCURL) {
      auto modified_val = dot(get<VALUE>(u), inv_J_);
      if constexpr (dim == 2) {
        auto modified_derivative = get<DERIVATIVE>(u) / detJ_;
        return smith::tuple{modified_val, modified_derivative};
      }
      if constexpr (dim == 3) {
        auto modified_derivative = dot(get<DERIVATIVE>(u), transpose(J_));
        return smith::tuple{modified_val, modified_derivative};
      }
    }
  }

  /**
   * @brief Modify the quadrature function return value using the correct shape-adjusted to reference transformation for
   * the underlying test function space
   *
   * @tparam test_space The test function finite element space (e.g. H1, L2, double)
   * @tparam q_func_type The type of the unmodified qfunction return
   *
   * @param v The unmodified q function return in shape-adjusted coordinates
   *
   * @return The modified return after applying the appropriate shape-adjusted to reference transformation
   */
  template <typename test_space, typename q_func_type>
  SMITH_HOST_DEVICE auto modify_shape_aware_qf_return(test_space /*test*/, const q_func_type& v)
  {
    if constexpr (std::is_same_v<test_space, double>) {
      return detJ_ * v;
    } else {
      if constexpr (test_space{}.family == Family::H1 || test_space{}.family == Family::L2) {
        auto modified_flux = dot(get<FLUX>(v), inv_JT_) * detJ_;
        auto modified_source = get<SOURCE>(v) * detJ_;

        return smith::tuple{modified_source, modified_flux};
      }
      if constexpr (test_space{}.family == Family::HCURL) {
        auto modified_source = dot(get<SOURCE>(v), inv_JT_) * detJ_;
        if constexpr (dim == 3) {
          auto modified_flux = dot(get<FLUX>(v), J_);
          return smith::tuple{modified_source, modified_flux};
        } else {
          return smith::tuple{modified_source, get<FLUX>(v)};
        }
      }
    }
  }

 private:
  /// @cond
  using jacobian_type = std::remove_reference_t<decltype(get<DERIVATIVE>(std::declval<shape_type>()))>;
  using detJ_type = decltype(det(std::declval<jacobian_type>()));
  using inv_J_type = decltype(inv(std::declval<jacobian_type>()));
  using inv_JT_type = decltype(inv(transpose(std::declval<jacobian_type>())));
  /// @endcond

  /// @brief The Jacobian of the shape-adjusted transformation (x = X + p, J = dx/dX)
  jacobian_type J_;

  /// @brief The determinant of the Jacobian
  detJ_type detJ_;

  /// @brief Inverse of the Jacobian
  inv_J_type inv_J_;

  /// @brief Inverse transpose of the Jacobian
  inv_JT_type inv_JT_;
};

/**
 * @brief Compute the boundary area correction term for boundary integrals with a shape displacement field
 *
 * @tparam position_type The position input argument type
 * @tparam shape_type The shape displacement input argument type
 *
 * @param X The input position (value and gradient)
 * @param shape The input shape displacement (value and gradient)
 *
 * @return The area correction factor transforming the boundary integral from the reference domain area into the
 * shape-displaced area
 */
template <typename position_type, typename shape_type>
SMITH_HOST_DEVICE auto compute_boundary_area_correction(const position_type& X, const shape_type& shape)
{
  auto x_prime = X + shape;

  auto n = cross(get<DERIVATIVE>(x_prime));

  // smith::Functional's boundary integrals multiply the q-function output by
  // norm(cross(dX_dxi)) at that quadrature point, but if we impose a shape displacement
  // then that weight needs to be corrected. The new weight should be
  // norm(cross(dX_dxi + dp_dxi)), so we multiply by the ratio w_new / w_old
  // to get
  //   q * area_correction * w_old
  // = q * (w_new / w_old) * w_old
  // = q * w_new

  auto area_correction = norm(n) / norm(cross(get<DERIVATIVE>(X)));

  return area_correction;
}

/**
 * @brief A helper function to modify all of the trial function input derivatives according to the given shape
 * displacement for integrands without state variables
 *
 * @tparam lambda The q-function type
 * @tparam coord_type The input position type
 * @tparam shape_type The type of the shape displacement argument
 * @tparam space_types The type of the input finite element space tuple for the trial functions
 * @tparam trial_types The type of the input finite element argument tuple (values and derivatives)
 * @tparam correction_type The type of the shape correction struct
 * @tparam i Indices for accessing the individual arguments for the underlying q-function
 *
 * @param qf The q-function integrand with expects shape-adjusted arguments
 * @param t The time at which to evaluate the integrand
 * @param position The quadrature point spatial coordinates and isoparametric derivatives
 * @param shape The space displacement at which to evaluate the integrand
 * @param space_tuple The tuple of finite element spaces used by the input trial functions
 * @param arg_tuple The tuple of input arguments for the trial functions (value and gradient)
 * @param correction The shape correction struct containing the data and methods used to calculate the appropriate
 * pullback mappings
 *
 * @return The q-function value using the shape-modified input arguments. Note that the returned value and flux have not
 * been modified to reflect the shape displacement.
 */
template <typename lambda, typename coord_type, typename shape_type, typename space_types, typename trial_types,
          typename correction_type, int... i>
SMITH_HOST_DEVICE auto apply_shape_aware_qf_helper(const lambda& qf, double t, const coord_type& position,
                                                   const shape_type& shape, const space_types& space_tuple,
                                                   const trial_types& arg_tuple, const correction_type& correction,
                                                   std::integer_sequence<int, i...>)
{
  static_assert(tuple_size<trial_types>::value == tuple_size<space_types>::value,
                "Argument and finite element space tuples are not the same size.");

  auto x = smith::tuple{get<VALUE>(position) + get<VALUE>(shape),

                        // x := X + u,
                        // so, dx/dxi = dX/dxi + du/dxi
                        //            = dX/dxi + du/dX * dX/dxi
                        get<DERIVATIVE>(position) + get<DERIVATIVE>(shape) * get<DERIVATIVE>(position)};

  return qf(t, x, correction.modify_trial_argument(smith::get<i>(space_tuple), smith::get<i>(arg_tuple))...);
}

/**
 * @brief A helper function to modify all of the trial function input derivatives according to the given shape
 * displacement for integrands without state variables
 *
 * @tparam lambda The q-function type
 * @tparam coord_type The input position type
 * @tparam state_type The quadrature data container type
 * @tparam shape_type The type of the shape displacement argument
 * @tparam space_types The type of the input finite element space tuple for the trial functions
 * @tparam trial_types The type of the input finite element argument tuple (values and derivatives)
 * @tparam correction_type The type of the shape correction struct
 * @tparam i Indices for accessing the individual arguments for the underlying q-function
 *
 * @param qf The q-function integrand with expects shape-adjusted arguments
 * @param t The time at which to evaluate the integrand
 * @param position The quadrature point spatial coordinates and isoparametric derivatives
 * @param state The quadrature data at which to evaluate the integrand
 * @param shape The space displacement at which to evaluate the integrand
 * @param space_tuple The tuple of finite element spaces used by the input trial functions
 * @param arg_tuple The tuple of input arguments for the trial functions (value and gradient)
 * @param correction The shape correction struct containing the data and methods used to calculate the appropriate
 * pullback mappings
 *
 * @return The q-function value using the shape-modified input arguments. Note that the returned value and flux have not
 * been modified to reflect the shape displacement.
 */
template <typename lambda, typename coord_type, typename state_type, typename shape_type, typename space_types,
          typename trial_types, typename correction_type, int... i>
SMITH_HOST_DEVICE auto apply_shape_aware_qf_helper_with_state(const lambda& qf, double t, const coord_type& position,
                                                              state_type& state, const shape_type& shape,
                                                              const space_types& space_tuple,
                                                              const trial_types& arg_tuple,
                                                              const correction_type& correction,
                                                              std::integer_sequence<int, i...>)
{
  static_assert(tuple_size<trial_types>::value == tuple_size<space_types>::value,
                "Argument and finite element space tuples are not the same size.");

  auto x = smith::tuple{get<VALUE>(position) + get<VALUE>(shape),

                        // x := X + u,
                        // so, dx/dxi = dX/dxi + du/dxi
                        //            = dX/dxi + du/dX * dX/dxi
                        get<DERIVATIVE>(position) + get<DERIVATIVE>(shape) * get<DERIVATIVE>(position)};

  return qf(t, x, state, correction.modify_trial_argument(smith::get<i>(space_tuple), smith::get<i>(arg_tuple))...);
}

}  // namespace detail

/// @cond
template <typename T1, typename T2, ExecutionSpace exec = smith::ExecutionSpace::CPU>
class ShapeAwareFunctional;
/// @endcond

/**
 * @brief This is a small wrapper around smith::Functional for shape-displaced domains of integration
 *
 * If a finite element kernel is defined on a domain \f$x = X + p\f$ where \f$X\f$ is the reference
 * configuration and \f$p\f$ is a shape displacement field. This shape displacement is typically a input
 * parameter used in shape optimization problems. This wrapper correctly modifies domain and boundary
 * integrals for L2 and H1 trial functions and H1, L2, and QOI (double) test functions for vector-valued
 * H1 shape displacement fields.
 *
 * See smith::Functional for more details about the underlying finite element abstraction and smith::HeatTransfer
 * for an example of how to use this wrapper.
 *
 * @tparam test The space of test function to use
 * @tparam shape The space of the shape displacement function to use
 * @tparam trials The space of trial functions to use
 * @tparam exec whether to carry out calculations on CPU or GPU
 */
template <typename test, typename shape, typename... trials, ExecutionSpace exec>
class ShapeAwareFunctional<shape, test(trials...), exec> {
  /// @brief The compile-time trial function finite element spaces
  static constexpr tuple<trials...> trial_spaces{};

  /// @brief The compile-time test function finite element space
  static constexpr test test_space{};

  /// @brief The compile-time shape displacement finite element space
  static constexpr shape shape_space{};

  /// @brief The number of input trial functions
  static constexpr uint32_t num_trial_spaces = sizeof...(trials);

 public:
  /**
   * @brief Constructs using @p mfem::ParFiniteElementSpace objects corresponding to the test/trial spaces
   *
   * @tparam test_type The finite element space of the test functions (used to disable this method if called with test =
   * double)
   *
   * @param[in] shape_fes The shape displacement finite element space
   * @param[in] test_fes The (non-qoi) test finite element space
   * @param[in] trial_fes The trial finite element spaces
   */
  template <typename test_type = test, typename = std::enable_if_t<!std::is_same_v<double, test_type>>>
  ShapeAwareFunctional(const mfem::ParFiniteElementSpace* shape_fes, const mfem::ParFiniteElementSpace* test_fes,
                       std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_fes)
  {
    static_assert(shape_space.family == Family::H1, "Only H1 spaces allowed for shape displacements");

    std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces + 1> prepended_spaces;

    prepended_spaces[0] = shape_fes;

    for (uint32_t i = 0; i < num_trial_spaces; ++i) {
      prepended_spaces[1 + i] = trial_fes[i];
    }

    functional_ = std::make_unique<Functional<test(shape, trials...), exec>>(test_fes, prepended_spaces);
  }

  /**
   * @brief Constructs a QOI functional using @p mfem::ParFiniteElementSpace objects corresponding to the trial spaces
   *
   * @tparam test_type The finite element space of the test functions (used to disable this method if called with test
   * != double)
   *
   * @param[in] shape_fes The shape displacement finite element space
   * @param[in] trial_fes The trial finite element spaces
   */
  template <typename test_type = test, typename = std::enable_if_t<std::is_same_v<double, test_type>>>
  ShapeAwareFunctional(const mfem::ParFiniteElementSpace* shape_fes,
                       std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_fes)
  {
    static_assert(shape_space.family == Family::H1, "Only H1 spaces allowed for shape displacements");

    std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces + 1> prepended_spaces;

    prepended_spaces[0] = shape_fes;

    for (uint32_t i = 0; i < num_trial_spaces; ++i) {
      prepended_spaces[1 + i] = trial_fes[i];
    }

    functional_ = std::make_unique<Functional<double(shape, trials...), exec>>(prepended_spaces);
  }

  /**
   * @brief Functor representing a shape-aware integrand.  Used instead of an extended generic
   * lambda for compatibility with NVCC.
   */
  template <typename Integrand, int dim, int... args>
  struct ShapeAwareIntegrandWrapper {
    /// @brief Constructor for functor
    ShapeAwareIntegrandWrapper(Integrand integrand) : integrand_(integrand) {}
    /// @brief Integrand
    Integrand integrand_;
    /**
     * @brief integrand call
     *
     * @tparam PositionType position type
     * @tparam ShapeValueType shape value type
     * @tparam QFuncArgs type of variadic pack to forward to qfunc
     * @param[in] time time
     * @param[in] x position
     * @param[in] shape_val shape
     * @param[in] qfunc_args qfunc parameter pack
     * @return Shape aware qf value
     */
    template <typename PositionType, typename ShapeValueType, typename... QFuncArgs>
    SMITH_HOST_DEVICE auto operator()(double time, PositionType x, ShapeValueType shape_val,
                                      QFuncArgs... qfunc_args) const
    {
      auto qfunc_tuple = make_tuple(qfunc_args...);
      auto reduced_trial_space_tuple = make_tuple(get<args>(trial_spaces)...);

      detail::ShapeCorrection shape_correction(Dimension<dim>{}, shape_val);
      // TODO(CUDA): When this is compiled to device code, the below make_integer_sequence will
      // need to change to a camp integer sequence.
      auto unmodified_qf_return = detail::apply_shape_aware_qf_helper(
          integrand_, time, x, shape_val, reduced_trial_space_tuple, qfunc_tuple, shape_correction,
          std::make_integer_sequence<int, sizeof...(qfunc_args)>{});
      return shape_correction.modify_shape_aware_qf_return(test_space, unmodified_qf_return);
    }
  };

  /**
   * @brief Functor representing a shape-aware integrand with state.  Used instead of an extended generic
   * lambda for compatibility with NVCC.
   */
  template <typename Integrand, int dim, int... args>
  struct ShapeAwareIntegrandWrapperWithState {
    /// @brief Constructor for functor
    ShapeAwareIntegrandWrapperWithState(Integrand integrand) : integrand_(integrand) {}
    /// @brief integrand
    Integrand integrand_;
    /**
     * @brief integrand call
     *
     * @tparam PositionType position type
     * @tparam StateType state type
     * @tparam ShapeValueType shape value type
     * @tparam QFuncArgs type of variadic pack to forward to qfunc
     * @param[in] time time
     * @param[in] x position
     * @param[in] state reference to state
     * @param[in] shape_val shape
     * @param[in] qfunc_args qfunc parameter pack
     * @return shape aware integrand value
     */
    template <typename PositionType, typename StateType, typename ShapeValueType, typename... QFuncArgs>
    SMITH_HOST_DEVICE auto operator()(double time, PositionType x, StateType& state, ShapeValueType shape_val,
                                      QFuncArgs... qfunc_args) const
    {
      auto qfunc_tuple = make_tuple(qfunc_args...);
      auto reduced_trial_space_tuple = make_tuple(get<args>(trial_spaces)...);

      detail::ShapeCorrection shape_correction(Dimension<dim>{}, shape_val);
      // TODO(CUDA): When this is compiled to device code, the below make_integer_sequence will
      // need to change to a camp integer sequence.
      auto unmodified_qf_return = detail::apply_shape_aware_qf_helper_with_state(
          integrand_, time, x, state, shape_val, reduced_trial_space_tuple, qfunc_tuple, shape_correction,
          std::make_integer_sequence<int, sizeof...(qfunc_args)>{});
      return shape_correction.modify_shape_aware_qf_return(test_space, unmodified_qf_return);
    }
  };

  /**
   * @brief Adds a domain integral term to the weak formulation of the PDE
   *
   * @tparam dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam args The type of the trial function input arguments
   * @tparam lambda The type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   *
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   * @param[inout] qdata The data for each quadrature point
   *
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int dim, int... args, typename lambda, typename qpt_data_type = Nothing>
  void AddDomainIntegral(Dimension<dim>, DependsOn<args...>, const lambda& integrand, Domain& domain,
                         std::shared_ptr<QuadratureData<qpt_data_type>> qdata = NoQData)
  {
    if constexpr (std::is_same_v<qpt_data_type, Nothing>) {
      functional_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
                                     ShapeAwareIntegrandWrapper<lambda, dim, args...>(integrand), domain, qdata);
    } else {
      functional_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
                                     ShapeAwareIntegrandWrapperWithState<lambda, dim, args...>(integrand), domain,
                                     qdata);
    }
  }

  /**
   * @brief Functor representing a shape-aware integrand.  Used instead of an extended generic
   * lambda for compatibility with NVCC.
   */
  template <typename test_type, typename Integrand, int dim, int... args>
  struct ShapeAwareInteriorBoundaryIntegrandWrapper {
    /// @brief Constructor for functor
    ShapeAwareInteriorBoundaryIntegrandWrapper(Integrand integrand) : integrand_(integrand) {}
    /// @brief integrand
    Integrand integrand_;
    /**
     * @brief integrand call
     *
     * @tparam PositionType position type
     * @tparam ShapeValueType shape value type
     * @tparam QFuncArgs type of variadic pack to forward to qfunc
     * @param[in] time time
     * @param[in] x position
     * @param[in] shape_val shape
     * @param[in] qfunc_args qfunc parameter pack
     * @return qf function corrected for boundary area
     */
    template <typename PositionType, typename ShapeValueType, typename... QFuncArgs>
    SMITH_HOST_DEVICE auto operator()(double time, PositionType x, ShapeValueType shape_val,
                                      QFuncArgs... qfunc_args) const
    {
      auto unmodified_qf_returns = integrand_(time, x + shape_val, qfunc_args...);
      auto correction = detail::compute_boundary_area_correction(x, shape_val);
      return smith::tuple{correction * smith::get<0>(unmodified_qf_returns),
                          correction * smith::get<1>(unmodified_qf_returns)};
    }
  };

  /**
   * @brief Functor representing a shape-aware integrand.  Used instead of an extended generic
   * lambda for compatibility with NVCC.
   */
  template <typename Integrand, int dim, int... args>
  struct ShapeAwareInteriorBoundaryIntegrandWrapper<double, Integrand, dim, args...> {
    /// @brief Constructor for functor
    ShapeAwareInteriorBoundaryIntegrandWrapper(Integrand integrand) : integrand_(integrand) {}
    /// @brief integrand
    Integrand integrand_;
    /**
     * @brief integrand call
     *
     * @tparam PositionType position type
     * @tparam ShapeValueType shape value type
     * @tparam QFuncArgs type of variadic pack to forward to qfunc
     * @param[in] time time
     * @param[in] x position
     * @param[in] shape_val shape
     * @param[in] qfunc_args qfunc parameter pack
     * @return qf function corrected for boundary area
     */
    template <typename PositionType, typename ShapeValueType, typename... QFuncArgs>
    SMITH_HOST_DEVICE auto operator()(double time, PositionType x, ShapeValueType shape_val,
                                      QFuncArgs... qfunc_args) const
    {
      auto unmodified_qf_returns = integrand_(time, x + shape_val, qfunc_args...);
      auto correction = detail::compute_boundary_area_correction(x, shape_val);
      return correction * unmodified_qf_returns;
    }
  };

  /**
   * @brief Adds an interior face integral term to the weak formulation of the PDE
   *
   * @tparam dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam args The type of the trial function input arguments
   * @tparam lambda The type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam domain_type The type of the integration domain (either smith::Domain or mfem::Mesh)
   *
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   *
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int dim, int... args, typename lambda>
  void AddInteriorFaceIntegral(Dimension<dim>, DependsOn<args...>, const lambda& integrand, Domain& domain)
  {
    functional_->AddInteriorFaceIntegral(
        Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
        ShapeAwareInteriorBoundaryIntegrandWrapper<test, lambda, dim, args...>(integrand), domain);
  }

  /**
   * @brief Functor representing a shape-aware integrand.  Used instead of an extended generic
   * lambda for compatibility with NVCC.
   */
  template <typename Integrand, int dim, int... args>
  struct ShapeAwareBoundaryIntegrandWrapper {
    /// @brief Constructor for functor
    ShapeAwareBoundaryIntegrandWrapper(Integrand integrand) : integrand_(integrand) {}
    /// @brief integrand
    Integrand integrand_;
    /**
     * @brief integrand call
     *
     * @tparam PositionType position type
     * @tparam ShapeValueType shape value type
     * @tparam QFuncArgs type of variadic pack to forward to qfunc
     * @param[in] time time
     * @param[in] x position
     * @param[in] shape_val shape
     * @param[in] qfunc_args qfunc parameter pack
     * @return qf function corrected for boundary area
     */
    template <typename PositionType, typename ShapeValueType, typename... QFuncArgs>
    SMITH_HOST_DEVICE auto operator()(double time, PositionType x, ShapeValueType shape_val,
                                      QFuncArgs... qfunc_args) const
    {
      auto unmodified_qf_return = integrand_(time, x + shape_val, qfunc_args...);
      return unmodified_qf_return * detail::compute_boundary_area_correction(x, shape_val);
    }
  };

  /**
   * @brief Adds a boundary integral term to the weak formulationx of the PDE
   *
   * @tparam dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam args The type of the trial function input arguments
   * @tparam lambda The type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam domain_type The type of the integration domain (either smith::Domain or mfem::Mesh)
   *
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   *
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int dim, int... args, typename lambda>
  void AddBoundaryIntegral(Dimension<dim>, DependsOn<args...>, const lambda& integrand, Domain& domain)
  {
    functional_->AddBoundaryIntegral(Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
                                     ShapeAwareBoundaryIntegrandWrapper<lambda, dim, args...>(integrand), domain);
  }

  /**
   * @brief This function lets the user evaluate the smith::ShapeAwareFunctional with the given trial space values
   *
   * @note The first argument after time in the argument list is always the shape displacement field.
   *
   * @tparam T The types of the arguments passed in
   * @tparam wrt The index of the argument to take the gradient with respect to. Index 0 is always the shape
   * displacement
   *
   * @param t The time
   * @param args The trial space dofs used to carry out the calculation. The first argument is always the shape
   * displacement.
   *
   * @return Either the evaluated integral value or a tuple of the integral value and the requested derivative.
   */
  template <uint32_t wrt, typename... T>
  auto operator()(DifferentiateWRT<wrt>, double t, const T&... args)
  {
    return (*functional_)(DifferentiateWRT<wrt>{}, t, args...);
  }

  /**
   * @brief This function lets the user evaluate the smith::ShapeAwareFunctional with the given trial space values
   *
   * @note The first argument after time in the argument list is always the shape displacement field.
   *
   * @tparam T The types of the arguments passed in
   *
   * @param t The time
   * @param args The trial space dofs used to carry out the calculation. The first argument is always the shape
   * displacement. To compute a derivative, at most one argument can be of type `differentiate_wrt_this(mfem::Vector)`.
   *
   * @return Either the evaluated integral value or a tuple of the integral value and the requested derivative.
   */
  template <typename... T>
  auto operator()(double t, const T&... args)
  {
    return (*functional_)(t, args...);
  }

  /**
   * @brief A flag to update the quadrature data for this operator following the computation
   *
   * Typically this is set to false during nonlinear solution iterations and is set to true for the
   * final pass once equilibrium is found.
   *
   * @param update_flag A flag to update the related quadrature data
   */
  void updateQdata(bool update_flag) { functional_->updateQdata(update_flag); }

 private:
  /// @brief The underlying pure Functional object
  std::unique_ptr<Functional<test(shape, trials...), exec>> functional_;
};

}  // namespace smith
