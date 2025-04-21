// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file functional_residual.hpp
 *
 * @brief Implements the residual interface using serac::ShapeAwareFunctional.
 * Allows for generic specification of body and boundary integrals
 */

#pragma once

#include "serac/physics/residual.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"

namespace serac {

template <int spatial_dim, typename ShapeDispSpace, typename OutputSpace, typename inputs = Parameters<>,
          typename input_indices = std::make_integer_sequence<int, inputs::n>>
class FunctionalResidual;

/**
 * @brief The nonlinear residual class
 *
 * This uses Functional to compute fairly arbitrary residuals and tangent
 * stiffness matrices based on body and boundary integrals.
 *
 */
template <int spatial_dim, typename ShapeDispSpace, typename OutputSpace, typename... InputSpaces, int... input_indices>
class FunctionalResidual<spatial_dim, ShapeDispSpace, OutputSpace, Parameters<InputSpaces...>,
                         std::integer_sequence<int, input_indices...>> : public Residual {
 public:
  using SpacesT = std::vector<const mfem::ParFiniteElementSpace*>;  ///< typedef

  /**
   * @brief Construct a new FunctionalResidual object
   *
   * @param physics_name A name for the physics module instance
   * @param mesh The serac mesh
   * @param shape_disp_space Shape displacement space
   * @param output_mfem_space Test space
   * @param input_mfem_spaces Vector of finite element spaces which are arguments to the residual
   */
  FunctionalResidual(std::string physics_name, std::shared_ptr<Mesh> mesh,
                     const mfem::ParFiniteElementSpace& shape_disp_space,
                     const mfem::ParFiniteElementSpace& output_mfem_space, const SpacesT& input_mfem_spaces)
      : Residual(physics_name), mesh_(mesh)
  {
    std::array<const mfem::ParFiniteElementSpace*, sizeof...(InputSpaces)> trial_spaces;

    SLIC_ERROR_ROOT_IF(
        sizeof...(InputSpaces) != input_mfem_spaces.size(),
        axom::fmt::format("{} parameter spaces given in the template argument but {} input mfem spaces were supplied.",
                          sizeof...(InputSpaces), input_mfem_spaces.size()));

    if constexpr (sizeof...(InputSpaces) > 0) {
      for_constexpr<sizeof...(InputSpaces)>([&](auto i) { trial_spaces[i] = input_mfem_spaces[i]; });
    }

    residual_ = std::make_unique<ShapeAwareFunctional<ShapeDispSpace, OutputSpace(InputSpaces...)>>(
        &shape_disp_space, &output_mfem_space, trial_spaces);
  }

  /**
   * @brief Add a body integral contribution to the residual
   *
   * @tparam BodyIntegralType The type of the body integral
   * // DependsOn<active_parameters...> can be indices into fields which the body integral may depend on
   * @param body_name The name of the registered domain over which the body force is applied. If nothing is supplied
   * the entire domain is
   * @param body_integral A function describing the body force applied
   * used.
   * @pre body_integral must be a object that can be called with the following arguments:
   *    1. `double t` the time
   *    2. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    3. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   */
  template <int... active_parameters, typename BodyIntegralType>
  void addBodyIntegral(DependsOn<active_parameters...>, std::string body_name, BodyIntegralType body_integral)
  {
    residual_->AddDomainIntegral(Dimension<spatial_dim>{}, DependsOn<active_parameters...>{}, body_integral,
                                 mesh_->domain(body_name));
  }

  /// @overload
  template <typename BodyForceType>
  void addBodyIntegral(std::string body_name, BodyForceType body_integral)
  {
    addBodyIntegral(DependsOn<>{}, body_name, body_integral);
  }

  /**
   * @brief Set the Neumann boundary condition
   *
   * @tparam NeumannType The type of the traction load
   * * // DependsOn<active_parameters...> can be indices into fields which the body integral may depend on
   * @param boundary_name The name of the registered domain over which the traction is applied. If nothing is supplied
   * the entire boundary is
   * @param surface_function A function describing the traction applied to a boundary
   * used.
   * @pre NeumannType must be a object that can be called with the following arguments:
   *    1. `double t` the time
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    3. `tensor<T,dim> n` the outward-facing unit normal for the quadrature point
   *    4. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   */
  template <int... active_parameters, typename NeumannType>
  void addBoundaryIntegral(DependsOn<active_parameters...>, std::string boundary_name, NeumannType surface_function)
  {
    residual_->AddBoundaryIntegral(
        Dimension<spatial_dim - 1>{}, DependsOn<active_parameters...>{},
        [surface_function](double t, auto X, auto... params) {
          auto n = cross(get<DERIVATIVE>(X));
          return surface_function(t, get<VALUE>(X), normalize(n), params...);
        },
        mesh_->domain(boundary_name));
  }

  /// @overload
  template <typename NeumannType>
  void addBoundaryIntegral(std::string boundary_name, NeumannType surface_function)
  {
    addBoundaryIntegral(DependsOn<>{}, boundary_name, surface_function);
  }

  /// @overload
  mfem::Vector residual(double time, double dt, const std::vector<FieldPtr>& fields, int block_row = 0) const override
  {
    SLIC_ERROR_IF(block_row != 0, "Invalid block row and column requested in fieldJacobian for FunctionalResidual");
    dt_ = dt;
    auto ret = (*residual_)(time, *fields[0], *fields[input_indices + 1]...);
    return ret;
  }

  /// @overload
  std::unique_ptr<mfem::HypreParMatrix> jacobian(double time, double dt, const std::vector<FieldPtr>& fields,
                                                 const std::vector<double>& jacobian_weights,
                                                 int block_row = 0) const override
  {
    SLIC_ERROR_IF(block_row != 0, "Invalid block row and column requested in fieldJacobian for FunctionalResidual");

    dt_ = dt;

    std::unique_ptr<mfem::HypreParMatrix> J;

    auto addToJ = [&J](double factor, std::unique_ptr<mfem::HypreParMatrix> jac_contrib) {
      if (J) {
        SLIC_ERROR_IF(J->N() != jac_contrib->N(),
                      "Multiple nonzero jacobian weights are being used on inconsistently sized input arguments.");
        SLIC_ERROR_IF(J->M() != jac_contrib->M(),
                      "Multiple nonzero jacobian weights are being used on inconsistently sized input arguments.");
        J->Add(factor, *jac_contrib);
      } else {
        J.reset(jac_contrib.release());
        if (factor != 1.0) (*J) *= factor;
      }
    };

    auto jacs = jacobianFunctions(std::make_integer_sequence<int, sizeof...(input_indices) + 1>{}, time, fields);

    for (size_t input_col = 0; input_col < jacobian_weights.size(); ++input_col) {
      if (jacobian_weights[input_col] != 0.0) {
        auto K = serac::get<DERIVATIVE>(jacs[input_col](time, fields));
        addToJ(jacobian_weights[input_col], assemble(K));
      }
    }

    return J;
  }

  /// @overload
  void jvp(double time, double dt, const std::vector<FieldPtr>& fields, const std::vector<FieldPtr>& vFields,
           std::vector<DualFieldPtr>& jvpReactions) const override
  {
    SLIC_ERROR_IF(vFields.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");
    SLIC_ERROR_IF(jvpReactions.size() != 1, "FunctionalResidual nonlinear systems only supports 1 output residual");

    dt_ = dt;
    auto jacs = jacobianFunctions(std::make_integer_sequence<int, sizeof...(input_indices) + 1>{}, time, fields);

    *jvpReactions[0] = 0.0;

    for (size_t input_col = 0; input_col < fields.size(); ++input_col) {
      if (vFields[input_col] != nullptr) {
        auto K = serac::get<DERIVATIVE>(jacs[input_col](time, fields));
        K.AddMult(*vFields[input_col], *jvpReactions[0]);
      }
    }
  }

  /// @overload
  void vjp(double time, double dt, const std::vector<FieldPtr>& fields, const std::vector<DualFieldPtr>& vReactions,
           std::vector<FieldPtr>& vjpFields) const override
  {
    SLIC_ERROR_IF(vjpFields.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");
    SLIC_ERROR_IF(vReactions.size() != 1, "FunctionalResidual nonlinear systems only supports 1 output residual");

    dt_ = dt;
    auto jacs = jacobianFunctions(std::make_integer_sequence<int, sizeof...(input_indices) + 1>{}, time, fields);

    for (size_t input_col = 0; input_col < fields.size(); ++input_col) {
      auto K = serac::get<DERIVATIVE>(jacs[input_col](time, fields));
      std::unique_ptr<mfem::HypreParMatrix> J = assemble(K);
      J->AddMultTranspose(*vReactions[0], *vjpFields[input_col]);
    }
  }

  /// @brief Accessor to get a reference to the underlying ShapeAwareFunctional in case more direct access is needed.
  /// @return Reference to ShapeAwareFunctional instance.
  ShapeAwareFunctional<ShapeDispSpace, OutputSpace(InputSpaces...)>& getFunctional() { return *residual_; }

 protected:
  /// @brief Utility to get array of jacobian functions, one for each input field in fs
  template <int... i>
  auto jacobianFunctions(std::integer_sequence<int, i...>, double time, const std::vector<FieldPtr>& fs) const
  {
    using JacFuncType = std::function<decltype((*residual_)(DifferentiateWRT<1>{}, time, *fs[i]...))(
        double, const std::vector<FieldPtr>&)>;
    return std::array<JacFuncType, sizeof...(i)>{[=](double _time, const std::vector<FieldPtr>& _fs) {
      return (*residual_)(DifferentiateWRT<i>{}, _time, *_fs[i]...);
    }...};
  };

  /// @brief timestep, this needs to be held here and modified for rate dependent applications
  mutable double dt_ = std::numeric_limits<double>::max();

  /// @brief primary mesh
  std::shared_ptr<Mesh> mesh_;

  /// @brief functional residual evaluator, shape aware
  std::unique_ptr<ShapeAwareFunctional<ShapeDispSpace, OutputSpace(InputSpaces...)>> residual_;
};

/// @brief Helper function to construct vector of spaces from an existing vector of FiniteElementState.
/// @param states vector of FiniteElementState
inline std::vector<const mfem::ParFiniteElementSpace*> getSpaces(const std::vector<serac::FiniteElementState>& states)
{
  std::vector<const mfem::ParFiniteElementSpace*> spaces;
  for (auto& f : states) {
    spaces.push_back(&f.space());
  }
  return spaces;
}

}  // namespace serac
