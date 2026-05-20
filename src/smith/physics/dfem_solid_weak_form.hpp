// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file dfem_solid_weak_form.hpp
 *
 * @brief Implements the WeakForm interface for solid mechanics physics using dFEM. Derives from DfemWeakForm.
 */

#pragma once

#include "smith/smith_config.hpp"

#ifdef SMITH_USE_ENZYME

#include "smith/physics/dfem_weak_form.hpp"

#include "smith/infrastructure/accelerator.hpp"

namespace smith {

template <int Idx>
struct ScalarParameter {
  static constexpr int index = Idx;
  using QFunctionInput = double;
  template <int FieldId>
  using QFunctionFieldOp = mfem::future::Value<FieldId>;
};

/**
 * @brief Boundary q-function that applies a user-supplied reference-configuration
 *        traction t(X, time) to the residual.
 *
 * Output convention follows the rest of DfemSolidWeakForm: residual = -external_force,
 * so the contribution written into the test-function `Value<...>` slot is `-t * dA * w`.
 *
 * `time_ptr` points at the parent DfemWeakForm's mutable current_time_, which is updated
 * at the top of each residual()/jacobian() call. That avoids re-registering the
 * qfunction every load step.
 *
 * Reference-configuration traction is independent of the displacement field, so the
 * boundary integrator is added with an empty derivative_ids list — it contributes to
 * the residual only.
 *
 * The qfunction takes Value<> of DISPLACEMENT, VELOCITY, ACCELERATION as ignored inputs.
 * mfem::future::DifferentiableOperator asserts that every field id in the operator's
 * union(infds, outfds) is referenced by every integrator's qfunction, so even unused
 * inputs must appear in the signature.
 */
template <int Dim, typename TractionFn>
struct DfemTractionQFunction {
  SMITH_HOST_DEVICE inline void operator()(
      const mfem::future::tensor<mfem::real_t, Dim>& /*u*/,
      const mfem::future::tensor<mfem::real_t, Dim>& /*v*/,
      const mfem::future::tensor<mfem::real_t, Dim>& /*a*/,
      const mfem::future::tensor<mfem::real_t, Dim, Dim - 1>& dX_dxi, mfem::real_t w,
      mfem::future::tensor<mfem::real_t, Dim>& out) const
  {
    const auto t_vec       = traction_fn(*time_ptr);
    const auto surface_dxi = mfem::future::weight(dX_dxi);
    for (int i = 0; i < Dim; ++i) {
      out[i] = -t_vec[i] * surface_dxi * w;
    }
  }

  TractionFn    traction_fn;  ///< user-supplied function (t) -> tensor<double, Dim>
  const double* time_ptr;     ///< points to parent DfemWeakForm::current_time_
};

/**
 * @brief Domain q-function that applies a user-supplied reference-configuration
 *        body force b(time) to the residual.
 *
 * Output is `-b * det(dX/dxi) * w` so the total volume integral matches
 * ∫_V b dV (per residual sign convention: load enters with a negative sign).
 *
 * Used as a workaround for boundary traction while `LocalQFBackend` boundary
 * support is broken upstream — applying a body force whose total volume
 * integral equals the intended surface traction's total force produces an
 * equivalent net load (though distributed through-thickness rather than on a
 * single face).
 */
template <int Dim, typename BodyForceFn>
struct DfemBodyForceQFunction {
  SMITH_HOST_DEVICE inline void operator()(
      const mfem::future::tensor<mfem::real_t, Dim>& /*u*/,
      const mfem::future::tensor<mfem::real_t, Dim>& /*v*/,
      const mfem::future::tensor<mfem::real_t, Dim>& /*a*/,
      const mfem::future::tensor<mfem::real_t, Dim, Dim>& dX_dxi, mfem::real_t w,
      mfem::future::tensor<mfem::real_t, Dim>& out) const
  {
    const auto b_vec = body_force_fn(*time_ptr);
    const auto detJ  = mfem::future::det(dX_dxi);
    for (int i = 0; i < Dim; ++i) {
      out[i] = -b_vec[i] * detJ * w;
    }
  }

  BodyForceFn   body_force_fn;
  const double* time_ptr;
};

template <typename Material, typename... Parameters>
struct StressDivQFunction {
  SMITH_HOST_DEVICE inline void operator()(
      // mfem::real_t dt, // TODO: figure out how to pass this in
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& du_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dv_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>&,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dX_dxi, mfem::real_t weight,
      typename Parameters::QFunctionInput... params,
      mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& out) const
  {
    auto dxi_dX = mfem::future::inv(dX_dxi);
    auto du_dX = mfem::future::dot(du_dxi, dxi_dX);
    auto dv_dX = mfem::future::dot(dv_dxi, dxi_dX);
    double dt = 1.0;  // TODO: figure out how to pass this in to the qfunction
    auto P = mfem::future::get<0>(material.pkStress(dt, du_dX, dv_dX, params...));
    auto JxW = mfem::future::det(dX_dxi) * weight * mfem::future::transpose(dxi_dX);
    out = -P * JxW;
  }

  Material material;  ///< the material model to use for computing the stress
};

/**
 * @brief The weak form for solid mechanics
 *
 * This uses dFEM to compute the solid mechanics residuals and tangent stiffness matrices.
 */
class DfemSolidWeakForm : public DfemWeakForm {
 public:
  /// @brief a container holding quadrature point data of the specified type
  /// @tparam T the type of data to store at each quadrature point
  template <typename T>
  using qdata_type = std::shared_ptr<QuadratureData<T>>;

  /// @brief disp, velo, accel
  static constexpr int NUM_STATE_VARS = 4;

  /// @brief enumeration of the required states
  enum STATE
  {
    DISPLACEMENT,
    VELOCITY,
    ACCELERATION,
    COORDINATES,
    NUM_STATES
  };

  /**
   * @brief Construct a new DfemSolidWeakForm object
   *
   * @param physics_name A name for the physics module instance
   * @param mesh The smith Mesh
   * @param test_space Test space
   * @param parameter_fe_spaces Vector of parameters spaces
   */
  DfemSolidWeakForm(std::string physics_name, std::shared_ptr<Mesh> mesh, const mfem::ParFiniteElementSpace& test_space,
                    std::vector<const mfem::ParFiniteElementSpace*> parameter_fe_spaces = {})
      : DfemWeakForm(physics_name, mesh, test_space, makeInputSpaces(test_space, mesh, parameter_fe_spaces))
  {
  }

  /**
   * @brief Set the material stress response and mass properties for the physics module
   *
   * @tparam MaterialType The solid material type
   * @tparam ParameterTypes types that contains the internal variables for MaterialType
   * @param domain_attributes Array of MFEM element attributes over which to compute the integral
   * @param material A material that provides a function to evaluate PK1 stress
   * @pre material must be a object that has a pkStress method with the following arguments:
   *    1. `double dt` the timestep size
   *    2. `tensor<double,dim,dim> du_dX` the displacement gradient at this quadrature point
   *    3. `tensor<double,dim,dim> dv_dX` the velocity gradient at this quadrature point
   *    4. Additional arguments for the dependent parameters of the material
   *
   * @pre MaterialType must have a public method `density` which can take FiniteElementState parameter inputs
   * @pre MaterialType must have a public method 'pkStress' which returns the first Piola-Kirchhoff stress
   *
   */
  template <typename MaterialType, typename... ParameterTypes>
  void setMaterial(const mfem::Array<int>& domain_attributes, const MaterialType& material,
                   const mfem::IntegrationRule& displacement_ir)
  {
    SLIC_ERROR_IF(material.dim != DfemWeakForm::mesh_->mfemParMesh().Dimension(),
                  "Material model dimension does not match mesh dimension.");
    auto stress_div_integral = StressDivQFunction<MaterialType, ParameterTypes...>{.material = material};
    mfem::future::tuple<mfem::future::Gradient<DISPLACEMENT>, mfem::future::Gradient<VELOCITY>,
                        mfem::future::Gradient<ACCELERATION>, mfem::future::Gradient<COORDINATES>, mfem::future::Weight,
                        typename ParameterTypes::template QFunctionFieldOp<NUM_STATE_VARS + ParameterTypes::index>...>
        stress_div_integral_inputs{};
    mfem::future::tuple<mfem::future::Gradient<NUM_STATE_VARS + sizeof...(ParameterTypes)>>
        stress_div_integral_outputs{};
    DfemWeakForm::addBodyIntegral(domain_attributes, stress_div_integral, stress_div_integral_inputs,
                                  stress_div_integral_outputs, displacement_ir,
                                  std::index_sequence<DISPLACEMENT, NUM_STATE_VARS + ParameterTypes::index...>{});
  }

  /**
   * @brief Apply a reference-configuration traction load over a set of boundary
   *        attributes.
   *
   * The user-supplied function is called per quadrature point on the boundary as
   * `traction_fn(X, time)`, where X is the reference position of the QP and
   * `time` is the current end-of-step time pulled from DfemWeakForm::current_time_.
   * The returned vector is the traction (force per unit reference area) applied at
   * that point.
   *
   * Implementation notes:
   *  - Reference-configuration traction is independent of the displacement field,
   *    so the integrator is registered with an empty derivative_ids list. The
   *    Jacobian assembled by DfemWeakForm::jacobian() therefore picks up no
   *    contribution from this term — which is physically correct for `setTraction`
   *    (use `setPressure`-style follower loads when ∂t/∂u ≠ 0).
   *  - Reverse-mode (vjp) on boundary integrators is not yet wired through dFEM;
   *    DfemWeakForm::addBoundaryIntegral with empty derivative_ids skips that path.
   *  - v1 only supports parameter-free DfemSolidWeakForm (no FE parameter spaces
   *    passed to the constructor). The MFEM dFEM operator requires every integrator
   *    to reference every field id in `union(infds, outfds)`, so adding parameter
   *    fields to the constructor without also threading them through the traction
   *    qfunction would trip an assert. Extend with parameter inputs when needed.
   *
   * @tparam Dim Spatial dimension of the problem (2 or 3). Templated because
   *             DfemSolidWeakForm itself is not currently templated on dim.
   * @param boundary_attributes Marker array selecting the boundary attribute(s) to
   *                            integrate over (one entry per bdr_attribute,
   *                            non-zero to include).
   * @param traction_fn Callable `(tensor<double, Dim>, double) -> tensor<double, Dim>`.
   * @param boundary_ir Integration rule on the boundary geometry.
   */
  template <int Dim, typename TractionFn>
  void setTraction(const mfem::Array<int>& boundary_attributes, TractionFn traction_fn,
                   const mfem::IntegrationRule& boundary_ir)
  {
    SLIC_ERROR_IF(Dim != DfemWeakForm::mesh_->mfemParMesh().Dimension(),
                  "DfemSolidWeakForm::setTraction Dim template argument does not match mesh dimension.");
    SLIC_ERROR_IF(DfemWeakForm::input_mfem_spaces_.size() != NUM_STATE_VARS,
                  "DfemSolidWeakForm::setTraction v1 only supports parameter-free weak forms. "
                  "Construct DfemSolidWeakForm with an empty parameter_fe_spaces list.");

    DfemTractionQFunction<Dim, TractionFn> traction_qf{
        .traction_fn = std::move(traction_fn),
        .time_ptr    = DfemWeakForm::currentTimePtr(),
    };

    constexpr int output_field_id = NUM_STATE_VARS;

    mfem::future::tuple<mfem::future::Value<DISPLACEMENT>, mfem::future::Value<VELOCITY>,
                        mfem::future::Value<ACCELERATION>, mfem::future::Gradient<COORDINATES>,
                        mfem::future::Weight>
        traction_inputs{};
    mfem::future::tuple<mfem::future::Value<output_field_id>> traction_outputs{};

    DfemWeakForm::addBoundaryIntegral(boundary_attributes, traction_qf, traction_inputs, traction_outputs,
                                      boundary_ir, std::index_sequence<>{});
  }

  /**
   * @brief Apply a reference-configuration body force load over a set of domain
   *        attributes. Used as a workaround for boundary traction while the
   *        upstream LocalQFBackend boundary path is not entity-aware.
   *
   * Equivalence: a body force `b` over volume `V` produces total force `b * V`,
   * matching a uniform top-face traction `t` of total `t * A_top` when
   * `b = t * A_top / V`.
   */
  template <int Dim, typename BodyForceFn>
  void setBodyForce(const mfem::Array<int>& domain_attributes, BodyForceFn body_force_fn,
                    const mfem::IntegrationRule& domain_ir)
  {
    SLIC_ERROR_IF(Dim != DfemWeakForm::mesh_->mfemParMesh().Dimension(),
                  "DfemSolidWeakForm::setBodyForce Dim template argument does not match mesh dimension.");
    SLIC_ERROR_IF(DfemWeakForm::input_mfem_spaces_.size() != NUM_STATE_VARS,
                  "DfemSolidWeakForm::setBodyForce v1 only supports parameter-free weak forms.");

    DfemBodyForceQFunction<Dim, BodyForceFn> body_force_qf{
        .body_force_fn = std::move(body_force_fn),
        .time_ptr      = DfemWeakForm::currentTimePtr(),
    };

    constexpr int output_field_id = NUM_STATE_VARS;

    mfem::future::tuple<mfem::future::Value<DISPLACEMENT>, mfem::future::Value<VELOCITY>,
                        mfem::future::Value<ACCELERATION>, mfem::future::Gradient<COORDINATES>,
                        mfem::future::Weight>
        body_force_inputs{};
    mfem::future::tuple<mfem::future::Value<output_field_id>> body_force_outputs{};

    DfemWeakForm::addBodyIntegral(domain_attributes, body_force_qf, body_force_inputs, body_force_outputs, domain_ir,
                                  std::index_sequence<>{});
  }

 protected:
  /**
   * @brief Creates a list of MFEM input spaces compatible with dFEM
   *
   * @param test_space Space the q-function will be integrated against
   * @param mesh Problem mesh
   * @param parameter_fe_spaces Vector of finite element spaces which are parameter arguments to the residual
   * @return Vector of input spaces that are passed along to the dFEM differentiable operator constructor
   */
  std::vector<const mfem::ParFiniteElementSpace*> makeInputSpaces(
      const mfem::ParFiniteElementSpace& test_space, const std::shared_ptr<Mesh>& mesh,
      const std::vector<const mfem::ParFiniteElementSpace*>& parameter_fe_spaces)
  {
    std::vector<const mfem::ParFiniteElementSpace*> input_spaces;
    input_spaces.reserve(NUM_STATE_VARS + parameter_fe_spaces.size());
    for (int i = 0; i < 3; ++i) {
      input_spaces.push_back(&test_space);
    }
    input_spaces.push_back(static_cast<const mfem::ParFiniteElementSpace*>(mesh->mfemParMesh().GetNodalFESpace()));
    for (auto space : parameter_fe_spaces) {
      input_spaces.push_back(space);
    }
    return input_spaces;
  }
};

}  // namespace smith

#endif
