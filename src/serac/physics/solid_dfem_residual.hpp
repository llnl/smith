// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_dfem_residual.hpp
 *
 * @brief Implements the residual interface for solid mechanics physics.
 * Derives from dfem_residual.
 */

#pragma once

#include "serac/physics/dfem_residual.hpp"

namespace serac {

template <typename Material, typename... Parameters>
struct StressDivQFunction {
  SERAC_HOST_DEVICE inline auto operator()(
      // mfem::real_t dt, // TODO: figure out how to pass this in
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& du_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dv_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>&,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dX_dxi, mfem::real_t weight,
      Parameters... params) const
  {
    auto dxi_dX = mfem::future::inv(dX_dxi);
    auto du_dX = mfem::future::dot(du_dxi, dxi_dX);
    auto dv_dX = mfem::future::dot(dv_dxi, dxi_dX);
    double dt = 1.0;  // TODO: figure out how to pass this in to the qfunction
    auto P = mfem::future::get<0>(material.pkStress(dt, du_dX, dv_dX, params...));
    auto JxW = mfem::future::det(dX_dxi) * weight * mfem::future::transpose(dxi_dX);
    return mfem::future::tuple{-P * JxW};
  }

  Material material;  ///< the material model to use for computing the stress
};

template <typename Material, typename... Parameters>
struct AccelerationQFunction {
  SERAC_HOST_DEVICE inline auto operator()(
      // mfem::real_t dt, // TODO: figure out how to pass this in
      const mfem::future::tensor<mfem::real_t, Material::dim>&,
      const mfem::future::tensor<mfem::real_t, Material::dim>&,
      const mfem::future::tensor<mfem::real_t, Material::dim>& a,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dX_dxi, mfem::real_t weight,
      Parameters... params) const
  {
    auto rho = material.density(params...);
    auto J = mfem::future::det(dX_dxi) * weight;
    return mfem::future::tuple{-rho * a * J};
  }

  Material material;  ///< the material model to use for computing the density
};

/**
 * @brief The nonlinear residual class
 *
 * This uses Functional to compute the solid mechanics residuals and tangent
 * stiffness matrices.
 *
 * @tparam order The order of the discretization of the displacement and velocity fields
 * @tparam dim The spatial dimension of the mesh
 */
template <bool IsQuasiStatic = false, bool UseLumpedMass = false>
class SolidDfemResidual : public DfemResidual {
 public:
  /// @brief disp, velo, accel, coord
  static constexpr int NUM_STATE_VARS = 4;

  enum FieldIDs
  {
    DISP,   ///< displacement
    VELO,   ///< velocity
    ACCEL,  ///< acceleration
    COORD   ///< coordinates
  };

  /**
   * @brief Construct a new SolidResidual object
   *
   * @param physics_name A name for the physics module instance
   * @param mesh The serac Mesh
   */
  SolidDfemResidual(std::string physics_name, std::shared_ptr<Mesh> mesh, const mfem::ParFiniteElementSpace& test_space,
                    const std::vector<const mfem::ParFiniteElementSpace*>& parameter_fe_spaces = {})
      : DfemResidual(physics_name, mesh, test_space, makeInputSpaces(test_space, mesh, parameter_fe_spaces))
  {
  }

  /**
   * @brief Set the material stress response and mass properties for the physics module
   *
   * @tparam MaterialType The solid material type
   * @tparam StateType the type that contains the internal variables for MaterialType
   * @param body_name string name for a registered body Domain on the mesh
   * @param material A material that provides a function to evaluate stress
   * @pre material must be a object that can be called with the following arguments:
   *    1. `MaterialType::State & state` an mutable reference to the internal variables for this quadrature point
   *    2. `tensor<T,dim,dim> du_dx` the displacement gradient at this quadrature point
   *    3. `tuple{value, derivative}`, a tuple of values and derivatives for each parameter field
   *            specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @param qdata the buffer of material internal variables at each quadrature point
   *
   * @pre MaterialType must have a public method `density` which can take FiniteElementState parameter inputs
   * @pre MaterialType must have a public method 'pkStress' which returns the first Piola-Kirchhoff stress
   *
   */
  template <typename MaterialType, typename... ParameterTypes>
  void setMaterial(const mfem::Array<int>& domain_attributes, const MaterialType& material,
                   const mfem::IntegrationRule& displacement_ir)
  {
    auto stress_div_integral = StressDivQFunction<MaterialType, double>{.material = material};
    mfem::future::tuple<mfem::future::Gradient<DISP>, mfem::future::Gradient<VELO>, mfem::future::Gradient<ACCEL>,
                        mfem::future::Gradient<COORD>, mfem::future::Weight, ParameterTypes...>
        stress_div_integral_inputs{};
    mfem::future::tuple<mfem::future::Gradient<DISP>> stress_div_integral_outputs{};
    DfemResidual::addBodyIntegral(domain_attributes, stress_div_integral, stress_div_integral_inputs,
                                  stress_div_integral_outputs, displacement_ir, std::index_sequence<DISP>{});

    if constexpr (!IsQuasiStatic) {
      auto acceleration_integral = AccelerationQFunction<MaterialType, double>{.material = material};
      mfem::future::tuple<mfem::future::Value<DISP>, mfem::future::Value<VELO>, mfem::future::Value<ACCEL>,
                          mfem::future::Gradient<COORD>, mfem::future::Weight, ParameterTypes...>
          acceleration_integral_inputs{};
      mfem::future::tuple<mfem::future::Value<ACCEL>> acceleration_integral_outputs{};
      if constexpr (UseLumpedMass) {
        SLIC_ERROR_IF(DfemResidual::input_mfem_spaces_[DISP]->IsVariableOrder(),
                      "Lumped mass matrix is not supported for variable order finite element spaces.");
        auto& mesh = DfemResidual::mesh_->mfemParMesh();
        SLIC_ERROR_IF(mesh.GetNumGeometries(mesh.Dimension()) != 1 ||
                          !(mesh.HasGeometry(mfem::Geometry::SQUARE) || mesh.HasGeometry(mfem::Geometry::CUBE)),
                      "Lumped mass matrix is only supported for 2D and 3D meshes with square or cubic elements.");
        // use lumped mass matrix via integration rule at nodes
        auto& fe_coll = *DfemResidual::input_mfem_spaces_[DISP]->FEColl();
        mfem::IntegrationRule rule_1d;
        mfem::QuadratureFunctions1D::GaussLobatto(fe_coll.GetOrder() + 1, &rule_1d);
        auto spatial_dim = DfemResidual::input_mfem_spaces_[DISP]->GetVDim();
        switch (spatial_dim) {
          case 1:
            nodal_ir_ = std::make_unique<mfem::IntegrationRule>(rule_1d);
            break;
          case 2:
            nodal_ir_ = std::make_unique<mfem::IntegrationRule>(rule_1d, rule_1d);
            break;
          case 3:
            nodal_ir_ = std::make_unique<mfem::IntegrationRule>(rule_1d, rule_1d, rule_1d);
            break;
          default:
            SLIC_ERROR_ROOT("Unsupported number of dimensions for nodal integration rule.");
        }
        DfemResidual::addBodyIntegral(domain_attributes, acceleration_integral, acceleration_integral_inputs,
                                      acceleration_integral_outputs, *nodal_ir_, std::index_sequence<ACCEL>{});
      } else {
        // use consistent mass matrix
        DfemResidual::addBodyIntegral(domain_attributes, acceleration_integral, acceleration_integral_inputs,
                                      acceleration_integral_outputs, displacement_ir, std::index_sequence<ACCEL>{});
      }
    }
  }

  void massMatrix(const std::vector<ConstFieldPtr>& fields, const mfem::Vector& direction_t,
                  mfem::Vector& result_t) const
  {
    static_assert(!IsQuasiStatic, "Mass matrix is not defined for quasi-static solid mechanics problems.");
    std::vector<mfem::Vector*> sol_fields({&fields[0]->gridFunction()});
    std::vector<mfem::Vector*> other_fields;
    other_fields.reserve(fields.size() - 1);
    for (size_t i = 1; i < fields.size(); ++i) {
      other_fields.push_back(&fields[i]->gridFunction());
    }
    auto deriv_op = DfemResidual::differentiable_operators_[0]->GetDerivative(ACCEL, sol_fields, other_fields);
    deriv_op->Mult(direction_t, result_t);
  }

 protected:
  std::unique_ptr<mfem::IntegrationRule> nodal_ir_;

 private:
  std::vector<const mfem::ParFiniteElementSpace*> makeInputSpaces(
      const mfem::ParFiniteElementSpace& test_space, const std::shared_ptr<Mesh>& mesh,
      const std::vector<const mfem::ParFiniteElementSpace*>& parameter_fe_spaces)
  {
    std::vector<const mfem::ParFiniteElementSpace*> input_spaces;
    input_spaces.reserve(4 + parameter_fe_spaces.size());
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

}  // namespace serac
