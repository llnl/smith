// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file dfem_solid_weak_form.hpp
 *
 * @brief Implements the WeakForm interface for solid mechanics physics using dFEM. Derives from DfemWeakForm.
 */

#pragma once

#include "serac/serac_config.hpp"

#ifdef SERAC_USE_DFEM

#include "serac/physics/dfem_weak_form.hpp"

#include "serac/infrastructure/accelerator.hpp"

namespace serac {

template <int Idx>
struct ScalarParameter {
  static constexpr int index = Idx;
  using QFunctionInput = double;
  template <int FieldId>
  using QFunctionFieldOp = mfem::future::Value<FieldId>;
};

template <int Idx, int NumVars>
struct InternalVariableParameter {
  static constexpr int index = Idx;
  using QFunctionInput = mfem::future::tensor<mfem::real_t, NumVars>;
  template <int FieldId>
  using QFunctionFieldOp = mfem::future::Identity<FieldId>;
};

template <typename Material, typename... Parameters>
struct StressDivQFunction {
  SERAC_HOST_DEVICE inline auto operator()(
      // mfem::real_t dt, // TODO: figure out how to pass this in
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& du_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dv_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>&,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dX_dxi, mfem::real_t weight,
      Parameters::QFunctionInput... params) const
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
      Parameters::QFunctionInput... params) const
  {
    auto rho = material.density(params...);
    auto J = mfem::future::det(dX_dxi) * weight;
    return mfem::future::tuple{-rho * a * J};
  }

  Material material;  ///< the material model to use for computing the density
};

/**
 * @brief The weak form for solid mechanics
 *
 * This uses dFEM to compute the solid mechanics residuals and tangent stiffness matrices.
 */
template <bool IsQuasiStatic = false, bool UseLumpedMass = false>
class DfemSolidWeakForm : public DfemWeakForm {
 public:
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
   * @param mesh The serac Mesh
   * @param test_space Test space
   * @param parameter_fe_spaces Vector of parameter finite element spaces
   * @param parameter_quadrature_spaces Vector of parameter quadrature spaces
   */
  DfemSolidWeakForm(std::string physics_name, std::shared_ptr<Mesh> mesh, const mfem::ParFiniteElementSpace& test_space,
                    const std::vector<const mfem::ParFiniteElementSpace*>& parameter_fe_spaces = {},
                    const std::vector<const mfem::future::ParameterSpace*> parameter_quadrature_spaces = {})
      : DfemWeakForm(physics_name, mesh, test_space, makeInputSpaces(test_space, mesh, parameter_fe_spaces),
                     parameter_quadrature_spaces)
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
                        typename ParameterTypes::template QFunctionFieldOp<NUM_STATES + ParameterTypes::index>...>
        stress_div_integral_inputs{};
    mfem::future::tuple<mfem::future::Gradient<NUM_STATES + sizeof...(ParameterTypes)>> stress_div_integral_outputs{};
    DfemWeakForm::addBodyIntegral(domain_attributes, stress_div_integral, stress_div_integral_inputs,
                                  stress_div_integral_outputs, displacement_ir,
                                  std::index_sequence<DISPLACEMENT, NUM_STATES + ParameterTypes::index...>{});

    if constexpr (!IsQuasiStatic) {
      auto acceleration_integral = AccelerationQFunction<MaterialType, ParameterTypes...>{.material = material};
      mfem::future::tuple<mfem::future::Value<DISPLACEMENT>, mfem::future::Value<VELOCITY>,
                          mfem::future::Value<ACCELERATION>, mfem::future::Gradient<COORDINATES>, mfem::future::Weight,
                          typename ParameterTypes::template QFunctionFieldOp<NUM_STATES + ParameterTypes::index>...>
          acceleration_integral_inputs{};
      mfem::future::tuple<mfem::future::Value<NUM_STATES + sizeof...(ParameterTypes)>> acceleration_integral_outputs{};
      if constexpr (UseLumpedMass) {
        SLIC_ERROR_IF(DfemWeakForm::input_mfem_spaces_[DISPLACEMENT]->IsVariableOrder(),
                      "Lumped mass matrix is not supported for variable order finite element spaces.");
        auto& mesh = DfemWeakForm::mesh_->mfemParMesh();
        SLIC_ERROR_IF(mesh.GetNumGeometries(mesh.Dimension()) != 1 ||
                          !(mesh.HasGeometry(mfem::Geometry::SQUARE) || mesh.HasGeometry(mfem::Geometry::CUBE)),
                      "Lumped mass matrix is only supported for 2D and 3D meshes with square or cubic elements.");
        // use lumped mass matrix via integration rule at nodes
        auto& fe_coll = *DfemWeakForm::input_mfem_spaces_[DISPLACEMENT]->FEColl();
        mfem::IntegrationRule rule_1d;
        mfem::QuadratureFunctions1D::GaussLobatto(fe_coll.GetOrder() + 1, &rule_1d);
        auto spatial_dim = DfemWeakForm::input_mfem_spaces_[DISPLACEMENT]->GetVDim();
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
        DfemWeakForm::addBodyIntegral(domain_attributes, acceleration_integral, acceleration_integral_inputs,
                                      acceleration_integral_outputs, *nodal_ir_, std::index_sequence<ACCELERATION>{});
      } else {
        // use consistent mass matrix
        DfemWeakForm::addBodyIntegral(domain_attributes, acceleration_integral, acceleration_integral_inputs,
                                      acceleration_integral_outputs, displacement_ir,
                                      std::index_sequence<ACCELERATION>{});
      }
    }
  }

  void massMatrix(const std::vector<ConstFieldPtr>& fields,
    const mfem::Vector& direction_t, mfem::Vector& result_t) const
  {
    static_assert(!IsQuasiStatic, "Mass matrix is not defined for quasi-static solid mechanics problems.");
    // Pass empty quad field vector, assuming mass matrix cannot depend on internal state variables
    auto deriv_op = DfemWeakForm::weak_form_.GetDerivative(ACCELERATION, {&fields[0]->gridFunction()},
                                                           DfemWeakForm::getLVectors(fields, {}));
    deriv_op->Mult(direction_t, result_t);
  }

 protected:
  std::unique_ptr<mfem::IntegrationRule> nodal_ir_;

 private:
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

#endif
