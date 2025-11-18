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

#ifdef SMITH_USE_DFEM

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

template <typename Material, typename... Parameters>
struct StressDivQFunction {
  SMITH_HOST_DEVICE inline auto operator()(
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
