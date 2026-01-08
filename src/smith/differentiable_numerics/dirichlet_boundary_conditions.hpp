// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file dirichlet_boundary_conditions.hpp
 *
 * @brief Contains DirichletBoundaryConditions class for interaction with the differentiable solve interfaces
 */

#pragma once

#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"

namespace smith {

class Mesh;

/// @brief A generic class for setting Dirichlet boundary conditions on arbitrary physics
class DirichletBoundaryConditions {
 public:
  /// @brief Construct from mfem::ParMesh
  DirichletBoundaryConditions(const mfem::ParMesh& mfem_mesh, mfem::ParFiniteElementSpace& space);

  /// @brief Construct from smith::Mesh
  DirichletBoundaryConditions(const Mesh& mesh, mfem::ParFiniteElementSpace& space);

  /// @brief Specify time and space varying Dirichlet boundary conditions over a domain.
  /// @param domain All dofs in this domain have boundary conditions applied to it.
  /// @param components vectors of computents.  The applied_displacement function returns the full vector, this
  /// specifies which subset of those should have dirichlet boundary conditions applied.  direction to apply boundary
  /// condition to if the underlying field is a vector-field.
  /// @param applied_displacement applied_displacement is a functor which takes time, and a
  /// smith::tensor<double,spatial_dim> corresponding to the spatial coordinate.  The functor must return a
  /// smith::Tensor<double,field_dim>, where field_dim is the dimension of the vector space for the field.  For example:
  /// [](double t, smith::tensor<double, dim> X) { return smith::tensor<double,2>{}; }
  template <int spatial_dim, typename AppliedDisplacementFunction>
  void setVectorBCs(const Domain& domain, std::vector<int> components, AppliedDisplacementFunction applied_displacement)
  {
    int field_dim = space_.GetVDim();
    for (auto component : components) {
      SLIC_ERROR_IF(component >= field_dim || component < 0,
                    axom::fmt::format("Trying to set boundary conditions on a field with dim {}, using component {}",
                                      field_dim, component));
      auto mfem_coefficient_function = [applied_displacement, component](const mfem::Vector& X_mfem, double t) {
        auto X = make_tensor<spatial_dim>([&X_mfem](int k) { return X_mfem[k]; });
        return applied_displacement(t, X)[component];
      };

      auto dof_list = domain.dof_list(&space_);
      // scalar ldofs -> vector ldofs
      space_.DofsToVDofs(static_cast<int>(component), dof_list);

      auto component_disp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(mfem_coefficient_function);
      bcs_.addEssential(dof_list, component_disp_bdr_coef_, space_, static_cast<int>(component));
    }
  }

  /// @overload
  template <int spatial_dim, typename AppliedDisplacementFunction>
  void setVectorBCs(const Domain& domain, AppliedDisplacementFunction applied_displacement)
  {
    const int field_dim = space_.GetVDim();
    std::vector<int> components(static_cast<size_t>(field_dim));
    for (int component = 0; component < field_dim; ++component) {
      components[static_cast<size_t>(component)] = component;
    }
    setVectorBCs<spatial_dim>(domain, components, applied_displacement);
  }

  /// @brief Specify time and space varying Dirichlet boundary conditions over a domain.
  /// @param domain All dofs in this domain have boundary conditions applied to it.
  /// @param component component direction to apply boundary condition to if the underlying field is a vector-field.
  /// @param applied_displacement applied_displacement is a functor which takes time, and a
  /// smith::tensor<double,spatial_dim> corresponding to the spatial coordinate.  The functor must return a double.  For
  /// example: [](double t, smith::tensor<double, dim> X) { return 1.0; }
  template <int spatial_dim, typename AppliedDisplacementFunction>
  void setVectorBCs(const Domain& domain, int component, AppliedDisplacementFunction applied_displacement)
  {
    const int field_dim = space_.GetVDim();
    SLIC_ERROR_IF(component >= field_dim,
                  axom::fmt::format("Trying to set boundary conditions on a field with dim {}, using component {}",
                                    field_dim, component));
    auto mfem_coefficient_function = [applied_displacement](const mfem::Vector& X_mfem, double t) {
      auto X = make_tensor<spatial_dim>([&X_mfem](int k) { return X_mfem[k]; });
      return applied_displacement(t, X);
    };

    auto dof_list = domain.dof_list(&space_);
    // scalar ldofs -> vector ldofs
    space_.DofsToVDofs(component, dof_list);

    auto component_disp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(mfem_coefficient_function);
    bcs_.addEssential(dof_list, component_disp_bdr_coef_, space_, component);
  }

  /// @brief Specify time and space varying Dirichlet boundary conditions over a domain.
  /// @param domain All dofs in this domain have boundary conditions applied to it.
  /// @param applied_displacement applied_displacement is a functor which takes time, and a
  /// smith::tensor<double,spatial_dim> corresponding to the spatial coordinate.  The functor must return a double.  For
  /// example: [](double t, smith::tensor<double, dim> X) { return 1.0; }
  template <int spatial_dim, typename AppliedDisplacementFunction>
  void setScalarBCs(const Domain& domain, AppliedDisplacementFunction applied_displacement)
  {
    setScalarBCs<spatial_dim>(domain, 0, applied_displacement);
  }

  /// @brief Constrain the dofs of a scalar field over a domain
  template <int spatial_dim>
  void setFixedScalarBCs(const Domain& domain)
  {
    setVectorBCs<spatial_dim>(domain, [](auto, auto) { return 0.0; });
  }

  /// @brief Constrain the dofs of a scalar field over a domain
  template <int spatial_dim>
  void setFixedVectorBCs(const Domain& domain, int component)
  {
    setScalarBCs<spatial_dim>(domain, component, [](auto, auto) { return 0.0; });
  }

  /// @brief Constrain the vector dofs over a domain corresponding to a subset of the vector components
  template <int spatial_dim>
  void setFixedVectorBCs(const Domain& domain, std::vector<int> components)
  {
    setVectorBCs<spatial_dim>(domain, components, [](auto, auto) { return smith::tensor<double, spatial_dim>{}; });
  }

  /// @brief Constrain all the vector dofs over a domain
  template <int spatial_dim>
  void setFixedVectorBCs(const Domain& domain)
  {
    const int field_dim = space_.GetVDim();
    SLIC_ERROR_IF(field_dim != spatial_dim,
                  "Vector boundary conditions current only work if they match the spatial dimension");
    std::vector<int> components(static_cast<size_t>(field_dim));
    for (int component = 0; component < field_dim; ++component) {
      components[static_cast<size_t>(component)] = component;
    }
    setFixedVectorBCs<spatial_dim>(domain, components);
  }

  /// @brief Return the smith BoundaryConditionManager
  const smith::BoundaryConditionManager& getBoundaryConditionManager() const { return bcs_; }

 private:
  smith::BoundaryConditionManager bcs_;  ///< boundary condition manager that does the heavy lifting
  mfem::ParFiniteElementSpace& space_;   ///< save the space for the field which will be constrained
};

}  // namespace smith
