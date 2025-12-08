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

class DirichletBoundaryConditions {
 public:
  DirichletBoundaryConditions(const mfem::ParMesh& mfem_mesh, mfem::ParFiniteElementSpace& space);

  DirichletBoundaryConditions(const Mesh& mesh, mfem::ParFiniteElementSpace& space);

  template <int spatial_dim, typename AppliedDisplacementFunction>
  void setVectorBCs(const Domain& domain, std::vector<int> components, AppliedDisplacementFunction applied_displacement)
  {
    int field_dim = space_.GetVDim();
    for (auto component : components) {
      SLIC_ERROR_IF(component >= field_dim,
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

  template <int spatial_dim, typename AppliedDisplacementFunction>
  void setScalarBCs(const Domain& domain, int component, AppliedDisplacementFunction applied_displacement)
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

  template <int spatial_dim, typename AppliedDisplacementFunction>
  void setScalarBCs(const Domain& domain, AppliedDisplacementFunction applied_displacement)
  {
    setScalarBCs<spatial_dim>(domain, 0, applied_displacement);
  }

  template <int spatial_dim>
  void setFixedScalarBCs(const Domain& domain, int component = 0)
  {
    setScalarBCs<spatial_dim>(domain, component, [](auto, auto) { return 0.0; });
  }

  template <int spatial_dim>
  void setFixedVectorBCs(const Domain& domain, std::vector<int> components)
  {
    setVectorBCs<spatial_dim>(domain, components, [](auto, auto) { return smith::tensor<double, spatial_dim>{}; });
  }

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

  const smith::BoundaryConditionManager& getBoundaryConditionManager() const { return bcs_; }

 private:
  smith::BoundaryConditionManager bcs_;
  mfem::ParFiniteElementSpace& space_;
};

}  // namespace smith