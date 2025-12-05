// Copyright (c), Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file reaction.hpp
 *
 * @brief Reaction class which is a names combination of a weak form and a set of dirichlet constrained nodes.
 */

#pragma once

#include <string>
#include "smith/physics/weak_form.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"

namespace smith {

// inline DoubleState netReaction(const ResultantState& r, int dir, serac::Domain& domain) {
//   return gretl::create_state<double, double>(
//     gretl::defaultInitializeZeroDual<double, double>(),
//     [](FEDualPtr A) {

//       return smith::innerProduct(*A, *B);
//     },
//     [](FEDualPtr A, FEDualPtr B, double, FEFieldPtr& A_, FEFieldPtr& B_, double product_) {
//       A_->Add(product_, *B);
//       B_->Add(product_, *A);
//     },
//     a, b);
//   DoubleState f = r.create_state<double,double>({r}, default_)
// }

class Reaction {
 public:
  Reaction(std::shared_ptr<WeakForm> weak_form, std::shared_ptr<DirichletBoundaryConditions> bcs,
           size_t index_of_field_with_residual_space, std::string name)
      : weak_form_(weak_form),
        bcs_(bcs),
        index_of_field_with_residual_space_(index_of_field_with_residual_space),
        name_(name)
  {
  }

  /// @brief gretl-function implementation which evaluates the residual force (which is minus the mechanical force)
  /// given
  /// shape displacement, states and params.  The inertial index denotes which index in the state corresponds to the
  /// highest time derivative term (e.g., acceleration for solid mechanics).
  inline ResultantState evaluate(TimeInfo time_info, FieldState shape_disp, const std::vector<FieldState>& field_states)
  {
    size_t field_index_for_residual_size = index_of_field_with_residual_space_;
    std::shared_ptr<WeakForm> weak_form = weak_form_;

    std::cout << "weak forn = " << weak_form << std::endl;

    std::vector<gretl::StateBase> all_state_bases{shape_disp};
    for (auto& f : field_states) all_state_bases.push_back(f);

    std::cout << "all state, all field states = " << all_state_bases.size() << " " << field_states.size() << std::endl;

    printf("a\n");

    auto z = shape_disp.create_state<FEDualPtr, FEFieldPtr>(all_state_bases, zero_state_from_dual());

    printf("b\n");

    z.set_eval([=](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
      SMITH_MARK_FUNCTION;

      printf("c\n");

      size_t num_fields = inputs.size();
      std::vector<ConstFieldPtr> fields;
      fields.reserve(num_fields-1);  // set up fields vector

      printf("d\n");

      for (size_t field_index = 1; field_index < num_fields; ++field_index) {
        fields.push_back(inputs[field_index].get<FEFieldPtr>().get());
      }

      FEDualPtr R = std::make_shared<FiniteElementDual>(fields[field_index_for_residual_size]->space(),
                                                        "residual");  // set up output pointer

      // evaluate the residual with zero acceleration contribution
      std::cout << "time info = " << time_info.time() << std::endl;
      std::cout << "num fields = " << fields.size() << std::endl;
      std::cout << "shape disp name = " << inputs[0].get<FEFieldPtr>().get()->name() << std::endl;
      for (auto& f : fields) {
        std::cout << "name = " << f->name() << std::endl;
      }
      mfem::Vector tmp = weak_form->residual(time_info, inputs[0].get<FEFieldPtr>().get(), fields);
      *R = tmp;
      printf("e\n");
      output.set<FEDualPtr, FEFieldPtr>(R);
      printf("f\n");
    });

    z.set_vjp([=](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
      SMITH_MARK_FUNCTION;

      const FEDualPtr Z = output.get<FEDualPtr, FEFieldPtr>();
      const FEFieldPtr Z_dual = output.get_dual<FEFieldPtr, FEDualPtr>();
      FiniteElementState Z_dual_state(Z_dual->space(), Z_dual->name());
      Z_dual_state = *Z_dual;

      // get the input values and store them in corrected_fields
      size_t num_fields = inputs.size();
      std::vector<ConstFieldPtr> fields;
      fields.reserve(num_fields-1);  // set up fields vector
      for (size_t field_index = 1; field_index < num_fields; ++field_index) {
        fields.push_back(inputs[field_index].get<FEFieldPtr>().get());
      }

      std::vector<DualFieldPtr> field_sensitivities;
      field_sensitivities.reserve(num_fields);
      for (size_t field_index = 1; field_index < num_fields; ++field_index) {
        field_sensitivities.push_back(inputs[field_index].get_dual<FEDualPtr, FEFieldPtr>().get());
      }

      auto shape_disp_ptr = inputs[0].get<FEFieldPtr>();
      auto shape_disp_sensitivity_ptr = inputs[0].get_dual<FEDualPtr, FEFieldPtr>();

      // set the dual fields for each input, using the call to residual that pulls the derivative
      weak_form->vjp(time_info, shape_disp_ptr.get(), fields, {}, &Z_dual_state, shape_disp_sensitivity_ptr.get(),
                     field_sensitivities, {});
    });

    return z.finalize();
  }

  std::string name() const { return name_; }

  std::shared_ptr<WeakForm> weak_form_;
  std::shared_ptr<DirichletBoundaryConditions> bcs_;
  size_t index_of_field_with_residual_space_;
  std::string name_;
};

}  // namespace smith