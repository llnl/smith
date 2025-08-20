
#include "serac/physics/weak_form.hpp"
#include "serac/differentiable_numerics/field_state.hpp"
#include "serac/differentiable_numerics/lumped_mass_weak_form.hpp"
#include "serac/physics/boundary_conditions/boundary_condition_manager.hpp"

namespace serac {

inline std::vector<FieldPtr> getResidualFieldPtrs(const std::vector<FEFieldPtr>& shared_ptrs)
{
  std::vector<FieldPtr> field_ptrs;
  for (auto& s : shared_ptrs) {
    field_ptrs.push_back(s.get());
  }
  return field_ptrs;
}

inline FieldState computeLumpedMass(const WeakForm* mass_residual_eval, const FieldState& shape_u,
                                    const FieldState& lumped_field, const FieldState& rho)
{
  std::vector<gretl::StateBase> inputs{shape_u, rho, lumped_field};

  FieldState z = lumped_field.clone(inputs);

  z.set_eval([=](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const FEFieldPtr& ShapeDisp = upstreams[0].get<FEFieldPtr>();
    const FEFieldPtr& Rho = upstreams[1].get<FEFieldPtr>();
    const FEFieldPtr& LumpedOutputFieldForSizing = upstreams[2].get<FEFieldPtr>();

    FEFieldPtr Z = std::make_shared<FiniteElementState>(
        LumpedOutputFieldForSizing->space(),
        "diag_mass");  // create a pointer to a new FE space for our new values to live in

    auto m_diagonal = mass_residual_eval->residual(
        0.0, 0.0, ShapeDisp.get(),
        getConstFieldPointers(Rho));  // diagonal of the diagonalized lumped mass matrix, in mfem vector format
    *Z = m_diagonal;

    downstream.set<FEFieldPtr, FEDualPtr>(Z);  // Set the output to our new values
  });

  z.set_vjp([=](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    const FEDualPtr& Z_dual = downstream.get_dual<FEDualPtr>();
    const FEFieldPtr& ShapeDisp = upstreams[0].get<FEFieldPtr>();  // shape disp tate
    const FEFieldPtr& Rho = upstreams[1].get<FEFieldPtr>();        // density parameter state
    FEDualPtr& Shape_dual = upstreams[0].get_dual<FEDualPtr>();   // dual of shape parameter state
    FEDualPtr& Rho_dual = upstreams[1].get_dual<FEDualPtr>();     // dual of density parameter state

    FiniteElementState Z_dual_state(Z_dual->space(), Z_dual->name());
    Z_dual_state = *Z_dual;

    mass_residual_eval->vjp(0.0, 0.0, ShapeDisp.get(), getConstFieldPointers(Rho), {},
                            getConstFieldPointers(Z_dual_state), Shape_dual.get(), getFieldPointers(Rho_dual), {});
  });

  return z.finalize();
}

inline FieldState diagInverse(const FieldState& x)
{
  auto z = x.clone({x});
  z.set_eval([](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const FEFieldPtr& X = upstreams[0].get<FEFieldPtr>();
    FEFieldPtr Z = std::make_shared<FiniteElementState>(X->space(), "diag_inverse");
    FiniteElementState& X_ = *upstreams[0].get<FEFieldPtr>();
    auto Z_ = *Z;
    int sz = X_.Size();
    for (int index = 0; index < sz; index++) {
      Z_[index] = 1 / X_[index];
    }
    *Z = Z_;
    downstream.set<FEFieldPtr, FEDualPtr>(Z);
  });

  z.set_vjp([](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    FiniteElementDual& Z_dual = *downstream.get_dual<FEDualPtr>();
    FiniteElementDual& X_dual = *upstreams[0].get_dual<FEDualPtr>();
    FiniteElementState& X_ = *upstreams[0].get<FEFieldPtr>();
    int sz = X_.Size();
    for (int index = 0; index < sz; index++) {
      X_dual[index] += -1 * Z_dual[index] / (std::pow(X_[index], 2));
    }
  });

  return z.finalize();
}

inline FieldState evalResidual(const WeakForm* residual_eval, FieldState shape_disp,
                               const std::vector<FieldState>& states, const std::vector<FieldState>& params,
                               double time, double dt, size_t inertial_index)
{
  std::vector<gretl::StateBase> allStateBases;
  for (auto& s : states) allStateBases.push_back(s);
  for (auto& p : params) allStateBases.push_back(p);
  allStateBases.push_back(shape_disp);
  auto z = states[inertial_index].clone(allStateBases);

  z.set_eval([=](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
    SERAC_MARK_FUNCTION;

    size_t num_fields = inputs.size() - 1;          // get the number of non-shapedisp input fields
    std::vector<ConstFieldPtr> fields(num_fields);  // set up fields vector

    // Convert from gretl FieldState to FiniteElementState pointers, stored in the FEFieldPointer array corrected_fields
    // Fields should be in the order shape_u, u, v, a
    for (size_t field_index = 0; field_index < num_fields; ++field_index) {
      fields[field_index] = inputs[field_index].get<FEFieldPtr>().get();
    }

    FEFieldPtr R =
        std::make_shared<FiniteElementState>(fields[inertial_index]->space(), "residual");  // set up output pointer

    // set the acceleration field equal to zero here, so that when we evaluate the residual, we get zero contribution
    // from the acceleration and mass, because acceleration and mass are being accounted for elsewhere
    FiniteElementState zero_accel(*fields[inertial_index]);
    fields[inertial_index] = &zero_accel;

    // evaluate the residual with zero acceleration contribution
    *R = residual_eval->residual(time, dt, inputs[num_fields].get<FEFieldPtr>().get(), fields);

    output.set<FEFieldPtr, FEDualPtr>(R);
  });

  z.set_vjp([=](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
    SERAC_MARK_FUNCTION;

    const FEFieldPtr Z = output.get<FEFieldPtr>();
    const FEDualPtr Z_dual = output.get_dual<FEDualPtr>();
    FiniteElementState Z_dual_state(Z_dual->space(), Z_dual->name());
    Z_dual_state = *Z_dual;

    // get the input values and store them in corrected_fields
    size_t num_fields = inputs.size() - 1;
    std::vector<ConstFieldPtr> fields(num_fields);
    for (size_t field_index = 0; field_index < num_fields; ++field_index) {
      fields[field_index] = inputs[field_index].get<FEFieldPtr>().get();
    }

    // set the acceleration field equal to zero here, so that when we evaluate the residual, we get zero contribution
    // from the acceleration and mass, because acceleration and mass are being accounted for elsewhere
    FiniteElementState zero_accel(*fields[inertial_index]);
    fields[inertial_index] = &zero_accel;

    std::vector<DualFieldPtr> field_sensitivities(num_fields);
    for (size_t field_index = 0; field_index < num_fields; ++field_index) {
      field_sensitivities[field_index] = inputs[field_index].get_dual<FEDualPtr>().get();
    }
    // setting the field sensitivity to nullptr means if will not be computed in the vjp call
    field_sensitivities[inertial_index] = nullptr;

    auto shape_disp_ptr = inputs[num_fields].get<FEFieldPtr>();
    auto shape_disp_sensitivity_ptr = inputs[num_fields].get_dual<FEDualPtr>();

    // set the dual fields for each input, using the call to residual that pulls the derivative
    residual_eval->vjp(time, dt, shape_disp_ptr.get(), fields, {}, getConstFieldPointers(Z_dual_state),
                       shape_disp_sensitivity_ptr.get(), field_sensitivities, {});
  });

  return z.finalize();
}

inline FieldState componentWiseMult(const FieldState& x, const FieldState& y,
                                    const BoundaryConditionManager* bc_manager)
{
  SLIC_ERROR_IF(x.get()->Size() != y.get()->Size(), "Trying to component wise multiple vectors with different sizes");
  auto z = x.clone({x, y});

  z.set_eval([=](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const FEFieldPtr& X = upstreams[0].get<FEFieldPtr>();
    FEFieldPtr Z = std::make_shared<FiniteElementState>(X->space(), "ComponentMult");
    const FiniteElementState& X_ = *upstreams[0].get<FEFieldPtr>();
    const FiniteElementState& Y_ = *upstreams[1].get<FEFieldPtr>();
    auto Z_ = *Z;

    int sz = X_.Size();
    for (int index = 0; index < sz; index++) {
      Z_[index] = X_[index] * Y_[index];
    }

    *Z = Z_;

    // enforce zero acceleration at fixed BCs
    if (bc_manager) {
      Z->SetSubVector(bc_manager->allEssentialTrueDofs(), 0.0);
    }

    downstream.set<FEFieldPtr, FEDualPtr>(Z);
  });

  z.set_vjp([=](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    auto Z_dual = *downstream.get_dual<FEDualPtr>();
    const FiniteElementState& X = *upstreams[0].get<FEFieldPtr>();
    const FiniteElementState& Y = *upstreams[1].get<FEFieldPtr>();
    FiniteElementDual& X_dual = *upstreams[0].get_dual<FEDualPtr>();
    FiniteElementDual& Y_dual = *upstreams[1].get_dual<FEDualPtr>();

    // enforce zero acceleration at fixed BCs
    if (bc_manager) {
      Z_dual.SetSubVector(bc_manager->allEssentialTrueDofs(), 0.0);
    }

    int sz = X.Size();
    for (int index = 0; index < sz; index++) {
      X_dual[index] += Z_dual[index] * Y[index];
      Y_dual[index] += Z_dual[index] * X[index];
    }
  });

  return z.finalize();
}

inline FieldState negativeComponentWiseMult(const FieldState& x, const FieldState& y,
                                            const BoundaryConditionManager* bc_manager)
{
  SLIC_ERROR_IF(x.get()->Size() != y.get()->Size(), "Trying to component wise multiple vectors with different sizes");
  auto z = x.clone({x, y});

  z.set_eval([=](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const FEFieldPtr& X = upstreams[0].get<FEFieldPtr>();
    FEFieldPtr Z = std::make_shared<FiniteElementState>(X->space(), "ComponentMult");
    const FiniteElementState& X_ = *upstreams[0].get<FEFieldPtr>();
    const FiniteElementState& Y_ = *upstreams[1].get<FEFieldPtr>();
    auto Z_ = *Z;

    int sz = X_.Size();
    for (int index = 0; index < sz; index++) {
      Z_[index] = -X_[index] * Y_[index];
    }

    *Z = Z_;

    // enforce zero acceleration at fixed BCs
    if (bc_manager) {
      Z->SetSubVector(bc_manager->allEssentialTrueDofs(), 0.0);
    }

    downstream.set<FEFieldPtr, FEDualPtr>(Z);
  });

  z.set_vjp([=](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    auto Z_dual = *downstream.get_dual<FEDualPtr>();
    const FiniteElementState& X = *upstreams[0].get<FEFieldPtr>();
    const FiniteElementState& Y = *upstreams[1].get<FEFieldPtr>();
    FiniteElementDual& X_dual = *upstreams[0].get_dual<FEDualPtr>();
    FiniteElementDual& Y_dual = *upstreams[1].get_dual<FEDualPtr>();

    // enforce zero acceleration at fixed BCs
    if (bc_manager) {
      Z_dual.SetSubVector(bc_manager->allEssentialTrueDofs(), 0.0);
    }

    int sz = X.Size();
    for (int index = 0; index < sz; index++) {
      X_dual[index] -= Z_dual[index] * Y[index];
      Y_dual[index] -= Z_dual[index] * X[index];
    }
  });

  return z.finalize();
}

}  // namespace serac
