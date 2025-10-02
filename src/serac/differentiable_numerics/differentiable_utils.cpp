#include "serac/differentiable_numerics/differentiable_utils.hpp"
#include "serac/gretl/data_store.hpp"

namespace serac {

DoubleState evaluate_objective(const TimeInfo& time_info, const FieldState& shape_disp,
                               const std::vector<FieldState>& inputs, const ScalarObjective* objective)
{
  std::vector<gretl::StateBase> all_states{shape_disp};
  all_states.insert(all_states.end(), inputs.begin(), inputs.end());
  DoubleState value =
      shape_disp.create_state<double, double>(all_states, gretl::defaultInitializeZeroDual<double, double>{});

  value.set_eval([time_info, objective](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    auto shape = upstreams[0].get<FEFieldPtr>();
    std::vector<FEFieldPtr> Inputs;
    for (size_t i = 1; i < upstreams.size(); ++i) {
      Inputs.push_back(upstreams[i].get<FEFieldPtr>());
    }
    downstream.set<double, double>(objective->evaluate(time_info, shape.get(), getConstFieldPointers(Inputs)));
  });

  value.set_vjp([time_info, objective](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    auto shape = upstreams[0].get<FEFieldPtr>();
    std::vector<FEFieldPtr> fields;
    for (size_t i = 1; i < upstreams.size(); ++i) {
      fields.push_back(upstreams[i].get<FEFieldPtr>());
    }

    const double downstream_dual = downstream.get_dual<double, double>();

    upstreams[0].get_dual<FEDualPtr, FEFieldPtr>()->Add(
        downstream_dual, objective->mesh_coordinate_gradient(time_info, shape.get(), getConstFieldPointers(fields)));
    for (size_t i = 1; i < upstreams.size(); ++i) {
      upstreams[i].get_dual<FEDualPtr, FEFieldPtr>()->Add(
          downstream_dual, objective->gradient(time_info, shape.get(), getConstFieldPointers(fields), i - 1));
    }
  });

  return value.finalize();
}

}  // namespace serac