#include "serac/differentiable_numerics/field_state.hpp"

namespace serac {

FieldState weighted_sum(std::vector<double> weights, std::vector<FieldState> weighted_fields,
                        std::vector<gretl::State<double>> differentiable_weights,
                        std::vector<FieldState> differentiably_weighted_fields)
{
  SLIC_ERROR_IF(weights.size() != weighted_fields.size(),
                "weights and the fields they are weighting do not match in size");
  SLIC_ERROR_IF(differentiable_weights.size() != differentiably_weighted_fields.size(),
                "differentiable weights and the fields they are weighting do not match in size");
  SLIC_ERROR_IF((weights.size() == 0) && (differentiable_weights.size() == 0),
                "At least 1 weight must be passed to a weighted sum");

  std::vector<gretl::StateBase> inputs;
  inputs.insert(inputs.end(), weighted_fields.begin(), weighted_fields.end());
  inputs.insert(inputs.end(), differentiable_weights.begin(), differentiable_weights.end());
  inputs.insert(inputs.end(), differentiably_weighted_fields.begin(), differentiably_weighted_fields.end());

  auto x = weights.size() ? weighted_fields[0] : differentiably_weighted_fields[0];
  auto z = x.clone(inputs);

  z.set_eval([weights](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    size_t num_weights = weights.size();
    size_t num_diffable_weights = (upstreams.size() - num_weights) / 2;

    auto X = weights.size() ? upstreams[0].get<FEFieldPtr>()
                            : upstreams[num_weights + num_diffable_weights].get<FEFieldPtr>();

    FEFieldPtr Z = std::make_shared<FiniteElementState>(X->space(), "weighted_sum");

    if (num_weights > 0) {
      double weightOld = weights[0];
      auto vecOld = upstreams[0].get<FEFieldPtr>();
      if (num_weights == 1) {
        Z->Set(weightOld, *vecOld);
      }
      for (size_t i = 1; i < num_weights; ++i) {
        double weightNew = weights[i];
        add(weightOld, *vecOld, weightNew, *upstreams[i].get<FEFieldPtr>(), *Z);
        weightOld = 1.0;
        vecOld = Z;
      }
    }

    if (num_diffable_weights > 0) {
      size_t start_index = 0;
      double weightOld = 1.0;
      FEFieldPtr vecOld = Z;

      if (weights.size() == 0) {
        start_index = 1;
        weightOld = upstreams[num_weights].get<double>();
        vecOld = upstreams[num_weights + num_diffable_weights].get<FEFieldPtr>();
        if (num_diffable_weights == 1) {
          Z->Set(weightOld, *vecOld);
        }
      }

      for (size_t i = start_index; i < num_diffable_weights; ++i) {
        double weightNew = upstreams[num_weights + i].get<double>();
        add(weightOld, *vecOld, weightNew, *upstreams[num_weights + num_diffable_weights + i].get<FEFieldPtr>(), *Z);
        weightOld = 1.0;
        vecOld = Z;
      }
    }

    downstream.set<FEFieldPtr, FEDualPtr>(Z);
  });

  z.set_vjp([weights](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    size_t num_weights = weights.size();
    size_t num_diffable_weights = (upstreams.size() - num_weights) / 2;

    const FEDualPtr& Z_dual = downstream.get_dual<FEDualPtr, FEFieldPtr>();

    for (size_t i = 0; i < num_weights; ++i) {
      FEDualPtr& V_dual = upstreams[i].get_dual<FEDualPtr, FEFieldPtr>();
      double weight = weights[i];
      add(*V_dual, weight, *Z_dual, *V_dual);
    }

    for (size_t i = 0; i < num_diffable_weights; ++i) {
      double& weight_dual = upstreams[num_weights + i].get_dual<double, double>();
      FEDualPtr& V_dual = upstreams[num_weights + num_diffable_weights + i].get_dual<FEDualPtr, FEFieldPtr>();
      double weight = upstreams[num_weights + i].get<double>();
      FEFieldPtr V = upstreams[num_weights + num_diffable_weights + i].get<FEFieldPtr>();
      add(*V_dual, weight, *Z_dual, *V_dual);
      weight_dual += serac::innerProduct(*Z_dual, *V);
    }
  });

  return z.finalize();
}

FieldStateWeightedSum operator*(double a, const FieldState& b) { return FieldStateWeightedSum({a},{b}); }

FieldStateWeightedSum operator*(const FieldState& b, double a) { return a * b; }

FieldStateWeightedSum operator*(const gretl::State<double>& a, const FieldState& b) { return FieldStateWeightedSum({a},{b}); }

FieldStateWeightedSum operator*(const FieldState& b, const gretl::State<double>& a) { return a * b; }

FieldStateWeightedSum operator+(const FieldStateWeightedSum& ax, const FieldStateWeightedSum& by)
{
 FieldStateWeightedSum c = ax;
 return c += by;
};

FieldStateWeightedSum operator+(const FieldStateWeightedSum& ax, const FieldState& y)
{
 FieldStateWeightedSum y1({1.0}, {y});
 return ax + y1;
};

FieldStateWeightedSum operator+(const FieldState& y, const FieldStateWeightedSum& ax)
{
  return ax + y;
};

}