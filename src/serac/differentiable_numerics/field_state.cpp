#include "serac/differentiable_numerics/field_state.hpp"

namespace serac {

/// @brief axpby using State<double> and FieldState
FieldState axpby(const gretl::State<double>& a, const FieldState& x, const gretl::State<double>& b, const FieldState& y)
{
  auto z = x.clone({a, x, b, y});

  z.set_eval([](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    double A = upstreams[0].get<double>();
    const FEFieldPtr& X = upstreams[1].get<FEFieldPtr>();
    double B = upstreams[2].get<double>();
    const FEFieldPtr& Y = upstreams[3].get<FEFieldPtr>();

    FEFieldPtr Z = std::make_shared<FiniteElementState>(X->space(), "axpby");
    add(A, *X, B, *Y, *Z);
    downstream.set<FEFieldPtr, FEDualPtr>(Z);
  });

  z.set_vjp([](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    double A = upstreams[0].get<double>();
    const FEFieldPtr& X = upstreams[1].get<FEFieldPtr>();
    double B = upstreams[2].get<double>();
    const FEFieldPtr& Y = upstreams[3].get<FEFieldPtr>();

    const FEDualPtr& Z_dual = downstream.get_dual<FEDualPtr, FEFieldPtr>();
    double& A_dual = upstreams[0].get_dual<double, double>();
    FEDualPtr& X_dual = upstreams[1].get_dual<FEDualPtr, FEFieldPtr>();
    double& B_dual = upstreams[2].get_dual<double, double>();
    FEDualPtr& Y_dual = upstreams[3].get_dual<FEDualPtr, FEFieldPtr>();

    add(*X_dual, A, *Z_dual, *X_dual);
    add(*Y_dual, B, *Z_dual, *Y_dual);
    A_dual += serac::innerProduct(*Z_dual, *X);
    B_dual += serac::innerProduct(*Z_dual, *Y);
  });

  return z.finalize();
}

/// @brief compute the differentiable weighted sum of fields, weighted by both double weights, and also
/// gret::State<double> differentiable weights.  The differentiable_scale_factors are applied to the differentiable
/// weights to enable negation and scalar muliplication of weights.
FieldState weighted_sum(const std::vector<double>& weights, const std::vector<FieldState>& weighted_fields,
                        const std::vector<gretl::State<double>>& differentiable_weights,
                        const std::vector<FieldState>& differentiably_weighted_fields,
                        const std::vector<double>& differentiable_scale_factors)
{
  SLIC_ERROR_IF(weights.size() != weighted_fields.size(),
                "weights and the fields they are weighting do not match in size");
  SLIC_ERROR_IF(differentiable_weights.size() != differentiably_weighted_fields.size(),
                "differentiable weights and the fields they are weighting do not match in size");
  SLIC_ERROR_IF(differentiable_weights.size() != differentiable_scale_factors.size(),
                "differentiable weights and the vector of fixed scale factors do not match in size");
  SLIC_ERROR_IF((weights.size() == 0) && (differentiable_weights.size() == 0),
                "At least 1 weight must be passed to a weighted sum");

  std::vector<gretl::StateBase> inputs;
  inputs.insert(inputs.end(), weighted_fields.begin(), weighted_fields.end());
  inputs.insert(inputs.end(), differentiable_weights.begin(), differentiable_weights.end());
  inputs.insert(inputs.end(), differentiably_weighted_fields.begin(), differentiably_weighted_fields.end());

  auto x = weights.size() ? weighted_fields[0] : differentiably_weighted_fields[0];
  auto z = x.clone(inputs);

  z.set_eval([weights, differentiable_scale_factors](const gretl::UpstreamStates& upstreams,
                                                     gretl::DownstreamState& downstream) {
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
        double scale = differentiable_scale_factors[0];
        weightOld = scale * upstreams[num_weights].get<double>();
        vecOld = upstreams[num_weights + num_diffable_weights].get<FEFieldPtr>();
        if (num_diffable_weights == 1) {
          Z->Set(weightOld, *vecOld);
        }
      }

      for (size_t i = start_index; i < num_diffable_weights; ++i) {
        double scale = differentiable_scale_factors[i];
        double weightNew = scale * upstreams[num_weights + i].get<double>();
        add(weightOld, *vecOld, weightNew, *upstreams[num_weights + num_diffable_weights + i].get<FEFieldPtr>(), *Z);
        weightOld = 1.0;
        vecOld = Z;
      }
    }

    downstream.set<FEFieldPtr, FEDualPtr>(Z);
  });

  z.set_vjp([weights, differentiable_scale_factors](gretl::UpstreamStates& upstreams,
                                                    const gretl::DownstreamState& downstream) {
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
      double scale = differentiable_scale_factors[i];
      double weight = scale * upstreams[num_weights + i].get<double>();
      FEFieldPtr V = upstreams[num_weights + num_diffable_weights + i].get<FEFieldPtr>();
      add(*V_dual, weight, *Z_dual, *V_dual);
      weight_dual += scale * serac::innerProduct(*Z_dual, *V);
    }
  });

  return z.finalize();
}

FieldStateWeightedSum& FieldStateWeightedSum::operator+=(const FieldStateWeightedSum& b)
{
  weights_.insert(weights_.end(), b.weights_.begin(), b.weights_.end());
  weighted_fields_.insert(weighted_fields_.end(), b.weighted_fields_.begin(), b.weighted_fields_.end());
  differentiable_weights_.insert(differentiable_weights_.end(), b.differentiable_weights_.begin(),
                                 b.differentiable_weights_.end());
  differentiably_weighted_fields_.insert(differentiably_weighted_fields_.end(),
                                         b.differentiably_weighted_fields_.begin(),
                                         b.differentiably_weighted_fields_.end());
  differentiable_scale_factors_.insert(differentiable_scale_factors_.end(), b.differentiable_scale_factors_.begin(),
                                       b.differentiable_scale_factors_.end());
  return *this;
}

FieldStateWeightedSum& FieldStateWeightedSum::operator-=(const FieldStateWeightedSum& b)
{
  const size_t num_initial_weights = weights_.size();

  weights_.insert(weights_.end(), b.weights_.begin(), b.weights_.end());
  for (size_t n = num_initial_weights; n < weights_.size(); ++n) {
    weights_[n] *= -1.0;
  }

  weighted_fields_.insert(weighted_fields_.end(), b.weighted_fields_.begin(), b.weighted_fields_.end());

  differentiable_weights_.insert(differentiable_weights_.end(), b.differentiable_weights_.begin(),
                                 b.differentiable_weights_.end());

  differentiably_weighted_fields_.insert(differentiably_weighted_fields_.end(),
                                         b.differentiably_weighted_fields_.begin(),
                                         b.differentiably_weighted_fields_.end());

  const size_t num_initial_differentiable_weights = differentiable_scale_factors_.size();

  differentiable_scale_factors_.insert(differentiable_scale_factors_.end(), b.differentiable_scale_factors_.begin(),
                                       b.differentiable_scale_factors_.end());
  for (size_t n = num_initial_differentiable_weights; n < differentiable_scale_factors_.size(); ++n) {
    differentiable_scale_factors_[n] *= -1.0;
  }
  return *this;
}

FieldStateWeightedSum FieldStateWeightedSum::operator-() const
{
  FieldStateWeightedSum zero(std::vector<double>{}, std::vector<FieldState>{});
  return zero -= *this;
}

FieldStateWeightedSum::operator FieldState() const
{
  return weighted_sum(weights_, weighted_fields_, differentiable_weights_, differentiably_weighted_fields_,
                      differentiable_scale_factors_);
}

FieldStateWeightedSum& FieldStateWeightedSum::operator*=(double weight)
{
  for (auto& w : weights_) {
    w *= weight;
  }
  for (auto& w : differentiable_scale_factors_) {
    w *= weight;
  }
  return *this;
}

FieldStateWeightedSum operator*(double a, const FieldState& b) { return FieldStateWeightedSum({a}, {b}); }

FieldStateWeightedSum operator*(const FieldState& b, double a) { return a * b; }

FieldStateWeightedSum operator*(double a, const FieldStateWeightedSum& b)
{
  FieldStateWeightedSum z = b;
  return z *= a;
}

FieldStateWeightedSum operator*(const FieldStateWeightedSum& b, double a)
{
  FieldStateWeightedSum z = b;
  return z *= a;
}

FieldStateWeightedSum operator*(const gretl::State<double>& a, const FieldState& b)
{
  return FieldStateWeightedSum({a}, {b});
}

FieldStateWeightedSum operator*(const FieldState& b, const gretl::State<double>& a) { return a * b; }

FieldStateWeightedSum operator+(const FieldState& x, const FieldState& y)
{
  return FieldStateWeightedSum({1.0, 1.0}, {x, y});
}

FieldStateWeightedSum operator-(const FieldState& x, const FieldState& y)
{
  return FieldStateWeightedSum({1.0, -1.0}, {x, y});
}

FieldStateWeightedSum operator+(const FieldStateWeightedSum& ax, const FieldStateWeightedSum& by)
{
  FieldStateWeightedSum c = ax;
  return c += by;
}

FieldStateWeightedSum operator-(const FieldStateWeightedSum& ax, const FieldStateWeightedSum& by)
{
  FieldStateWeightedSum c = ax;
  return c -= by;
}

FieldStateWeightedSum operator+(const FieldStateWeightedSum& ax, const FieldState& y)
{
  FieldStateWeightedSum y1({1.0}, {y});
  return ax + y1;
}

FieldStateWeightedSum operator+(const FieldState& y, const FieldStateWeightedSum& ax) { return ax + y; }

FieldStateWeightedSum operator-(const FieldStateWeightedSum& ax, const FieldState& y)
{
  FieldStateWeightedSum z = ax;
  return z += FieldStateWeightedSum({-1.0}, {y});
}

FieldStateWeightedSum operator-(const FieldState& x, const FieldStateWeightedSum& by)
{
  FieldStateWeightedSum z = -by;
  return z += FieldStateWeightedSum({1.0}, {x});
}

}  // namespace serac