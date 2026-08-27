// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "axom/core/Array.hpp"
#include "smith/numerics/functional/dual.hpp"
#include "smith/numerics/functional/tensor.hpp"

namespace smith {

/**
 * @brief Non-owning evaluator for shape-preserving piecewise cubic Hermite data.
 *
 * Each interval uses cubic Bernstein control points. Values outside
 * the tabulated coordinate range clamp to endpoint values.
 *
 * @note Referenced PchipData must outlive every use of this view.
 */
class PchipView {
 public:
  /**
   * @brief Evaluate interpolated value.
   */
  template <typename T>
  SMITH_HOST_DEVICE auto operator()(T input) const
  {
    const double input_value = get_value(input);
    if (input_value < coordinates_[0]) {
      return control_points_[0][0] + 0.0 * input;
    }
    if (input_value > coordinates_[num_points_ - 1]) {
      return control_points_[num_points_ - 2][3] + 0.0 * input;
    }

    axom::IndexType lower = 0;
    axom::IndexType upper = num_points_ - 1;
    while (upper - lower > 1) {
      const axom::IndexType middle = lower + (upper - lower) / 2;
      if (input_value < coordinates_[middle]) {
        upper = middle;
      } else {
        lower = middle;
      }
    }

    const auto local_coordinate = (input - coordinates_[lower]) / (coordinates_[lower + 1] - coordinates_[lower]);
    const auto complement = 1.0 - local_coordinate;
    const auto first_linear = complement * control_points_[lower][0] + local_coordinate * control_points_[lower][1];
    const auto second_linear = complement * control_points_[lower][1] + local_coordinate * control_points_[lower][2];
    const auto third_linear = complement * control_points_[lower][2] + local_coordinate * control_points_[lower][3];
    const auto first_quadratic = complement * first_linear + local_coordinate * second_linear;
    const auto second_quadratic = complement * second_linear + local_coordinate * third_linear;
    return complement * first_quadratic + local_coordinate * second_quadratic;
  }

 private:
  friend class PchipData;

  PchipView(const double* coordinates, const tensor<double, 4>* control_points, axom::IndexType num_points)
      : coordinates_(coordinates), control_points_(control_points), num_points_(num_points)
  {
  }

  const double* coordinates_;
  const tensor<double, 4>* control_points_;
  axom::IndexType num_points_;
};

/**
 * @brief Owns runtime-sized PCHIP data in unified memory.
 */
class PchipData {
 public:
  /**
   * @brief Construct from strictly increasing coordinates and tabulated values.
   */
  PchipData(std::span<const double> coordinates, std::span<const double> values)
      : coordinates_(checkedSize(coordinates, values)), control_points_(checkedSize(coordinates, values) - 1)
  {
    std::vector<double> widths(coordinates.size() - 1);
    std::vector<double> secant_slopes(coordinates.size() - 1);
    std::vector<double> nodal_slopes(coordinates.size());

    for (std::size_t i = 0; i < coordinates.size(); ++i) {
      if (!std::isfinite(coordinates[i]) || !std::isfinite(values[i])) {
        throw std::invalid_argument("Pchip coordinates and values must be finite");
      }
      if (i > 0 && coordinates[i] <= coordinates[i - 1]) {
        throw std::invalid_argument("Pchip coordinates must be strictly increasing");
      }
      coordinates_[static_cast<axom::IndexType>(i)] = coordinates[i];
    }

    for (std::size_t i = 0; i + 1 < coordinates.size(); ++i) {
      widths[i] = coordinates[i + 1] - coordinates[i];
      secant_slopes[i] = (values[i + 1] - values[i]) / widths[i];
    }

    if (coordinates.size() == 2) {
      nodal_slopes[0] = secant_slopes[0];
      nodal_slopes[1] = secant_slopes[0];
    } else {
      nodal_slopes[0] = endpointSlope(widths[0], widths[1], secant_slopes[0], secant_slopes[1]);
      const std::size_t last = coordinates.size() - 1;
      nodal_slopes[last] =
          endpointSlope(widths[last - 1], widths[last - 2], secant_slopes[last - 1], secant_slopes[last - 2]);

      for (std::size_t i = 1; i < last; ++i) {
        const double left_slope = secant_slopes[i - 1];
        const double right_slope = secant_slopes[i];
        if (left_slope == 0.0 || right_slope == 0.0 || std::signbit(left_slope) != std::signbit(right_slope)) {
          nodal_slopes[i] = 0.0;
        } else {
          const double left_weight = 2.0 * widths[i] + widths[i - 1];
          const double right_weight = widths[i] + 2.0 * widths[i - 1];
          nodal_slopes[i] = (left_weight + right_weight) / (left_weight / left_slope + right_weight / right_slope);
        }
      }
    }

    for (std::size_t i = 0; i + 1 < coordinates.size(); ++i) {
      auto& control_points = control_points_[static_cast<axom::IndexType>(i)];
      control_points[0] = values[i];
      control_points[1] = values[i] + widths[i] * nodal_slopes[i] / 3.0;
      control_points[2] = values[i + 1] - widths[i] * nodal_slopes[i + 1] / 3.0;
      control_points[3] = values[i + 1];
    }
  }

  /**
   * @brief Return lightweight callable view.
   */
  PchipView view() const { return {coordinates_.data(), control_points_.data(), coordinates_.size()}; }

 private:
  static axom::IndexType checkedSize(std::span<const double> coordinates, std::span<const double> values)
  {
    if (coordinates.size() != values.size()) {
      throw std::invalid_argument("Pchip coordinates and values must have matching sizes");
    }
    if (coordinates.size() < 2) {
      throw std::invalid_argument("Pchip requires at least two points");
    }
    if (coordinates.size() > static_cast<std::size_t>(std::numeric_limits<axom::IndexType>::max())) {
      throw std::length_error("Pchip point count exceeds supported index range");
    }
    return static_cast<axom::IndexType>(coordinates.size());
  }

  /**
   * @brief Compute shape-preserving derivative at one endpoint.
   *
   * PCHIP first forms a one-sided, three-point derivative estimate
   *
   * \f[ m = \frac{(2h_0 + h_1)d_0 - h_0d_1}{h_0 + h_1}, \f]
   *
   * where \f$h_i\f$ are interval widths and \f$d_i\f$ are secant slopes.
   * If \f$m\f$ points opposite \f$d_0\f$, setting \f$m=0\f$ prevents the
   * interpolant from initially moving away from the first interval data. When
   * \f$d_0\f$ and \f$d_1\f$ have opposite signs, limiting \f$|m|\f$ to
   * \f$3|d_0|\f$ prevents endpoint overshoot near the neighboring extremum.
   *
   * Explicit zero checks precede signbit comparisons. Comparing sign bits
   * avoids multiplying slopes solely to determine whether signs differ.
   * Reversing interval arguments applies the same rule at the right endpoint.
   */
  static double endpointSlope(double first_width, double second_width, double first_secant, double second_secant)
  {
    double slope = ((2.0 * first_width + second_width) * first_secant - first_width * second_secant) /
                   (first_width + second_width);
    if (slope == 0.0 || first_secant == 0.0 || std::signbit(slope) != std::signbit(first_secant)) {
      return 0.0;
    }
    if (std::signbit(first_secant) != std::signbit(second_secant) && std::abs(slope) > 3.0 * std::abs(first_secant)) {
      slope = 3.0 * first_secant;
    }
    return slope;
  }

  axom::Array<double> coordinates_;
  axom::Array<tensor<double, 4>> control_points_;
};

}  // namespace smith
