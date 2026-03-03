#pragma once

namespace smith {

struct ConvergenceMonitor {

  /**
   * @brief Compute absolute increment
   *        ||u_new - u_old||
   */
  static double
  absoluteIncrement(
    smith::FieldState const &current,
    smith::FieldState const &previous
  ) {

    auto const &u_new = *current.get();
    auto const &u_old = *previous.get();

    auto diff(u_new);
    diff.Add(-1.0, u_old);

    return smith::norm(diff);

  }

  /**
   * @brief Compute relative increment
   *        ||u_new - u_old|| / ||u_new||
   *
   * Falls back to absolute increment if ||u_new|| is near zero.
   */
  static double
  relativeIncrement(
    smith::FieldState const &current,
    smith::FieldState const &previous
  ) {

    double diff = absoluteIncrement(current, previous);

    auto const &u_new = *current.get();
    double mag = smith::norm(u_new);

    if (mag < std::numeric_limits<double>::epsilon()) {
      return diff;
    }

    return diff / mag;

  }

};

}  // namespace smith
