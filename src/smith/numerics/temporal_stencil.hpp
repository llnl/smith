#pragma once

namespace smith {

/**
 * @brief Fixed-capacity temporal stencil for multistep time-integration schemes.
 *
 * A TemporalStencil stores a bounded sequence of lagged states associated with
 * a multistep time discretization (e.g., BDF, Adams–Bashforth, projection methods).
 *
 * The stencil is ordered from newest to oldest:
 *
 *   buffer_[0]  = state at time tⁿ
 *   buffer_[1]  = state at time tⁿ⁻¹
 *   buffer_[2]  = state at time tⁿ⁻²
 *   ...
 *
 * Only the most recent `capacity` states are retained. When a new state is pushed
 * and the stencil exceeds capacity, the oldest entry is discarded.
 *
 * This container does not own any temporal policy (e.g., weights, order selection,
 * cycle logic). It is a pure storage abstraction. Interpretation of the stored
 * states is the responsibility of temporal operators (e.g., BDF, extrapolation).
 *
 * @tparam State Type of the stored state (e.g., FieldState, FiniteElementState, scalar value, vector type).
 *
 * @note This class is intended for small stencil sizes typical of multistep methods (order ≤ 6).
 *       It is not designed for archival storage of full simulation history.
 */
template <typename State>
class TemporalStencil {

public:

  /**
   * @brief Construct a temporal stencil with fixed capacity.
   *
   * @param capacity Maximum number of lag states to retain.
   *
   * @pre capacity ≥ 0
   *
   * If capacity is zero, the stencil will not retain any states.
   *
   * @note The capacity should be chosen to satisfy the highest order
   *       required by any temporal operator that consumes this stencil.
   */
  explicit TemporalStencil(
    int capacity
  ) : capacity_(capacity) {

    SLIC_ERROR_ROOT_IF(
      capacity_ < 0,
      "TemporalStencil capacity must be nonnegative."
    );

  }

  /**
   * @brief Insert a new state into the stencil.
   *
   * The inserted state becomes the newest entry (index 0).
   * If insertion causes the stencil to exceed its capacity, the oldest state is removed.
   *
   * @param s State to insert (typically representing the solution at the current time level).
   *
   * @post size() ≤ capacity
   *
   * @note States are stored by value.
   */
  void
  push(
    State const &s
  ) {

    buffer_.insert(buffer_.begin(), s);

    if (static_cast<int>(buffer_.size()) > capacity_) {
      buffer_.pop_back();
    }

  }

  /**
   * @brief Return the number of currently stored lag states.
   *
   * @return Number of entries in the stencil.
   *
   * The returned value is always less than or equal to the configured capacity.
   */
  int size() const {

    return static_cast<int>(buffer_.size());

  }

  /**
   * @brief Access the full stencil (newest → oldest).
   *
   * @return Const reference to the underlying container.
   *
   * The returned sequence is ordered from most recent to least recent state.
   *
   * @note No copy is performed.
   */
  std::vector<State> const &
  view() const {

    return buffer_;

  }

  /**
   * @brief Access the first n entries of the stencil.
   *
   * Returns the n most recent states in newest-to-oldest order.
   *
   * @param n Number of entries to retrieve.
   *
   * @return Vector containing the first n states.
   *
   * @pre 0 ≤ n ≤ size()
   *
   * @throws Error if n exceeds the number of stored states.
   *
   * @note A new vector is returned (copy).
   *       This is typically used by temporal operators that require only a subset of the available stencil.
   */
  std::vector<State>
  view(
    int n
  ) const {

    SLIC_ERROR_ROOT_IF(n < 0, "Requested negative stencil length.");
    SLIC_ERROR_ROOT_IF(n > size(), "Requested more stencil entries than available.");

    return std::vector<State>(buffer_.begin(), buffer_.begin() + n);

  }

  /**
   * @brief Remove the most recently inserted state.
   *
   * This operation discards the newest entry in the stencil.
   *
   * @pre size() > 0
   *
   * @note Intended for use in rollback scenarios where a time step must be reverted due to solver failure or rejection.
   */
  void
  rollback() {

    SLIC_ERROR_ROOT_IF(
      buffer_.empty(),
      "Rollback called on empty TemporalStencil."
    );

    buffer_.erase(buffer_.begin());

  }

private:

  /// @brief Discrete temporal support (newest → oldest) for a multistep method.
  ///
  /// Stores the sequence { uⁿ, uⁿ⁻¹, … } required by temporal operators.
  /// The ordering is strictly newest-first to align with standard
  /// multistep notation:
  ///
  ///   φ^{n-m} ↔ buffer_[m]
  ///
  /// The container size never exceeds capacity_.
  std::vector<State> buffer_;

  /// @brief Width of the temporal stencil (maximum number of lag states).
  ///
  /// Defines the maximum multistep order supported by this stencil.
  /// For example:
  ///   capacity_ = 2 → supports second-order schemes
  ///   capacity_ = 1 → supports first-order schemes
  ///
  /// This parameter controls storage only; it does not determine
  /// time-integration weights.
  int capacity_;

};

}  // namespace smith
