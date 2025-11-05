// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file stdfunction_operator.hpp
 *
 * @brief Classes for defining an mfem::Operator with std::functions
 */

#pragma once

#include <functional>

#include "mfem.hpp"

namespace smith::mfem_ext {

/**
 * @brief StdFunctionOperator is a class wrapping mfem::Operator
 *   so that the user can use std::function to define the implementations of
 *   mfem::Operator::Mult and
 *   mfem::Operator::GetGradient
 *
 * The main benefit of this approach is that lambda capture lists allow
 * for a flexible inline representation of the overloaded functions,
 * without having to manually define a separate functor class.
 */
class StdFunctionOperator : public mfem::Operator {
 public:
  /**
   * @brief Default constructor for creating a square uninitialized StdFunctionOperator
   *
   * @param[in] n The size of the square operator
   */
  StdFunctionOperator(int n) : mfem::Operator(n) {}

  /**
   * @brief Default constructor for creating a rectangular uninitialized StdFunctionOperator
   *
   * @param[in] h The height of the operator
   * @param[in] w The width of the operator
   */
  StdFunctionOperator(int h, int w) : mfem::Operator(h, w) {}

  /**
   * @brief Constructor for a square StdFunctionOperator that only defines mfem::Operator::Mult
   *
   * @param[in] n The size fo the operator
   * @param[in] function The function that defines the mult (typically residual evaluation) method
   */
  StdFunctionOperator(int n, std::function<void(const mfem::Vector&, mfem::Vector&)> function)
      : mfem::Operator(n), function_(function)
  {
  }

  /**
   * @brief Constructor for a rectangular StdFunctionOperator that only defines mfem::Operator::Mult
   *
   * @param[in] h The height of the operator
   * @param[in] w The width of the operator
   * @param[in] function The function that defines the mult (typically residual evaluation) method
   */
  StdFunctionOperator(int h, int w, std::function<void(const mfem::Vector&, mfem::Vector&)> function)
      : mfem::Operator(h, w), function_(function)
  {
  }

  /**
   * @brief Constructor for a square StdFunctionOperator that defines mfem::Operator::Mult and
   * mfem::Operator::GetGradient
   *
   * @param[in] n The size of the operator
   * @param[in] function The function that defines the mult (typically residual evaluation) method
   * @param[in] jacobian The function that defines the GetGradient (typically residual jacobian evaluation) method
   */
  StdFunctionOperator(int n, std::function<void(const mfem::Vector&, mfem::Vector&)> function,
                      std::function<mfem::Operator&(const mfem::Vector&)> jacobian)
      : mfem::Operator(n), function_(function), jacobian_(jacobian)
  {
  }

  /**
   * @brief Constructor for a rectangular StdFunctionOperator that defines mfem::Operator::Mult and
   * mfem::Operator::GetGradient
   *
   * @param[in] h The height of the operator
   * @param[in] w The width of the operator
   * @param[in] function The function that defines the mult (typically residual evaluation) method
   * @param[in] jacobian The function that defines the GetGradient (typically residual jacobian evaluation) method
   */
  StdFunctionOperator(int h, int w, std::function<void(const mfem::Vector&, mfem::Vector&)> function,
                      std::function<mfem::Operator&(const mfem::Vector&)> jacobian)
      : mfem::Operator(h, w), function_(function), jacobian_(jacobian)
  {
  }

  /**
   * @brief The underlying mult (e.g. residual evaluation) method
   *
   * @param[in] k state input vector
   * @param[out] y output residual vector
   */
  void Mult(const mfem::Vector& k, mfem::Vector& y) const { function_(k, y); };

  /**
   * @brief The underlying GetGradient (e.g. residual jacobian evaluation) method
   *
   * @param[in] k The current state input vector
   * @return A non-owning reference to the gradient operator
   */
  mfem::Operator& GetGradient(const mfem::Vector& k) const { return jacobian_(k); };

 private:
  /**
   * @brief the function that is used to implement mfem::Operator::Mult
   */
  std::function<void(const mfem::Vector&, mfem::Vector&)> function_;

  /**
   * @brief the function that is used to implement mfem::Operator::GetGradient
   */
  std::function<mfem::Operator&(const mfem::Vector&)> jacobian_;
};

}  // namespace smith::mfem_ext
