// Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file residual.hpp
 *
 * @brief Specifies interface for evaluating residuals and their gradients
 */

#pragma once

#include "serac/physics/common.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"

namespace serac {

class FiniteElementState;
class FiniteElementDual;

/// Abstract residual class
class Residual {
  public:
  Residual(std::string name) : name_(name) {}
  virtual ~Residual() {}

  using FieldPtr = FiniteElementState*;
  using DualFieldPtr = FiniteElementDual*;

  /// provided  computes residual outputs
  virtual mfem::Vector residual(double time, const std::vector<FieldPtr>& fields, int block_row=0) const = 0;

  // computes jacobian terms
  virtual std::unique_ptr<mfem::HypreParMatrix> jacobian(double time, const std::vector<FieldPtr>& fields, const std::vector<double>& argument_tangents, int block_row=0) const = 0;

  // computes for each residual output: dr/du * fieldsV + dr/dp * parametersV
  virtual void jvp(double time,
                   const std::vector<FieldPtr>& fields,
                   const std::vector<FieldPtr>& fieldsV, // consider a way to turn off components? through nullptrs? yikes?
                   std::vector<DualFieldPtr>& jacobianVectorProducts) const = 0;

  // computes for each input field  (dr/du).T * vResidual
  // computes for each input parameter (dr/dp).T * vResidual
  // can early out if the vectors being requested are sized to 0?
  virtual void vjp(double time, 
                   const std::vector<FieldPtr>& fields,
                   const std::vector<DualFieldPtr>& vResiduals,
                   std::vector<DualFieldPtr>& fieldSensitivities) const = 0;

 std::string name() const { return name_; }

 private:
  std::string name_;
};

}