// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/mesh.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/boundary_conditions/boundary_condition.hpp"

#include <algorithm>
#include <cmath>

namespace smith {

namespace {

constexpr double bc_time_fd_step = 1.0e-4;

class SecondTimeDerivativeScalarCoefficient : public mfem::Coefficient {
 public:
  explicit SecondTimeDerivativeScalarCoefficient(std::shared_ptr<mfem::Coefficient> source) : source_(std::move(source))
  {
  }

  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override
  {
    const double t0 = GetTime();
    const double h = bc_time_fd_step * std::max(1.0, std::abs(t0));
    source_->SetTime(t0);
    const double f0 = source_->Eval(T, ip);
    source_->SetTime(t0 + h);
    const double f1 = source_->Eval(T, ip);
    source_->SetTime(t0 + 2.0 * h);
    const double f2 = source_->Eval(T, ip);
    source_->SetTime(t0);
    return (f2 - 2.0 * f1 + f0) / (h * h);
  }

 private:
  std::shared_ptr<mfem::Coefficient> source_;
};

class SecondTimeDerivativeVectorCoefficient : public mfem::VectorCoefficient {
 public:
  explicit SecondTimeDerivativeVectorCoefficient(std::shared_ptr<mfem::VectorCoefficient> source)
      : mfem::VectorCoefficient(source->GetVDim()), source_(std::move(source))
  {
  }

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override
  {
    const double t0 = GetTime();
    const double h = bc_time_fd_step * std::max(1.0, std::abs(t0));
    mfem::Vector v0(vdim), v1(vdim), v2(vdim);
    source_->SetTime(t0);
    source_->Eval(v0, T, ip);
    source_->SetTime(t0 + h);
    source_->Eval(v1, T, ip);
    source_->SetTime(t0 + 2.0 * h);
    source_->Eval(v2, T, ip);
    source_->SetTime(t0);
    V = v2;
    V.Add(-2.0, v1);
    V += v0;
    V /= (h * h);
  }

 private:
  std::shared_ptr<mfem::VectorCoefficient> source_;
};

}  // namespace

DirichletBoundaryConditions::DirichletBoundaryConditions(const mfem::ParMesh& mfem_mesh,
                                                         mfem::ParFiniteElementSpace& space)
    : bcs_(mfem_mesh), space_(space)
{
}

DirichletBoundaryConditions::DirichletBoundaryConditions(const Mesh& mesh, mfem::ParFiniteElementSpace& space)
    : DirichletBoundaryConditions(mesh.mfemParMesh(), space)
{
}

void DirichletBoundaryConditions::setSecondTimeDerivativeBCsMatchingDofs(const BoundaryConditionManager& source)
{
  for (const auto& bc : source.essentials()) {
    if (is_vector_valued(bc.coefficient())) {
      auto accel_coef = std::make_shared<SecondTimeDerivativeVectorCoefficient>(
          get<std::shared_ptr<mfem::VectorCoefficient>>(bc.coefficient()));
      bcs_.addEssentialByTrueDofs(bc.getTrueDofList(), accel_coef, space_);
    } else {
      auto accel_coef = std::make_shared<SecondTimeDerivativeScalarCoefficient>(
          get<std::shared_ptr<mfem::Coefficient>>(bc.coefficient()));
      bcs_.addEssential(bc.getLocalDofList(), accel_coef, space_, bc.component());
    }
  }
}

}  // namespace smith
