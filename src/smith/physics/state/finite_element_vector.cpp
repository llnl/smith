// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/state/finite_element_vector.hpp"

#include <cstring>
#include <ostream>
#include <utility>

#include "smith/smith_config.hpp"
#include "smith/infrastructure/logger.hpp"

namespace smith {

/**
 * @brief Check if two finite element spaces are the same
 *
 * @param left
 * @param right
 * @return Bool which is true if the spaces are the same, otherwise false
 */
bool sameFiniteElementSpace(const mfem::FiniteElementSpace& left, const mfem::FiniteElementSpace& right)
{
  bool sameMesh = (left.GetMesh() == right.GetMesh());
  bool equivalentFEColl = strcmp(left.FEColl()->Name(), right.FEColl()->Name()) == 0;
  bool sameVectorDimension = (left.GetVDim() == right.GetVDim());
  bool sameOrdering = (left.GetOrdering() == right.GetOrdering());
  return sameMesh && equivalentFEColl && sameVectorDimension && sameOrdering;
}

FiniteElementVector::FiniteElementVector(const mfem::ParFiniteElementSpace& space, const std::string& name)
    : mesh_(*space.GetParMesh()),
      coll_(std::unique_ptr<mfem::FiniteElementCollection>(mfem::FiniteElementCollection::New(space.FEColl()->Name()))),
      space_(std::make_unique<mfem::ParFiniteElementSpace>(space, &mesh_.get(), coll_.get())),
      name_(name)
{
  SLIC_ERROR_ROOT_IF(space_->GetVDim() > 1 && space_->GetOrdering() != smith::ordering,
                     "Smith only operates on finite element spaces ordered by "
                         << (smith::ordering == mfem::Ordering::byVDIM ? "VDIM" : "NODES"));

  // Construct a hypre par vector based on the new finite element space
  HypreParVector new_vector(space_.get());

  // Move the data from this new hypre vector into this object without doubly allocating the data
  auto* parallel_vec = new_vector.StealParVector();
  WrapHypreParVector(parallel_vec);

  // Initialize the vector to zero
  HypreParVector::operator=(0.0);

  // Use device (if available)
  UseDevice(true);
}

FiniteElementVector::FiniteElementVector(FiniteElementVector&& input_vector)
    : mesh_(input_vector.mesh()),
      coll_(std::move(input_vector.coll_)),
      space_(std::move(input_vector.space_)),
      name_(std::move(input_vector.name_))
{
  // Grab the allocated data from the input argument for the underlying Hypre vector
  auto* parallel_vec = input_vector.StealParVector();
  WrapHypreParVector(parallel_vec);
}

FiniteElementVector& FiniteElementVector::operator=(const mfem::HypreParVector& rhs)
{
  SLIC_ERROR_IF(Size() != rhs.Size(),
                axom::fmt::format("Finite element vector of size '{}' assigned to a HypreParVector of size '{}'",
                                  Size(), rhs.Size()));

  HypreParVector::operator=(rhs);
  return *this;
}

FiniteElementVector& FiniteElementVector::operator=(const mfem::Vector& rhs)
{
  Vector::operator=(rhs);
  return *this;
}

FiniteElementVector& FiniteElementVector::operator=(const FiniteElementVector& rhs)
{
  SLIC_ERROR_IF(!sameFiniteElementSpace(*space_, *rhs.space_), "Finite element vectors have different spaces");

  SLIC_ERROR_IF(Size() != rhs.Size(),
                axom::fmt::format("Finite element vector of size '{}' assigned to a HypreParVector of size '{}'",
                                  Size(), rhs.Size()));

  HypreParVector::operator=(rhs);

  return *this;
}

FiniteElementVector& FiniteElementVector::operator=(FiniteElementVector&& rhs)
{
  mesh_ = rhs.mesh_;
  coll_ = std::move(rhs.coll_);
  space_ = std::move(rhs.space_);
  name_ = rhs.name_;

  auto* parallel_vec = rhs.StealParVector();
  WrapHypreParVector(parallel_vec);

  return *this;
}

FiniteElementVector& FiniteElementVector::operator=(const double value)
{
  HypreParVector::operator=(value);
  return *this;
}

double avg(const FiniteElementVector& fe_vector)
{
  double global_sum;
  double local_sum = fe_vector.Sum();
  int global_size;
  int local_size = fe_vector.Size();
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, fe_vector.comm());
  MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, fe_vector.comm());
  return global_sum / global_size;
}

double max(const FiniteElementVector& fe_vector)
{
  double global_max;
  double local_max = fe_vector.Max();
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, fe_vector.comm());
  return global_max;
}

double min(const FiniteElementVector& fe_vector)
{
  double global_min;
  double local_min = fe_vector.Min();
  MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, fe_vector.comm());
  return global_min;
}

double innerProduct(const FiniteElementVector& v1, const FiniteElementVector& v2)
{
  SLIC_ERROR_IF(
      v1.Size() != v2.Size(),
      axom::fmt::format("Finite element vector of size '{}' can not inner product with another vector of size '{}'",
                        v1.Size(), v2.Size()));
  SLIC_ERROR_IF(v1.comm() != v2.comm(),
                "Cannot compute inner products between finite element vectors with different mpi communicators");
  SLIC_ERROR_IF(!sameFiniteElementSpace(v1.space(), v2.space()),
                "Currently cannot compute inner products between finite element vectors with different mfem spaces");

  double global_ip;
  double local_ip = mfem::InnerProduct(v1, v2);
  MPI_Allreduce(&local_ip, &global_ip, 1, MPI_DOUBLE, MPI_SUM, v1.comm());
  return global_ip;
}

}  // namespace smith
