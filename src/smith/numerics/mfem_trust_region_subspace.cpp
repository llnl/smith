// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/trust_region_subspace_cache.hpp"

namespace smith {

TrustRegionSubspaceResult solveSubspaceProblem(const std::vector<const mfem::Vector*>& directions,
                                               const std::vector<const mfem::Vector*>& A_directions,
                                               const mfem::Vector& b, double delta, int num_leftmost, MPI_Comm comm)
{
  TrustRegionSubspaceCache cache;
  cache.prepare(directions, A_directions, b, num_leftmost, comm);
  return cache.solve(delta);
}

}  // namespace smith
