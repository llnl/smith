// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "mfem.hpp"

namespace smith {

/**
 * @brief Compile-time alias for a dimension
 */
template <int d>
struct Dimension {
  /**
   * @brief Returns the dimension
   */
  constexpr operator int() { return d; }
};

/**
 * @brief return the number of quadrature points in a Gauss-Legendre rule
 * with parameter "Q"
 *
 * @tparam g the element geometry
 * @tparam Q the number of quadrature points per dimension
 */
constexpr int num_quadrature_points(mfem::Geometry::Type g, int Q)
{
  if (g == mfem::Geometry::SEGMENT) {
    return Q;
  }
  if (g == mfem::Geometry::TRIANGLE) {
    return (Q * (Q + 1)) / 2;
  }
  if (g == mfem::Geometry::SQUARE) {
    return Q * Q;
  }
  if (g == mfem::Geometry::TETRAHEDRON) {
    return (Q * (Q + 1) * (Q + 2)) / 6;
  }
  if (g == mfem::Geometry::CUBE) {
    return Q * Q * Q;
  }
  return -1;
}

/**
 * @brief Returns the dimension of an element geometry
 * @param[in] g The @p Geometry to retrieve the dimension of
 */
constexpr int dimension_of(mfem::Geometry::Type g)
{
  if (g == mfem::Geometry::SEGMENT) {
    return 1;
  }

  if (g == mfem::Geometry::TRIANGLE || g == mfem::Geometry::SQUARE) {
    return 2;
  }

  if (g == mfem::Geometry::TETRAHEDRON || g == mfem::Geometry::CUBE) {
    return 3;
  }

  return -1;
}

/**
 * @brief count the number of elements of each geometry in a mesh
 * @param mesh the mesh to count
 */
inline std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> geometry_counts(const mfem::Mesh& mesh)
{
  std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> counts{};
  for (int i = 0; i < mesh.GetNE(); i++) {
    counts[uint64_t(mesh.GetElementGeometry(i))]++;
  }
  return counts;
}

/**
 * @brief count the number of boundary elements of each geometry in a mesh
 * @param mesh the mesh to count
 */
inline std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> boundary_geometry_counts(const mfem::Mesh& mesh)
{
  std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> counts{};
  for (int f = 0; f < mesh.GetNumFaces(); f++) {
    // skip interior faces
    if (mesh.GetFaceInformation(f).IsInterior()) continue;

    counts[uint64_t(mesh.GetFaceGeometry(f))]++;
  }
  return counts;
}

}  // namespace smith
