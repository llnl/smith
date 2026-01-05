// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file segment_Hcurl.inl
 *
 * @brief Specialization of finite_element for Hcurl on segment geometry
 */

// specialization of finite_element for Hcurl on segment geometry
//
// this specialization defines shape functions (and their gradients) that
// interpolate at Gauss-Lobatto nodes for the appropriate polynomial order
//
// note: mfem assumes the parent element domain is [0,1]
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p, int c>
struct finite_element<mfem::Geometry::SEGMENT, Hcurl<p, c> > {
  static constexpr auto geometry = mfem::Geometry::SEGMENT;
  static constexpr auto family = Family::HCURL;
  static constexpr int components = c;
  static constexpr int dim = 1;
  static constexpr int ndof = (p + 1);

  using dof_type = tensor<double, c, ndof>;
  using dof_type_if = dof_type;

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_functions(double xi)
  {
    return GaussLegendreInterpolation<ndof>(xi);
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_function_gradients(double xi)
  {
    return GaussLegendreInterpolationDerivative<ndof>(xi);
  }
};
/// @endcond
