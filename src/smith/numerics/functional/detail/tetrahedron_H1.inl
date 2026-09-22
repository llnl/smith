// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file tetrahedron_H1.inl
 *
 * @brief Specialization of finite_element for H1 on tetrahedron geometry
 */

// this specialization defines shape functions (and their gradients) that
// interpolate at Gauss-Lobatto nodes for the appropriate polynomial order
//
// note: mfem assumes the parent element domain is the convex hull of {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}}
// for additional information on the finite_element concept requirements, see finite_element.hpp
//
// for detailed information about node locations for different polynomial orders, see
// simplex_basis_function_unit_tests.cpp
/// @cond
template <int p, int c>
struct finite_element<mfem::Geometry::TETRAHEDRON, H1<p, c> > {
  static constexpr auto geometry = mfem::Geometry::TETRAHEDRON;
  static constexpr auto family = Family::H1;
  static constexpr int components = c;
  static constexpr int dim = 3;
  static constexpr int n = (p + 1);
  static constexpr int ndof = (p + 1) * (p + 2) * (p + 3) / 6;
  static constexpr int order = p;
  static constexpr int nqpts(int q) { return num_quadrature_points(mfem::Geometry::TETRAHEDRON, q); }

  static constexpr int VALUE = 0, GRADIENT = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  using dof_type = tensor<double, c, ndof>;

  using value_type = typename std::conditional<components == 1, double, tensor<double, components> >::type;
  using derivative_type =
      typename std::conditional<components == 1, tensor<double, dim>, tensor<double, components, dim> >::type;
  using qf_input_type = tuple<value_type, derivative_type>;

  SMITH_HOST_DEVICE static constexpr double shape_function([[maybe_unused]] tensor<double, dim> xi, int i)
  {
    if constexpr (p == 1) {
      switch (i) {
        case 0:
          return 1 - xi[0] - xi[1] - xi[2];
        case 1:
          return xi[0];
        case 2:
          return xi[1];
        case 3:
          return xi[2];
      }
    }
    if constexpr (p == 2) {
      switch (i) {
        case 0:
          return (-1 + xi[0] + xi[1] + xi[2]) * (-1 + 2 * xi[0] + 2 * xi[1] + 2 * xi[2]);
        case 1:
          return -4 * xi[0] * (-1 + xi[0] + xi[1] + xi[2]);
        case 2:
          return xi[0] * (-1 + 2 * xi[0]);
        case 3:
          return -4 * xi[1] * (-1 + xi[0] + xi[1] + xi[2]);
        case 4:
          return 4 * xi[0] * xi[1];
        case 5:
          return xi[1] * (-1 + 2 * xi[1]);
        case 6:
          return -4 * xi[2] * (-1 + xi[0] + xi[1] + xi[2]);
        case 7:
          return 4 * xi[0] * xi[2];
        case 8:
          return 4 * xi[1] * xi[2];
        case 9:
          return xi[2] * (-1 + 2 * xi[2]);
      }
    }

    if constexpr (p == 3) {
      constexpr double sqrt5 = 2.23606797749978981;
      switch (i) {
        case 0:
          return -((-1 + xi[0] + xi[1] + xi[2]) *
                   (1 + 5 * xi[0] * xi[0] + 5 * xi[1] * xi[1] + 5 * (-1 + xi[2]) * xi[2] + xi[1] * (-5 + 11 * xi[2]) +
                    xi[0] * (-5 + 11 * xi[1] + 11 * xi[2])));
        case 1:
          return (5 * xi[0] * (-1 + xi[0] + xi[1] + xi[2]) *
                  (-1 - sqrt5 + 2 * sqrt5 * xi[0] + (3 + sqrt5) * xi[1] + (3 + sqrt5) * xi[2])) /
                 2.;
        case 2:
          return (-5 * xi[0] * (-1 + xi[0] + xi[1] + xi[2]) *
                  (1 - sqrt5 + 2 * sqrt5 * xi[0] + (-3 + sqrt5) * xi[1] + (-3 + sqrt5) * xi[2])) /
                 2.;
        case 3:
          return xi[0] * (1 + 5 * xi[0] * xi[0] + xi[1] - xi[1] * xi[1] + xi[2] - xi[1] * xi[2] - xi[2] * xi[2] -
                          xi[0] * (5 + xi[1] + xi[2]));
        case 4:
          return (5 * xi[1] * (-1 + xi[0] + xi[1] + xi[2]) *
                  (-1 - sqrt5 + (3 + sqrt5) * xi[0] + 2 * sqrt5 * xi[1] + (3 + sqrt5) * xi[2])) /
                 2.;
        case 5:
          return -27 * xi[0] * xi[1] * (-1 + xi[0] + xi[1] + xi[2]);
        case 6:
          return (5 * xi[0] * xi[1] * (-2 + (3 + sqrt5) * xi[0] - (-3 + sqrt5) * xi[1])) / 2.;
        case 7:
          return (-5 * xi[1] * (-1 + xi[0] + xi[1] + xi[2]) *
                  (1 - sqrt5 + (-3 + sqrt5) * xi[0] + 2 * sqrt5 * xi[1] + (-3 + sqrt5) * xi[2])) /
                 2.;
        case 8:
          return (-5 * xi[0] * xi[1] * (2 + (-3 + sqrt5) * xi[0] - (3 + sqrt5) * xi[1])) / 2.;
        case 9:
          return xi[1] * (1 - xi[0] * xi[0] + 5 * xi[1] * xi[1] + xi[2] - xi[2] * xi[2] - xi[1] * (5 + xi[2]) -
                          xi[0] * (-1 + xi[1] + xi[2]));
        case 10:
          return (5 * xi[2] * (-1 + xi[0] + xi[1] + xi[2]) *
                  (-439204 - 196418 * sqrt5 + (710647 + 317811 * sqrt5) * xi[0] + (710647 + 317811 * sqrt5) * xi[1] +
                   606965 * xi[2] + 271443 * sqrt5 * xi[2])) /
                 (271443 + 121393 * sqrt5);
        case 11:
          return -27 * xi[0] * xi[2] * (-1 + xi[0] + xi[1] + xi[2]);
        case 12:
          return (5 * xi[0] * xi[2] * (-5 - 3 * sqrt5 + (15 + 7 * sqrt5) * xi[0] + 2 * sqrt5 * xi[2])) /
                 (5 + 3 * sqrt5);
        case 13:
          return -27 * xi[1] * xi[2] * (-1 + xi[0] + xi[1] + xi[2]);
        case 14:
          return 27 * xi[0] * xi[1] * xi[2];
        case 15:
          return (5 * xi[1] * xi[2] * (-5 - 3 * sqrt5 + (15 + 7 * sqrt5) * xi[1] + 2 * sqrt5 * xi[2])) /
                 (5 + 3 * sqrt5);
        case 16:
          return (5 * xi[2] * (-1 + xi[0] + xi[1] + xi[2]) *
                  (88555 + 39603 * sqrt5 + (54730 + 24476 * sqrt5) * xi[0] + (54730 + 24476 * sqrt5) * xi[1] -
                   5 * (64079 + 28657 * sqrt5) * xi[2])) /
                 (143285 + 64079 * sqrt5);
        case 17:
          return (-5 * xi[0] * xi[2] * (2 + (-3 + sqrt5) * xi[0] - (3 + sqrt5) * xi[2])) / 2.;
        case 18:
          return (-5 * xi[1] * xi[2] * (2 + (-3 + sqrt5) * xi[1] - (3 + sqrt5) * xi[2])) / 2.;
        case 19:
          return -(xi[2] * (-1 + xi[0] * xi[0] + xi[1] * xi[1] + xi[1] * (-1 + xi[2]) - 5 * (-1 + xi[2]) * xi[2] +
                            xi[0] * (-1 + xi[1] + xi[2])));
      }
    }

    return 0.0;
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, dim> shape_function_gradient(
      [[maybe_unused]] tensor<double, dim> xi, int i)
  {
    if (p == 1) {
      switch (i) {
        case 0:
          return {-1, -1, -1};
        case 1:
          return {1, 0, 0};
        case 2:
          return {0, 1, 0};
        case 3:
          return {0, 0, 1};
      }
    }
    if (p == 2) {
      switch (i) {
        case 0:
          return {-3 + 4 * xi[0] + 4 * xi[1] + 4 * xi[2], -3 + 4 * xi[0] + 4 * xi[1] + 4 * xi[2],
                  -3 + 4 * xi[0] + 4 * xi[1] + 4 * xi[2]};
        case 1:
          return {-4 * (-1 + 2 * xi[0] + xi[1] + xi[2]), -4 * xi[0], -4 * xi[0]};
        case 2:
          return {-1 + 4 * xi[0], 0, 0};
        case 3:
          return {-4 * xi[1], -4 * (-1 + xi[0] + 2 * xi[1] + xi[2]), -4 * xi[1]};
        case 4:
          return {4 * xi[1], 4 * xi[0], 0};
        case 5:
          return {0, -1 + 4 * xi[1], 0};
        case 6:
          return {-4 * xi[2], -4 * xi[2], -4 * (-1 + xi[0] + xi[1] + 2 * xi[2])};
        case 7:
          return {4 * xi[2], 0, 4 * xi[0]};
        case 8:
          return {0, 4 * xi[2], 4 * xi[1]};
        case 9:
          return {0, 0, -1 + 4 * xi[2]};
      }
    }

    if (p == 3) {
      constexpr double sqrt5 = 2.23606797749978981;
      switch (i) {
        case 0:
          return {-6 - 15 * xi[0] * xi[0] - 16 * xi[1] * xi[1] + xi[1] * (21 - 33 * xi[2]) + (21 - 16 * xi[2]) * xi[2] -
                      4 * xi[0] * (-5 + 8 * xi[1] + 8 * xi[2]),
                  -6 - 16 * xi[0] * xi[0] + 20 * xi[1] + xi[0] * (21 - 32 * xi[1] - 33 * xi[2]) + 21 * xi[2] -
                      (3 * xi[1] + 4 * xi[2]) * (5 * xi[1] + 4 * xi[2]),
                  -6 - 16 * xi[0] * xi[0] + 21 * xi[1] + xi[0] * (21 - 33 * xi[1] - 32 * xi[2]) + 20 * xi[2] -
                      (4 * xi[1] + 3 * xi[2]) * (4 * xi[1] + 5 * xi[2])};
        case 1:
          return {(5 * (6 * sqrt5 * xi[0] * xi[0] +
                        xi[0] * (-2 - 6 * sqrt5 + 6 * (1 + sqrt5) * xi[1] + 6 * (1 + sqrt5) * xi[2]) +
                        (-1 + xi[1] + xi[2]) * (-1 - sqrt5 + (3 + sqrt5) * xi[1] + (3 + sqrt5) * xi[2]))) /
                      2.,
                  (5 * xi[0] *
                   (-4 - 2 * sqrt5 + 3 * (1 + sqrt5) * xi[0] + 2 * (3 + sqrt5) * xi[1] + 2 * (3 + sqrt5) * xi[2])) /
                      2.,
                  (5 * xi[0] *
                   (-4 - 2 * sqrt5 + 3 * (1 + sqrt5) * xi[0] + 2 * (3 + sqrt5) * xi[1] + 2 * (3 + sqrt5) * xi[2])) /
                      2.};
        case 2:
          return {-15 * sqrt5 * xi[0] * xi[0] -
                      (5 * (-1 + xi[1] + xi[2]) * (1 - sqrt5 + (-3 + sqrt5) * xi[1] + (-3 + sqrt5) * xi[2])) / 2. -
                      5 * xi[0] * (1 - 3 * sqrt5 + 3 * (-1 + sqrt5) * xi[1] + 3 * (-1 + sqrt5) * xi[2]),
                  (-5 * xi[0] *
                   (4 - 2 * sqrt5 + 3 * (-1 + sqrt5) * xi[0] + 2 * (-3 + sqrt5) * xi[1] + 2 * (-3 + sqrt5) * xi[2])) /
                      2.,
                  (-5 * xi[0] *
                   (4 - 2 * sqrt5 + 3 * (-1 + sqrt5) * xi[0] + 2 * (-3 + sqrt5) * xi[1] + 2 * (-3 + sqrt5) * xi[2])) /
                      2.};
        case 3:
          return {1 + 15 * xi[0] * xi[0] + xi[1] - xi[1] * xi[1] + xi[2] - xi[1] * xi[2] - xi[2] * xi[2] -
                      2 * xi[0] * (5 + xi[1] + xi[2]),
                  -(xi[0] * (-1 + xi[0] + 2 * xi[1] + xi[2])), -(xi[0] * (-1 + xi[0] + xi[1] + 2 * xi[2]))};
        case 4:
          return {(5 * xi[1] *
                   (-2 * (2 + sqrt5) + 2 * (3 + sqrt5) * xi[0] + 3 * (1 + sqrt5) * xi[1] + 2 * (3 + sqrt5) * xi[2])) /
                      2.,
                  15 * sqrt5 * xi[1] * xi[1] +
                      5 * xi[1] * (-1 - 3 * sqrt5 + 3 * (1 + sqrt5) * xi[0] + 3 * (1 + sqrt5) * xi[2]) +
                      (5 * (-1 + xi[0] + xi[2]) * (-1 - sqrt5 + (3 + sqrt5) * xi[0] + (3 + sqrt5) * xi[2])) / 2.,
                  (5 * xi[1] *
                   (-2 * (2 + sqrt5) + 2 * (3 + sqrt5) * xi[0] + 3 * (1 + sqrt5) * xi[1] + 2 * (3 + sqrt5) * xi[2])) /
                      2.};
        case 5:
          return {-27 * xi[1] * (-1 + 2 * xi[0] + xi[1] + xi[2]), -27 * xi[0] * (-1 + xi[0] + 2 * xi[1] + xi[2]),
                  -27 * xi[0] * xi[1]};
        case 6:
          return {(-5 * xi[1] * (2 - 2 * (3 + sqrt5) * xi[0] + (-3 + sqrt5) * xi[1])) / 2.,
                  (5 * xi[0] * (-2 + (3 + sqrt5) * xi[0] - 2 * (-3 + sqrt5) * xi[1])) / 2., 0};
        case 7:
          return {(-5 * xi[1] *
                   (4 - 2 * sqrt5 + 2 * (-3 + sqrt5) * xi[0] + 3 * (-1 + sqrt5) * xi[1] + 2 * (-3 + sqrt5) * xi[2])) /
                      2.,
                  -15 * sqrt5 * xi[1] * xi[1] -
                      (5 * (-1 + xi[0] + xi[2]) * (1 - sqrt5 + (-3 + sqrt5) * xi[0] + (-3 + sqrt5) * xi[2])) / 2. -
                      5 * xi[1] * (1 - 3 * sqrt5 + 3 * (-1 + sqrt5) * xi[0] + 3 * (-1 + sqrt5) * xi[2]),
                  (-5 * xi[1] *
                   (4 - 2 * sqrt5 + 2 * (-3 + sqrt5) * xi[0] + 3 * (-1 + sqrt5) * xi[1] + 2 * (-3 + sqrt5) * xi[2])) /
                      2.};
        case 8:
          return {(5 * xi[1] * (-2 - 2 * (-3 + sqrt5) * xi[0] + (3 + sqrt5) * xi[1])) / 2.,
                  (-5 * xi[0] * (2 + (-3 + sqrt5) * xi[0] - 2 * (3 + sqrt5) * xi[1])) / 2., 0};
        case 9:
          return {-(xi[1] * (-1 + 2 * xi[0] + xi[1] + xi[2])),
                  1 - xi[0] * xi[0] + 15 * xi[1] * xi[1] + xi[2] - xi[2] * xi[2] - 2 * xi[1] * (5 + xi[2]) -
                      xi[0] * (-1 + 2 * xi[1] + xi[2]),
                  -(xi[1] * (-1 + xi[0] + xi[1] + 2 * xi[2]))};
        case 10:
          return {
              (5 * xi[2] *
               (-2 * (2 + sqrt5) + 2 * (3 + sqrt5) * xi[0] + 2 * (3 + sqrt5) * xi[1] + 3 * (1 + sqrt5) * xi[2])) /
                  2.,
              (5 * xi[2] *
               (-2 * (2 + sqrt5) + 2 * (3 + sqrt5) * xi[0] + 2 * (3 + sqrt5) * xi[1] + 3 * (1 + sqrt5) * xi[2])) /
                  2.,
              (5 * (1 + sqrt5 + (3 + sqrt5) * xi[0] * xi[0] - 2 * (2 + sqrt5) * xi[1] + (3 + sqrt5) * xi[1] * xi[1] +
                    6 * (1 + sqrt5) * xi[1] * xi[2] + 2 * xi[2] * (-1 - 3 * sqrt5 + 3 * sqrt5 * xi[2]) +
                    2 * xi[0] * (-2 - sqrt5 + (3 + sqrt5) * xi[1] + 3 * (1 + sqrt5) * xi[2]))) /
                  2.};
        case 11:
          return {-27 * xi[2] * (-1 + 2 * xi[0] + xi[1] + xi[2]), -27 * xi[0] * xi[2],
                  -27 * xi[0] * (-1 + xi[0] + xi[1] + 2 * xi[2])};
        case 12:
          return {(-5 * xi[2] * (2 - 2 * (3 + sqrt5) * xi[0] + (-3 + sqrt5) * xi[2])) / 2., 0,
                  (5 * xi[0] * (-2 + (3 + sqrt5) * xi[0] - 2 * (-3 + sqrt5) * xi[2])) / 2.};
        case 13:
          return {-27 * xi[1] * xi[2], -27 * xi[2] * (-1 + xi[0] + 2 * xi[1] + xi[2]),
                  -27 * xi[1] * (-1 + xi[0] + xi[1] + 2 * xi[2])};
        case 14:
          return {27 * xi[1] * xi[2], 27 * xi[0] * xi[2], 27 * xi[0] * xi[1]};
        case 15:
          return {0, (-5 * xi[2] * (2 - 2 * (3 + sqrt5) * xi[1] + (-3 + sqrt5) * xi[2])) / 2.,
                  (5 * xi[1] * (-2 + (3 + sqrt5) * xi[1] - 2 * (-3 + sqrt5) * xi[2])) / 2.};
        case 16:
          return {
              (-5 * xi[2] *
               (4 - 2 * sqrt5 + 2 * (-3 + sqrt5) * xi[0] + 2 * (-3 + sqrt5) * xi[1] + 3 * (-1 + sqrt5) * xi[2])) /
                  2.,
              (-5 * xi[2] *
               (4 - 2 * sqrt5 + 2 * (-3 + sqrt5) * xi[0] + 2 * (-3 + sqrt5) * xi[1] + 3 * (-1 + sqrt5) * xi[2])) /
                  2.,
              (-5 * (-3 + sqrt5) * xi[0] * xi[0]) / 2. -
                  5 * xi[0] * (2 - sqrt5 + (-3 + sqrt5) * xi[1] + 3 * (-1 + sqrt5) * xi[2]) -
                  (5 * (-1 + sqrt5 + (-3 + sqrt5) * xi[1] * xi[1] + 2 * xi[2] * (1 - 3 * sqrt5 + 3 * sqrt5 * xi[2]) +
                        xi[1] * (4 - 2 * sqrt5 + 6 * (-1 + sqrt5) * xi[2]))) /
                      2.};
        case 17:
          return {(5 * xi[2] * (-2 - 2 * (-3 + sqrt5) * xi[0] + (3 + sqrt5) * xi[2])) / 2., 0,
                  (-5 * xi[0] * (2 + (-3 + sqrt5) * xi[0] - 2 * (3 + sqrt5) * xi[2])) / 2.};
        case 18:
          return {0, (5 * xi[2] * (-2 - 2 * (-3 + sqrt5) * xi[1] + (3 + sqrt5) * xi[2])) / 2.,
                  (-5 * xi[1] * (2 + (-3 + sqrt5) * xi[1] - 2 * (3 + sqrt5) * xi[2])) / 2.};
        case 19:
          return {-(xi[2] * (-1 + 2 * xi[0] + xi[1] + xi[2])), -(xi[2] * (-1 + xi[0] + 2 * xi[1] + xi[2])),
                  1 + xi[0] - xi[0] * xi[0] + xi[1] - xi[0] * xi[1] - xi[1] * xi[1] - 2 * (5 + xi[0] + xi[1]) * xi[2] +
                      15 * xi[2] * xi[2]};
      }
    }

    return {};
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_functions(tensor<double, dim> xi)
  {
    tensor<double, ndof> output{};
    for (int i = 0; i < ndof; i++) {
      output[i] = shape_function(xi, i);
    }
    return output;
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_function_gradients(tensor<double, dim> xi)
  {
    tensor<double, ndof, dim> output{};
    for (int i = 0; i < ndof; i++) {
      output[i] = shape_function_gradient(xi, i);
    }
    return output;
  }

  template <typename in_t, int q>
  static auto batch_apply_shape_fn(int j, tensor<in_t, nqpts(q)> input, const TensorProductQuadratureRule<q>&)
  {
    using source_t = decltype(get<0>(get<0>(in_t{})) + dot(get<1>(get<0>(in_t{})), tensor<double, dim>{}));
    using flux_t = decltype(get<0>(get<1>(in_t{})) + dot(get<1>(get<1>(in_t{})), tensor<double, dim>{}));

    constexpr auto xi = GaussLegendreNodes<q, mfem::Geometry::TETRAHEDRON>();

    tensor<tuple<source_t, flux_t>, nqpts(q)> output;

    for (int i = 0; i < nqpts(q); i++) {
      double phi_j = shape_function(xi[i], j);
      tensor<double, dim> dphi_j_dxi = shape_function_gradient(xi[i], j);

      const auto& d00 = get<0>(get<0>(input(i)));
      const auto& d01 = get<1>(get<0>(input(i)));
      const auto& d10 = get<0>(get<1>(input(i)));
      const auto& d11 = get<1>(get<1>(input(i)));

      output[i] = {d00 * phi_j + dot(d01, dphi_j_dxi), d10 * phi_j + dot(d11, dphi_j_dxi)};
    }

    return output;
  }

  template <int q>
  SMITH_HOST_DEVICE static auto interpolate(const tensor<double, c, ndof>& X, const TensorProductQuadratureRule<q>&)
  {
    constexpr auto xi = GaussLegendreNodes<q, mfem::Geometry::TETRAHEDRON>();

    // transpose the quadrature data into a flat tensor of tuples
    union {
      tensor<tuple<tensor<double, c>, tensor<double, c, dim> >, nqpts(q)> unflattened;
      tensor<qf_input_type, nqpts(q)> flattened;
    } output{};

    for (int i = 0; i < c; i++) {
      for (int j = 0; j < nqpts(q); j++) {
        for (int k = 0; k < ndof; k++) {
          get<VALUE>(output.unflattened[j])[i] += X(i, k) * shape_function(xi[j], k);
          get<GRADIENT>(output.unflattened[j])[i] += X(i, k) * shape_function_gradient(xi[j], k);
        }
      }
    }

    return output.flattened;
  }

  template <typename source_type, typename flux_type, int q>
  SMITH_HOST_DEVICE static void integrate(const tensor<tuple<source_type, flux_type>, nqpts(q)>& qf_output,
                                          const TensorProductQuadratureRule<q>&,
                                          tensor<double, c, ndof>* element_residual, int step = 1)
  {
    if constexpr (is_zero<source_type>{} && is_zero<flux_type>{}) {
      return;
    }

    using source_component_type = std::conditional_t<is_zero<source_type>{}, zero, double>;
    using flux_component_type = std::conditional_t<is_zero<flux_type>{}, zero, tensor<double, dim> >;

    constexpr int ntrial = std::max(size(source_type{}), size(flux_type{}) / dim) / c;
    constexpr auto integration_points = GaussLegendreNodes<q, mfem::Geometry::TETRAHEDRON>();
    constexpr auto integration_weights = GaussLegendreWeights<q, mfem::Geometry::TETRAHEDRON>();

    for (int j = 0; j < ntrial; j++) {
      for (int i = 0; i < c; i++) {
        for (int Q = 0; Q < nqpts(q); Q++) {
          tensor<double, dim> xi = integration_points[Q];
          double wt = integration_weights[Q];

          source_component_type source;
          if constexpr (!is_zero<source_type>{}) {
            source = reinterpret_cast<const double*>(&get<SOURCE>(qf_output[Q]))[i * ntrial + j];
          }

          flux_component_type flux;
          if constexpr (!is_zero<flux_type>{}) {
            for (int k = 0; k < dim; k++) {
              flux[k] = reinterpret_cast<const double*>(&get<FLUX>(qf_output[Q]))[(i * dim + k) * ntrial + j];
            }
          }

          for (int k = 0; k < ndof; k++) {
            element_residual[j * step](i, k) +=
                (source * shape_function(xi, k) + dot(flux, shape_function_gradient(xi, k))) * wt;
          }
        }
      }
    }
  }
};
/// @endcond
