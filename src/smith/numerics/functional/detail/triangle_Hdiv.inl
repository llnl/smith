// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file triangle_Hdiv.inl
 *
 * @brief Specialization of finite_element for Hdiv (Raviart-Thomas) on triangle geometry
 */

// This specialization defines RT shape functions on the reference triangle
// with vertices at {(0,0), (1,0), (0,1)}.
//
// Convention: Hdiv<p> corresponds to RT_FECollection(p-1, dim) in mfem,
// matching the tensor-product convention where Hcurl<p> <-> ND_FECollection(p).
// The de Rham sequence is:  H1<p> -> Hcurl<p> -> Hdiv<p> -> L2<p-1>
//
// RT_{p-1} on triangles has ndof = p*(p+2) DOFs.
//   Hdiv<1> -> RT_0: 3 DOFs (one per edge)
//   Hdiv<2> -> RT_1: 8 DOFs (2 per edge + 2 interior)
//
// The shape functions are vector-valued and the relevant derivative is the scalar divergence.
// DOF ordering follows mfem's RT_FECollection convention.
//
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p>
struct finite_element<mfem::Geometry::TRIANGLE, Hdiv<p> > {
  static constexpr auto geometry = mfem::Geometry::TRIANGLE;
  static constexpr auto family = Family::HDIV;
  static constexpr int dim = 2;
  static constexpr int n = p;
  static constexpr int ndof = p * (p + 2);  // = (RT_{p-1} on triangle) = p*(p+2)
  static constexpr int components = 1;

  static constexpr int VALUE = 0, DIV = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  using dof_type = tensor<double, ndof>;
  using dof_type_if = dof_type;

  using value_type = tensor<double, dim>;
  using derivative_type = double;
  using qf_input_type = tuple<value_type, derivative_type>;

  /*
    Hdiv<1> -> RT_0 on triangle: 3 DOFs (one per edge, normal component)

    Edge 0: (0,0)-(1,0),  outward normal (0,-1)
    Edge 1: (1,0)-(0,1),  outward normal (1,1)/sqrt(2)
    Edge 2: (0,1)-(0,0),  outward normal (-1,0)

    The RT0 basis functions (mfem convention, scaled for unit normal flux):
      phi_0 = (x, y-1)       div = 2
      phi_1 = (x, y)         div = 2
      phi_2 = (x-1, y)       div = 2
  */

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_functions(
      [[maybe_unused]] tensor<double, dim> xi)
  {
    tensor<double, ndof, dim> N{};

    if constexpr (p == 1) {
      // Hdiv<1> -> RT_0: 3 basis functions
      double x = xi[0], y = xi[1];
      N[0] = {x, y - 1.0};
      N[1] = {x, y};
      N[2] = {x - 1.0, y};
    }

    if constexpr (p == 2) {
      // Hdiv<2> -> RT_1: 8 basis functions
      // TODO: verify against mfem::RT_TriangleElement::CalcVShape
      double x = xi[0], y = xi[1];
      double l0 = 1.0 - x - y;  // barycentric coord for vertex 0

      N[0] = {-2.0 * x * l0, -2.0 * y * l0};
      N[1] = {2.0 * x * x, 2.0 * x * y};
      N[2] = {2.0 * x * y, 2.0 * y * y};
      N[3] = {-x * (1.0 - 2.0 * y), -y * (1.0 - 2.0 * y)};
      N[4] = {x * (2.0 * x - 1.0), y * (2.0 * x - 1.0)};
      N[5] = {x * (1.0 - 2.0 * x), y * (1.0 - 2.0 * x)};
      // Interior DOFs (2)
      N[6] = {x, 0.0};
      N[7] = {0.0, y};
    }

    return N;
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_function_div(
      [[maybe_unused]] tensor<double, dim> xi)
  {
    tensor<double, ndof> div{};

    if constexpr (p == 1) {
      // Hdiv<1> -> RT_0: constant divergence = 2
      div[0] = 2.0;
      div[1] = 2.0;
      div[2] = 2.0;
    }

    if constexpr (p == 2) {
      // Hdiv<2> -> RT_1: linear divergence
      double x = xi[0], y = xi[1];
      div[0] = -2.0 * (1.0 - 2.0 * x - 2.0 * y) + 2.0 * (1.0 - x - y) * 2.0;
      div[1] = 4.0 * x + 2.0 * x;
      div[2] = 2.0 * y + 4.0 * y;
      div[3] = -(1.0 - 2.0 * y) - (1.0 - 2.0 * y);
      div[4] = (2.0 * x - 1.0) + (2.0 * x - 1.0);
      div[5] = (1.0 - 2.0 * x) + (1.0 - 2.0 * x);
      div[6] = 1.0;
      div[7] = 1.0;
    }

    return div;
  }

  template <typename in_t, int q>
  static auto batch_apply_shape_fn(int j, tensor<in_t, q*(q + 1) / 2> input, const TensorProductQuadratureRule<q>&)
  {
    using source_t = decltype(dot(get<0>(get<0>(in_t{})), tensor<double, dim>{}) + get<1>(get<0>(in_t{})) * double{});
    using flux_t = decltype(dot(get<0>(get<1>(in_t{})), tensor<double, dim>{}) + get<1>(get<1>(in_t{})) * double{});

    constexpr auto xi = GaussLegendreNodes<q, mfem::Geometry::TRIANGLE>();

    static constexpr int Q = q * (q + 1) / 2;
    tensor<tuple<source_t, flux_t>, Q> output;

    for (int i = 0; i < Q; i++) {
      auto all_shapes = shape_functions(xi[i]);
      auto all_divs = shape_function_div(xi[i]);

      tensor<double, dim> phi_j = all_shapes[j];
      double div_phi_j = all_divs[j];

      const auto& d00 = get<0>(get<0>(input(i)));
      const auto& d01 = get<1>(get<0>(input(i)));
      const auto& d10 = get<0>(get<1>(input(i)));
      const auto& d11 = get<1>(get<1>(input(i)));

      output[i] = {dot(d00, phi_j) + d01 * div_phi_j, dot(d10, phi_j) + d11 * div_phi_j};
    }

    return output;
  }

  template <int q>
  SMITH_HOST_DEVICE static auto interpolate(const dof_type& element_values, const TensorProductQuadratureRule<q>&)
  {
    constexpr auto xi = GaussLegendreNodes<q, mfem::Geometry::TRIANGLE>();
    constexpr int Q = q * (q + 1) / 2;

    tensor<tuple<tensor<double, dim>, double>, Q> qf_inputs;

    for (int i = 0; i < Q; i++) {
      auto N = shape_functions(xi[i]);
      auto divN = shape_function_div(xi[i]);

      tensor<double, dim> val{};
      double div_val = 0.0;

      for (int j = 0; j < ndof; j++) {
        for (int d = 0; d < dim; d++) {
          val[d] += N[j][d] * element_values[j];
        }
        div_val += divN[j] * element_values[j];
      }

      get<VALUE>(qf_inputs(i)) = val;
      get<DIV>(qf_inputs(i)) = div_val;
    }

    return qf_inputs;
  }

  template <typename source_type, typename flux_type, int q>
  SMITH_HOST_DEVICE static void integrate(const tensor<tuple<source_type, flux_type>, q*(q + 1) / 2>& qf_output,
                                          const TensorProductQuadratureRule<q>&, dof_type* element_residual,
                                          [[maybe_unused]] int step = 1)
  {
    constexpr auto xi = GaussLegendreNodes<q, mfem::Geometry::TRIANGLE>();
    [[maybe_unused]] constexpr auto weights = GaussLegendreWeights<q, mfem::Geometry::TRIANGLE>();
    constexpr int Q = q * (q + 1) / 2;

    if constexpr (is_zero<source_type>{} && is_zero<flux_type>{}) return;

    for (int i = 0; i < Q; i++) {
      auto N = shape_functions(xi[i]);
      auto divN = shape_function_div(xi[i]);

      for (int j = 0; j < ndof; j++) {
        // source is dual to value, flux is dual to divergence
        double contrib = 0.0;
        if constexpr (!is_zero<source_type>{}) {
          tensor<double, dim> s{get<SOURCE>(qf_output[i])};
          contrib += dot(s, N[j]);
        }
        if constexpr (!is_zero<flux_type>{}) {
          double f = get<FLUX>(qf_output[i]);
          contrib += f * divN[j];
        }
        element_residual[0][j] += contrib * weights[i];
      }
    }
  }
};
/// @endcond
