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
//   Hdiv<3> -> RT_2: 15 DOFs (3 per edge + 6 interior)
//
// Shape functions are computed via a Vandermonde approach matching mfem:
//   1. Build a raw polynomial basis (Chebyshev polynomials in barycentric coords)
//   2. Build the Vandermonde matrix T at DOF nodes (Gauss-Legendre open points)
//   3. Shape functions = inv(T) * raw_basis
//
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p>
struct finite_element<mfem::Geometry::TRIANGLE, Hdiv<p> > {
  static constexpr auto geometry = mfem::Geometry::TRIANGLE;
  static constexpr auto family = Family::HDIV;
  static constexpr int dim = 2;
  static constexpr int n = p;
  static constexpr int ndof = p * (p + 2);
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

  // mfem polynomial order (RT_{mp} element)
  static constexpr int mp = p - 1;

  // Evaluate Chebyshev polynomials T_0(2x-1), ..., T_order(2x-1) at x
  SMITH_HOST_DEVICE static constexpr void chebyshev_eval(int order, double x, double* u)
  {
    u[0] = 1.0;
    if (order < 1) return;
    double z = 2.0 * x - 1.0;
    u[1] = z;
    for (int i = 1; i < order; i++) {
      u[i + 1] = 2.0 * z * u[i] - u[i - 1];
    }
  }

  // Evaluate Chebyshev polynomials and their derivatives (w.r.t. x, not z)
  SMITH_HOST_DEVICE static constexpr void chebyshev_eval_d(int order, double x, double* u, double* d)
  {
    u[0] = 1.0;
    d[0] = 0.0;
    if (order < 1) return;
    double z = 2.0 * x - 1.0;
    u[1] = z;
    d[1] = 2.0;
    for (int i = 1; i < order; i++) {
      u[i + 1] = 2.0 * z * u[i] - u[i - 1];
      d[i + 1] = double(i + 1) * (z * d[i] / double(i) + 2.0 * u[i]);
    }
  }

  // Raw (pre-Vandermonde) basis functions at point xi, matching mfem convention.
  // Face DOFs: (s, 0) and (0, s)  where s = T_i(2x-1)*T_j(2y-1)*T_k(2(1-x-y)-1)
  // Interior DOFs: ((x-1/3)*s, (y-1/3)*s)  where s = T_i(2x-1)*T_j(2y-1)
  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> raw_vshape(tensor<double, dim> xi)
  {
    double sx[mp + 1], sy[mp + 1], sl[mp + 1];
    chebyshev_eval(mp, xi[0], sx);
    chebyshev_eval(mp, xi[1], sy);
    chebyshev_eval(mp, 1.0 - xi[0] - xi[1], sl);

    tensor<double, ndof, dim> raw{};
    int o = 0;
    for (int j = 0; j <= mp; j++) {
      for (int i = 0; i + j <= mp; i++) {
        double s = sx[i] * sy[j] * sl[mp - i - j];
        raw[o][0] = s;
        raw[o][1] = 0.0;
        o++;
        raw[o][0] = 0.0;
        raw[o][1] = s;
        o++;
      }
    }
    constexpr double c = 1.0 / 3.0;
    for (int i = 0; i <= mp; i++) {
      double s = sx[i] * sy[mp - i];
      raw[o][0] = (xi[0] - c) * s;
      raw[o][1] = (xi[1] - c) * s;
      o++;
    }
    return raw;
  }

  // Raw divergence of the pre-Vandermonde basis
  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> raw_divshape(tensor<double, dim> xi)
  {
    double sx[mp + 1], sy[mp + 1], sl[mp + 1];
    double dsx[mp + 1], dsy[mp + 1], dsl[mp + 1];
    chebyshev_eval_d(mp, xi[0], sx, dsx);
    chebyshev_eval_d(mp, xi[1], sy, dsy);
    chebyshev_eval_d(mp, 1.0 - xi[0] - xi[1], sl, dsl);

    tensor<double, ndof> raw{};
    int o = 0;
    for (int j = 0; j <= mp; j++) {
      for (int i = 0; i + j <= mp; i++) {
        int k = mp - i - j;
        // div(s,0) = ds/dx;  dl/dx = -1 gives the subtraction
        raw[o++] = (dsx[i] * sl[k] - sx[i] * dsl[k]) * sy[j];
        // div(0,s) = ds/dy;  dl/dy = -1
        raw[o++] = (dsy[j] * sl[k] - sy[j] * dsl[k]) * sx[i];
      }
    }
    constexpr double c = 1.0 / 3.0;
    for (int i = 0; i <= mp; i++) {
      int j = mp - i;
      // div((x-c)*s, (y-c)*s) = 2*s + (x-c)*ds/dx + (y-c)*ds/dy
      raw[o++] = (sx[i] + (xi[0] - c) * dsx[i]) * sy[j] + (sy[j] + (xi[1] - c) * dsy[j]) * sx[i];
    }
    return raw;
  }

  // DOF node locations (matching mfem's RT_TriangleElement constructor)
  // Edge DOFs at Gauss-Legendre open points, interior DOFs at barycentric-weighted points
  static constexpr auto nodes = [] {
    tensor<double, ndof, dim> nd{};
    constexpr auto bop = GaussLegendreNodes<p, mfem::Geometry::SEGMENT>();
    int o = 0;
    // edge 0: (0,0)-(1,0), y=0
    for (int i = 0; i < p; i++) {
      nd[o++] = {bop[i], 0.0};
    }
    // edge 1: (1,0)-(0,1)
    for (int i = 0; i < p; i++) {
      nd[o++] = {bop[p - 1 - i], bop[i]};
    }
    // edge 2: (0,1)-(0,0), x=0
    for (int i = 0; i < p; i++) {
      nd[o++] = {0.0, bop[p - 1 - i]};
    }
    // interior DOFs (two per interior point, same location, different normals)
    if constexpr (p > 1) {
      constexpr auto iop = GaussLegendreNodes<p - 1, mfem::Geometry::SEGMENT>();
      for (int j = 0; j < p - 1; j++) {
        for (int i = 0; i + j < p - 1; i++) {
          double w = iop[i] + iop[j] + iop[p - 2 - i - j];
          nd[o++] = {iop[i] / w, iop[j] / w};
          nd[o++] = {iop[i] / w, iop[j] / w};
        }
      }
    }
    return nd;
  }();

  // Normal directions associated with each DOF (for the DOF functional phi_i(x_j) . dir_j = delta_ij)
  static constexpr auto directions = [] {
    tensor<double, ndof, dim> d{};
    int o = 0;
    for (int i = 0; i < p; i++) d[o++] = {0.0, -1.0};   // edge 0
    for (int i = 0; i < p; i++) d[o++] = {1.0, 1.0};    // edge 1
    for (int i = 0; i < p; i++) d[o++] = {-1.0, 0.0};   // edge 2
    if constexpr (p > 1) {
      for (int j = 0; j < p - 1; j++) {
        for (int i = 0; i + j < p - 1; i++) {
          d[o++] = {0.0, -1.0};   // interior: nk index 0
          d[o++] = {-1.0, 0.0};   // interior: nk index 2
        }
      }
    }
    return d;
  }();

  // Inverse Vandermonde matrix: Ti[i][j] transforms raw basis j into shape function i
  // T[raw_idx][dof_idx] = raw_vshape[raw_idx](node[dof_idx]) . direction[dof_idx]
  static constexpr auto Ti = [] {
    tensor<double, ndof, ndof> T{};
    for (int k = 0; k < ndof; k++) {
      auto raw = raw_vshape(nodes[k]);
      for (int r = 0; r < ndof; r++) {
        T[r][k] = raw[r][0] * directions[k][0] + raw[r][1] * directions[k][1];
      }
    }
    return inv(T);
  }();

  // Evaluate all ndof shape functions (vector-valued) at point xi
  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_functions(
      [[maybe_unused]] tensor<double, dim> xi)
  {
    auto raw = raw_vshape(xi);
    tensor<double, ndof, dim> N{};
    for (int i = 0; i < ndof; i++) {
      for (int j = 0; j < ndof; j++) {
        N[i][0] += Ti[i][j] * raw[j][0];
        N[i][1] += Ti[i][j] * raw[j][1];
      }
    }
    return N;
  }

  // Evaluate divergence of all ndof shape functions at point xi
  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_function_div(
      [[maybe_unused]] tensor<double, dim> xi)
  {
    auto raw = raw_divshape(xi);
    tensor<double, ndof> d{};
    for (int i = 0; i < ndof; i++) {
      for (int j = 0; j < ndof; j++) {
        d[i] += Ti[i][j] * raw[j];
      }
    }
    return d;
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
