// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file tetrahedron_Hdiv.inl
 *
 * @brief Specialization of finite_element for Hdiv (Raviart-Thomas) on tetrahedron geometry
 */

// This specialization defines RT shape functions on the reference tetrahedron
// with vertices at {(0,0,0), (1,0,0), (0,1,0), (0,0,1)}.
//
// Convention: Hdiv<p> corresponds to RT_FECollection(p-1, dim) in mfem.
// RT_{p-1} on tetrahedra has ndof = p*(p+1)*(p+3)/2 DOFs.
//   Hdiv<1> -> RT_0: 4 DOFs (one per face)
//   Hdiv<2> -> RT_1: 15 DOFs (3 per face + 3 interior)
//   Hdiv<3> -> RT_2: 36 DOFs (6 per face + 12 interior)
//
// Shape functions are computed via a Vandermonde approach matching mfem:
//   1. Build a raw polynomial basis (Chebyshev polynomials in barycentric coords)
//   2. Build the Vandermonde matrix T at DOF nodes
//   3. Shape functions = inv(T) * raw_basis
//
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p>
struct finite_element<mfem::Geometry::TETRAHEDRON, Hdiv<p> > {
  static constexpr auto geometry = mfem::Geometry::TETRAHEDRON;
  static constexpr auto family = Family::HDIV;
  static constexpr int dim = 3;
  static constexpr int n = p;
  static constexpr int ndof = p * (p + 1) * (p + 3) / 2;
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

  // Evaluate Chebyshev polynomials and their derivatives (w.r.t. x)
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
  // Face DOFs: (s,0,0), (0,s,0), (0,0,s) where s = Tx_i * Ty_j * Tz_k * Tl_{mp-i-j-k}
  // Interior DOFs: ((x-1/4)*s, (y-1/4)*s, (z-1/4)*s) where s = Tx_i * Ty_j * Tz_{mp-i-j}
  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> raw_vshape(tensor<double, dim> xi)
  {
    double sx[mp + 1], sy[mp + 1], sz[mp + 1], sl[mp + 1];
    chebyshev_eval(mp, xi[0], sx);
    chebyshev_eval(mp, xi[1], sy);
    chebyshev_eval(mp, xi[2], sz);
    chebyshev_eval(mp, 1.0 - xi[0] - xi[1] - xi[2], sl);

    tensor<double, ndof, dim> raw{};
    int o = 0;
    for (int k = 0; k <= mp; k++) {
      for (int j = 0; j + k <= mp; j++) {
        for (int i = 0; i + j + k <= mp; i++) {
          double s = sx[i] * sy[j] * sz[k] * sl[mp - i - j - k];
          raw[o][0] = s;  raw[o][1] = 0;  raw[o][2] = 0;  o++;
          raw[o][0] = 0;  raw[o][1] = s;  raw[o][2] = 0;  o++;
          raw[o][0] = 0;  raw[o][1] = 0;  raw[o][2] = s;  o++;
        }
      }
    }
    constexpr double c = 1.0 / 4.0;
    for (int j = 0; j <= mp; j++) {
      for (int i = 0; i + j <= mp; i++) {
        double s = sx[i] * sy[j] * sz[mp - i - j];
        raw[o][0] = (xi[0] - c) * s;
        raw[o][1] = (xi[1] - c) * s;
        raw[o][2] = (xi[2] - c) * s;
        o++;
      }
    }
    return raw;
  }

  // Raw divergence of the pre-Vandermonde basis
  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> raw_divshape(tensor<double, dim> xi)
  {
    double sx[mp + 1], sy[mp + 1], sz[mp + 1], sl[mp + 1];
    double dsx[mp + 1], dsy[mp + 1], dsz[mp + 1], dsl[mp + 1];
    chebyshev_eval_d(mp, xi[0], sx, dsx);
    chebyshev_eval_d(mp, xi[1], sy, dsy);
    chebyshev_eval_d(mp, xi[2], sz, dsz);
    chebyshev_eval_d(mp, 1.0 - xi[0] - xi[1] - xi[2], sl, dsl);

    tensor<double, ndof> raw{};
    int o = 0;
    for (int k = 0; k <= mp; k++) {
      for (int j = 0; j + k <= mp; j++) {
        for (int i = 0; i + j + k <= mp; i++) {
          int l = mp - i - j - k;
          // div(s,0,0) = ds/dx;  dl/dx = -1
          raw[o++] = (dsx[i] * sl[l] - sx[i] * dsl[l]) * sy[j] * sz[k];
          // div(0,s,0) = ds/dy;  dl/dy = -1
          raw[o++] = (dsy[j] * sl[l] - sy[j] * dsl[l]) * sx[i] * sz[k];
          // div(0,0,s) = ds/dz;  dl/dz = -1
          raw[o++] = (dsz[k] * sl[l] - sz[k] * dsl[l]) * sx[i] * sy[j];
        }
      }
    }
    constexpr double c = 1.0 / 4.0;
    for (int j = 0; j <= mp; j++) {
      for (int i = 0; i + j <= mp; i++) {
        int k = mp - i - j;
        // div((x-c)*s, (y-c)*s, (z-c)*s) = 3*s + (x-c)*ds/dx + (y-c)*ds/dy + (z-c)*ds/dz
        raw[o++] = (sx[i] + (xi[0] - c) * dsx[i]) * sy[j] * sz[k] +
                   (sy[j] + (xi[1] - c) * dsy[j]) * sx[i] * sz[k] +
                   (sz[k] + (xi[2] - c) * dsz[k]) * sx[i] * sy[j];
      }
    }
    return raw;
  }

  // DOF node locations matching mfem's RT_TetrahedronElement constructor.
  // Face DOFs at barycentric GL points on each face, interior DOFs at barycentric GL points.
  // face_dofs_per_face = p*(p+1)/2
  static constexpr auto nodes = [] {
    tensor<double, ndof, dim> nd{};
    constexpr auto bop = GaussLegendreNodes<p, mfem::Geometry::SEGMENT>();
    int o = 0;

    // face 0: vertices (1,0,0),(0,1,0),(0,0,1)  -- nk[0]=(1,1,1)
    for (int j = 0; j <= mp; j++) {
      for (int i = 0; i + j <= mp; i++) {
        double w = bop[i] + bop[j] + bop[mp - i - j];
        nd[o++] = {bop[mp - i - j] / w, bop[i] / w, bop[j] / w};
      }
    }
    // face 1: vertices (0,0,0),(0,0,1),(0,1,0), x=0  -- nk[1]=(-1,0,0)
    for (int j = 0; j <= mp; j++) {
      for (int i = 0; i + j <= mp; i++) {
        double w = bop[i] + bop[j] + bop[mp - i - j];
        nd[o++] = {0.0, bop[j] / w, bop[i] / w};
      }
    }
    // face 2: vertices (0,0,0),(1,0,0),(0,0,1), y=0  -- nk[2]=(0,-1,0)
    for (int j = 0; j <= mp; j++) {
      for (int i = 0; i + j <= mp; i++) {
        double w = bop[i] + bop[j] + bop[mp - i - j];
        nd[o++] = {bop[i] / w, 0.0, bop[j] / w};
      }
    }
    // face 3: vertices (0,0,0),(0,1,0),(1,0,0), z=0  -- nk[3]=(0,0,-1)
    for (int j = 0; j <= mp; j++) {
      for (int i = 0; i + j <= mp; i++) {
        double w = bop[i] + bop[j] + bop[mp - i - j];
        nd[o++] = {bop[j] / w, bop[i] / w, 0.0};
      }
    }

    // interior DOFs (3 per interior point)
    if constexpr (p > 1) {
      constexpr auto iop = GaussLegendreNodes<p - 1, mfem::Geometry::SEGMENT>();
      for (int k = 0; k < p - 1; k++) {
        for (int j = 0; j + k < p - 1; j++) {
          for (int i = 0; i + j + k < p - 1; i++) {
            double w = iop[i] + iop[j] + iop[k] + iop[p - 2 - i - j - k];
            nd[o++] = {iop[i] / w, iop[j] / w, iop[k] / w};
            nd[o++] = {iop[i] / w, iop[j] / w, iop[k] / w};
            nd[o++] = {iop[i] / w, iop[j] / w, iop[k] / w};
          }
        }
      }
    }
    return nd;
  }();

  // Normal directions associated with each DOF
  static constexpr auto directions = [] {
    tensor<double, ndof, dim> d{};
    int o = 0;
    constexpr int fdofs = p * (p + 1) / 2;  // DOFs per face
    for (int i = 0; i < fdofs; i++) d[o++] = {1.0, 1.0, 1.0};    // face 0
    for (int i = 0; i < fdofs; i++) d[o++] = {-1.0, 0.0, 0.0};   // face 1
    for (int i = 0; i < fdofs; i++) d[o++] = {0.0, -1.0, 0.0};   // face 2
    for (int i = 0; i < fdofs; i++) d[o++] = {0.0, 0.0, -1.0};   // face 3
    if constexpr (p > 1) {
      for (int k = 0; k < p - 1; k++) {
        for (int j = 0; j + k < p - 1; j++) {
          for (int i = 0; i + j + k < p - 1; i++) {
            d[o++] = {-1.0, 0.0, 0.0};   // nk[1]
            d[o++] = {0.0, -1.0, 0.0};   // nk[2]
            d[o++] = {0.0, 0.0, -1.0};   // nk[3]
          }
        }
      }
    }
    return d;
  }();

  // Inverse Vandermonde matrix
  static constexpr auto Ti = [] {
    tensor<double, ndof, ndof> T{};
    for (int k = 0; k < ndof; k++) {
      auto raw = raw_vshape(nodes[k]);
      for (int r = 0; r < ndof; r++) {
        T[r][k] = raw[r][0] * directions[k][0] + raw[r][1] * directions[k][1] +
                  raw[r][2] * directions[k][2];
      }
    }
    return inv(T);
  }();

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_functions(
      [[maybe_unused]] tensor<double, dim> xi)
  {
    auto raw = raw_vshape(xi);
    tensor<double, ndof, dim> N{};
    for (int i = 0; i < ndof; i++) {
      for (int j = 0; j < ndof; j++) {
        N[i][0] += Ti[i][j] * raw[j][0];
        N[i][1] += Ti[i][j] * raw[j][1];
        N[i][2] += Ti[i][j] * raw[j][2];
      }
    }
    return N;
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_function_div(
      [[maybe_unused]] tensor<double, dim> xi)
  {
    auto raw = raw_divshape(xi);
    tensor<double, ndof> div{};
    for (int i = 0; i < ndof; i++) {
      for (int j = 0; j < ndof; j++) {
        div[i] += Ti[i][j] * raw[j];
      }
    }
    return div;
  }

  template <typename in_t, int q>
  static auto batch_apply_shape_fn(int j, tensor<in_t, q*(q + 1)*(q + 2) / 6> input,
                                   const TensorProductQuadratureRule<q>&)
  {
    using source_t = decltype(dot(get<0>(get<0>(in_t{})), tensor<double, dim>{}) + get<1>(get<0>(in_t{})) * double{});
    using flux_t = decltype(dot(get<0>(get<1>(in_t{})), tensor<double, dim>{}) + get<1>(get<1>(in_t{})) * double{});

    constexpr auto xi = GaussLegendreNodes<q, mfem::Geometry::TETRAHEDRON>();

    static constexpr int Q = q * (q + 1) * (q + 2) / 6;
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
    constexpr auto xi = GaussLegendreNodes<q, mfem::Geometry::TETRAHEDRON>();
    constexpr int Q = q * (q + 1) * (q + 2) / 6;

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
  SMITH_HOST_DEVICE static void integrate(
      const tensor<tuple<source_type, flux_type>, q*(q + 1)*(q + 2) / 6>& qf_output,
      const TensorProductQuadratureRule<q>&, dof_type* element_residual,
      [[maybe_unused]] int step = 1)
  {
    constexpr auto xi = GaussLegendreNodes<q, mfem::Geometry::TETRAHEDRON>();
    [[maybe_unused]] constexpr auto weights = GaussLegendreWeights<q, mfem::Geometry::TETRAHEDRON>();
    constexpr int Q = q * (q + 1) * (q + 2) / 6;

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
