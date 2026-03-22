// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file hexahedron_Hdiv.inl
 *
 * @brief Specialization of finite_element for Hdiv on hexahedron geometry
 */

// this specialization defines shape functions (and their divergences) that
// interpolate at Gauss-Lobatto nodes for closed intervals, and Gauss-Legendre
// nodes for open intervals.
//
// note 1: mfem assumes the parent element domain is [0,1]x[0,1]x[0,1]
// note 2: dofs are numbered by direction and then lexicographically in space.
// note 3: H(div) is the "dual" of H(curl) -- the roles of open and closed
//         nodes are swapped relative to hexahedron_Hcurl.inl:
//         - Hcurl: open (Legendre) in component direction, closed (Lobatto) in transverse
//         - Hdiv:  closed (Lobatto) in component direction, open (Legendre) in transverse
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p>
struct finite_element<mfem::Geometry::CUBE, Hdiv<p>> {
  static constexpr auto geometry = mfem::Geometry::CUBE;
  static constexpr auto family = Family::HDIV;
  static constexpr int dim = 3;
  static constexpr int n = p + 1;
  static constexpr int ndof = 3 * p * p * (p + 1);
  static constexpr int components = 1;

  static constexpr int VALUE = 0, DIV = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components>>::type;

  // DOF layout: each component has closed (Lobatto, p+1) in its own direction,
  // open (Legendre, p) in the two transverse directions.
  // This is the transpose of Hcurl's layout.
  struct dof_type {
    tensor<double, p, p, p + 1> x;    // open-z, open-y, closed-x
    tensor<double, p, p + 1, p> y;    // open-z, closed-y, open-x
    tensor<double, p + 1, p, p> z;    // closed-z, open-y, open-x
  };

  template <int q>
  using cpu_batched_values_type = tensor<tensor<double, 3>, q, q, q>;

  template <int q>
  using cpu_batched_derivatives_type = tensor<double, q, q, q>;

  static constexpr auto directions = [] {
    int dof_per_direction = p * p * (p + 1);

    tensor<double, ndof, dim> directions{};
    for (int i = 0; i < dof_per_direction; i++) {
      directions[i + 0 * dof_per_direction] = {1.0, 0.0, 0.0};
      directions[i + 1 * dof_per_direction] = {0.0, 1.0, 0.0};
      directions[i + 2 * dof_per_direction] = {0.0, 0.0, 1.0};
    }
    return directions;
  }();

  static constexpr auto nodes = []() {
    auto legendre_nodes = GaussLegendreNodes<p, mfem::Geometry::SEGMENT>();
    auto lobatto_nodes = GaussLobattoNodes<p + 1>();

    tensor<double, ndof, dim> nodes{};

    int count = 0;
    // x-facing DOFs: closed in x, open in y, open in z
    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          nodes[count++] = {lobatto_nodes[i], legendre_nodes[j], legendre_nodes[k]};
        }
      }
    }

    // y-facing DOFs: open in x, closed in y, open in z
    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          nodes[count++] = {legendre_nodes[i], lobatto_nodes[j], legendre_nodes[k]};
        }
      }
    }

    // z-facing DOFs: open in x, open in y, closed in z
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p; i++) {
          nodes[count++] = {legendre_nodes[i], legendre_nodes[j], lobatto_nodes[k]};
        }
      }
    }

    return nodes;
  }();

  // B1/B2/G2 basis evaluation matrices — delegated to shared free functions
  // in detail/tensor_product_basis.hpp
  template <bool apply_weights, int q>
  static constexpr auto calculate_B1() { return basis_detail::calculate_B1<p, apply_weights, q>(); }

  template <bool apply_weights, int q>
  static constexpr auto calculate_B2() { return basis_detail::calculate_B2<p, apply_weights, q>(); }

  template <bool apply_weights, int q>
  static constexpr auto calculate_G2() { return basis_detail::calculate_G2<p, apply_weights, q>(); }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_functions(tensor<double, dim> xi)
  {
    tensor<double, ndof, dim> N{};

    // f = open (Legendre), g = closed (Lobatto)
    tensor<double, p> f[3] = {GaussLegendreInterpolation<p>(xi[0]), GaussLegendreInterpolation<p>(xi[1]),
                              GaussLegendreInterpolation<p>(xi[2])};

    tensor<double, p + 1> g[3] = {GaussLobattoInterpolation<p + 1>(xi[0]), GaussLobattoInterpolation<p + 1>(xi[1]),
                                  GaussLobattoInterpolation<p + 1>(xi[2])};

    int count = 0;

    // x-facing DOFs: closed in x, open in y, open in z
    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          N[count++] = {g[0][i] * f[1][j] * f[2][k], 0.0, 0.0};
        }
      }
    }

    // y-facing DOFs: open in x, closed in y, open in z
    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          N[count++] = {0.0, f[0][i] * g[1][j] * f[2][k], 0.0};
        }
      }
    }

    // z-facing DOFs: open in x, open in y, closed in z
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p; i++) {
          N[count++] = {0.0, 0.0, f[0][i] * f[1][j] * g[2][k]};
        }
      }
    }

    return N;
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_function_div(tensor<double, dim> xi)
  {
    tensor<double, ndof> div{};

    tensor<double, p> f[3] = {GaussLegendreInterpolation<p>(xi[0]), GaussLegendreInterpolation<p>(xi[1]),
                              GaussLegendreInterpolation<p>(xi[2])};

    tensor<double, p + 1> dg[3] = {GaussLobattoInterpolationDerivative<p + 1>(xi[0]),
                                   GaussLobattoInterpolationDerivative<p + 1>(xi[1]),
                                   GaussLobattoInterpolationDerivative<p + 1>(xi[2])};

    int count = 0;

    // x-facing DOFs: d(g(x))/dx * f(y) * f(z)
    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          div[count++] = dg[0][i] * f[1][j] * f[2][k];
        }
      }
    }

    // y-facing DOFs: f(x) * d(g(y))/dy * f(z)
    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          div[count++] = f[0][i] * dg[1][j] * f[2][k];
        }
      }
    }

    // z-facing DOFs: f(x) * f(y) * d(g(z))/dz
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p; i++) {
          div[count++] = f[0][i] * f[1][j] * dg[2][k];
        }
      }
    }

    return div;
  }

  template <typename in_t, int q>
  static auto batch_apply_shape_fn(int j, tensor<in_t, q * q * q> input, const TensorProductQuadratureRule<q>&)
  {
    constexpr bool apply_weights = false;
    constexpr tensor<double, q, p> B1 = calculate_B1<apply_weights, q>();
    constexpr tensor<double, q, p + 1> B2 = calculate_B2<apply_weights, q>();
    constexpr tensor<double, q, p + 1> G2 = calculate_G2<apply_weights, q>();

    int dof_per_direction = p * p * (p + 1);
    int jx, jy, jz;
    int dir = j / dof_per_direction;
    int remainder = j % dof_per_direction;
    switch (dir) {
      case 0:  // x-direction: closed-x(p+1), open-y(p), open-z(p)
        jx = remainder % n;
        jy = (remainder / n) % p;
        jz = remainder / (n * p);
        break;

      case 1:  // y-direction: open-x(p), closed-y(p+1), open-z(p)
        jx = remainder % p;
        jy = (remainder / p) % n;
        jz = remainder / (p * n);
        break;

      case 2:  // z-direction: open-x(p), open-y(p), closed-z(p+1)
        jx = remainder % p;
        jy = (remainder / p) % p;
        jz = remainder / (p * p);
        break;
    }

    using source_t = decltype(dot(get<0>(get<0>(in_t{})), vec3{}) + get<1>(get<0>(in_t{})) * double{});
    using flux_t = decltype(dot(get<0>(get<1>(in_t{})), vec3{}) + get<1>(get<1>(in_t{})) * double{});

    tensor<tuple<source_t, flux_t>, q * q * q> output;

    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          tensor<double, 3> phi_j{};
          double div_phi_j = 0.0;

          switch (dir) {
            case 0:
              phi_j[0] = B2(qx, jx) * B1(qy, jy) * B1(qz, jz);
              div_phi_j = G2(qx, jx) * B1(qy, jy) * B1(qz, jz);
              break;

            case 1:
              phi_j[1] = B1(qx, jx) * B2(qy, jy) * B1(qz, jz);
              div_phi_j = B1(qx, jx) * G2(qy, jy) * B1(qz, jz);
              break;

            case 2:
              phi_j[2] = B1(qx, jx) * B1(qy, jy) * B2(qz, jz);
              div_phi_j = B1(qx, jx) * B1(qy, jy) * G2(qz, jz);
              break;
          }

          int Q = (qz * q + qy) * q + qx;
          const auto& d00 = get<0>(get<0>(input(Q)));
          const auto& d01 = get<1>(get<0>(input(Q)));
          const auto& d10 = get<0>(get<1>(input(Q)));
          const auto& d11 = get<1>(get<1>(input(Q)));

          output[Q] = {dot(d00, phi_j) + d01 * div_phi_j, dot(d10, phi_j) + d11 * div_phi_j};
        }
      }
    }

    return output;
  }

  template <int q>
  SMITH_HOST_DEVICE static auto interpolate(const dof_type& element_values, const TensorProductQuadratureRule<q>&)
  {
    constexpr bool apply_weights = false;
    constexpr tensor<double, q, p> B1 = calculate_B1<apply_weights, q>();
    constexpr tensor<double, q, p + 1> B2 = calculate_B2<apply_weights, q>();
    constexpr tensor<double, q, p + 1> G2 = calculate_G2<apply_weights, q>();

    tensor<tensor<double, q, q, q>, 3> value{};
    tensor<double, q, q, q> div{};

    // to clarify which contractions correspond to which spatial dimensions
    constexpr int x = 2, y = 1, z = 0;

    // clang-format off
    // x-component: closed(B2) in x, open(B1) in y, open(B1) in z
    // element_values.x is (p, p, p+1) = (z_open, y_open, x_closed)
    {
      auto A1  = contract< x, 1 >(element_values.x, B2);
      auto A2  = contract< y, 1 >(A1,  B1);
      value[0] = contract< z, 1 >(A2, B1);

      // divergence from x: dg/dx in x (G2), open in y,z (B1)
      A1       = contract< x, 1 >(element_values.x, G2);
      A2       = contract< y, 1 >(A1,  B1);
      div      = contract< z, 1 >(A2, B1);
    }

    // y-component: open(B1) in x, closed(B2) in y, open(B1) in z
    // element_values.y is (p, p+1, p) = (z_open, y_closed, x_open)
    {
      auto A1  = contract< y, 1 >(element_values.y, B2);
      auto A2  = contract< x, 1 >(A1,  B1);
      value[1] = contract< z, 1 >(A2, B1);

      // divergence from y: open in x (B1), dg/dy in y (G2), open in z (B1)
      A1       = contract< y, 1 >(element_values.y, G2);
      A2       = contract< x, 1 >(A1,  B1);
      div     += contract< z, 1 >(A2, B1);
    }

    // z-component: open(B1) in x, open(B1) in y, closed(B2) in z
    // element_values.z is (p+1, p, p) = (z_closed, y_open, x_open)
    {
      auto A1  = contract< z, 1 >(element_values.z, B2);
      auto A2  = contract< y, 1 >(A1,  B1);
      value[2] = contract< x, 1 >(A2, B1);

      // divergence from z: open in x,y (B1), dg/dz in z (G2)
      A1       = contract< z, 1 >(element_values.z, G2);
      A2       = contract< y, 1 >(A1,  B1);
      div     += contract< x, 1 >(A2, B1);
    }
    // clang-format on

    tensor<tuple<tensor<double, 3>, double>, q * q * q> qf_inputs;

    int count = 0;
    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          for (int i = 0; i < 3; i++) {
            get<VALUE>(qf_inputs(count))[i] = value[i](qz, qy, qx);
          }
          get<DIV>(qf_inputs(count)) = div(qz, qy, qx);
          count++;
        }
      }
    }

    return qf_inputs;
  }

  template <typename source_type, typename flux_type, int q>
  SMITH_HOST_DEVICE static void integrate(const tensor<tuple<source_type, flux_type>, q * q * q>& qf_output,
                                          const TensorProductQuadratureRule<q>&, dof_type* element_residual,
                                          [[maybe_unused]] int step = 1)
  {
    if constexpr (is_zero<source_type>{} && is_zero<flux_type>{}) return;

    constexpr bool apply_weights = true;
    constexpr tensor<double, q, p> B1 = calculate_B1<apply_weights, q>();
    constexpr tensor<double, q, p + 1> B2 = calculate_B2<apply_weights, q>();
    constexpr tensor<double, q, p + 1> G2 = calculate_G2<apply_weights, q>();

    using source_buf_t = std::conditional_t<is_zero<source_type>{}, zero, tensor<double, 3, q, q, q>>;
    using flux_buf_t   = std::conditional_t<is_zero<flux_type>{},   zero, tensor<double, q, q, q>>;

    source_buf_t source{};
    flux_buf_t flux{};

    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          int k = (qz * q + qy) * q + qx;
          if constexpr (!is_zero<source_type>{}) {
            tensor<double, 3> s{get<SOURCE>(qf_output[k])};
            for (int i = 0; i < 3; i++) {
              source(i, qz, qy, qx) = s[i];
            }
          }
          if constexpr (!is_zero<flux_type>{}) {
            flux(qz, qy, qx) = get<FLUX>(qf_output[k]);
          }
        }
      }
    }

    // to clarify which contractions correspond to which spatial dimensions
    constexpr int x = 2, y = 1, z = 0;

    // clang-format off
    if constexpr (!is_zero<source_type>{}) {
      // x-component: source[0] tested with B2(x)*B1(y)*B1(z)
      {
        auto A2  = contract< z, 0 >(source[0], B1);
        auto A1  = contract< y, 0 >(A2, B1);
        element_residual[0].x += contract< x, 0 >(A1, B2);
      }
      // y-component: source[1] tested with B1(x)*B2(y)*B1(z)
      {
        auto A2  = contract< z, 0 >(source[1], B1);
        auto A1  = contract< x, 0 >(A2, B1);
        element_residual[0].y += contract< y, 0 >(A1, B2);
      }
      // z-component: source[2] tested with B1(x)*B1(y)*B2(z)
      {
        auto A2  = contract< y, 0 >(source[2], B1);
        auto A1  = contract< x, 0 >(A2, B1);
        element_residual[0].z += contract< z, 0 >(A1, B2);
      }
    }

    if constexpr (!is_zero<flux_type>{}) {
      // x-component: flux tested with G2(x)*B1(y)*B1(z)
      {
        auto A2  = contract< z, 0 >(flux, B1);
        auto A1  = contract< y, 0 >(A2, B1);
        element_residual[0].x += contract< x, 0 >(A1, G2);
      }
      // y-component: flux tested with B1(x)*G2(y)*B1(z)
      {
        auto A2  = contract< z, 0 >(flux, B1);
        auto A1  = contract< x, 0 >(A2, B1);
        element_residual[0].y += contract< y, 0 >(A1, G2);
      }
      // z-component: flux tested with B1(x)*B1(y)*G2(z)
      {
        auto A2  = contract< y, 0 >(flux, B1);
        auto A1  = contract< x, 0 >(A2, B1);
        element_residual[0].z += contract< z, 0 >(A1, G2);
      }
    }
    // clang-format on
  }
};
/// @endcond
