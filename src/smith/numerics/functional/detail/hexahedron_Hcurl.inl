// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file hexahedron_Hcurl.inl
 *
 * @brief Specialization of finite_element for Hcurl on hexahedron geometry
 */

// this specialization defines shape functions (and their curls) that
// interpolate at Gauss-Lobatto nodes for closed intervals, and Gauss-Legendre
// nodes for open intervals.
//
// note 1: mfem assumes the parent element domain is [0,1]x[0,1]x[0,1]
// note 2: dofs are numbered by direction and then lexicographically in space.
//         see quadrilateral_hcurl.inl for more information
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p>
struct finite_element<mfem::Geometry::CUBE, Hcurl<p>> {
  static constexpr auto geometry = mfem::Geometry::CUBE;
  static constexpr auto family = Family::HCURL;
  static constexpr int dim = 3;
  static constexpr int n = p + 1;
  static constexpr int ndof = 3 * p * (p + 1) * (p + 1);
  static constexpr int components = 1;

  static constexpr int VALUE = 0, CURL = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  // TODO: delete this in favor of dof_type
  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components>>::type;

  // this is how mfem provides the data to us for these elements
  struct dof_type {
    tensor<double, p + 1, p + 1, p> x;
    tensor<double, p + 1, p, p + 1> y;
    tensor<double, p, p + 1, p + 1> z;
  };

  template <int q>
  using cpu_batched_values_type = tensor<tensor<double, 3>, q, q, q>;

  template <int q>
  using cpu_batched_derivatives_type = tensor<tensor<double, 3>, q, q, q>;

  static constexpr auto directions = [] {
    int dof_per_direction = p * (p + 1) * (p + 1);

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
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          nodes[count++] = {legendre_nodes[i], lobatto_nodes[j], lobatto_nodes[k]};
        }
      }
    }

    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          nodes[count++] = {lobatto_nodes[i], legendre_nodes[j], lobatto_nodes[k]};
        }
      }
    }

    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          nodes[count++] = {lobatto_nodes[i], lobatto_nodes[j], legendre_nodes[k]};
        }
      }
    }

    return nodes;
  }();

  /**
   * @brief B1(i,j) is the
   *  jth 1D Gauss-Legendre interpolating polynomial,
   *  evaluated at the ith 1D quadrature point
   *
   * @tparam apply_weights optionally multiply the rows of B1 by the associated quadrature weight
   * @tparam q the number of quadrature points in the 1D rule
   *
   * @return the matrix B1 of 1D polynomial evaluations
   */
  template <bool apply_weights, int q>
  static constexpr auto calculate_B1()
  {
    constexpr auto points1D = GaussLegendreNodes<q>();
    [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q>();
    tensor<double, q, p> B1{};
    for (int i = 0; i < q; i++) {
      B1[i] = GaussLegendreInterpolation<p>(points1D[i]);
      if constexpr (apply_weights) B1[i] = B1[i] * weights1D[i];
    }
    return B1;
  }

  /**
   * @brief B2(i,j) is the
   *  jth 1D Gauss-Lobatto interpolating polynomial,
   *  evaluated at the ith 1D quadrature point
   *
   * @tparam apply_weights optionally multiply the rows of B2 by the associated quadrature weight
   * @tparam q the number of quadrature points in the 1D rule
   *
   * @return the matrix B2 of 1D polynomial evaluations
   */
  template <bool apply_weights, int q>
  static constexpr auto calculate_B2()
  {
    constexpr auto points1D = GaussLegendreNodes<q>();
    [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q>();
    tensor<double, q, p + 1> B2{};
    for (int i = 0; i < q; i++) {
      B2[i] = GaussLobattoInterpolation<p + 1>(points1D[i]);
      if constexpr (apply_weights) B2[i] = B2[i] * weights1D[i];
    }
    return B2;
  }

  /**
   * @brief G2(i,j) is the derivative of the
   *  jth 1D Gauss-Lobatto interpolating polynomial,
   *  evaluated at the ith 1D quadrature point
   *
   * @tparam apply_weights optionally multiply the rows of G by the associated quadrature weight
   * @tparam q the number of quadrature points in the 1D rule
   *
   * @return the matrix G2 of 1D polynomial evaluations
   */
  template <bool apply_weights, int q>
  static constexpr auto calculate_G2()
  {
    constexpr auto points1D = GaussLegendreNodes<q>();
    [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q>();
    tensor<double, q, p + 1> G2{};
    for (int i = 0; i < q; i++) {
      G2[i] = GaussLobattoInterpolationDerivative<p + 1>(points1D[i]);
      if constexpr (apply_weights) G2[i] = G2[i] * weights1D[i];
    }
    return G2;
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_functions(tensor<double, dim> xi)
  {
    tensor<double, ndof, dim> N{};

    tensor<double, p> f[3] = {GaussLegendreInterpolation<p>(xi[0]), GaussLegendreInterpolation<p>(xi[1]),
                              GaussLegendreInterpolation<p>(xi[2])};

    tensor<double, p + 1> g[3] = {GaussLobattoInterpolation<p + 1>(xi[0]), GaussLobattoInterpolation<p + 1>(xi[1]),
                                  GaussLobattoInterpolation<p + 1>(xi[2])};

    int count = 0;

    // do all the x-facing nodes first
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          N[count++] = {f[0][i] * g[1][j] * g[2][k], 0.0, 0.0};
        }
      }
    }

    // then all the y-facing nodes
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          N[count++] = {0.0, g[0][i] * f[1][j] * g[2][k], 0.0};
        }
      }
    }

    // then, finally, all the z-facing nodes
    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          N[count++] = {0.0, 0.0, g[0][i] * g[1][j] * f[2][k]};
        }
      }
    }

    return N;
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_function_curl(tensor<double, dim> xi)
  {
    tensor<double, ndof, dim> curl{};

    tensor<double, p> f[3] = {GaussLegendreInterpolation<p>(xi[0]), GaussLegendreInterpolation<p>(xi[1]),
                              GaussLegendreInterpolation<p>(xi[2])};

    tensor<double, p + 1> g[3] = {GaussLobattoInterpolation<p + 1>(xi[0]), GaussLobattoInterpolation<p + 1>(xi[1]),
                                  GaussLobattoInterpolation<p + 1>(xi[2])};

    tensor<double, p + 1> dg[3] = {GaussLobattoInterpolationDerivative<p + 1>(xi[0]),
                                   GaussLobattoInterpolationDerivative<p + 1>(xi[1]),
                                   GaussLobattoInterpolationDerivative<p + 1>(xi[2])};

    int count = 0;

    // do all the x-facing nodes first
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p; i++) {
          // curl({f(x) g(y) g(z), 0.0, 0.0}) == {0.0, f(x) g(y) g'(z), -f(x) g'(y) g(z)};
          curl[count++] = {0.0, f[0][i] * g[1][j] * dg[2][k], -f[0][i] * dg[1][j] * g[2][k]};
        }
      }
    }

    // then all the y-facing nodes
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p; j++) {
        for (int i = 0; i < p + 1; i++) {
          // curl({0.0, g(x) f(y) g(z), 0.0}) == {-g(x) f(y) g'(z), 0.0, g'(x) f(y) g(z)};
          curl[count++] = {-g[0][i] * f[1][j] * dg[2][k], 0.0, dg[0][i] * f[1][j] * g[2][k]};
        }
      }
    }

    // then, finally, all the z-facing nodes
    for (int k = 0; k < p; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          // curl({0.0, 0.0, g(x) g(y) f(z)}) == {g(x) g'(y) f(z), -g'(x) g(y) f(z), 0.0};
          curl[count++] = {g[0][i] * dg[1][j] * f[2][k], -dg[0][i] * g[1][j] * f[2][k], 0.0};
        }
      }
    }

    return curl;
  }

  template <typename in_t, int q>
  static auto batch_apply_shape_fn(int j, tensor<in_t, q * q * q> input, const TensorProductQuadratureRule<q>&)
  {
    constexpr bool apply_weights = false;
    constexpr tensor<double, q, p> B1 = calculate_B1<apply_weights, q>();
    constexpr tensor<double, q, p + 1> B2 = calculate_B2<apply_weights, q>();
    constexpr tensor<double, q, p + 1> G2 = calculate_G2<apply_weights, q>();

    // figure out which node and which direction
    // correspond to the dof index "j"
    int jx, jy, jz;
    int dir = j / (p * (p + 1) * (p + 1));
    int remainder = j % (p * (p + 1) * (p + 1));
    switch (dir) {
      case 0:  // x-direction
        jx = remainder % p;
        jy = (remainder % (p * (p + 1))) / p;
        jz = remainder / (p * (p + 1));
        break;

      case 1:  // y-direction
        jx = remainder % (p + 1);
        jy = (remainder % (p * (p + 1))) / (p + 1);
        jz = remainder / (p * (p + 1));
        break;

      case 2:  // z-direction
        jx = remainder % (p + 1);
        jy = (remainder % ((p + 1) * (p + 1))) / (p + 1);
        jz = remainder / ((p + 1) * (p + 1));
        break;
    }

    using source_t = decltype(dot(get<0>(get<0>(in_t{})), vec3{}) + dot(get<1>(get<0>(in_t{})), vec3{}));
    using flux_t = decltype(dot(get<0>(get<1>(in_t{})), vec3{}) + dot(get<1>(get<1>(in_t{})), vec3{}));

    tensor<tuple<source_t, flux_t>, q * q * q> output;

    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          tensor<double, 3> phi_j{};
          tensor<double, 3> curl_phi_j{};

          switch (dir) {
            case 0:
              phi_j[0] = B1(qx, jx) * B2(qy, jy) * B2(qz, jz);
              curl_phi_j[1] = B1(qx, jx) * B2(qy, jy) * G2(qz, jz);
              curl_phi_j[2] = -B1(qx, jx) * G2(qy, jy) * B2(qz, jz);
              break;

            case 1:
              curl_phi_j[0] = -B2(qx, jx) * B1(qy, jy) * G2(qz, jz);
              phi_j[1] = B2(qx, jx) * B1(qy, jy) * B2(qz, jz);
              curl_phi_j[2] = G2(qx, jx) * B1(qy, jy) * B2(qz, jz);
              break;

            case 2:
              curl_phi_j[0] = B2(qx, jx) * G2(qy, jy) * B1(qz, jz);
              curl_phi_j[1] = -G2(qx, jx) * B2(qy, jy) * B1(qz, jz);
              phi_j[2] = B2(qx, jx) * B2(qy, jy) * B1(qz, jz);
              break;
          }

          int Q = (qz * q + qy) * q + qx;
          const auto& d00 = get<0>(get<0>(input(Q)));
          const auto& d01 = get<1>(get<0>(input(Q)));
          const auto& d10 = get<0>(get<1>(input(Q)));
          const auto& d11 = get<1>(get<1>(input(Q)));

          output[Q] = {dot(d00, phi_j) + dot(d01, curl_phi_j), dot(d10, phi_j) + dot(d11, curl_phi_j)};
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
    tensor<tensor<double, q, q, q>, 3> curl{};

    // to clarify which contractions correspond to which spatial dimensions
    constexpr int x = 2, y = 1, z = 0;

    // clang-format off
    {
      auto A1  = contract< x, 1 >(element_values.x, B1);
      auto A20 = contract< y, 1 >(A1,  B2);
      auto A21 = contract< y, 1 >(A1,  G2);
      value[0] = contract< z, 1 >(A20, B2);
      curl[1] += contract< z, 1 >(A20, G2);
      curl[2] -= contract< z, 1 >(A21, B2);
    }

    {
      auto A1  = contract< y, 1 >(element_values.y, B1);
      auto A20 = contract< z, 1 >(A1,  B2);
      auto A21 = contract< z, 1 >(A1,  G2);
      value[1] = contract< x, 1 >(A20, B2);
      curl[2] += contract< x, 1 >(A20, G2);
      curl[0] -= contract< x, 1 >(A21, B2);
    }

    {
      auto A1  = contract< z, 1 >(element_values.z, B1);
      auto A20 = contract< x, 1 >(A1,  B2);
      auto A21 = contract< x, 1 >(A1,  G2);
      value[2] = contract< y, 1 >(A20, B2);
      curl[0] += contract< y, 1 >(A20, G2);
      curl[1] -= contract< y, 1 >(A21, B2);
    }
    // clang-format on

    tensor<tuple<tensor<double, 3>, tensor<double, 3>>, q * q * q> qf_inputs;

    int count = 0;
    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          for (int i = 0; i < 3; i++) {
            get<VALUE>(qf_inputs(count))[i] = value[i](qz, qy, qx);
            get<CURL>(qf_inputs(count))[i] = curl[i](qz, qy, qx);
          }
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
    constexpr bool apply_weights = true;
    constexpr tensor<double, q, p> B1 = calculate_B1<apply_weights, q>();
    constexpr tensor<double, q, p + 1> B2 = calculate_B2<apply_weights, q>();
    constexpr tensor<double, q, p + 1> G2 = calculate_G2<apply_weights, q>();

    tensor<double, 3, q, q, q> source{};
    tensor<double, 3, q, q, q> flux{};

    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          int k = (qz * q + qy) * q + qx;
          tensor<double, 3> s{get<SOURCE>(qf_output[k])};
          tensor<double, 3> f{get<FLUX>(qf_output[k])};
          for (int i = 0; i < 3; i++) {
            source(i, qz, qy, qx) = s[i];
            flux(i, qz, qy, qx) = f[i];
          }
        }
      }
    }

    // to clarify which contractions correspond to which spatial dimensions
    constexpr int x = 2, y = 1, z = 0;

    // clang-format off
    //  r(0, dz, dy, dx) = s(0, qz, qy, qx) * B2(qz, dz) * B2(qy, dy) * B1(qx, dx)
    //                   + f(1, qz, qy, qx) * G2(qz, dz) * B2(qy, dy) * B1(qx, dx)
    //                   - f(2, qz, qy, qx) * B2(qz, dz) * G2(qy, dy) * B1(qx, dx);
    {
      auto A20 = contract< z, 0 >(source[0], B2) + contract< z, 0 >(flux[1], G2);
      auto A21 = contract< z, 0 >(flux[2], B2);
      auto A1 = contract< y, 0 >(A20, B2) - contract< y, 0 >(A21, G2);
      element_residual[0].x += contract< x, 0 >(A1, B1);
    }

    //  r(1, dz, dy, dx) = s(1, qz, qy, qx) * B2(qz, dz) * B1(qy, dy) * B2(qx, dx)
    //                   - f(0, qz, qy, qx) * G2(qz, dz) * B1(qy, dy) * B2(qx, dx)
    //                   + f(2, qz, qy, qx) * B2(qz, dz) * B1(qy, dy) * G2(qx, dx);
    {
      auto A20 = contract< x, 0 >(source[1], B2) + contract< x, 0 >(flux[2], G2);
      auto A21 = contract< x, 0 >(flux[0], B2);
      auto A1 = contract< z, 0 >(A20, B2) - contract< z, 0 >(A21, G2);
      element_residual[0].y += contract< y, 0 >(A1, B1);
    }

    //  r(2, dz, dy, dx) = s(2, qz, qy, qx) * B1(qz, dz) * B2(qy, dy) * B2(qx, dx) 
    //                   + f(0, qz, qy, qx) * B1(qz, dz) * G2(qy, dy) * B2(qx, dx) 
    //                   - f(1, qz, qy, qx) * B1(qz, dz) * B2(qy, dy) * G2(qx, dx);
    {
      auto A20 = contract< y, 0 >(source[2], B2) + contract< y, 0 >(flux[0], G2);
      auto A21 = contract< y, 0 >(flux[1], B2);
      auto A1 = contract< x, 0 >(A20, B2) - contract< x, 0 >(A21, G2);
      element_residual[0].z += contract< z, 0 >(A1, B1);
    }
    // clang-format on
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0

  template <int q>
  static SMITH_DEVICE auto interpolate(const dof_type& element_values, const tensor<double, dim, dim>& J,
                                       const TensorProductQuadratureRule<q>& rule, cache_type<q>& cache)
  {
    int tidx = threadIdx.x % q;
    int tidy = (threadIdx.x % (q * q)) / q;
    int tidz = threadIdx.x / (q * q);

    static constexpr auto points1D = GaussLegendreNodes<q>();

    static constexpr auto B1_ = [=]() {
      tensor<double, q, p> B1{};
      for (int i = 0; i < q; i++) {
        B1[i] = GaussLegendreInterpolation<p>(points1D[i]);
      }
      return B1;
    }();

    static constexpr auto B2_ = [=]() {
      tensor<double, q, n> B2{};
      for (int i = 0; i < q; i++) {
        B2[i] = GaussLobattoInterpolation<n>(points1D[i]);
      }
      return B2;
    }();

    static constexpr auto G2_ = [=]() {
      tensor<double, q, n> G2{};
      for (int i = 0; i < q; i++) {
        G2[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      }
      return G2;
    }();

    __shared__ tensor<double, q, p> B1;
    __shared__ tensor<double, q, n> B2;
    __shared__ tensor<double, q, n> G2;
    for (int entry = threadIdx.x; entry < n * q; entry += q * q) {
      int i = entry % n;
      int j = entry / n;
      if (i < p) {
        B1(j, i) = B1_(j, i);
      }
      B2(j, i) = B2_(j, i);
      G2(j, i) = G2_(j, i);
    }
    __syncthreads();

    tensor<double, dim> value{};
    tensor<double, dim> curl{};

    /////////////////////////////////
    ////////// X-component //////////
    /////////////////////////////////
    for (int dz = tidz; dz < p + 1; dz += q) {
      for (int dy = tidy; dy < p + 1; dy += q) {
        for (int qx = tidx; qx < q; qx += q) {
          double sum = 0.0;
          for (int dx = 0; dx < p; dx++) {
            sum += B1(qx, dx) * element_values.x(dz, dy, dx);
          }
          cache.A1(dz, dy, qx) = sum;
        }
      }
    }
    __syncthreads();

    for (int dz = tidz; dz < p + 1; dz += q) {
      for (int qy = tidy; qy < q; qy += q) {
        for (int qx = tidx; qx < q; qx += q) {
          double sum[2]{};
          for (int dy = 0; dy < (p + 1); dy++) {
            sum[0] += B2(qy, dy) * cache.A1(dz, dy, qx);
            sum[1] += G2(qy, dy) * cache.A1(dz, dy, qx);
          }
          cache.A2(0, dz, qy, qx) = sum[0];
          cache.A2(1, dz, qy, qx) = sum[1];
        }
      }
    }
    __syncthreads();

    for (int qz = tidz; qz < q; qz += q) {
      for (int qy = tidy; qy < q; qy += q) {
        for (int qx = tidx; qx < q; qx += q) {
          double sum[3]{};
          for (int dz = 0; dz < (p + 1); dz++) {
            sum[0] += B2(qz, dz) * cache.A2(0, dz, qy, qx);
            sum[1] += G2(qz, dz) * cache.A2(0, dz, qy, qx);
            sum[2] += B2(qz, dz) * cache.A2(1, dz, qy, qx);
          }
          value[0] += sum[0];
          curl[1] += sum[1];
          curl[2] -= sum[2];
        }
      }
    }
    __syncthreads();

    /////////////////////////////////
    ////////// Y-component //////////
    /////////////////////////////////
    for (int dz = tidz; dz < p + 1; dz += q) {
      for (int dx = tidx; dx < p + 1; dx += q) {
        for (int qy = tidy; qy < q; qy += q) {
          double sum = 0.0;
          for (int dy = 0; dy < p; dy++) {
            sum += B1(qy, dy) * element_values.y(dz, dy, dx);
          }
          cache.A1(dz, dx, qy) = sum;
        }
      }
    }
    __syncthreads();

    for (int dz = tidz; dz < p + 1; dz += q) {
      for (int qy = tidy; qy < q; qy += q) {
        for (int qx = tidx; qx < q; qx += q) {
          double sum[2]{};
          for (int dx = 0; dx < (p + 1); dx++) {
            sum[0] += B2(qx, dx) * cache.A1(dz, dx, qy);
            sum[1] += G2(qx, dx) * cache.A1(dz, dx, qy);
          }
          cache.A2(0, dz, qy, qx) = sum[0];
          cache.A2(1, dz, qy, qx) = sum[1];
        }
      }
    }
    __syncthreads();

    for (int qz = tidz; qz < q; qz += q) {
      for (int qy = tidy; qy < q; qy += q) {
        for (int qx = tidx; qx < q; qx += q) {
          double sum[3]{};
          for (int dz = 0; dz < (p + 1); dz++) {
            sum[0] += B2(qz, dz) * cache.A2(0, dz, qy, qx);
            sum[1] += G2(qz, dz) * cache.A2(0, dz, qy, qx);
            sum[2] += B2(qz, dz) * cache.A2(1, dz, qy, qx);
          }
          value[1] += sum[0];
          curl[2] += sum[2];
          curl[0] -= sum[1];
        }
      }
    }
    __syncthreads();

    /////////////////////////////////
    ////////// Z-component //////////
    /////////////////////////////////
    for (int dy = tidy; dy < p + 1; dy += q) {
      for (int dx = tidx; dx < p + 1; dx += q) {
        for (int qz = tidz; qz < q; qz += q) {
          double sum = 0.0;
          for (int k = 0; k < p; k++) {
            sum += B1(qz, k) * element_values.z(k, dy, dx);
          }
          cache.A1(dy, dx, qz) = sum;
        }
      }
    }
    __syncthreads();

    for (int dy = tidy; dy < p + 1; dy += q) {
      for (int qz = tidz; qz < q; qz += q) {
        for (int qx = tidx; qx < q; qx += q) {
          double sum[2]{};
          for (int dx = 0; dx < (p + 1); dx++) {
            sum[0] += B2(qx, dx) * cache.A1(dy, dx, qz);
            sum[1] += G2(qx, dx) * cache.A1(dy, dx, qz);
          }
          cache.A2(0, dy, qz, qx) = sum[0];
          cache.A2(1, dy, qz, qx) = sum[1];
        }
      }
    }
    __syncthreads();

    for (int qz = tidz; qz < q; qz += q) {
      for (int qy = tidy; qy < q; qy += q) {
        for (int qx = tidx; qx < q; qx += q) {
          double sum[3]{};
          for (int dy = 0; dy < (p + 1); dy++) {
            sum[0] += B2(qy, dy) * cache.A2(0, dy, qz, qx);
            sum[1] += G2(qy, dy) * cache.A2(0, dy, qz, qx);
            sum[2] += B2(qy, dy) * cache.A2(1, dy, qz, qx);
          }
          value[2] += sum[0];
          curl[0] += sum[1];
          curl[1] -= sum[2];
        }
      }
    }

    // apply covariant Piola transformation to go
    // from parent element -> physical element
    value = linear_solve(transpose(J), value);
    curl  = dot(J, curl) / det(J);

    return tuple{value, curl};
  }

  template <typename T1, typename T2, int q>
  static SMITH_DEVICE void integrate(tuple<T1, T2>& response, const tensor<double, dim, dim>& J,
                                     const TensorProductQuadratureRule<q>& rule, cache_type<q>& cache,
                                     dof_type& residual)
  {
    int tidx = threadIdx.x % q;
    int tidy = (threadIdx.x % (q * q)) / q;
    int tidz = threadIdx.x / (q * q);

    static constexpr auto points1D  = GaussLegendreNodes<q>();
    static constexpr auto weights1D = GaussLegendreWeights<q>();

    static constexpr auto B1_ = [=]() {
      tensor<double, q, p> B1{};
      for (int i = 0; i < q; i++) {
        B1[i] = GaussLegendreInterpolation<p>(points1D[i]);
      }
      return B1;
    }();

    static constexpr auto B2_ = [=]() {
      tensor<double, q, n> B2{};
      for (int i = 0; i < q; i++) {
        B2[i] = GaussLobattoInterpolation<n>(points1D[i]);
      }
      return B2;
    }();

    static constexpr auto G2_ = [=]() {
      tensor<double, q, n> G2{};
      for (int i = 0; i < q; i++) {
        G2[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      }
      return G2;
    }();

    __shared__ tensor<double, q, p> B1;
    __shared__ tensor<double, q, n> B2;
    __shared__ tensor<double, q, n> G2;
    for (int entry = threadIdx.x; entry < n * q; entry += q * q) {
      int i = entry % n;
      int j = entry / n;
      if (i < p) {
        B1(j, i) = B1_(j, i);
      }
      B2(j, i) = B2_(j, i);
      G2(j, i) = G2_(j, i);
    }
    __syncthreads();

    // transform the source and flux terms from values on the physical element,
    // to values on the parent element. Also, the source/flux values are scaled
    // according to the weight of their quadrature point, so that when we add them
    // together, it approximates the integral over the element
    auto detJ = det(J);
    auto dv   = detJ * weights1D[tidx] * weights1D[tidy] * weights1D[tidz];

    auto source = linear_solve(J, get<0>(response)) * dv;
    auto flux   = dot(get<1>(response), J) * (dv / detJ);

    /////////////////////////////////
    ////////// X-component //////////
    /////////////////////////////////
    for (int qz = tidz; qz < q; qz += q) {
      for (int qy = tidy; qy < q; qy += q) {
        for (int dx = tidx; dx < n; dx += q) {
          cache.A2(0, dx, qy, qz) = 0.0;
          cache.A2(1, dx, qy, qz) = 0.0;
        }
      }
    }
    __syncthreads();

    for (int offset = 0; offset < n; offset++) {
      int  dz  = (tidz + offset) % n;
      auto sum = B2(tidz, dz) * source[0] + G2(tidz, dz) * flux[1];
      atomicAdd(&cache.A2(0, dz, tidy, tidx), sum);
      atomicAdd(&cache.A2(1, dz, tidy, tidx), -B2(tidz, dz) * flux[2]);
    }
    __syncthreads();

    for (int dz = tidz; dz < p + 1; dz += q) {
      for (int dy = tidy; dy < p + 1; dy += q) {
        for (int qx = tidx; qx < q; qx += q) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B2(qy, dy) * cache.A2(0, dz, qy, qx);
            sum += G2(qy, dy) * cache.A2(1, dz, qy, qx);
          }
          cache.A1(dz, dy, qx) = sum;
        }
      }
    }
    __syncthreads();

    for (int dz = tidz; dz < p + 1; dz += q) {
      for (int dy = tidy; dy < p + 1; dy += q) {
        for (int dx = tidx; dx < p; dx += q) {
          double sum = 0.0;
          for (int qx = 0; qx < q; qx++) {
            sum += B1(qx, dx) * cache.A1(dz, dy, qx);
          }
          residual.x(dz, dy, dx) += sum;
        }
      }
    }
    __syncthreads();

    /////////////////////////////////
    ////////// Y-component //////////
    /////////////////////////////////
    for (int qz = tidz; qz < q; qz += q) {
      for (int qy = tidy; qy < q; qy += q) {
        for (int dx = tidx; dx < n; dx += q) {
          cache.A2(0, dx, qy, qz) = 0.0;
          cache.A2(1, dx, qy, qz) = 0.0;
        }
      }
    }
    __syncthreads();

    for (int offset = 0; offset < n; offset++) {
      int  dz  = (tidz + offset) % n;
      auto sum = B2(tidz, dz) * source[1] - G2(tidz, dz) * flux[0];
      atomicAdd(&cache.A2(0, dz, tidy, tidx), sum);
      atomicAdd(&cache.A2(1, dz, tidy, tidx), B2(tidz, dz) * flux[2]);
    }
    __syncthreads();

    for (int dz = tidz; dz < p + 1; dz += q) {
      for (int dx = tidx; dx < p + 1; dx += q) {
        for (int qy = tidy; qy < q; qy += q) {
          double sum = 0.0;
          for (int qx = 0; qx < q; qx++) {
            sum += B2(qx, dx) * cache.A2(0, dz, qy, qx);
            sum += G2(qx, dx) * cache.A2(1, dz, qy, qx);
          }
          cache.A1(dz, dx, qy) = sum;
        }
      }
    }
    __syncthreads();

    for (int dz = tidz; dz < p + 1; dz += q) {
      for (int dy = tidy; dy < p; dy += q) {
        for (int dx = tidx; dx < p + 1; dx += q) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B1(qy, dy) * cache.A1(dz, dx, qy);
          }
          residual.y(dz, dy, dx) += sum;
        }
      }
    }
    __syncthreads();

    /////////////////////////////////
    ////////// Z-component //////////
    /////////////////////////////////
    for (int qz = tidz; qz < q; qz += q) {
      for (int qy = tidy; qy < q; qy += q) {
        for (int dx = tidx; dx < n; dx += q) {
          cache.A2(0, dx, qy, qz) = 0.0;
          cache.A2(1, dx, qy, qz) = 0.0;
        }
      }
    }
    __syncthreads();

    for (int offset = 0; offset < n; offset++) {
      int  dx  = (tidx + offset) % n;
      auto sum = B2(tidx, dx) * source[2] - G2(tidx, dx) * flux[1];
      atomicAdd(&cache.A2(0, dx, tidz, tidy), sum);
      atomicAdd(&cache.A2(1, dx, tidz, tidy), B2(tidx, dx) * flux[0]);
    }
    __syncthreads();

    for (int dy = tidy; dy < p + 1; dy += q) {
      for (int dx = tidx; dx < p + 1; dx += q) {
        for (int qz = tidz; qz < q; qz += q) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B2(qy, dy) * cache.A2(0, dx, qz, qy);
            sum += G2(qy, dy) * cache.A2(1, dx, qz, qy);
          }
          cache.A1(dy, dx, qz) = sum;
        }
      }
    }
    __syncthreads();

    for (int dz = tidz; dz < p; dz += q) {
      for (int dy = tidy; dy < p + 1; dy += q) {
        for (int dx = tidx; dx < p + 1; dx += q) {
          double sum = 0.0;
          for (int qz = 0; qz < q; qz++) {
            sum += B1(qz, dz) * cache.A1(dy, dx, qz);
          }
          residual.z(dz, dy, dx) += sum;
        }
      }
    }
  }

#endif
};
/// @endcond
