// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file hexahedron_L2.inl
 *
 * @brief Specialization of finite_element for L2 on hexahedron geometry
 */

// this specialization defines shape functions (and their gradients) that
// interpolate at Gauss-Lobatto nodes for the appropriate polynomial order
//
// note: mfem assumes the parent element domain is [0,1]x[0,1]x[0,1]
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p, int c>
struct finite_element<mfem::Geometry::CUBE, L2<p, c> > {
  static constexpr auto geometry = mfem::Geometry::CUBE;
  static constexpr auto family = Family::L2;
  static constexpr int components = c;
  static constexpr int dim = 3;
  static constexpr int n = (p + 1);
  static constexpr int ndof = (p + 1) * (p + 1) * (p + 1);
  static constexpr int order = p;

  static constexpr int VALUE = 0, GRADIENT = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  // TODO: remove this
  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  using dof_type = tensor<double, c, p + 1, p + 1, p + 1>;

  using value_type = typename std::conditional<components == 1, double, tensor<double, components> >::type;
  using derivative_type =
      typename std::conditional<components == 1, tensor<double, dim>, tensor<double, components, dim> >::type;
  using qf_input_type = tuple<value_type, derivative_type>;

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_functions(tensor<double, dim> xi)
  {
    auto N_xi = GaussLobattoInterpolation<p + 1>(xi[0]);
    auto N_eta = GaussLobattoInterpolation<p + 1>(xi[1]);
    auto N_zeta = GaussLobattoInterpolation<p + 1>(xi[2]);

    int count = 0;

    tensor<double, ndof> N{};
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          N[count++] = N_xi[i] * N_eta[j] * N_zeta[k];
        }
      }
    }
    return N;
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_function_gradients(tensor<double, dim> xi)
  {
    auto N_xi = GaussLobattoInterpolation<p + 1>(xi[0]);
    auto N_eta = GaussLobattoInterpolation<p + 1>(xi[1]);
    auto N_zeta = GaussLobattoInterpolation<p + 1>(xi[2]);
    auto dN_xi = GaussLobattoInterpolationDerivative<p + 1>(xi[0]);
    auto dN_eta = GaussLobattoInterpolationDerivative<p + 1>(xi[1]);
    auto dN_zeta = GaussLobattoInterpolationDerivative<p + 1>(xi[2]);

    int count = 0;

    // clang-format off
    tensor<double, ndof, dim> dN{};
    for (int k = 0; k < p + 1; k++) {
      for (int j = 0; j < p + 1; j++) {
        for (int i = 0; i < p + 1; i++) {
          dN[count++] = {
            dN_xi[i] *  N_eta[j] *  N_zeta[k], 
             N_xi[i] * dN_eta[j] *  N_zeta[k],
             N_xi[i] *  N_eta[j] * dN_zeta[k]
          };
        }
      }
    }
    return dN;
    // clang-format on
  }

  /**
   * @brief B(i,j) is the
   *  jth 1D Gauss-Lobatto interpolating polynomial,
   *  evaluated at the ith 1D quadrature point
   *
   * @tparam apply_weights optionally multiply the rows of B by the associated quadrature weight
   * @tparam q the number of quadrature points in the 1D rule
   *
   * @return the matrix B of 1D polynomial evaluations
   */
  template <bool apply_weights, int q>
  static constexpr auto calculate_B()
  {
    constexpr auto points1D = GaussLegendreNodes<q, mfem::Geometry::SEGMENT>();
    [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
    tensor<double, q, n> B{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(points1D[i]);
      if constexpr (apply_weights) B[i] = B[i] * weights1D[i];
    }
    return B;
  }

  /**
   * @brief G(i,j) is the derivative of the
   *  jth 1D Gauss-Lobatto interpolating polynomial,
   *  evaluated at the ith 1D quadrature point
   *
   * @tparam apply_weights optionally multiply the rows of G by the associated quadrature weight
   * @tparam q the number of quadrature points in the 1D rule
   *
   * @return the matrix G of 1D polynomial evaluations
   */
  template <bool apply_weights, int q>
  static constexpr auto calculate_G()
  {
    constexpr auto points1D = GaussLegendreNodes<q, mfem::Geometry::SEGMENT>();
    [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
    tensor<double, q, n> G{};
    for (int i = 0; i < q; i++) {
      G[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      if constexpr (apply_weights) G[i] = G[i] * weights1D[i];
    }
    return G;
  }

  template <typename in_t, int q>
  static auto batch_apply_shape_fn(int j, tensor<in_t, q * q * q> input, const TensorProductQuadratureRule<q>&)
  {
    static constexpr bool apply_weights = false;
    static constexpr auto B = calculate_B<apply_weights, q>();
    static constexpr auto G = calculate_G<apply_weights, q>();

    int jx = j % n;
    int jy = (j % (n * n)) / n;
    int jz = j / (n * n);

    using source_t = decltype(get<0>(get<0>(in_t{})) + dot(get<1>(get<0>(in_t{})), tensor<double, dim>{}));
    using flux_t = decltype(get<0>(get<1>(in_t{})) + dot(get<1>(get<1>(in_t{})), tensor<double, dim>{}));

    tensor<tuple<source_t, flux_t>, q * q * q> output;

    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          double phi_j = B(qx, jx) * B(qy, jy) * B(qz, jz);
          tensor<double, dim> dphi_j_dxi = {G(qx, jx) * B(qy, jy) * B(qz, jz), B(qx, jx) * G(qy, jy) * B(qz, jz),
                                            B(qx, jx) * B(qy, jy) * G(qz, jz)};

          int Q = (qz * q + qy) * q + qx;
          const auto& d00 = get<0>(get<0>(input(Q)));
          const auto& d01 = get<1>(get<0>(input(Q)));
          const auto& d10 = get<0>(get<1>(input(Q)));
          const auto& d11 = get<1>(get<1>(input(Q)));

          output[Q] = {d00 * phi_j + dot(d01, dphi_j_dxi), d10 * phi_j + dot(d11, dphi_j_dxi)};
        }
      }
    }

    return output;
  }

  template <int q>
  SMITH_HOST_DEVICE static auto interpolate(const dof_type& X, const TensorProductQuadratureRule<q>&)
  {
    // we want to compute the following:
    //
    // X_q(u, v, w) := (B(u, i) * B(v, j) * B(w, k)) * X_e(i, j, k)
    //
    // where
    //   X_q(u, v, w) are the quadrature-point values at position {u, v, w},
    //   B(u, i) is the i^{th} 1D interpolation/differentiation (shape) function,
    //           evaluated at the u^{th} 1D quadrature point, and
    //   X_e(i, j, k) are the values at node {i, j, k} to be interpolated
    //
    // this algorithm carries out the above calculation in 3 steps:
    //
    // A1(dz, dy, qx)  := B(qx, dx) * X_e(dz, dy, dx)
    // A2(dz, qy, qx)  := B(qy, dy) * A1(dz, dy, qx)
    // X_q(qz, qy, qx) := B(qz, dz) * A2(dz, qy, qx)
    static constexpr bool apply_weights = false;
    static constexpr auto B = calculate_B<apply_weights, q>();
    static constexpr auto G = calculate_G<apply_weights, q>();

    tensor<double, c, q, q, q> value{};
    tensor<double, c, dim, q, q, q> gradient{};

    for (int i = 0; i < c; i++) {
      auto A10 = contract<2, 1>(X[i], B);
      auto A11 = contract<2, 1>(X[i], G);

      auto A20 = contract<1, 1>(A10, B);
      auto A21 = contract<1, 1>(A11, B);
      auto A22 = contract<1, 1>(A10, G);

      value(i) = contract<0, 1>(A20, B);
      gradient(i, 0) = contract<0, 1>(A21, B);
      gradient(i, 1) = contract<0, 1>(A22, B);
      gradient(i, 2) = contract<0, 1>(A20, G);
    }

    // transpose the quadrature data into a flat tensor of tuples
    union {
      tensor<qf_input_type, q * q * q> one_dimensional;
      tensor<tuple<tensor<double, c>, tensor<double, c, dim> >, q, q, q> three_dimensional;
    } output;

    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        for (int qx = 0; qx < q; qx++) {
          for (int i = 0; i < c; i++) {
            get<VALUE>(output.three_dimensional(qz, qy, qx))[i] = value(i, qz, qy, qx);
            for (int j = 0; j < dim; j++) {
              get<GRADIENT>(output.three_dimensional(qz, qy, qx))[i][j] = gradient(i, j, qz, qy, qx);
            }
          }
        }
      }
    }

    return output.one_dimensional;
  }

  template <typename source_type, typename flux_type, int q>
  SMITH_HOST_DEVICE static void integrate(const tensor<tuple<source_type, flux_type>, q * q * q>& qf_output,
                                          const TensorProductQuadratureRule<q>&, dof_type* element_residual,
                                          int step = 1)
  {
    if constexpr (is_zero<source_type>{} && is_zero<flux_type>{}) {
      return;
    }

    constexpr int ntrial = std::max(size(source_type{}), size(flux_type{}) / dim) / c;

    using s_buffer_type = std::conditional_t<is_zero<source_type>{}, zero, tensor<double, q, q, q> >;
    using f_buffer_type = std::conditional_t<is_zero<flux_type>{}, zero, tensor<double, dim, q, q, q> >;

    static constexpr bool apply_weights = true;
    static constexpr auto B = calculate_B<apply_weights, q>();
    static constexpr auto G = calculate_G<apply_weights, q>();

    for (int j = 0; j < ntrial; j++) {
      for (int i = 0; i < c; i++) {
        s_buffer_type source;
        f_buffer_type flux;

        for (int qz = 0; qz < q; qz++) {
          for (int qy = 0; qy < q; qy++) {
            for (int qx = 0; qx < q; qx++) {
              int Q = (qz * q + qy) * q + qx;
              if constexpr (!is_zero<source_type>{}) {
                source(qz, qy, qx) = reinterpret_cast<const double*>(&get<SOURCE>(qf_output[Q]))[i * ntrial + j];
              }
              if constexpr (!is_zero<flux_type>{}) {
                for (int k = 0; k < dim; k++) {
                  flux(k, qz, qy, qx) =
                      reinterpret_cast<const double*>(&get<FLUX>(qf_output[Q]))[(i * dim + k) * ntrial + j];
                }
              }
            }
          }
        }

        auto A20 = contract<2, 0>(source, B) + contract<2, 0>(flux(0), G);
        auto A21 = contract<2, 0>(flux(1), B);
        auto A22 = contract<2, 0>(flux(2), B);

        auto A10 = contract<1, 0>(A20, B) + contract<1, 0>(A21, G);
        auto A11 = contract<1, 0>(A22, B);

        element_residual[j * step](i) += contract<0, 0>(A10, B) + contract<0, 0>(A11, G);
      }
    }
  }

#if 0

  template <int q>
  static SMITH_DEVICE auto interpolate(const dof_type& X, const tensor<double, dim, dim>& J,
                                       const TensorProductQuadratureRule<q>& rule, cache_type<q>& cache)
  {
    // we want to compute the following:
    //
    // X_q(u, v, w) := (B(u, i) * B(v, j) * B(w, k)) * X_e(i, j, k)
    //
    // where
    //   X_q(u, v, w) are the quadrature-point values at position {u, v, w},
    //   B(u, i) is the i^{th} 1D interpolation/differentiation (shape) function,
    //           evaluated at the u^{th} 1D quadrature point, and
    //   X_e(i, j, k) are the values at node {i, j, k} to be interpolated
    //
    // this algorithm carries out the above calculation in 3 steps:
    //
    // A1(dz, dy, qx)  := B(qx, dx) * X_e(dz, dy, dx)
    // A2(dz, qy, qx)  := B(qy, dy) * A1(dz, dy, qx)
    // X_q(qz, qy, qx) := B(qz, dz) * A2(dz, qy, qx)

    int tidx = threadIdx.x % q;
    int tidy = (threadIdx.x % (q * q)) / q;
    int tidz = threadIdx.x / (q * q);

    static constexpr auto points1D = GaussLegendreNodes<q>();
    static constexpr auto B_       = [=]() {
      tensor<double, q, n> B{};
      for (int i = 0; i < q; i++) {
        B[i] = GaussLobattoInterpolation<n>(points1D[i]);
      }
      return B;
    }();

    static constexpr auto G_ = [=]() {
      tensor<double, q, n> G{};
      for (int i = 0; i < q; i++) {
        G[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      }
      return G;
    }();

    __shared__ tensor<double, q, n> B;
    __shared__ tensor<double, q, n> G;
    for (int entry = threadIdx.x; entry < n * q; entry += q * q * q) {
      int i   = entry % n;
      int j   = entry / n;
      B(j, i) = B_(j, i);
      G(j, i) = G_(j, i);
    }
    __syncthreads();

    tuple<tensor<double, c>, tensor<double, c, 3> > qf_input{};

    for (int i = 0; i < c; i++) {
      for (int dz = tidz; dz < n; dz += q) {
        for (int dy = tidy; dy < n; dy += q) {
          for (int qx = tidx; qx < q; qx += q) {
            double sum[2]{};
            for (int dx = 0; dx < n; dx++) {
              sum[0] += B(qx, dx) * X(i, dz, dy, dx);
              sum[1] += G(qx, dx) * X(i, dz, dy, dx);
            }
            cache.A1(0, dz, dy, qx) = sum[0];
            cache.A1(1, dz, dy, qx) = sum[1];
          }
        }
      }
      __syncthreads();

      for (int dz = tidz; dz < n; dz += q) {
        for (int qy = tidy; qy < q; qy += q) {
          for (int qx = tidx; qx < q; qx += q) {
            double sum[3]{};
            for (int dy = 0; dy < n; dy++) {
              sum[0] += B(qy, dy) * cache.A1(0, dz, dy, qx);
              sum[1] += B(qy, dy) * cache.A1(1, dz, dy, qx);
              sum[2] += G(qy, dy) * cache.A1(0, dz, dy, qx);
            }
            cache.A2(0, dz, qy, qx) = sum[0];
            cache.A2(1, dz, qy, qx) = sum[1];
            cache.A2(2, dz, qy, qx) = sum[2];
          }
        }
      }
      __syncthreads();

      for (int qz = tidz; qz < q; qz += q) {
        for (int qy = tidy; qy < q; qy += q) {
          for (int qx = tidx; qx < q; qx += q) {
            for (int dz = 0; dz < n; dz++) {
              get<0>(qf_input)[i] += B(qz, dz) * cache.A2(0, dz, qy, qx);
              get<1>(qf_input)[i][0] += B(qz, dz) * cache.A2(1, dz, qy, qx);
              get<1>(qf_input)[i][1] += B(qz, dz) * cache.A2(2, dz, qy, qx);
              get<1>(qf_input)[i][2] += G(qz, dz) * cache.A2(0, dz, qy, qx);
            }
          }
        }
      }
    }

    get<1>(qf_input) = dot(get<1>(qf_input), inv(J));

    return qf_input;
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
    static constexpr auto B_        = [=]() {
      tensor<double, q, n> B{};
      for (int i = 0; i < q; i++) {
        B[i] = GaussLobattoInterpolation<n>(points1D[i]);
      }
      return B;
    }();

    static constexpr auto G_ = [=]() {
      tensor<double, q, n> G{};
      for (int i = 0; i < q; i++) {
        G[i] = GaussLobattoInterpolationDerivative<n>(points1D[i]);
      }
      return G;
    }();

    __shared__ tensor<double, q, n> B;
    __shared__ tensor<double, q, n> G;
    for (int entry = threadIdx.x; entry < n * q; entry += q * q * q) {
      int i   = entry % n;
      int j   = entry / n;
      B(j, i) = B_(j, i);
      G(j, i) = G_(j, i);
    }
    __syncthreads();

    auto dv = det(J) * weights1D[tidx] * weights1D[tidy] * weights1D[tidz];

    get<0>(response) = get<0>(response) * dv;
    get<1>(response) = dot(get<1>(response), inv(transpose(J))) * dv;

    for (int i = 0; i < c; i++) {
      // this first contraction is performed a little differently, since `response` is not
      // in shared memory, so each thread can only access its own values
      for (int qz = tidz; qz < q; qz += q) {
        for (int qy = tidy; qy < q; qy += q) {
          for (int dx = tidx; dx < n; dx += q) {
            cache.A2(0, dx, qy, qz) = 0.0;
            cache.A2(1, dx, qy, qz) = 0.0;
            cache.A2(2, dx, qy, qz) = 0.0;
          }
        }
      }
      __syncthreads();

      for (int offset = 0; offset < n; offset++) {
        int  dx  = (tidx + offset) % n;
        auto sum = B(tidx, dx) * get<0>(response)(i) + G(tidx, dx) * get<1>(response)(i, 0);
        atomicAdd(&cache.A2(0, dx, tidz, tidy), sum);
        atomicAdd(&cache.A2(1, dx, tidz, tidy), B(tidx, dx) * get<1>(response)(i, 1));
        atomicAdd(&cache.A2(2, dx, tidz, tidy), B(tidx, dx) * get<1>(response)(i, 2));
      }
      __syncthreads();

      for (int qz = tidz; qz < q; qz += q) {
        for (int dy = tidy; dy < n; dy += q) {
          for (int dx = tidx; dx < n; dx += q) {
            double sum[2]{};
            for (int qy = 0; qy < q; qy++) {
              sum[0] += B(qy, dy) * cache.A2(0, dx, qz, qy);
              sum[0] += G(qy, dy) * cache.A2(1, dx, qz, qy);
              sum[1] += B(qy, dy) * cache.A2(2, dx, qz, qy);
            }
            cache.A1(0, qz, dy, dx) = sum[0];
            cache.A1(1, qz, dy, dx) = sum[1];
          }
        }
      }
      __syncthreads();

      for (int dz = tidz; dz < n; dz += q) {
        for (int dy = tidy; dy < n; dy += q) {
          for (int dx = tidx; dx < n; dx += q) {
            double sum = 0.0;
            for (int qz = 0; qz < q; qz++) {
              sum += B(qz, dz) * cache.A1(0, qz, dy, dx);
              sum += G(qz, dz) * cache.A1(1, qz, dy, dx);
            }
            residual(i, dz, dy, dx) += sum;
          }
        }
      }
    }
  }

#endif
};
/// @endcond
