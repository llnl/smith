// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrilateral_H1.inl
 *
 * @brief Specialization of finite_element for H1 on quadrilateral geometry
 */

// this specialization defines shape functions (and their gradients) that
// interpolate at Gauss-Lobatto nodes for the appropriate polynomial order
//
// note: mfem assumes the parent element domain is [0,1]x[0,1]
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p, int c>
struct finite_element<mfem::Geometry::SQUARE, H1<p, c> > {
  static constexpr auto geometry = mfem::Geometry::SQUARE;
  static constexpr auto family = Family::H1;
  static constexpr int components = c;
  static constexpr int dim = 2;
  static constexpr int order = p;
  static constexpr int n = (p + 1);
  static constexpr int ndof = (p + 1) * (p + 1);

  static constexpr int VALUE = 0, GRADIENT = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  using dof_type = tensor<double, c, p + 1, p + 1>;
  using dof_type_if = dof_type;

  using value_type = typename std::conditional<components == 1, double, tensor<double, components> >::type;
  using derivative_type =
      typename std::conditional<components == 1, tensor<double, dim>, tensor<double, components, dim> >::type;
  using qf_input_type = tuple<value_type, derivative_type>;

  /*

    interpolation nodes and their associated numbering:

        linear
    2-----------3
    |           |
    |           |
    |           |
    |           |
    |           |
    0-----------1


      quadratic
    6-----7-----8
    |           |
    |           |
    3     4     5
    |           |
    |           |
    0-----1-----2


        cubic
    12-13--14--15
    |           |
    8   9  10  11
    |           |
    4   5   6   7
    |           |
    0---1---2---3

  */

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_functions(tensor<double, dim> xi)
  {
    auto N_xi = GaussLobattoInterpolation<p + 1>(xi[0]);
    auto N_eta = GaussLobattoInterpolation<p + 1>(xi[1]);

    int count = 0;

    tensor<double, ndof> N{};
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
        N[count++] = N_xi[i] * N_eta[j];
      }
    }
    return N;
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_function_gradients(tensor<double, dim> xi)
  {
    auto N_xi = GaussLobattoInterpolation<p + 1>(xi[0]);
    auto N_eta = GaussLobattoInterpolation<p + 1>(xi[1]);
    auto dN_xi = GaussLobattoInterpolationDerivative<p + 1>(xi[0]);
    auto dN_eta = GaussLobattoInterpolationDerivative<p + 1>(xi[1]);

    int count = 0;

    tensor<double, ndof, dim> dN{};
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
        dN[count++] = {dN_xi[i] * N_eta[j], N_xi[i] * dN_eta[j]};
      }
    }
    return dN;
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
  static auto batch_apply_shape_fn(int j, tensor<in_t, q * q> input, const TensorProductQuadratureRule<q>&)
  {
    static constexpr bool apply_weights = false;
    static constexpr auto B = calculate_B<apply_weights, q>();
    static constexpr auto G = calculate_G<apply_weights, q>();

    int jx = j % n;
    int jy = j / n;

    using source_t = decltype(get<0>(get<0>(in_t{})) + dot(get<1>(get<0>(in_t{})), tensor<double, 2>{}));
    using flux_t = decltype(get<0>(get<1>(in_t{})) + dot(get<1>(get<1>(in_t{})), tensor<double, 2>{}));

    tensor<tuple<source_t, flux_t>, q * q> output;

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double phi_j = B(qx, jx) * B(qy, jy);
        tensor<double, dim> dphi_j_dxi = {G(qx, jx) * B(qy, jy), B(qx, jx) * G(qy, jy)};

        int Q = qy * q + qx;
        const auto& d00 = get<0>(get<0>(input(Q)));
        const auto& d01 = get<1>(get<0>(input(Q)));
        const auto& d10 = get<0>(get<1>(input(Q)));
        const auto& d11 = get<1>(get<1>(input(Q)));

        output[Q] = {d00 * phi_j + dot(d01, dphi_j_dxi), d10 * phi_j + dot(d11, dphi_j_dxi)};
      }
    }

    return output;
  }

  // we want to compute the following:
  //
  // X_q(u, v) := (B(u, i) * B(v, j)) * X_e(i, j)
  //
  // where
  //   X_q(u, v) are the quadrature-point values at position {u, v},
  //   B(u, i) is the i^{th} 1D interpolation/differentiation (shape) function,
  //           evaluated at the u^{th} 1D quadrature point, and
  //   X_e(i, j) are the values at node {i, j} to be interpolated
  //
  // this algorithm carries out the above calculation in 2 steps:
  //
  // A(dy, qx)  := B(qx, dx) * X_e(dy, dx)
  // X_q(qy, qx) := B(qy, dy) * A(dy, qx)
  template <int q>
  SMITH_HOST_DEVICE static auto interpolate(const dof_type& X, const TensorProductQuadratureRule<q>&)
  {
    static constexpr bool apply_weights = false;
    static constexpr auto B = calculate_B<apply_weights, q>();
    static constexpr auto G = calculate_G<apply_weights, q>();

    tensor<double, c, q, q> value{};
    tensor<double, c, dim, q, q> gradient{};

    // apply the shape functions
    for (int i = 0; i < c; i++) {
      auto A0 = contract<1, 1>(X[i], B);
      auto A1 = contract<1, 1>(X[i], G);

      value(i) = contract<0, 1>(A0, B);
      gradient(i, 0) = contract<0, 1>(A1, B);
      gradient(i, 1) = contract<0, 1>(A0, G);
    }

    // transpose the quadrature data into a flat tensor of tuples
    union {
      tensor<qf_input_type, q * q> one_dimensional;
      tensor<tuple<tensor<double, c>, tensor<double, c, dim> >, q, q> two_dimensional;
    } output;

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        for (int i = 0; i < c; i++) {
          get<VALUE>(output.two_dimensional(qy, qx))[i] = value(i, qy, qx);
          for (int j = 0; j < dim; j++) {
            get<GRADIENT>(output.two_dimensional(qy, qx))[i][j] = gradient(i, j, qy, qx);
          }
        }
      }
    }

    return output.one_dimensional;
  }

  // source can be one of: {zero, double, tensor<double,dim>, tensor<double,dim,dim>}
  // flux can be one of: {zero, tensor<double,dim>, tensor<double,dim,dim>, tensor<double,dim,dim,dim>,
  // tensor<double,dim,dim,dim>}
  template <typename source_type, typename flux_type, int q>
  SMITH_HOST_DEVICE static void integrate(const tensor<tuple<source_type, flux_type>, q * q>& qf_output,
                                          const TensorProductQuadratureRule<q>&, dof_type* element_residual,
                                          int step = 1)
  {
    if constexpr (is_zero<source_type>{} && is_zero<flux_type>{}) {
      return;
    }

    constexpr int ntrial = std::max(size(source_type{}), size(flux_type{}) / dim) / c;

    using s_buffer_type = std::conditional_t<is_zero<source_type>{}, zero, tensor<double, q, q> >;
    using f_buffer_type = std::conditional_t<is_zero<flux_type>{}, zero, tensor<double, dim, q, q> >;

    static constexpr bool apply_weights = true;
    static constexpr auto B = calculate_B<apply_weights, q>();
    static constexpr auto G = calculate_G<apply_weights, q>();

    for (int j = 0; j < ntrial; j++) {
      for (int i = 0; i < c; i++) {
        s_buffer_type source;
        f_buffer_type flux;

        for (int qy = 0; qy < q; qy++) {
          for (int qx = 0; qx < q; qx++) {
            [[maybe_unused]] int Q = qy * q + qx;
            if constexpr (!is_zero<source_type>{}) {
              source(qy, qx) = reinterpret_cast<const double*>(&get<SOURCE>(qf_output[Q]))[i * ntrial + j];
            }

            if constexpr (!is_zero<flux_type>{}) {
              for (int k = 0; k < dim; k++) {
                flux(k, qy, qx) = reinterpret_cast<const double*>(&get<FLUX>(qf_output[Q]))[(i * dim + k) * ntrial + j];
              }
            }
          }
        }

        auto A0 = contract<1, 0>(source, B) + contract<1, 0>(flux(0), G);
        auto A1 = contract<1, 0>(flux(1), B);

        element_residual[j * step](i) += contract<0, 0>(A0, B) + contract<0, 0>(A1, G);
      }
    }
  }

#if 0

  template <int q>
  static SMITH_DEVICE auto interpolate(const dof_type& X, const tensor<double, dim, dim>& J,
                                       const TensorProductQuadratureRule<q>& rule, cache_type<q>& A)
  {
    int tidx = threadIdx.x % q;
    int tidy = threadIdx.x / q;

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
    for (int entry = threadIdx.x; entry < n * q; entry += q * q) {
      int i   = entry % n;
      int j   = entry / n;
      B(j, i) = B_(j, i);
      G(j, i) = G_(j, i);
    }
    __syncthreads();

    tuple<tensor<double, c>, tensor<double, c, dim> > qf_input{};

    for (int i = 0; i < c; i++) {
      for (int dy = tidy; dy < n; dy += q) {
        for (int qx = tidx; qx < q; qx += q) {
          double sum[2]{};
          for (int dx = 0; dx < n; dx++) {
            sum[0] += B(qx, dx) * X(i, dy, dx);
            sum[1] += G(qx, dx) * X(i, dy, dx);
          }
          A(0, dy, qx) = sum[0];
          A(1, dy, qx) = sum[1];
        }
      }
      __syncthreads();

      for (int qy = tidy; qy < q; qy += q) {
        for (int qx = tidx; qx < q; qx += q) {
          for (int dy = 0; dy < n; dy++) {
            get<0>(qf_input)[i] += B(qy, dy) * A(0, dy, qx);
            get<1>(qf_input)[i][0] += B(qy, dy) * A(1, dy, qx);
            get<1>(qf_input)[i][1] += G(qy, dy) * A(0, dy, qx);
          }
        }
      }
    }

    get<1>(qf_input) = dot(get<1>(qf_input), inv(J));

    return qf_input;
  }

  template <typename T1, typename T2, int q>
  static SMITH_DEVICE void integrate(tuple<T1, T2>& response, const tensor<double, dim, dim>& J,
                                     const TensorProductQuadratureRule<q>& rule, cache_type<q>& A, dof_type& residual)
  {
    int tidx = threadIdx.x % q;
    int tidy = threadIdx.x / q;

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
    for (int entry = threadIdx.x; entry < n * q; entry += q * q) {
      int i   = entry % n;
      int j   = entry / n;
      B(j, i) = B_(j, i);
      G(j, i) = G_(j, i);
    }
    __syncthreads();

    auto dv = det(J) * weights1D[tidx] * weights1D[tidy];

    get<0>(response) = get<0>(response) * dv;
    get<1>(response) = dot(get<1>(response), inv(transpose(J))) * dv;

    for (int i = 0; i < c; i++) {
      // this first contraction is performed a little differently, since `response` is not
      // in shared memory, so each thread can only access its own values
      for (int qy = tidy; qy < q; qy += q) {
        for (int dx = tidx; dx < n; dx += q) {
          A(0, dx, qy) = 0.0;
          A(1, dx, qy) = 0.0;
        }
      }
      __syncthreads();

      for (int offset = 0; offset < n; offset++) {
        int  dx  = (tidx + offset) % n;
        auto sum = B(tidx, dx) * get<0>(response)(i) + G(tidx, dx) * get<1>(response)(i, 0);
        atomicAdd(&A(0, dx, tidy), sum);
        atomicAdd(&A(1, dx, tidy), B(tidx, dx) * get<1>(response)(i, 1));
      }
      __syncthreads();

      for (int dy = tidy; dy < n; dy += q) {
        for (int dx = tidx; dx < n; dx += q) {
          double sum = 0.0;
          for (int qy = 0; qy < q; qy++) {
            sum += B(qy, dy) * A(0, dx, qy);
            sum += G(qy, dy) * A(1, dx, qy);
          }
          residual(i, dy, dx) += sum;
        }
      }
    }
  }

#endif
};
/// @endcond
