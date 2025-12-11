// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrilateral_Hcurl.inl
 *
 * @brief Specialization of finite_element for Hcurl on quadrilateral geometry
 */

// this specialization defines shape functions (and their curls) that
// interpolate at Gauss-Lobatto nodes for closed intervals, and Gauss-Legendre
// nodes for open intervals.
//
// note 1: mfem assumes the parent element domain is [0,1]x[0,1]
// note 2: dofs are numbered by direction and then lexicographically in space.
//         see below
// note 3: since this is a 2D element type, the "curl" returned is just the out-of-plane component,
//         rather than 3D vector along the z-direction.
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p>
struct finite_element<mfem::Geometry::SQUARE, Hcurl<p> > {
  static constexpr auto geometry = mfem::Geometry::SQUARE;
  static constexpr auto family = Family::HCURL;
  static constexpr int dim = 2;
  static constexpr int n = (p + 1);
  static constexpr int ndof = 2 * p * (p + 1);
  static constexpr int components = 1;

  static constexpr int VALUE = 0, CURL = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  // this is how mfem provides the data to us for these elements
  // if, instead, it was stored as simply tensor< double, 2, p + 1, p >,
  // the interpolation/integrate implementation would be considerably shorter
  struct dof_type {
    tensor<double, p + 1, p> x;
    tensor<double, p, p + 1> y;
  };
  using dof_type_if = dof_type;

  template <int q>
  using cpu_batched_values_type = tensor<tensor<double, 2>, q, q>;

  template <int q>
  using cpu_batched_derivatives_type = tensor<double, q, q>;

  static constexpr auto directions = [] {
    int dof_per_direction = p * (p + 1);

    tensor<double, ndof, dim> directions{};
    for (int i = 0; i < dof_per_direction; i++) {
      directions[i + 0 * dof_per_direction] = {1.0, 0.0};
      directions[i + 1 * dof_per_direction] = {0.0, 1.0};
    }
    return directions;
  }();

  static constexpr auto nodes = [] {
    auto legendre_nodes = GaussLegendreNodes<p, mfem::Geometry::SEGMENT>();
    auto lobatto_nodes = GaussLobattoNodes<p + 1>();

    tensor<double, ndof, dim> nodes{};

    int count = 0;
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        nodes[count++] = {legendre_nodes[i], lobatto_nodes[j]};
      }
    }

    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        nodes[count++] = {lobatto_nodes[i], legendre_nodes[j]};
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
    constexpr auto points1D = GaussLegendreNodes<q, mfem::Geometry::SEGMENT>();
    [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
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

  /*

    interpolate nodes/directions and their associated numbering:

                   linear

    o-----→-----o         o-----1-----o
    |           |         |           |
    |           |         |           |
    ↑           ↑         2           3
    |           |         |           |
    |           |         |           |
    o-----→-----o         o-----0-----o


                 quadratic

    o---→---→---o         o---4---5---o
    |           |         |           |
    ↑     ↑     ↑         9    10    11
    |   →   →   |         |   2   3   |
    ↑     ↑     ↑         6     7     8
    |           |         |           |
    o---→---→---o         o---0---1---o


                   cubic

    o--→--→--→--o         o--9-10-11--o
    ↑   ↑   ↑   ↑         20  21 22  23
    |  →  →  →  |         |  6  7  8  |
    ↑   ↑   ↑   ↑         16  17 18  19
    |  →  →  →  |         |  3  4  5  |
    ↑   ↑   ↑   ↑         12  13 14  15
    o--→--→--→--o         o--0--1--2--o

  */

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_functions(tensor<double, dim> xi)
  {
    int count = 0;
    tensor<double, ndof, dim> N{};

    // do all the x-facing nodes first
    tensor<double, p + 1> N_closed = GaussLobattoInterpolation<p + 1>(xi[1]);
    tensor<double, p> N_open = GaussLegendreInterpolation<p>(xi[0]);
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        N[count++] = {N_open[i] * N_closed[j], 0.0};
      }
    }

    // then all the y-facing nodes
    N_closed = GaussLobattoInterpolation<p + 1>(xi[0]);
    N_open = GaussLegendreInterpolation<p>(xi[1]);
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        N[count++] = {0.0, N_closed[i] * N_open[j]};
      }
    }
    return N;
  }

  // the curl of a 2D vector field is entirely out-of-plane, so we return just that component
  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_function_curl(tensor<double, dim> xi)
  {
    int count = 0;
    tensor<double, ndof> curl_z{};

    // do all the x-facing nodes first
    tensor<double, p + 1> dN_closed = GaussLobattoInterpolationDerivative<p + 1>(xi[1]);
    tensor<double, p> N_open = GaussLegendreInterpolation<p>(xi[0]);
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        curl_z[count++] = -N_open[i] * dN_closed[j];
      }
    }

    // then all the y-facing nodes
    dN_closed = GaussLobattoInterpolationDerivative<p + 1>(xi[0]);
    N_open = GaussLegendreInterpolation<p>(xi[1]);
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        curl_z[count++] = dN_closed[i] * N_open[j];
      }
    }

    return curl_z;
  }

  template <typename in_t, int q>
  static auto batch_apply_shape_fn(int j, tensor<in_t, q * q> input, const TensorProductQuadratureRule<q>&)
  {
    constexpr bool apply_weights = false;
    constexpr tensor<double, q, p> B1 = calculate_B1<apply_weights, q>();
    constexpr tensor<double, q, p + 1> B2 = calculate_B2<apply_weights, q>();
    constexpr tensor<double, q, p + 1> G2 = calculate_G2<apply_weights, q>();

    int jx, jy;
    int dir = j / ((p + 1) * p);
    if (dir == 0) {
      jx = j % p;
      jy = j / p;
    } else {
      jx = (j % ((p + 1) * p)) % n;
      jy = (j % ((p + 1) * p)) / n;
    }

    using source_t = decltype(dot(get<0>(get<0>(in_t{})), tensor<double, 2>{}) + get<1>(get<0>(in_t{})) * double{});
    using flux_t = decltype(dot(get<0>(get<1>(in_t{})), tensor<double, 2>{}) + get<1>(get<1>(in_t{})) * double{});

    tensor<tuple<source_t, flux_t>, q * q> output;

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        tensor<double, 2> phi_j{(dir == 0) * B1(qx, jx) * B2(qy, jy), (dir == 1) * B1(qy, jy) * B2(qx, jx)};

        double curl_phi_j = (dir == 0) * -B1(qx, jx) * G2(qy, jy) + (dir == 1) * B1(qy, jy) * G2(qx, jx);

        int Q = qy * q + qx;
        const auto& d00 = get<0>(get<0>(input(Q)));
        const auto& d01 = get<1>(get<0>(input(Q)));
        const auto& d10 = get<0>(get<1>(input(Q)));
        const auto& d11 = get<1>(get<1>(input(Q)));

        output[Q] = {dot(d00, phi_j) + d01 * curl_phi_j, dot(d10, phi_j) + d11 * curl_phi_j};
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

    tensor<double, 2, q, q> value{};
    tensor<double, q, q> curl{};

    // to clarify which contractions correspond to which spatial dimensions
    constexpr int x = 1, y = 0;

    auto A = contract<x, 1>(element_values.x, B1);
    value[0] = contract<y, 1>(A, B2);
    curl -= contract<y, 1>(A, G2);

    A = contract<y, 1>(element_values.y, B1);
    value[1] = contract<x, 1>(A, B2);
    curl += contract<x, 1>(A, G2);

    tensor<tuple<tensor<double, 2>, double>, q * q> qf_inputs;

    int count = 0;
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        for (int i = 0; i < dim; i++) {
          get<VALUE>(qf_inputs(count))[i] = value[i](qy, qx);
        }
        get<CURL>(qf_inputs(count)) = curl(qy, qx);
        count++;
      }
    }

    return qf_inputs;
  }

  template <typename source_type, typename flux_type, int q>
  SMITH_HOST_DEVICE static void integrate(const tensor<tuple<source_type, flux_type>, q * q>& qf_output,
                                          const TensorProductQuadratureRule<q>&, dof_type* element_residual,
                                          [[maybe_unused]] int step = 1)
  {
    constexpr bool apply_weights = true;
    constexpr tensor<double, q, p> B1 = calculate_B1<apply_weights, q>();
    constexpr tensor<double, q, p + 1> B2 = calculate_B2<apply_weights, q>();
    constexpr tensor<double, q, p + 1> G2 = calculate_G2<apply_weights, q>();

    tensor<double, 2, q, q> source{};
    tensor<double, q, q> flux{};

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        int Q = qy * q + qx;
        tensor<double, dim> s{get<SOURCE>(qf_output[Q])};
        for (int i = 0; i < dim; i++) {
          source(i, qy, qx) = s[i];
        }
        flux(qy, qx) = get<FLUX>(qf_output[Q]);
      }
    }

    // to clarify which contractions correspond to which spatial dimensions
    constexpr int x = 1, y = 0;

    auto A = contract<y, 0>(source[0], B2) - contract<y, 0>(flux, G2);
    element_residual[0].x += contract<x, 0>(A, B1);

    A = contract<x, 0>(source[1], B2) + contract<x, 0>(flux, G2);
    element_residual[0].y += contract<y, 0>(A, B1);
  }

#if 0

  template <int q>
  static SMITH_DEVICE auto interpolate(const dof_type& element_values, const tensor<double, dim, dim>& J,
                                       const TensorProductQuadratureRule<q>& rule, cache_type<q>& A)
  {
    int tidx = threadIdx.x % q;
    int tidy = threadIdx.x / q;

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

    tuple<tensor<double, dim>, double> qf_input{};

    /////////////////////////////////
    ////////// X-component //////////
    /////////////////////////////////
    for (int dy = tidy; dy < p + 1; dy += q) {
      for (int qx = tidx; qx < q; qx += q) {
        double sum = 0.0;
        for (int dx = 0; dx < p; dx++) {
          sum += B1(qx, dx) * element_values.x(dy, dx);
        }
        A(dy, qx) = sum;
      }
    }
    __syncthreads();

    for (int qy = tidy; qy < q; qy += q) {
      for (int qx = tidx; qx < q; qx += q) {
        double sum[2]{};
        for (int dy = 0; dy < (p + 1); dy++) {
          sum[0] += B2(qy, dy) * A(dy, qx);
          sum[1] += G2(qy, dy) * A(dy, qx);
        }
        smith::get<0>(qf_input)[0] += sum[0];
        smith::get<1>(qf_input) -= sum[1];
      }
    }
    __syncthreads();

    /////////////////////////////////
    ////////// Y-component //////////
    /////////////////////////////////
    for (int dx = tidx; dx < p + 1; dx += q) {
      for (int qy = tidy; qy < q; qy += q) {
        double sum = 0.0;
        for (int dy = 0; dy < p; dy++) {
          sum += B1(qy, dy) * element_values.y(dy, dx);
        }
        A(dx, qy) = sum;
      }
    }
    __syncthreads();

    for (int qy = tidy; qy < q; qy += q) {
      for (int qx = tidx; qx < q; qx += q) {
        double sum[3]{};
        for (int dx = 0; dx < (p + 1); dx++) {
          sum[0] += B2(qx, dx) * A(dx, qy);
          sum[1] += G2(qx, dx) * A(dx, qy);
        }
        smith::get<0>(qf_input)[1] += sum[0];
        smith::get<1>(qf_input) += sum[1];
      }
    }

    // apply covariant Piola transformation to go
    // from parent element -> physical element
    smith::get<0>(qf_input) = linear_solve(transpose(J), smith::get<0>(qf_input));
    smith::get<1>(qf_input) = smith::get<1>(qf_input) / det(J);

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
    auto dv   = detJ * weights1D[tidx] * weights1D[tidy];

    get<0>(response) = linear_solve(J, get<0>(response)) * dv;
    get<1>(response) = get<1>(response) * (dv / detJ);

    /////////////////////////////////
    ////////// X-component //////////
    /////////////////////////////////

    // this first contraction is performed a little differently, since `response` is not
    // in shared memory, so each thread can only access its own values
    for (int qy = tidy; qy < q; qy += q) {
      for (int dx = tidx; dx < n; dx += q) {
        A(dx, qy) = 0.0;
      }
    }
    __syncthreads();

    for (int offset = 0; offset < n; offset++) {
      int  dy  = (tidy + offset) % n;
      auto sum = B2(tidy, dy) * get<0>(response)[0] - G2(tidy, dy) * get<1>(response);
      atomicAdd(&A(dy, tidx), sum);
    }
    __syncthreads();

    for (int dy = tidy; dy < p + 1; dy += q) {
      for (int dx = tidx; dx < p; dx += q) {
        double sum = 0.0;
        for (int qx = 0; qx < q; qx++) {
          sum += B1(qx, dx) * A(dy, qx);
        }
        residual.x(dy, dx) += sum;
      }
    }
    __syncthreads();

    /////////////////////////////////
    ////////// Y-component //////////
    /////////////////////////////////

    // this first contraction is performed a little differently, since `response` is not
    // in shared memory, so each thread can only access its own values
    for (int qy = tidy; qy < q; qy += q) {
      for (int dx = tidx; dx < n; dx += q) {
        A(dx, qy) = 0.0;
      }
    }
    __syncthreads();

    for (int offset = 0; offset < n; offset++) {
      int  dx  = (tidx + offset) % n;
      auto sum = B2(tidx, dx) * get<0>(response)[1] + G2(tidx, dx) * get<1>(response);
      atomicAdd(&A(dx, tidy), sum);
    }
    __syncthreads();

    for (int dy = tidy; dy < p; dy += q) {
      for (int dx = tidx; dx < p + 1; dx += q) {
        double sum = 0.0;
        for (int qy = 0; qy < q; qy++) {
          sum += B1(qy, dy) * A(dx, qy);
        }
        residual.y(dy, dx) += sum;
      }
    }
  }

#endif
};
/// @endcond
