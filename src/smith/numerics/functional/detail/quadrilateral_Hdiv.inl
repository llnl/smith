// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrilateral_Hdiv.inl
 *
 * @brief Specialization of finite_element for Hdiv on quadrilateral geometry
 */

// this specialization defines shape functions (and their divergences) that
// interpolate at Gauss-Lobatto nodes for closed intervals, and Gauss-Legendre
// nodes for open intervals.
//
// note 1: mfem assumes the parent element domain is [0,1]x[0,1]
// note 2: dofs are numbered by direction and then lexicographically in space.
//         see below
// note 3: H(div) is the "dual" of H(curl) -- the roles of open and closed
//         nodes are swapped relative to quadrilateral_Hcurl.inl:
//         - Hcurl: open (Legendre) in component direction, closed (Lobatto) in transverse
//         - Hdiv:  closed (Lobatto) in component direction, open (Legendre) in transverse
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p>
struct finite_element<mfem::Geometry::SQUARE, Hdiv<p> > {
  static constexpr auto geometry = mfem::Geometry::SQUARE;
  static constexpr auto family = Family::HDIV;
  static constexpr int dim = 2;
  static constexpr int n = (p + 1);
  static constexpr int ndof = 2 * p * (p + 1);
  static constexpr int components = 1;

  static constexpr int VALUE = 0, DIV = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  // DOF layout: x-component has closed(p+1) in x, open(p) in y
  //             y-component has open(p) in x, closed(p+1) in y
  // This is the transpose of Hcurl's layout
  struct dof_type {
    tensor<double, p, p + 1> x;
    tensor<double, p + 1, p> y;
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
    // x-facing DOFs: closed (Lobatto) in x, open (Legendre) in y
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        nodes[count++] = {lobatto_nodes[i], legendre_nodes[j]};
      }
    }

    // y-facing DOFs: open (Legendre) in x, closed (Lobatto) in y
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        nodes[count++] = {legendre_nodes[i], lobatto_nodes[j]};
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

  /*

    interpolate nodes/directions and their associated numbering:

    For Hdiv, closed (Lobatto) nodes are in the component direction,
    and open (Legendre) nodes are in the transverse direction.

                   linear

    o-----↑-----o         o-----3-----o
    |           |         |           |
    |           |         |           |
    →           →         0           1
    |           |         |           |
    |           |         |           |
    o-----↑-----o         o-----2-----o


                 quadratic

    o---↑---↑---o         o---9--10---o
    |           |         |           |
    →     →     →         3     4     5
    |   ↑   ↑   |         |   7   8   |
    →     →     →         0     1     2
    |           |         |           |
    o---↑---↑---o         o---6---7---o

  */

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_functions(tensor<double, dim> xi)
  {
    int count = 0;
    tensor<double, ndof, dim> N{};

    // x-facing DOFs: closed (Lobatto) in x, open (Legendre) in y
    tensor<double, p + 1> N_closed = GaussLobattoInterpolation<p + 1>(xi[0]);
    tensor<double, p> N_open = GaussLegendreInterpolation<p>(xi[1]);
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        N[count++] = {N_closed[i] * N_open[j], 0.0};
      }
    }

    // y-facing DOFs: open (Legendre) in x, closed (Lobatto) in y
    N_closed = GaussLobattoInterpolation<p + 1>(xi[1]);
    N_open = GaussLegendreInterpolation<p>(xi[0]);
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        N[count++] = {0.0, N_open[i] * N_closed[j]};
      }
    }
    return N;
  }

  // the divergence of a 2D vector field is a scalar: d(Nx)/dx + d(Ny)/dy
  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_function_div(tensor<double, dim> xi)
  {
    int count = 0;
    tensor<double, ndof> div{};

    // x-facing DOFs: derivative of closed (Lobatto) in x, open (Legendre) in y
    tensor<double, p + 1> dN_closed = GaussLobattoInterpolationDerivative<p + 1>(xi[0]);
    tensor<double, p> N_open = GaussLegendreInterpolation<p>(xi[1]);
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        div[count++] = dN_closed[i] * N_open[j];
      }
    }

    // y-facing DOFs: open (Legendre) in x, derivative of closed (Lobatto) in y
    dN_closed = GaussLobattoInterpolationDerivative<p + 1>(xi[1]);
    N_open = GaussLegendreInterpolation<p>(xi[0]);
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        div[count++] = N_open[i] * dN_closed[j];
      }
    }

    return div;
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
      // x-facing: closed in x (p+1 nodes), open in y (p nodes)
      jx = j % n;
      jy = j / n;
    } else {
      // y-facing: open in x (p nodes), closed in y (p+1 nodes)
      jx = (j % ((p + 1) * p)) % p;
      jy = (j % ((p + 1) * p)) / p;
    }

    using source_t = decltype(dot(get<0>(get<0>(in_t{})), tensor<double, 2>{}) + get<1>(get<0>(in_t{})) * double{});
    using flux_t = decltype(dot(get<0>(get<1>(in_t{})), tensor<double, 2>{}) + get<1>(get<1>(in_t{})) * double{});

    tensor<tuple<source_t, flux_t>, q * q> output;

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        // Hdiv shape function: x-component uses B2 in x (closed), B1 in y (open)
        //                      y-component uses B1 in x (open), B2 in y (closed)
        tensor<double, 2> phi_j{(dir == 0) * B2(qx, jx) * B1(qy, jy), (dir == 1) * B1(qx, jx) * B2(qy, jy)};

        // divergence: x-component uses G2 in x, B1 in y
        //             y-component uses B1 in x, G2 in y
        double div_phi_j = (dir == 0) * G2(qx, jx) * B1(qy, jy) + (dir == 1) * B1(qx, jx) * G2(qy, jy);

        int Q = qy * q + qx;
        const auto& d00 = get<0>(get<0>(input(Q)));
        const auto& d01 = get<1>(get<0>(input(Q)));
        const auto& d10 = get<0>(get<1>(input(Q)));
        const auto& d11 = get<1>(get<1>(input(Q)));

        output[Q] = {dot(d00, phi_j) + d01 * div_phi_j, dot(d10, phi_j) + d11 * div_phi_j};
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
    tensor<double, q, q> div{};

    // to clarify which contractions correspond to which spatial dimensions
    constexpr int x = 1, y = 0;

    // x-component: closed (B2) in x, open (B1) in y
    // element_values.x is (p, p+1) = (y_open, x_closed)
    auto Ax = contract<x, 1>(element_values.x, B2);
    value[0] = contract<y, 1>(Ax, B1);
    // divergence contribution from x: dN_closed/dx in x (G2), open in y (B1)
    Ax = contract<x, 1>(element_values.x, G2);
    div = contract<y, 1>(Ax, B1);

    // y-component: open (B1) in x, closed (B2) in y
    // element_values.y is (p+1, p) = (y_closed, x_open)
    auto Ay = contract<x, 1>(element_values.y, B1);
    value[1] = contract<y, 1>(Ay, B2);
    // divergence contribution from y: open in x (B1), dN_closed/dy in y (G2)
    div += contract<y, 1>(Ay, G2);

    tensor<tuple<tensor<double, 2>, double>, q * q> qf_inputs;

    int count = 0;
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        for (int i = 0; i < dim; i++) {
          get<VALUE>(qf_inputs(count))[i] = value[i](qy, qx);
        }
        get<DIV>(qf_inputs(count)) = div(qy, qx);
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
    if constexpr (is_zero<source_type>{} && is_zero<flux_type>{}) return;

    constexpr bool apply_weights = true;
    constexpr tensor<double, q, p> B1 = calculate_B1<apply_weights, q>();
    constexpr tensor<double, q, p + 1> B2 = calculate_B2<apply_weights, q>();
    constexpr tensor<double, q, p + 1> G2 = calculate_G2<apply_weights, q>();

    using source_buf_t = std::conditional_t<is_zero<source_type>{}, zero, tensor<double, 2, q, q>>;
    using flux_buf_t   = std::conditional_t<is_zero<flux_type>{},   zero, tensor<double, q, q>>;

    source_buf_t source{};
    flux_buf_t flux{};

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        int Q = qy * q + qx;
        if constexpr (!is_zero<source_type>{}) {
          tensor<double, dim> s{get<SOURCE>(qf_output[Q])};
          for (int i = 0; i < dim; i++) {
            source(i, qy, qx) = s[i];
          }
        }
        if constexpr (!is_zero<flux_type>{}) {
          flux(qy, qx) = get<FLUX>(qf_output[Q]);
        }
      }
    }

    // to clarify which contractions correspond to which spatial dimensions
    constexpr int x = 1, y = 0;

    // x-component residual:
    //   source[0] is dual to the x-value -> tested with closed(x) * open(y) = B2(x) * B1(y)
    //   flux is dual to the divergence -> tested with dg_closed/dx * open(y) = G2(x) * B1(y)
    if constexpr (!is_zero<source_type>{}) {
      auto Ax = contract<y, 0>(source[0], B1);
      element_residual[0].x += contract<x, 0>(Ax, B2);
      auto Ay = contract<x, 0>(source[1], B1);
      element_residual[0].y += contract<y, 0>(Ay, B2);
    }

    if constexpr (!is_zero<flux_type>{}) {
      auto Ax = contract<y, 0>(flux, B1);
      element_residual[0].x += contract<x, 0>(Ax, G2);
      auto Ay = contract<x, 0>(flux, B1);
      element_residual[0].y += contract<y, 0>(Ay, G2);
    }
  }
};
/// @endcond
