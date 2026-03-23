// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file quadrilateral_HdivBoundary.inl
 *
 * @brief Specialization of finite_element for HdivBoundary on quadrilateral geometry
 */

// This element represents the face trace of a 3D Hdiv space on a quadrilateral
// (face) boundary. On each face of an RT_{p-1} (Hdiv<p>) hex element,
// there are p^2 DOFs representing the normal flux sigma dot n, located at
// Gauss-Legendre (open) node positions in a tensor product structure.
//
// The "value" is the scalar normal flux and the "divergence" is the
// tangential gradient of the normal flux along the face (2D vector).
//
// note: mfem assumes the parent element domain is [0,1]x[0,1]
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p, int c>
struct finite_element<mfem::Geometry::SQUARE, HdivBoundary<p, c> > {
  static constexpr auto geometry = mfem::Geometry::SQUARE;
  static constexpr auto family = Family::HDIV;
  static constexpr int components = c;
  static constexpr int dim = 2;  // dimension of the face
  static constexpr int n = p;    // number of 1D nodes
  static constexpr int ndof = p * p;

  static constexpr int VALUE = 0, DIV = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  using dof_type = tensor<double, c, p, p>;
  using dof_type_if = dof_type;

  using value_type = typename std::conditional<components == 1, double, tensor<double, components> >::type;
  using derivative_type =
      typename std::conditional<components == 1, tensor<double, dim>, tensor<double, components, dim> >::type;
  using qf_input_type = tuple<value_type, derivative_type>;

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_functions(tensor<double, dim> xi)
  {
    auto N_xi = GaussLegendreInterpolation<p>(xi[0]);
    auto N_eta = GaussLegendreInterpolation<p>(xi[1]);

    int count = 0;
    tensor<double, ndof> N{};
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p; i++) {
        N[count++] = N_xi[i] * N_eta[j];
      }
    }
    return N;
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_function_div(tensor<double, dim> xi)
  {
    auto N_xi = GaussLegendreInterpolation<p>(xi[0]);
    auto N_eta = GaussLegendreInterpolation<p>(xi[1]);
    auto dN_xi = GaussLegendreInterpolationDerivative<p>(xi[0]);
    auto dN_eta = GaussLegendreInterpolationDerivative<p>(xi[1]);

    int count = 0;
    tensor<double, ndof, dim> dN{};
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p; i++) {
        dN[count++] = {dN_xi[i] * N_eta[j], N_xi[i] * dN_eta[j]};
      }
    }
    return dN;
  }

  template <bool apply_weights, int q>
  static constexpr auto calculate_B()
  {
    constexpr auto points1D = GaussLegendreNodes<q, mfem::Geometry::SEGMENT>();
    [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
    tensor<double, q, n> B{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLegendreInterpolation<n>(points1D[i]);
      if constexpr (apply_weights) {
        B[i] = B[i] * weights1D[i];
      }
    }
    return B;
  }

  template <bool apply_weights, int q>
  static constexpr auto calculate_G()
  {
    constexpr auto points1D = GaussLegendreNodes<q, mfem::Geometry::SEGMENT>();
    [[maybe_unused]] constexpr auto weights1D = GaussLegendreWeights<q, mfem::Geometry::SEGMENT>();
    tensor<double, q, n> G{};
    for (int i = 0; i < q; i++) {
      G[i] = GaussLegendreInterpolationDerivative<n>(points1D[i]);
      if constexpr (apply_weights) {
        G[i] = G[i] * weights1D[i];
      }
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

  template <int q>
  SMITH_HOST_DEVICE static auto interpolate(const dof_type& X, const TensorProductQuadratureRule<q>&)
  {
    static constexpr bool apply_weights = false;
    static constexpr auto B = calculate_B<apply_weights, q>();
    static constexpr auto G = calculate_G<apply_weights, q>();

    tensor<double, c, q, q> value{};
    tensor<double, c, dim, q, q> gradient{};

    for (int i = 0; i < c; i++) {
      auto A0 = contract<1, 1>(X[i], B);
      auto A1 = contract<1, 1>(X[i], G);

      value[i] = contract<0, 1>(A0, B);
      gradient[i][0] = contract<0, 1>(A1, B);
      gradient[i][1] = contract<0, 1>(A0, G);
    }

    tensor<qf_input_type, q * q> output;

    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        for (int i = 0; i < c; i++) {
          if constexpr (c == 1) {
            get<VALUE>(output(qy * q + qx)) = value(0, qy, qx);
            for (int j = 0; j < dim; j++) {
              get<DIV>(output(qy * q + qx))[j] = gradient(0, j, qy, qx);
            }
          } else {
            get<VALUE>(output(qy * q + qx))[i] = value(i, qy, qx);
            for (int j = 0; j < dim; j++) {
              get<DIV>(output(qy * q + qx))[i][j] = gradient(i, j, qy, qx);
            }
          }
        }
      }
    }

    return output;
  }

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

        element_residual[j * step][i] += contract<0, 0>(A0, B) + contract<0, 0>(A1, G);
      }
    }
  }
};
/// @endcond
