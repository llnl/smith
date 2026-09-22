// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file segment_L2.inl
 *
 * @brief Specialization of finite_element for L2 on segment geometry
 */

// this specialization defines shape functions (and their gradients) that
// interpolate at Gauss-Lobatto nodes for the appropriate polynomial order
//
// note: mfem assumes the parent element domain is [0,1]
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p, int c>
struct finite_element<mfem::Geometry::SEGMENT, L2<p, c> > {
  static constexpr auto geometry = mfem::Geometry::SEGMENT;
  static constexpr auto family = Family::L2;
  static constexpr int components = c;
  static constexpr int dim = 1;
  static constexpr int n = (p + 1);
  static constexpr int ndof = (p + 1);

  static constexpr int VALUE = 0, GRADIENT = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using dof_type = tensor<double, c, n>;
  using dof_type_if = tensor<double, c, 2, ndof>;

  using value_type = typename std::conditional<components == 1, double, tensor<double, components> >::type;
  using derivative_type = value_type;
  using qf_input_type = tuple<value_type, derivative_type>;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_functions(double xi)
  {
    return GaussLobattoInterpolation<ndof>(xi);
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_function_gradients(double xi)
  {
    return GaussLobattoInterpolationDerivative<ndof>(xi);
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

  template <typename T, int q>
  static auto batch_apply_shape_fn(int jx, tensor<T, q> input, const TensorProductQuadratureRule<q>&)
  {
    static constexpr bool apply_weights = false;
    static constexpr auto B = calculate_B<apply_weights, q>();
    static constexpr auto G = calculate_G<apply_weights, q>();

    using source_t = decltype(get<0>(get<0>(T{})) + get<1>(get<0>(T{})));
    using flux_t = decltype(get<0>(get<1>(T{})) + get<1>(get<1>(T{})));

    tensor<tuple<source_t, flux_t>, q> output;

    for (int qx = 0; qx < q; qx++) {
      double phi_j = B(qx, jx);
      double dphi_j_dxi = G(qx, jx);

      auto& d00 = get<0>(get<0>(input(qx)));
      auto& d01 = get<1>(get<0>(input(qx)));
      auto& d10 = get<0>(get<1>(input(qx)));
      auto& d11 = get<1>(get<1>(input(qx)));

      output[qx] = {d00 * phi_j + d01 * dphi_j_dxi, d10 * phi_j + d11 * dphi_j_dxi};
    }

    return output;
  }

  template <typename T, int q>
  static auto batch_apply_shape_fn_interior_face(int jx, tensor<T, q> input, const TensorProductQuadratureRule<q>&)
  {
    static constexpr bool apply_weights = false;
    static constexpr auto B = calculate_B<apply_weights, q>();

    using source0_t = decltype(get<0>(get<0>(T{})) + get<1>(get<0>(T{})));
    using source1_t = decltype(get<0>(get<1>(T{})) + get<1>(get<1>(T{})));

    tensor<tuple<source0_t, source1_t>, q> output;

    for (int qx = 0; qx < q; qx++) {
      int j = jx % ndof;
      int s = jx / ndof;

      double phi0_j = B(qx, j) * (s == 0);
      double phi1_j = B(qx, j) * (s == 1);

      const auto& d00 = get<0>(get<0>(input(qx)));
      const auto& d01 = get<1>(get<0>(input(qx)));
      const auto& d10 = get<0>(get<1>(input(qx)));
      const auto& d11 = get<1>(get<1>(input(qx)));

      output[qx] = {d00 * phi0_j + d01 * phi1_j, d10 * phi0_j + d11 * phi1_j};
    }

    return output;
  }

  template <int q>
  SMITH_HOST_DEVICE static auto interpolate(const dof_type_if& X, const TensorProductQuadratureRule<q>&)
  {
    static constexpr bool apply_weights = false;
    static constexpr auto BT = transpose(calculate_B<apply_weights, q>());

    tensor<double, q> values{};

    tensor<tuple<value_type, value_type>, q> output{};

    // apply the shape functions
    for (int i = 0; i < c; i++) {
      values = dot(X[i][0], BT);
      for (int qx = 0; qx < q; qx++) {
        if constexpr (c == 1) {
          get<0>(output[qx]) = values[qx];
        } else {
          get<0>(output[qx])[i] = values[qx];
        }
      }

      values = dot(X[i][1], BT);
      for (int qx = 0; qx < q; qx++) {
        if constexpr (c == 1) {
          get<1>(output[qx]) = values[qx];
        } else {
          get<1>(output[qx])[i] = values[qx];
        }
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

    tensor<double, c, q> value{};
    tensor<double, c, q> gradient{};

    // apply the shape functions
    for (int i = 0; i < c; i++) {
      value(i) = dot(B, X[i]);
      gradient(i) = dot(G, X[i]);
    }

    // transpose the quadrature data into a tensor of tuples
    tensor<qf_input_type, q> output;

    for (int qx = 0; qx < q; qx++) {
      if constexpr (c == 1) {
        get<VALUE>(output(qx)) = value(0, qx);
        get<GRADIENT>(output(qx)) = gradient(0, qx);
      } else {
        for (int i = 0; i < c; i++) {
          get<VALUE>(output(qx))[i] = value(i, qx);
          get<GRADIENT>(output(qx))[i] = gradient(i, qx);
        }
      }
    }

    return output;
  }

  template <typename source_type, typename flux_type, int q>
  SMITH_HOST_DEVICE static void integrate(const tensor<tuple<source_type, flux_type>, q>& qf_output,
                                          const TensorProductQuadratureRule<q>&, dof_type* element_residual,
                                          [[maybe_unused]] int step = 1)
  {
    if constexpr (is_zero<source_type>{} && is_zero<flux_type>{}) {
      return;
    }

    constexpr int ntrial = std::max(size(source_type{}), size(flux_type{}) / dim) / c;

    using s_buffer_type = std::conditional_t<is_zero<source_type>{}, zero, tensor<double, q> >;
    using f_buffer_type = std::conditional_t<is_zero<flux_type>{}, zero, tensor<double, q> >;

    static constexpr bool apply_weights = true;
    static constexpr auto B = calculate_B<apply_weights, q>();
    static constexpr auto G = calculate_G<apply_weights, q>();

    for (int j = 0; j < ntrial; j++) {
      for (int i = 0; i < c; i++) {
        s_buffer_type source;
        f_buffer_type flux;

        for (int qx = 0; qx < q; qx++) {
          if constexpr (!is_zero<source_type>{}) {
            source(qx) = reinterpret_cast<const double*>(&get<SOURCE>(qf_output[qx]))[i * ntrial + j];
          }
          if constexpr (!is_zero<flux_type>{}) {
            flux(qx) = reinterpret_cast<const double*>(&get<FLUX>(qf_output[qx]))[i * ntrial + j];
          }
        }

        element_residual[j * step](i) += dot(source, B) + dot(flux, G);
      }
    }
  }

  template <typename T, int q>
  SMITH_HOST_DEVICE static void integrate(const tensor<tuple<T, T>, q>& qf_output,
                                          const TensorProductQuadratureRule<q>&, dof_type_if* element_residual,
                                          [[maybe_unused]] int step = 1)
  {
    constexpr int ntrial = size(T{}) / c;

    using buffer_type = tensor<double, q>;

    static constexpr bool apply_weights = true;
    static constexpr auto B = calculate_B<apply_weights, q>();

    for (int j = 0; j < ntrial; j++) {
      for (int i = 0; i < c; i++) {
        buffer_type source_0;
        buffer_type source_1;

        for (int qx = 0; qx < q; qx++) {
          source_0(qx) = reinterpret_cast<const double*>(&get<0>(qf_output[qx]))[i * ntrial + j];
          source_1(qx) = reinterpret_cast<const double*>(&get<1>(qf_output[qx]))[i * ntrial + j];
        }

        element_residual[j * step](i, 0) += dot(source_0, B);
        element_residual[j * step](i, 1) += dot(source_1, B);
      }
    }
  }
};
/// @endcond
