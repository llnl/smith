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
//
// The shape functions are vector-valued and the relevant derivative is the scalar divergence.
// DOF ordering follows mfem's RT_FECollection convention.
//
// for additional information on the finite_element concept requirements, see finite_element.hpp
/// @cond
template <int p>
struct finite_element<mfem::Geometry::TETRAHEDRON, Hdiv<p> > {
  static constexpr auto geometry = mfem::Geometry::TETRAHEDRON;
  static constexpr auto family = Family::HDIV;
  static constexpr int dim = 3;
  static constexpr int n = p;
  static constexpr int ndof = p * (p + 1) * (p + 3) / 2;  // = RT_{p-1} on tet
  static constexpr int components = 1;

  static constexpr int VALUE = 0, DIV = 1;
  static constexpr int SOURCE = 0, FLUX = 1;

  using residual_type =
      typename std::conditional<components == 1, tensor<double, ndof>, tensor<double, ndof, components> >::type;

  using dof_type = tensor<double, ndof>;

  using value_type = tensor<double, dim>;
  using derivative_type = double;
  using qf_input_type = tuple<value_type, derivative_type>;

  /*
    Hdiv<1> -> RT_0 on tetrahedron: 4 DOFs (one per face, normal component)

    The 4 faces of the reference tet and their outward normals:
    Face 0: {v1,v2,v3} = {(1,0,0),(0,1,0),(0,0,1)}, normal ~ (1,1,1)
    Face 1: {v0,v2,v3} = {(0,0,0),(0,1,0),(0,0,1)}, normal ~ (-1,0,0)
    Face 2: {v0,v1,v3} = {(0,0,0),(1,0,0),(0,0,1)}, normal ~ (0,-1,0)
    Face 3: {v0,v1,v2} = {(0,0,0),(1,0,0),(0,1,0)}, normal ~ (0,0,-1)

    RT0 basis functions (mfem convention):
      phi_0 = (x, y, z)             div = 3
      phi_1 = (x-1, y, z)           div = 3
      phi_2 = (x, y-1, z)           div = 3
      phi_3 = (x, y, z-1)           div = 3
  */

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof, dim> shape_functions(
      [[maybe_unused]] tensor<double, dim> xi)
  {
    tensor<double, ndof, dim> N{};

    if constexpr (p == 1) {
      // Hdiv<1> -> RT_0: 4 basis functions
      double x = xi[0], y = xi[1], z = xi[2];
      N[0] = {x, y, z};
      N[1] = {x - 1.0, y, z};
      N[2] = {x, y - 1.0, z};
      N[3] = {x, y, z - 1.0};
    }

    return N;
  }

  SMITH_HOST_DEVICE static constexpr tensor<double, ndof> shape_function_div(
      [[maybe_unused]] tensor<double, dim> xi)
  {
    tensor<double, ndof> div{};

    if constexpr (p == 1) {
      // Hdiv<1> -> RT_0: constant divergence = 3
      div[0] = 3.0;
      div[1] = 3.0;
      div[2] = 3.0;
      div[3] = 3.0;
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

    for (int i = 0; i < Q; i++) {
      auto N = shape_functions(xi[i]);
      auto divN = shape_function_div(xi[i]);

      tensor<double, dim> s{get<SOURCE>(qf_output[i])};
      double f = get<FLUX>(qf_output[i]);

      for (int j = 0; j < ndof; j++) {
        element_residual[0][j] += (dot(s, N[j]) + f * divN[j]) * weights[i];
      }
    }
  }
};
/// @endcond
