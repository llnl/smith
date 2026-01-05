// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/infrastructure/application_manager.hpp"

using namespace smith;

static constexpr double kronecker_tolerance = 1.0e-13;
static constexpr double curl_tolerance = 1.0e-10;  // coarser, since comparing to a finite difference approximation
static constexpr int num_points = 10;
static constexpr tensor random_numbers = {
    {-0.886787, -0.850126, 0.464212,  -0.0733101, -0.397738, 0.302355,   -0.570758, 0.977727,  0.282365,  -0.768947,
     0.6216,    0.43598,   -0.696321, 0.92545,    0.183003,  0.121761,   -0.877239, 0.0347577, -0.818463, -0.216474,
     -0.43894,  0.0178874, -0.869944, -0.733499,  0.255124,  -0.0561095, -0.34607,  -0.305958, 0.414472,  -0.744998}};

/*
iterate over the node/direction pairs in the element, and verify that
contributions from other shape functions in the element are orthogonal
*/
template <typename element_type>
void verify_kronecker_delta_property()
{
  static constexpr auto nodes = element_type::nodes;
  static constexpr auto directions = element_type::directions;
  static constexpr auto I = DenseIdentity<element_type::ndof>();

  for (int i = 0; i < element_type::ndof; i++) {
    double error = norm(I[i] - dot(element_type::shape_functions(nodes[i]), directions[i]));
    EXPECT_NEAR(error, 0.0, kronecker_tolerance);
  }
}

/*
  compare the direct curl evaluation to a finite difference approximation
*/
template <typename element_type>
void verify_curl_calculation()
{
  static constexpr double eps = 1.0e-6;
  static constexpr int dim = element_type::dim;
  static constexpr auto I = DenseIdentity<dim>();
  static constexpr auto random_points =
      make_tensor<num_points, dim>([](int i, int j) { return random_numbers[i * dim + j]; });

  constexpr auto N = element_type::shape_functions;
  constexpr auto curlN = element_type::shape_function_curl;

  for (int i = 0; i < num_points; i++) {
    auto x = random_points[i];
    if constexpr (dim == 2) {
      auto dN_dx = (N(x + eps * I[0]) - N(x - eps * I[0])) / (2.0 * eps);
      auto dN_dy = (N(x + eps * I[1]) - N(x - eps * I[1])) / (2.0 * eps);

      double relative_error = norm(curlN(x) - (dot(dN_dx, I[1]) - dot(dN_dy, I[0]))) / norm(curlN(x));
      EXPECT_NEAR(relative_error, 0.0, curl_tolerance);
    }

    if constexpr (dim == 3) {
      auto dN_dx = (N(x + eps * I[0]) - N(x - eps * I[0])) / (2.0 * eps);
      auto dN_dy = (N(x + eps * I[1]) - N(x - eps * I[1])) / (2.0 * eps);
      auto dN_dz = (N(x + eps * I[2]) - N(x - eps * I[2])) / (2.0 * eps);

      double relative_error = norm(tensor{{norm(transpose(curlN(x))[0] - (dot(dN_dy, I[2]) - dot(dN_dz, I[1]))),
                                           norm(transpose(curlN(x))[1] - (dot(dN_dz, I[0]) - dot(dN_dx, I[2]))),
                                           norm(transpose(curlN(x))[2] - (dot(dN_dx, I[1]) - dot(dN_dy, I[0])))}}) /
                              norm(curlN(x));
      EXPECT_NEAR(relative_error, 0.0, curl_tolerance);
    }
  }
}

TEST(VerifyKroneckerdelta, QuadrilateralLinear)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::SQUARE, Hcurl<1>>>();
}

TEST(VerifyKroneckerdelta, QuadrilateralQuadratic)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::SQUARE, Hcurl<2>>>();
}

TEST(VerifyKroneckerdelta, QuadrilateralCubic)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::SQUARE, Hcurl<3>>>();
}

TEST(VerifyKroneckerdelta, HexahedronLinear)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::CUBE, Hcurl<1>>>();
}

TEST(VerifyKroneckerdelta, HexahedronQuadratic)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::CUBE, Hcurl<2>>>();
}

TEST(VerifyKroneckerdelta, HexahedronCubic)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::CUBE, Hcurl<3>>>();
}

TEST(VerifyCurl, QuadrilateralLinear) { verify_curl_calculation<finite_element<::mfem::Geometry::SQUARE, Hcurl<1>>>(); }

TEST(VerifyCurl, QuadrilateralQuadratic)
{
  verify_curl_calculation<finite_element<::mfem::Geometry::SQUARE, Hcurl<2>>>();
}

TEST(VerifyCurl, QuadrilateralCubic) { verify_curl_calculation<finite_element<::mfem::Geometry::SQUARE, Hcurl<3>>>(); }

TEST(VerifyCurl, HexahedronLinear) { verify_curl_calculation<finite_element<::mfem::Geometry::CUBE, Hcurl<1>>>(); }

TEST(VerifyCurl, HexahedronQuadratic) { verify_curl_calculation<finite_element<::mfem::Geometry::CUBE, Hcurl<2>>>(); }

TEST(VerifyCurl, HexahedronCubic) { verify_curl_calculation<finite_element<::mfem::Geometry::CUBE, Hcurl<3>>>(); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
