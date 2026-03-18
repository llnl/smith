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
static constexpr double div_tolerance = 1.0e-10;  // coarser, since comparing to a finite difference approximation
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
  compare the direct divergence evaluation to a finite difference approximation
*/
template <typename element_type>
void verify_div_calculation()
{
  static constexpr double eps = 1.0e-6;
  static constexpr int dim = element_type::dim;
  static constexpr auto I = DenseIdentity<dim>();
  static constexpr auto random_points =
      make_tensor<num_points, dim>([](int i, int j) { return random_numbers[i * dim + j]; });

  constexpr auto N = element_type::shape_functions;
  constexpr auto divN = element_type::shape_function_div;

  for (int i = 0; i < num_points; i++) {
    auto x = random_points[i];
    if constexpr (dim == 2) {
      auto dN_dx = (N(x + eps * I[0]) - N(x - eps * I[0])) / (2.0 * eps);
      auto dN_dy = (N(x + eps * I[1]) - N(x - eps * I[1])) / (2.0 * eps);

      // divergence = d(Nx)/dx + d(Ny)/dy
      auto fd_div = dot(dN_dx, I[0]) + dot(dN_dy, I[1]);
      double relative_error = norm(divN(x) - fd_div) / norm(divN(x));
      EXPECT_NEAR(relative_error, 0.0, div_tolerance);
    }

    if constexpr (dim == 3) {
      auto dN_dx = (N(x + eps * I[0]) - N(x - eps * I[0])) / (2.0 * eps);
      auto dN_dy = (N(x + eps * I[1]) - N(x - eps * I[1])) / (2.0 * eps);
      auto dN_dz = (N(x + eps * I[2]) - N(x - eps * I[2])) / (2.0 * eps);

      // divergence = d(Nx)/dx + d(Ny)/dy + d(Nz)/dz
      auto fd_div = dot(dN_dx, I[0]) + dot(dN_dy, I[1]) + dot(dN_dz, I[2]);
      double relative_error = norm(divN(x) - fd_div) / norm(divN(x));
      EXPECT_NEAR(relative_error, 0.0, div_tolerance);
    }
  }
}

TEST(HdivKroneckerDelta, QuadrilateralLinear)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::SQUARE, Hdiv<1>>>();
}

TEST(HdivKroneckerDelta, QuadrilateralQuadratic)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::SQUARE, Hdiv<2>>>();
}

TEST(HdivKroneckerDelta, QuadrilateralCubic)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::SQUARE, Hdiv<3>>>();
}

TEST(HdivKroneckerDelta, HexahedronLinear)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::CUBE, Hdiv<1>>>();
}

TEST(HdivKroneckerDelta, HexahedronQuadratic)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::CUBE, Hdiv<2>>>();
}

TEST(HdivKroneckerDelta, HexahedronCubic)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::CUBE, Hdiv<3>>>();
}

TEST(HdivDiv, QuadrilateralLinear) { verify_div_calculation<finite_element<::mfem::Geometry::SQUARE, Hdiv<1>>>(); }

TEST(HdivDiv, QuadrilateralQuadratic) { verify_div_calculation<finite_element<::mfem::Geometry::SQUARE, Hdiv<2>>>(); }

TEST(HdivDiv, QuadrilateralCubic) { verify_div_calculation<finite_element<::mfem::Geometry::SQUARE, Hdiv<3>>>(); }

TEST(HdivDiv, HexahedronLinear) { verify_div_calculation<finite_element<::mfem::Geometry::CUBE, Hdiv<1>>>(); }

TEST(HdivDiv, HexahedronQuadratic) { verify_div_calculation<finite_element<::mfem::Geometry::CUBE, Hdiv<2>>>(); }

TEST(HdivDiv, HexahedronCubic) { verify_div_calculation<finite_element<::mfem::Geometry::CUBE, Hdiv<3>>>(); }

// Triangle (only p=1 shape functions fully implemented)
TEST(HdivKroneckerDelta, TriangleLinear)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::TRIANGLE, Hdiv<1>>>();
}

TEST(HdivDiv, TriangleLinear) { verify_div_calculation<finite_element<::mfem::Geometry::TRIANGLE, Hdiv<1>>>(); }

// Tetrahedron (only p=1 shape functions fully implemented)
TEST(HdivKroneckerDelta, TetrahedronLinear)
{
  verify_kronecker_delta_property<finite_element<::mfem::Geometry::TETRAHEDRON, Hdiv<1>>>();
}

TEST(HdivDiv, TetrahedronLinear) { verify_div_calculation<finite_element<::mfem::Geometry::TETRAHEDRON, Hdiv<1>>>(); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
