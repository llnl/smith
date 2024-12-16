#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <functional>

#include "refactor/mesh.hpp"
#include "refactor/domain.hpp"

#include <gtest/gtest.h>

#include "forall.hpp"
#include "serac/numerics/functional/tensor.hpp"

#include "containers/ndarray_conversions.hpp"

using namespace refactor;

// ----------------------------------------------------------------------------

template < typename vec_t, typename curl_t >
void integrate_flux_test(std::string filename,
                         std::function< vec_t(vec_t, int) > f,
                         std::function< curl_t(vec_t) > g,
                         std::function< double(int) > answer,
                         double tolerance) {

  using mat_t = decltype(outer(vec_t{}, vec_t{}));

  auto mesh = Mesh::load(SERAC_MESH_DIR + filename);

  std::cout << std::setprecision(15);

  // evaluate g at each quadrature point
  std::function< curl_t(vec_t, mat_t) > g_xi = [g](vec_t x, mat_t J) { 
    // note: detJ cancels out here, since it appears in the numerator from
    // the change of coordinates, but also in the denominator from the covariant piola 
    if constexpr (std::is_same< mat_t, mat2 >::value) {
      return g(x);
    } else {
      return dot(transpose(J), g(x));
    }
  };

  for (int p = 1; p < 4; p++) {

    std::function< double(vec_t, vec_t) > f_p = [f, p](vec_t x, vec_t n) { 
      return dot(f(x,p), n) ; 
    };

    Field u = create_field(mesh, Family::Hcurl, p, 1);
    nd::array<double,2> nodes = nodes_for(u, mesh); 
    nd::array<double,2> directions = directions_for(u, mesh); 
    u = forall(f_p, nodes, directions);

    BasisFunction phi(u);

    for (int q = p + 1; q < 5; q++) {

      Domain domain(mesh, MeshQuadratureRule(q));

      auto x_q = evaluate(mesh.X, domain);
      auto dx_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

      nd::array<double,2> g_q = forall(g, x_q);
      nd::array<double,2> g_xi_q = forall(g_xi, x_q, dx_dxi_q);

      Residual r1 = integrate(dot(g_q, curl(phi)), domain);
      Residual r2 = integrate(dot(g_xi_q, curl(phi)), isoparametric(domain));

      SCOPED_TRACE("p = " + std::to_string(p) + ", q = " + std::to_string(q));
      EXPECT_NEAR(dot(r1, u), answer(p), tolerance);
      EXPECT_NEAR(dot(r2, u), answer(p), tolerance);

    }
  }

}

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[{x_, y_}] := {3 + y, 2 - x};
    g[{x_, y_}] := x + y;
    Integrate[Curl[f[{x, y}], {x, y}] g[{x, y}], {x, y} \[Element] Rectangle[{0, 0}, {1, 1}]]
  
  --------------------
  
  Out[] := 
    -2.0
*/
std::function<vec2(vec2,int)> f1([](vec2 x, int p){ return vec2{3 + x[1], 2 - x[0]}; });
std::function<double(vec2)> g1([](vec2 x){ return x[0] + x[1]; });
std::function<double(int)> answer1([](int p){ return -2.0; });

// note: non affinely-transformed low-order quadrilateral elements can't exactly reproduce the 
//       functions in this test, so for p=1 those tests have some small finite error
//       which decreases with mesh refinement
TEST(IntegrateTest, FluxHcurlTris) { integrate_flux_test("patch_test_tris.json", f1, g1, answer1, 5.0e-14); }
TEST(IntegrateTest, FluxHcurlQuads) { integrate_flux_test("patch_test_quads.json", f1, g1, answer1, 1.0e-2); }
TEST(IntegrateTest, FluxHcurlQuadsFine) { integrate_flux_test("unit_square_of_quads_fine.json", f1, g1, answer1, 1.0e-5); }
TEST(IntegrateTest, FluxHcurlBoth) { integrate_flux_test("patch_test_tris_and_quads.json", f1, g1, answer1, 1.0e-3); }
TEST(IntegrateTest, FluxHcurlTrisFine) { integrate_flux_test("unit_square_of_tris_fine.json", f1, g1, answer1, 1.5e-13); }

// ----------------------------------------------------------------------------

/*
  In[] := 
  f[{x_, y_, z_}] := {y + 3, 2 - x - z, 1 + y};
  g[{x_, y_, z_}] := {y, z, 2 x};
  Integrate[Curl[f[{x, y, z}], {x, y, z}]  g[{x, y, z}], {x, y, z} \[Element] Tetrahedron[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]]
  Integrate[Curl[f[{x, y, z}], {x, y, z}]  g[{x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {1, 1, 1}]]
  Integrate[Curl[f[{x, y, z}], {x, y, z}]  g[{x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {2, 1, 1}]]
 
  --------------------
 
  Out[] := 
    -1.0 / 12.0
    -1.0
    -6.0
*/
std::function< vec3(vec3, int) > f2([](vec3 x, int p) { return vec3{3 + x[1], 2 - x[0] - x[2], 1 + x[1]}; });
std::function< vec3(vec3) > g2([](vec3 x) { return vec3{x[1], x[2], 2 * x[0]}; });
std::function< double(int)> answer2([](int p){ return -1.0 / 12.0; });
std::function< double(int)> answer3([](int p){ return -1.0; });
std::function< double(int)> answer4([](int p){ return -6.0; });

TEST(IntegrateTest, FluxHcurlOneTet) { integrate_flux_test("one_tet.json", f2, g2, answer2, 1.0e-13); }
TEST(IntegrateTest, FluxHcurlOneHex) { integrate_flux_test("one_hex.json", f2, g2, answer3, 1.0e-13); }

TEST(IntegrateTest, FluxHcurlTets) { integrate_flux_test("patch_test_tets.json", f2, g2, answer3, 1.0e-13); }
TEST(IntegrateTest, FluxHcurlHexes) { integrate_flux_test("patch_test_hexes.json", f2, g2, answer3, 1.0e-1); }
TEST(IntegrateTest, FluxHcurlHexesFine) { integrate_flux_test("unit_cube_of_hexes_fine.json", f2, g2, answer3, 5.0e-4); }
TEST(IntegrateTest, FluxHcurlTetsAndHexes) { integrate_flux_test("patch_test_tets_and_hexes.json", f2, g2, answer4, 1.0e-1); }
