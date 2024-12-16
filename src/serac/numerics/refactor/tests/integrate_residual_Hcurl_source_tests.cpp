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

template < typename vecd >
void integrate_source_test(std::string filename,
                           std::function< vecd(vecd, int) > f,
                           std::function< vecd(vecd) > g,
                           std::function< double(int) > answer,
                           double tolerance) {

  using matd = decltype(outer(vecd{}, vecd{}));

  auto mesh = Mesh::load(SERAC_MESH_DIR + filename);

  std::cout << std::setprecision(15);

  // evaluate g at each quadrature point
  std::function< vecd(vecd, matd) > g_xi = [g](vecd x, matd J) { 
    return dot(inv(J), g(x)) * det(J); 
  };

  for (int p = 1; p < 4; p++) {

    std::function< double(vecd, vecd) > f_p = [f, p](vecd x, vecd n) { return dot(f(x,p), n) ; };

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

      Residual r1 = integrate(g_q * phi, domain);
      Residual r2 = integrate(g_xi_q * phi, isoparametric(domain));

      SCOPED_TRACE("p = " + std::to_string(p) + ", q = " + std::to_string(q));
      EXPECT_NEAR(dot(r1, u), answer(p), tolerance);
      EXPECT_NEAR(dot(r2, u), answer(p), tolerance);

      EXPECT_NEAR(relative_error(r1.data, r2.data), 0.0, 1.0e-14);

    }
  }

}

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[{x_, y_}] := {1, 3};
    g[{x_, y_}] := {1 + y, 2 - x};
    Integrate[f[{x, y}] . g[{x, y}], {x, y} \[Element] Rectangle[{0, 0}, {1, 1}]]
  
  --------------------
  
  Out[] := 
    6.0
*/
std::function<vec2(vec2,int)> f1([](vec2 x, int p){ return vec2{1, 3}; });
std::function<vec2(vec2)> g1([](vec2 x){ return vec2{1 + x[1], 2 - x[0]}; });
std::function<double(int)> answer1([](int p){ return 6.0; });
TEST(IntegrateTest, SourceHcurlTris) { integrate_source_test("patch_test_tris.json", f1, g1, answer1, 1.5e-14); }
TEST(IntegrateTest, SourceHcurlQuads) { integrate_source_test("patch_test_quads.json", f1, g1, answer1, 1.0e-14); }
TEST(IntegrateTest, SourceHcurlBoth) { integrate_source_test("patch_test_tris_and_quads.json", f1, g1, answer1, 1.0e-14); }

/* 
  In[] := 
    f[{x_, y_}] := {1 + 2 y, 3 - 2 x};
    g[{x_, y_}] := {2, 3};
    Integrate[f[{x, y}] . g[{x, y}], {x, y} \[Element] Triangle[{{0, 0}, {1, 0}, {0, 1}}]]
    Integrate[f[{x, y}] . g[{x, y}], {x, y} \[Element] Triangle[{{0, 0}, {1, 0}, {1, 1}}]]
  
  --------------------
  
  Out[] := 
    31.0 / 6.0
    25.0 / 6.0
*/
std::function<vec2(vec2,int)> f2([](vec2 x, int p){ return vec2{1 + 2 * x[1], 3 - 2 * x[0]}; });
std::function<vec2(vec2)> g2([](vec2 x){ return vec2{2.0, 3.0}; });
std::function<double(int)> answer2([](int p){ return 31.0 / 6.0; });
std::function<double(int)> answer3([](int p){ return 25.0 / 6.0; });
TEST(IntegrateTest, SourceHcurlOneTri) { integrate_source_test("one_tri.json", f2, g2, answer2, 3.0e-14); }
TEST(IntegrateTest, SourceHcurlOneTriRotated) { integrate_source_test("one_tri_rotated.json", f2, g2, answer2, 3.0e-14); }
TEST(IntegrateTest, SourceHcurlOneTriSheared) { integrate_source_test("one_tri_sheared.json", f2, g2, answer3, 3.0e-14); }

// ----------------------------------------------------------------------------

/*
  In[] := 
    f[{x_, y_, z_}] := {1, 2, 3};
    g[{x_, y_, z_}] := {2, 2, 4};
    Integrate[f[{x, y, z}] . g[{x, y, z}], {x, y, z} \[Element] Tetrahedron[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]]
    Integrate[f[{x, y, z}] . g[{x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {1, 1, 1}]]
    Integrate[f[{x, y, z}] . g[{x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {2, 1, 1}]]
 
  --------------------
 
  Out[] := 
    3
    18
    36
*/
std::function< vec3(vec3, int) > f3([](vec3 x, int p) { return vec3{1, 2, 3}; });
std::function< vec3(vec3) > g3([](vec3 x) { return vec3{2, 2, 4}; });
std::function<double(int)> answer4([](int p){ return 3.0; });
std::function<double(int)> answer5([](int p){ return 18.0; });
std::function<double(int)> answer6([](int p){ return 36.0; });

TEST(IntegrateTest, SourceHcurlOneTet) { integrate_source_test("one_tet.json", f3, g3, answer4, 1.0e-13); }
TEST(IntegrateTest, SourceHcurlOneHex) { integrate_source_test("one_hex.json", f3, g3, answer5, 1.0e-13); }

TEST(IntegrateTest, SourceHcurlTets) { integrate_source_test("patch_test_tets.json", f3, g3, answer5, 1.0e-13); }
TEST(IntegrateTest, SourceHcurlHexes) { integrate_source_test("patch_test_hexes.json", f3, g3, answer5, 1.0e-13); }
TEST(IntegrateTest, SourceHcurlTetsAndHexes) { integrate_source_test("patch_test_tets_and_hexes.json", f3, g3, answer6, 1.0e-13); }
