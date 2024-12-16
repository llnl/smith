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
                           std::function< double(vecd, int) > f,
                           std::function< double(vecd) > g,
                           std::function< double(int) > answer,
                           double tolerance) {

  using matd = decltype(outer(vecd{}, vecd{}));

  auto mesh = Mesh::load(SERAC_MESH_DIR + filename);

  std::cout << std::setprecision(15);

  // evaluate g at each quadrature point
  std::function< double(vecd, matd) > g_detJ = [g](vecd x, matd J) { 
    return g(x) * det(J); 
  };

  for (int p = 1; p < 4; p++) {

    std::function< double(vecd) > f_p = [f, p](vecd x) { return f(x,p); };

    Field u = create_field(mesh, Family::H1, p, 1);
    nd::array<double,2> nodes = nodes_for(u, mesh); 
    u = forall(f_p, nodes);

    BasisFunction phi(u);

    for (int q = p + 1; q < 5; q++) {

      Domain domain(mesh, MeshQuadratureRule(q));

      auto x_q = evaluate(mesh.X, domain);
      auto dx_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

      auto g_q = forall(g, x_q);
      auto g_detJ_q = forall(g_detJ, x_q, dx_dxi_q);

      Residual r1 = integrate(g_q * phi, domain);
      Residual r2 = integrate(g_detJ_q * phi, isoparametric(domain));

      SCOPED_TRACE("p = " + std::to_string(p) + ", q = " + std::to_string(q));
      EXPECT_NEAR(dot(r1, u), answer(p), tolerance);
      EXPECT_NEAR(dot(r2, u), answer(p), tolerance);

    }
  }

}

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[x_] := x^p + 3;
    g[x_] := x;
    Integrate[f[x]  g[x], {x, 0, 1}]
  
  --------------------
  
  Out[] := 
    (3.0/2.0) + 1.0 / (2.0 + p)
*/
std::function<double(double,int)> f0([](double x, int p){ return pow(x, p) + 3.0; });
std::function<double(double)> g0([](double x){ return x; });
std::function<double(int)> answer0([](int p){ return 1.5 + 1.0 / (2 + p); });

TEST(IntegrateTest, SourceH1Edge) { integrate_source_test("patch_test_edges.json", f0, g0, answer0, 3.0e-2); }

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[{x_, y_}] := x^p - y + 3;
    g[{x_, y_}] := y;
    Integrate[f[{x, y}] g[{x, y}], {x, y} \[Element] Rectangle[{0, 0}, {1, 1}]]
  
  --------------------
  
  Out[] := 
    (7.0/6.0) + 1.0 /(2.0 + 2.0 * p)
*/
std::function<double(vec2,int)> f1([](vec2 x, int p){ return pow(x[0], p) - x[1] + 3.0; });
std::function<double(vec2)> g1([](vec2 x){ return x[1]; });
std::function<double(int)> answer1([](int p){ return (7.0/6.0) + 1.0 /(2.0 + 2.0 * p); });

TEST(IntegrateTest, SourceH1Tris) { integrate_source_test("patch_test_tris.json", f1, g1, answer1, 3.0e-15); }
TEST(IntegrateTest, SourceH1Quads) { integrate_source_test("patch_test_quads.json", f1, g1, answer1, 1.0e-15); }
TEST(IntegrateTest, SourceH1Both) { integrate_source_test("patch_test_tris_and_quads.json", f1, g1, answer1, 1.0e-15); }

// ----------------------------------------------------------------------------

/*
  In[] := 
    f[{x_, y_, z_}] := x^p - y - z + 3;
    g[{x_, y_, z_}] := 1;
    Integrate[f[{x, y, z}] g[{x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {1, 1, 1}]]
 
  --------------------
 
  Out[] := 
    2.0 + 1.0 / (1.0 + p)
*/
std::function<double(vec3,int)> f2([](vec3 x, int p){ return pow(x[0], p) - x[1] - x[2] + 3.0; });
std::function<double(vec3)> g2([](vec3 x){ return 1.0; });
std::function<double(int)> answer2([](int p){ return 2.0 + 1.0 / (1.0 + p); });

TEST(IntegrateTest, SourceH1Tets) { integrate_source_test("patch_test_tets.json", f2, g2, answer2, 7.0e-15); }
TEST(IntegrateTest, SourceH1Hexes) { integrate_source_test("patch_test_hexes.json", f2, g2, answer2, 1.0e-15); }
TEST(IntegrateTest, SourceH1TetsFine) { integrate_source_test("unit_cube_of_tets_fine.json", f2, g2, answer2, 5.0e-14); }
TEST(IntegrateTest, SourceH1HexesFine) { integrate_source_test("unit_cube_of_hexes_fine.json", f2, g2, answer2, 2.0e-14); }
