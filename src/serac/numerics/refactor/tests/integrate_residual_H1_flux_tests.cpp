#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <functional>

#include "serac/numerics/refactor/forall.hpp"
#include "serac/numerics/refactor/evaluate.hpp"
#include "serac/numerics/refactor/tests/common.hpp"

using namespace serac;
using namespace refactor;

// ----------------------------------------------------------------------------

template < typename vec_t >
void flux_test(std::string filename,
               std::function< double(vec_t, int) > f,
               std::function< vec_t(vec_t) > g,
               std::function< double(int) > answer,
               double tolerance) {

  constexpr int components = 1;
  constexpr int dim = size(vec_t{});
  using mat_t = decltype(outer(vec_t{}, vec_t{}));

  mfem::ParMesh mesh = load_parmesh(SERAC_MESH_DIR + filename);
  Domain domain = EntireDomain(mesh);

  Field X = mesh_coordinates(mesh);

  // evaluate g at each quadrature point
  std::function< vec_t(vec_t, mat_t) > g_xi = [g](vec_t x, mat_t J) { 
    return dot(serac::inv(J), g(x)) * det(J); 
  };

  for (int p = 1; p < 4; p++) {

    std::function< double(vec_t) > f_p = [f, p](vec_t x) { return f(x, p); };

    Field u(mesh, refactor::Family::H1, p, components);

    mfem::FunctionCoefficient mfem_f([&](const mfem::Vector & mfem_X) {
      vec_t X;
      if constexpr (dim == 1) {
        X = mfem_X[0];
      } else {
        for (int i = 0; i < dim; i++) {
          X[i] = mfem_X[i];
        }
      }
      return f(X, p);
    });

    u.project(mfem_f);

    BasisFunction phi(u);

    for (int q = p + 1; q < 5; q++) {

      MeshQuadratureRule qrule(static_cast<uint32_t>(q));

      auto X_q = evaluate(X, domain, qrule);
      auto dX_dxi_q = evaluate(grad(X), domain, qrule);

      auto g_q = forall(g, X_q);
      auto g_xi_q = forall(g_xi, X_q, dX_dxi_q);

      Residual r = integrate(dot(g_q, grad(phi)), domain, qrule);

      SCOPED_TRACE("p = " + std::to_string(p) + ", q = " + std::to_string(q));
      EXPECT_NEAR(dot(r, u), answer(p), tolerance);

    }
  }

}

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[x_] := x^p + 3;
    gradf[x_] := p  x^(p-1);
    g[x_] := x;
    Integrate[gradf[x] g[x], {x, 0, 1}]
  
   --------------------
  
  Out[] := 
    1/2
*/

//std::function<double(double,int)> f3([](double x, int p){ return pow(x, p); });
//std::function<double(double)> g3([](double x){ return x; });
//std::function<double(int)> answer3([](int p){ return double(p) / double(1 + p); });
//
//TEST(IntegrateTest, FluxH1Edges) { flux_test("patch_test_edges.json", f3, g3, answer3, 3.0e-15); }

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[{x_, y_}] := x^p - y + 3;
    gradf[{x_, y_}] := {p x^(-1 + p), -1};
    g[{x_, y_}] := {y, -x};
    Integrate[gradf[{x, y}] . g[{x, y}], {x, y} \[Element] Rectangle[{0, 0}, {1, 1}]]
  
   --------------------
  
  Out[] := 
    1
*/

std::function<double(vec2,int)> f4([](vec2 x, int p){ return pow(x[0], p) - x[1] + 3.0; });
std::function<vec2(vec2)> g4([](vec2 x){ return vec2{x[1], -x[0]}; });
std::function<double(int)> answer4([](int){ return 1.0; });

TEST(IntegrateTest, FluxH1Tris) { flux_test("patch_test_tris.json", f4, g4, answer4, 5.0e-15); }
TEST(IntegrateTest, FluxH1Quads) { flux_test("patch_test_quads.json", f4, g4, answer4, 5.0e-15); }
TEST(IntegrateTest, FluxH1Both) { flux_test("patch_test_tris_and_quads.json", f4, g4, answer4, 5.0e-15); }
TEST(IntegrateTest, FluxH1TrisFine) { flux_test("unit_square_of_tris_fine.json", f4, g4, answer4, 8.0e-14); }
TEST(IntegrateTest, FluxH1QuadsFine) { flux_test("unit_square_of_quads_fine.json", f4, g4, answer4, 8.0e-14); }

// ----------------------------------------------------------------------------

/* 
  In[] := 
    f[{x_, y_, z_}] := x^p - y - 2 z + 3;
    gradf[{x_, y_, z_}] := {p x^(-1 + p), -1, -2};
    g[{x_, y_, z_}] := {z, x, -y};
    Integrate[gradf[{x, y, z}] . g[{x, y, z}], {x, y, z} \[Element] Cuboid[{0, 0, 0}, {1, 1, 1}]]
  
  --------------------
  
  Out[] := 
    1
*/
std::function<double(vec3,int)> f5([](vec3 x, int p){ return pow(x[0], p) - x[1] - 2 * x[2] + 3.0; });
std::function<vec3(vec3)> g5([](vec3 x){ return vec3{x[2], x[0], -x[1]}; });
std::function<double(int)> answer5([](int){ return 1.0; });

TEST(IntegrateTest, FluxH1Tets) { flux_test("patch_test_tets.json", f5, g5, answer5, 5.0e-15); }
TEST(IntegrateTest, FluxH1Hexes) { flux_test("patch_test_hexes.json", f5, g5, answer5, 5.0e-15); }
TEST(IntegrateTest, FluxH1TetsFine) { flux_test("unit_cube_of_tets_fine.json", f5, g5, answer5, 5.0e-14); }
TEST(IntegrateTest, FluxH1HexesFine) { flux_test("unit_cube_of_hexes_fine.json", f5, g5, answer5, 5.0e-15); }
