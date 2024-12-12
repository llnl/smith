#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <functional>

#include "refactor/mesh.hpp"
#include "refactor/domain.hpp"

#include <gtest/gtest.h>

#include "forall.hpp"
#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray_conversions.hpp"

using namespace refactor;

// ----------------------------------------------------------------------------

template < typename vecd >
void flux_test(std::string filename,
               std::function< double(vecd, int) > f,
               std::function< vecd(vecd) > g,
               std::function< double(int) > answer,
               double tolerance) {

  constexpr int dim = dimension(vecd{});

  double scale_factor = (dim == 2) ? 5.0 : 14.0;

  using matd = decltype(outer(vecd{}, vecd{}));

  auto mesh = Mesh::load(SERAC_MESH_DIR + filename);

  // evaluate g at each quadrature point
  std::function< matd(vecd, matd) > qf = [g](vecd x, matd J) { 
    matd output{};
    for (int i = 0; i < dim; i++) {
      output[i] = dot(inv(J), g(x)) * det(J) * (i + 1); 
    }
    return output;
  };

  for (int p = 1; p < 4; p++) {

    std::function< vecd(vecd) > f_p = [f, p](vecd x) { 
      vecd output;
      for (int i = 0; i < dim; i++) {
        output[i] = f(x, p) * (i + 1);  
      }
      return output;
    };

    Field u = create_field(mesh, Family::H1, p, dim);
    nd::array<double,2> nodes = nodes_for(u, mesh); 
    u = forall(f_p, nodes);

    BasisFunction phi(u);

    for (int q = p + 1; q < 5; q++) {

      Domain domain(mesh, MeshQuadratureRule(q));

      auto x_q = evaluate(mesh.X, domain);
      auto dx_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

      auto g_q = forall(qf, x_q, dx_dxi_q);

      Residual r = integrate(dot(g_q, grad(phi)), isoparametric(domain));

      SCOPED_TRACE("p = " + std::to_string(p) + ", q = " + std::to_string(q));
      EXPECT_NEAR(dot(r, u), scale_factor * answer(p), tolerance);

    }
  }

}

// ----------------------------------------------------------------------------

// note: edge element tests omitted, since they are 1D elements so this test would
//       just duplicate what is already in integrate_H1_flux_tests.cpp

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

TEST(IntegrateTest, FluxH1Tris) { flux_test("patch_test_tris.json", f4, g4, answer4, 5.0e-14); }
TEST(IntegrateTest, FluxH1Quads) { flux_test("patch_test_quads.json", f4, g4, answer4, 5.0e-14); }
TEST(IntegrateTest, FluxH1Both) { flux_test("patch_test_tris_and_quads.json", f4, g4, answer4, 5.0e-14); }
TEST(IntegrateTest, FluxH1TrisFine) { flux_test("unit_square_of_tris_fine.json", f4, g4, answer4, 8.0e-13); }
TEST(IntegrateTest, FluxH1QuadsFine) { flux_test("unit_square_of_quads_fine.json", f4, g4, answer4, 8.0e-13); }

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

TEST(IntegrateTest, FluxH1Tets) { flux_test("patch_test_tets.json", f5, g5, answer5, 1.0e-13); }
TEST(IntegrateTest, FluxH1Hexes) { flux_test("patch_test_hexes.json", f5, g5, answer5, 1.0e-13); }
TEST(IntegrateTest, FluxH1TetsFine) { flux_test("unit_cube_of_tets_fine.json", f5, g5, answer5, 1.0e-13); }
TEST(IntegrateTest, FluxH1HexesFine) { flux_test("unit_cube_of_hexes_fine.json", f5, g5, answer5, 1.0e-13); }
