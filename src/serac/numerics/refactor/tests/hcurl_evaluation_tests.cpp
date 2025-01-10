#include <gtest/gtest.h>

#include <iostream>
#include <functional>

#include "refactor/mesh.hpp"
#include "refactor/field.hpp"
#include "refactor/piola_transformations.hpp"

#include "refactor/domain.hpp"

#include "misc/for_constexpr.hpp"
#include "forall.hpp"

#include "serac/numerics/functional/tensor.hpp"


using namespace serac;
using namespace refactor;

template < typename vec_t, typename curl_t>
void evaluation_test(std::string filename, 
                     std::function<vec_t(vec_t)> f,
                     std::function<curl_t(vec_t)> curl_f, 
                     int p, 
                     double tolerance) {

  constexpr int dim = dimension(vec_t{});

  using mat_t = mat<dim, dim>;

  auto mesh = Mesh::load(SERAC_MESH_DIR + filename);

  Field u = create_field(mesh, Family::HCURL, p);
  nd::array<double, 2> nodes = nodes_for(u, mesh);
  nd::array<double, 2> directions = directions_for(u, mesh);

  u = forall(std::function< double(vec_t, vec_t)> ([=](vec_t x, vec_t n){ 
    return dot(f(x), n); 
  }), nodes, directions);

  for (int q = 1; q <= 1; q++) {

    Domain domain(mesh, MeshQuadratureRule(q));

    auto x_q = evaluate(mesh.X, domain);
    auto dx_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

    auto u_q_1 = evaluate(u, domain);
    auto u_xi_q = evaluate(u, isoparametric(domain));
    auto curl_u_q_1 = evaluate(curl(u), domain);
    auto curl_u_xi_q = evaluate(curl(u), isoparametric(domain));
    auto u_q_2 = forall(contravariant_piola<vec_t, mat_t>, u_xi_q, dx_dxi_q);
    auto curl_u_q_2 = forall(covariant_piola<curl_t, mat_t>, curl_u_xi_q, dx_dxi_q);

    // evaluate f, df_dx directly at each quadrature point
    auto f_q = forall(f, x_q);
    auto curl_f_q = forall(curl_f, x_q);

    EXPECT_LT(relative_error(flatten(u_q_1), flatten(f_q)), tolerance);
    EXPECT_LT(relative_error(u_q_2, f_q), tolerance);
    EXPECT_LT(relative_error(flatten(curl_u_q_1), flatten(curl_f_q)), tolerance);
    EXPECT_LT(relative_error(curl_u_q_2, curl_f_q), tolerance);

  }

}

#if 1
#define GENERATE_TEST_2D(NAME, mesh, p, tolerance)                                        \
TEST(InterpolationTest2D, NAME) {                                                         \
  evaluation_test(                                                                        \
    mesh,                                                                                 \
    std::function< vec2(vec2) >([](vec2 x) { return vec2{x[1], -x[0]}; }),                \
    std::function< double(vec2) >([](vec2 x) { return -2; }),                             \
    p,                                                                                    \
    tolerance                                                                             \
  );                                                                                      \
}

GENERATE_TEST_2D(AffineTestQuadP1, "one_quad_pure_shear.json", 1, 1.0e-15);

//GENERATE_TEST_2D(PatchTestQuadP1, "patch_test_quads.json", 1, 1.0e-15);
GENERATE_TEST_2D(PatchTestQuadP2, "patch_test_quads.json", 2, 1.0e-14);
GENERATE_TEST_2D(PatchTestQuadP3, "patch_test_quads.json", 3, 1.0e-14);

GENERATE_TEST_2D(HcurlPatchTestTriP1, "patch_test_tris.json", 1, 1.0e-15);
GENERATE_TEST_2D(HcurlPatchTestTriP2, "patch_test_tris.json", 2, 1.0e-14);
GENERATE_TEST_2D(HcurlPatchTestTriP3, "patch_test_tris.json", 3, 1.5e-14);

//GENERATE_TEST_2D(PatchTestTriAndQuadP1, "patch_test_tris_and_quads.json", 1, 1.0e-15);
GENERATE_TEST_2D(PatchTestTriAndQuadP2, "patch_test_tris_and_quads.json", 2, 1.0e-14);
GENERATE_TEST_2D(PatchTestTriAndQuadP3, "patch_test_tris_and_quads.json", 3, 2.0e-14);

GENERATE_TEST_2D(WrenchTestP1, "wrench.json", 1, 1.0e-12);
GENERATE_TEST_2D(WrenchTestP2, "wrench.json", 2, 1.0e-12);
GENERATE_TEST_2D(WrenchTestP3, "wrench.json", 3, 1.0e-12);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

#define GENERATE_TEST_3D(NAME, mesh, p, tolerance)                                              \
TEST(InterpolationTest3D, NAME) {                                                               \
  evaluation_test(                                                                              \
    mesh,                                                                                       \
    std::function< vec3(vec3) >([](vec3 x) { return vec3{1 + 2*x[1], 3 - 2*x[0], 4}; }),        \
    std::function< vec3(vec3) >([](vec3 x) { return vec3{0, 0, -4}; }),                         \
    p,                                                                                          \
    tolerance                                                                                   \
  );                                                                                            \
}

GENERATE_TEST_3D(PatchTestHexesP1, "one_hex.json", 1, 2.0e-15);

//GENERATE_TEST_3D(PatchTestHexP1, "patch_test_hexes.json", 1, 2.0e-15);
GENERATE_TEST_3D(PatchTestHexP2, "patch_test_hexes.json", 2, 2.0e-14);
GENERATE_TEST_3D(PatchTestHexP3, "patch_test_hexes.json", 3, 4.5e-13);

GENERATE_TEST_3D(PatchTestTetP1, "patch_test_tets.json", 1, 3.5e-16);
GENERATE_TEST_3D(PatchTestTetP2, "patch_test_tets.json", 2, 5.0e-15);
GENERATE_TEST_3D(PatchTestTetP3, "patch_test_tets.json", 3, 2.0e-14);

//GENERATE_TEST_3D(PatchTestTetAndHexP1, "patch_test_tets_and_hexes.json", 1, 1.5e-15);
GENERATE_TEST_3D(PatchTestTetAndHexP2, "patch_test_tets_and_hexes.json", 2, 1.5e-14);
GENERATE_TEST_3D(PatchTestTetAndHexP3, "patch_test_tets_and_hexes.json", 3, 3.5e-13);
