#include <gtest/gtest.h>

#include <iostream>
#include <functional>

#include "refactor/mesh.hpp"
#include "refactor/field.hpp"
#include "refactor/piola_transformations.hpp"

#include "refactor/domain.hpp"

#include "misc/for_constexpr.hpp"
#include "forall.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"

using namespace refactor;

constexpr int dimension(double) { return 1; }

template < int dim >
constexpr int dimension(vec < dim >) { return dim; }

template < typename vec_t >
void evaluation_test(std::string filename, 
                     std::function<double(vec_t)> f,
                     std::function<vec_t(vec_t)> grad_f, 
                     int p, 
                     double tolerance) {

  constexpr int dim = dimension(vec_t{});

  using mat_t = mat<dim, dim>;

  auto mesh = Mesh::load(SERAC_MESH_DIR + filename);

  Field u = create_field(mesh, Family::H1, p);
  nd::array<double, 2> nodes = nodes_for(u, mesh);

  u = forall(f, nodes);

  for (int q = 1; q <= 4; q++) {

    Domain domain(mesh, MeshQuadratureRule(q));

    auto x_q = evaluate(mesh.X, domain);
    auto dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

    auto u_q = evaluate(u, domain);
    auto du_dxi_q = evaluate(grad(u), isoparametric(domain));
    auto du_dX_q_1 = evaluate(grad(u), domain);
    auto du_dX_q_2 = forall(contravariant_piola<vec_t, mat_t>, du_dxi_q, dX_dxi_q);

    // evaluate f, df_dx directly at each quadrature point
    auto f_q = forall(f, x_q);
    auto df_dX_q = forall(grad_f, x_q);

    EXPECT_LT(relative_error(flatten(u_q), flatten(f_q)), tolerance);
    EXPECT_LT(relative_error(flatten(du_dX_q_1), flatten(df_dX_q)), tolerance);
    EXPECT_LT(relative_error(du_dX_q_2, df_dX_q), tolerance);

  }

}

#define GENERATE_TEST_2D(NAME, mesh, p, tolerance)                                        \
TEST(InterpolationTest2D, NAME) {                                                         \
  evaluation_test(                                                                        \
    mesh,                                                                                 \
    std::function< double(vec2) >([](vec2 x) { return 3 * pow(x[0], p) + 2 * x[1] - 1; }),\
    std::function< vec2(vec2) >([](vec2 x) { return vec2{3 * p * pow(x[0], p - 1), 2}; }),\
    p,                                                                                    \
    tolerance                                                                             \
  );                                                                                      \
}

GENERATE_TEST_2D(AffineTestQuadP1, "one_quad_pure_shear.json", 1, 1.0e-15);

GENERATE_TEST_2D(PatchTestQuadP1, "patch_test_quads.json", 1, 1.0e-15);
GENERATE_TEST_2D(PatchTestQuadP2, "patch_test_quads.json", 2, 1.0e-14);
GENERATE_TEST_2D(PatchTestQuadP3, "patch_test_quads.json", 3, 1.0e-14);

GENERATE_TEST_2D(PatchTestTriP1, "patch_test_tris.json", 1, 1.0e-15);
GENERATE_TEST_2D(PatchTestTriP2, "patch_test_tris.json", 2, 1.0e-14);
GENERATE_TEST_2D(PatchTestTriP3, "patch_test_tris.json", 3, 1.5e-14);

GENERATE_TEST_2D(PatchTestTriAndQuadP1, "patch_test_tris_and_quads.json", 1, 1.0e-15);
GENERATE_TEST_2D(PatchTestTriAndQuadP2, "patch_test_tris_and_quads.json", 2, 1.0e-14);
GENERATE_TEST_2D(PatchTestTriAndQuadP3, "patch_test_tris_and_quads.json", 3, 2.0e-14);

GENERATE_TEST_2D(WrenchTestP1, "wrench.json", 1, 1.0e-12);
GENERATE_TEST_2D(WrenchTestP2, "wrench.json", 2, 1.0e-12);
GENERATE_TEST_2D(WrenchTestP3, "wrench.json", 3, 1.0e-12);

/////////////////////////////////////////////////////////////////////////////////////////////////

#define GENERATE_TEST_3D(NAME, mesh, p, tolerance)                                              \
TEST(InterpolationTest3D, NAME) {                                                               \
  evaluation_test(                                                                              \
    mesh,                                                                                       \
    std::function< double(vec3) >([](vec3 x) { return 3 * pow(x[0], p) + 2 * x[1] + x[2]; }),   \
    std::function< vec3(vec3) >([](vec3 x) { return vec3{3 * p * pow(x[0], p - 1), 2, 1}; }),   \
    p,                                                                                          \
    tolerance                                                                                   \
  );                                                                                            \
}

GENERATE_TEST_3D(PatchTestHexP1, "patch_test_hexes.json", 1, 2.0e-15);
GENERATE_TEST_3D(PatchTestHexP2, "patch_test_hexes.json", 2, 2.0e-14);
GENERATE_TEST_3D(PatchTestHexP3, "patch_test_hexes.json", 3, 4.5e-13);

GENERATE_TEST_3D(PatchTestTetP1, "patch_test_tets.json", 1, 5.0e-16);
GENERATE_TEST_3D(PatchTestTetP2, "patch_test_tets.json", 2, 5.0e-15);
GENERATE_TEST_3D(PatchTestTetP3, "patch_test_tets.json", 3, 2.0e-14);

GENERATE_TEST_3D(PatchTestTetAndHexP1, "patch_test_tets_and_hexes.json", 1, 1.5e-15);
GENERATE_TEST_3D(PatchTestTetAndHexP2, "patch_test_tets_and_hexes.json", 2, 1.5e-14);
GENERATE_TEST_3D(PatchTestTetAndHexP3, "patch_test_tets_and_hexes.json", 3, 3.5e-13);
