#include <gtest/gtest.h>

#include <iostream>
#include <functional>

#include "serac/numerics/refactor/forall.hpp"
#include "serac/numerics/refactor/evaluate.hpp"
#include "serac/numerics/refactor/tests/common.hpp"

using namespace refactor;

constexpr int dimension(double) { return 1; }

template < int dim >
constexpr int dimension(vec < dim >) { return dim; }

template < typename grad_t, typename jac_t >
auto contravariant_piola(const grad_t & du_dxi, const jac_t & dx_dxi) {
  return dot(du_dxi, inv(dx_dxi));
};

template < typename vec_t >
void evaluation_test(std::string filename, 
                     std::function<double(vec_t)> f,
                     std::function<vec_t(vec_t)> grad_f, 
                     int p, 
                     double tolerance) {

  constexpr int dim = dimension(vec_t{});
  constexpr int components = 1;

  using mat_t = mat<dim, dim>;

  mfem::ParMesh mesh = load_parmesh(SERAC_MESH_DIR + filename);
  Domain domain = EntireDomain(mesh);

  Field u(mesh, refactor::Family::H1, p, components);

  mfem::FunctionCoefficient mfem_f([&](const mfem::Vector & mfem_X) {
    vec_t X;
    for (int i = 0; i < dim; i++) {
      X[i] = mfem_X[i];
    }
    return f(X);
  });

  u.project(mfem_f);

  Field X = mesh_coordinates(mesh);

  //for (uint32_t q = 1; q <= 4; q++) {
  for (uint32_t q = 1; q <= 1; q++) {

    MeshQuadratureRule qrule(q);

    auto X_q = evaluate(X, domain, qrule);
    auto dX_dxi_q = evaluate(grad(X), domain, qrule);

    auto u_q = evaluate(u, domain, qrule);
    auto du_dxi_q = evaluate(grad(u), domain, qrule);
    auto du_dX_q_2 = forall(contravariant_piola<vec_t, mat_t>, du_dxi_q, dX_dxi_q);

    // evaluate f, df_dx directly at each quadrature point
    auto f_q = forall(f, X_q);
    auto df_dX_q = forall(grad_f, X_q);

    print(X_q);
    print(f_q);
    print(u_q);

    //return 3 * pow(x[0], p) + 2 * x[1] + x[2];

    EXPECT_LT(relative_error(flatten(u_q), flatten(f_q)), tolerance);
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

//GENERATE_TEST_2D(PatchTestTriP1, "patch2D_tris.mesh", 1, 1.0e-15);
//GENERATE_TEST_2D(PatchTestTriP2, "patch2D_tris.mesh", 2, 1.0e-14);
//GENERATE_TEST_2D(PatchTestTriP3, "patch2D_tris.mesh", 3, 1.5e-14);
//
//GENERATE_TEST_2D(PatchTestQuadP1, "patch2D_quads.mesh", 1, 1.0e-15);
//GENERATE_TEST_2D(PatchTestQuadP2, "patch2D_quads.mesh", 2, 1.0e-14);
//GENERATE_TEST_2D(PatchTestQuadP3, "patch2D_quads.mesh", 3, 1.0e-14);
//
//GENERATE_TEST_2D(PatchTestTriAndQuadP1, "patch2D_tris_and_quads.mesh", 1, 1.0e-15);
//GENERATE_TEST_2D(PatchTestTriAndQuadP2, "patch2D_tris_and_quads.mesh", 2, 1.0e-14);
//GENERATE_TEST_2D(PatchTestTriAndQuadP3, "patch2D_tris_and_quads.mesh", 3, 2.0e-14);

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

GENERATE_TEST_3D(OneTetP1, "onetet.mesh", 1, 5.0e-16);

//GENERATE_TEST_3D(PatchTestTetP1, "patch3D_tets.mesh", 1, 5.0e-16);
//GENERATE_TEST_3D(PatchTestTetP2, "patch3D_tets.mesh", 2, 5.0e-15);
//GENERATE_TEST_3D(PatchTestTetP3, "patch3D_tets.mesh", 3, 2.0e-14);
//
//GENERATE_TEST_3D(PatchTestHexP1, "patch3D_hexes.mesh", 1, 2.0e-15);
//GENERATE_TEST_3D(PatchTestHexP2, "patch3D_hexes.mesh", 2, 2.0e-14);
//GENERATE_TEST_3D(PatchTestHexP3, "patch3D_hexes.mesh", 3, 4.5e-13);
//
//GENERATE_TEST_3D(PatchTestTetAndHexP1, "patch3D_tets_and_hexes.mesh", 1, 1.5e-15);
//GENERATE_TEST_3D(PatchTestTetAndHexP2, "patch3D_tets_and_hexes.mesh", 2, 1.5e-14);
//GENERATE_TEST_3D(PatchTestTetAndHexP3, "patch3D_tets_and_hexes.mesh", 3, 3.5e-13);

int main(int argc, char* argv[]) {
  int num_procs, myid;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
