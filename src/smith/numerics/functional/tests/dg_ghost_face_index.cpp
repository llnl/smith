// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include <gtest/gtest.h>

#include "smith/infrastructure/application_manager.hpp"
#include "smith/smith_config.hpp"
#include "smith/mesh_utils/mesh_utils_base.hpp"
#include "smith/numerics/stdfunction_operator.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"

using namespace smith;
using namespace smith::profiling;

// This test initializes a DG field with nodal coordinates of the dofs, so that the
// discontinuous dof pairs across the interior faces have the value. For example
// on the interior face with the following dofs
//        {1, 2} | {5, 6}
//               |
//               |
//        {3, 4} | {7, 8}
// we have {1, 2} = {5, 6} and {3, 4} = {7, 8}.
// It then integrates the jump of dof values over all interior faces.
// If the ghost dof data is constructed correctly to align with locally owned data,
// then every entry in the residual vector should equal to zero. This is tested
// by the L2 norm of the residual equal to zero.
template <int dim, int p>
void L2_index_test(std::string meshfile)
{
  using test_space = L2<p, dim>;
  using trial_space = L2<p, dim>;

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 1);

  auto [test_fespace, test_fec] = smith::generateParFiniteElementSpace<test_space>(mesh.get());
  auto [trial_fespace, trial_fec] = smith::generateParFiniteElementSpace<trial_space>(mesh.get());

  // Initialize the ParGridFunction by dof coordinates
  mfem::ParGridFunction U_gf(trial_fespace.get());
  mfem::VectorFunctionCoefficient vcoef(dim, [](const mfem::Vector& X, mfem::Vector& F) {
    int d = X.Size();
    F.SetSize(d);
    for (int i = 0; i < d; ++i) {
      F(i) = X(i);
    }
  });
  U_gf.ProjectCoefficient(vcoef);

  mfem::Vector U(trial_fespace->TrueVSize());
  U_gf.GetTrueDofs(U);

  // Construct the new functional object using the specified test and trial spaces
  Functional<test_space(trial_space)> residual(test_fespace.get(), {trial_fespace.get()});

  Domain interior_faces = InteriorFaces(*mesh);

  // Define the integral of jumps over all interior faces
  residual.AddInteriorFaceIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [=](double /*t*/, auto X, auto velocity) {
        // compute the surface normal
        auto dX_dxi = get<DERIVATIVE>(X);
        auto n = normalize(cross(dX_dxi));

        // extract the velocity values from each side of the interface
        // note: the orientation convention is such that the normal
        //       computed as above will point from from side 1->2
        auto [u_1, u_2] = velocity;
        SLIC_INFO(axom::fmt::format("One size = {}, The other side = {}, Jump = {}", axom::fmt::streamed(u_1),
                                    axom::fmt::streamed(u_2), axom::fmt::streamed(u_1 - u_2)));

        auto a = dot(u_2 - u_1, n);

        auto f_1 = u_1 * a;
        auto f_2 = u_2 * a;
        return smith::tuple{f_1, f_2};
      },
      interior_faces);

  double t = 0.0;

  auto value = residual(t, U);
  EXPECT_NEAR(0., value.Norml2(), 1.e-12);
}

TEST(index, L2_test_tris_and_quads_linear)
{
  L2_index_test<2, 1>(SMITH_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh");
}
TEST(index, L2_test_tris_and_quads_quadratic)
{
  L2_index_test<2, 2>(SMITH_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh");
}

TEST(index, L2_test_tets_linear) { L2_index_test<3, 1>(SMITH_REPO_DIR "/data/meshes/patch3D_tets.mesh"); }
TEST(index, L2_test_tets_quadratic) { L2_index_test<3, 2>(SMITH_REPO_DIR "/data/meshes/patch3D_tets.mesh"); }

TEST(index, L2_test_hexes_linear) { L2_index_test<3, 1>(SMITH_REPO_DIR "/data/meshes/patch3D_hexes.mesh"); }
TEST(index, L2_test_hexes_quadratic) { L2_index_test<3, 2>(SMITH_REPO_DIR "/data/meshes/patch3D_hexes.mesh"); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
