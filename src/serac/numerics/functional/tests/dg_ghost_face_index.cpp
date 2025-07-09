// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/serac_config.hpp"
#include "serac/mesh_utils/mesh_utils_base.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"

using namespace serac;
using namespace serac::profiling;

template <int dim, int p>
void L2_index_test(std::string meshfile)
{
  using test_space = L2<p, dim>;
  using trial_space = L2<p, dim>;

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 1);

  auto [test_fespace, test_fec] = serac::generateParFiniteElementSpace<test_space>(mesh.get());
  auto [trial_fespace, trial_fec] = serac::generateParFiniteElementSpace<trial_space>(mesh.get());

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
        std::cout << "One side = " << u_1 << ", The other side = " << u_2 << ", Jump = " << u_1 - u_2 << std::endl;

        auto a = dot(u_2 - u_1, n);

        auto f_1 = u_1 * a;
        auto f_2 = u_2 * a;
        return serac::tuple{f_1, f_2};
      },
      interior_faces);

  double t = 0.0;

  auto value = residual(t, U);
  EXPECT_NEAR(0., value.Norml2(), 1.e-12);
}

TEST(index, L2_test_tris_and_quads_linear)
{
  L2_index_test<2, 1>(SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh");
}
TEST(index, L2_test_tris_and_quads_quadratic)
{
  L2_index_test<2, 2>(SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh");
}

TEST(index, L2_test_tets_linear) { L2_index_test<3, 1>(SERAC_REPO_DIR "/data/meshes/patch3D_tets.mesh"); }
TEST(index, L2_test_tets_quadratic) { L2_index_test<3, 2>(SERAC_REPO_DIR "/data/meshes/patch3D_tets.mesh"); }

TEST(index, L2_test_hexes_linear) { L2_index_test<3, 1>(SERAC_REPO_DIR "/data/meshes/patch3D_hexes.mesh"); }
TEST(index, L2_test_hexes_quadratic) { L2_index_test<3, 2>(SERAC_REPO_DIR "/data/meshes/patch3D_hexes.mesh"); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
