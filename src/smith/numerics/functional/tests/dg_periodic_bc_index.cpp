
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

template <int dim, int p>
void L2_periodic_index_test(mfem::Element::Type element_type)
{
  using test_space = L2<p, dim>;
  using trial_space = L2<p, dim>;

  auto initial_mesh = mfem::Mesh(mfem::Mesh::MakeCartesian3D(4, 4, 4, element_type, 1.0, 1.0, 1.0));

  mfem::Vector x_translation({1.0, 0.0, 0.0});
  mfem::Vector y_translation({0.0, 1.0, 0.0});
  mfem::Vector z_translation({0.0, 0.0, 1.0});
  std::vector<mfem::Vector> translations = {x_translation, y_translation, z_translation};
  double tol = 1e-6;

  std::vector<int> periodicMap = initial_mesh.CreatePeriodicVertexMapping(translations, tol);

  auto mesh = mesh::refineAndDistribute(mfem::Mesh::MakePeriodic(initial_mesh, periodicMap));

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

  Domain periodic_faces = Domain::ofInteriorFaces(*mesh, by_attr<dim>({1, 2, 3, 4, 5, 6}));

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
        SLIC_INFO(axom::fmt::format("One side = {}, The other side = {}, Jump = {}", axom::fmt::streamed(u_1),
                                    axom::fmt::streamed(u_2), axom::fmt::streamed(u_1 - u_2)));

        auto a = dot(u_1 - u_2, n) - 1.0;

        auto f_1 = u_1 * a;
        auto f_2 = u_2 * a;
        return smith::tuple{f_1, f_2};
      },
      periodic_faces);

  double t = 0.0;

  auto value = residual(t, U);
  EXPECT_NEAR(0., value.Norml2(), 1.e-12);
}

TEST(periodic_index, L2_test_tets_linear)
{
  L2_periodic_index_test<3, 1>(mfem::Element::Type::TETRAHEDRON);
}

TEST(periodic_index, L2_test_hex_linear)
{
  L2_periodic_index_test<3, 1>(mfem::Element::Type::HEXAHEDRON);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
