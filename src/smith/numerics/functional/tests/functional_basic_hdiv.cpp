// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "gtest/gtest.h"

#include "axom/slic/core/SimpleLogger.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/smith_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/stdfunction_operator.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/tests/check_gradient.hpp"

using namespace smith;

template <int p, int dim>
void hdiv_test_impl(std::unique_ptr<mfem::ParMesh>& mesh)
{
  using test_space = Hdiv<p>;
  using trial_space = Hdiv<p>;

  auto [fespace, fec] = smith::generateParFiniteElementSpace<Hdiv<p>>(mesh.get());

  mfem::Vector U(fespace->TrueVSize());
  U.Randomize(7);

  Functional<test_space(trial_space)> residual(fespace.get(), {fespace.get()});

  auto d00 = make_tensor<dim, dim>([](int i, int j) { return i + j * j - 1; });
  auto d01 = make_tensor<dim>([](int i) { return i * i + 3; });
  auto d10 = make_tensor<dim>([](int i) { return 3 * i - 2; });
  auto d11 = 1.0;

  Domain whole_domain = EntireDomain(*mesh);
  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto /*x*/, auto sigma) {
        auto [val, div_val] = sigma;
        auto source = dot(d00, val) + d01 * div_val;
        auto flux = dot(d10, val) + d11 * div_val;
        return smith::tuple{source, flux};
      },
      whole_domain);

  double t = 0.0;
  check_gradient(residual, t, U);
}

template <int p>
void hdiv_test(std::string meshfile)
{
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SMITH_REPO_DIR + meshfile), 1);

  if (mesh->Dimension() == 2) {
    hdiv_test_impl<p, 2>(mesh);
  }

  if (mesh->Dimension() == 3) {
    hdiv_test_impl<p, 3>(mesh);
  }
}

// 2D tests
TEST(basic, hdiv_quads) { hdiv_test<1>("/data/meshes/patch2D_quads.mesh"); }
TEST(basic, hdiv_tris) { hdiv_test<1>("/data/meshes/patch2D_tris.mesh"); }
TEST(basic, hdiv_tris_and_quads) { hdiv_test<1>("/data/meshes/patch2D_tris_and_quads.mesh"); }

// 3D tests
TEST(basic, hdiv_hexes) { hdiv_test<1>("/data/meshes/patch3D_hexes.mesh"); }
TEST(basic, hdiv_tets) { hdiv_test<1>("/data/meshes/patch3D_tets.mesh"); }
TEST(basic, hdiv_tets_and_hexes) { hdiv_test<1>("/data/meshes/patch3D_tets_and_hexes.mesh"); }

// Higher order (quads/hexes have full p support; simplices only p=1 for now)
TEST(basic, hdiv_quads_p2) { hdiv_test<2>("/data/meshes/patch2D_quads.mesh"); }
TEST(basic, hdiv_hexes_p2) { hdiv_test<2>("/data/meshes/patch3D_hexes.mesh"); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
