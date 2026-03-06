// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <iostream>
#include <complex>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/shape_aware_functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/dual.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/geometry.hpp"
#include "smith/numerics/functional/tuple.hpp"

using namespace smith;

double t = 0.0;

TEST(QoI, TetrahedronQuality)
{
  static constexpr int dim{3};

  double displacement_to_regular_tetrahedron[4][3] = {
      {0., 0., 0.}, {0.122462, 0., 0.}, {0.561231, -0.0279194, 0.}, {0.561231, 0.324027, -0.0835136}};

  tensor<double, 3, 3> regular_tet_correction = {
      {{1.00000, -0.577350, -0.408248}, {0, 1.15470, -0.408248}, {0, 0, 1.22474}}};

  auto mu = [](auto J) {
    using std::pow;
    return tr(dot(J, J)) / (3 * pow(smith::det(J), 2. / 3.)) - 1.0;
  };

  using shape_space = H1<1, dim>;

  std::string meshfile3D = SMITH_REPO_DIR "/data/meshes/onetet.mesh";
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), 0, 0);

  auto [fes, fec] = generateParFiniteElementSpace<shape_space>(mesh.get());

  Domain whole_domain = EntireDomain(*mesh);

  // Define the shape-aware QOI objects
  smith::ShapeAwareFunctional<shape_space, double()> saf_qoi(fes.get(), {});

  // Note that the integral does not have a shape parameter field. The transformations are handled under the hood
  // so the user only sees the modified x = X + p input arguments
  saf_qoi.AddDomainIntegral(
      smith::Dimension<3>{}, smith::DependsOn<>{},
      [=](double /*t*/, auto position) {
        auto [x, dx_dxi] = position;
        return mu(dot(regular_tet_correction, dx_dxi));
      },
      whole_domain);

  smith::Functional<double(shape_space)> qoi({fes.get()});

  qoi.AddDomainIntegral(
      smith::Dimension<3>{}, smith::DependsOn<0>{},
      [=](double /*t*/, auto position, auto displacement) {
        auto [X, dX_dxi] = position;  // <--- the position derivative term is w.r.t. xi, not X!
        auto [u, du_dX] = displacement;

        // x := X + u,
        // so, dx/dxi = dX/dxi + du/dxi
        //            = dX/dxi + du/dX * dX/dxi
        //            = (I + du/dX) * dX/dxi
        // auto dx_dxi = dot(I + du_dX, dX_dxi);
        auto dx_dxi = dX_dxi + dot(du_dX, dX_dxi);
        return mu(dot(regular_tet_correction, dx_dxi));
      },
      whole_domain);

  std::unique_ptr<mfem::HypreParVector> u(fes->NewTrueDofVector());
  *u = 0.0;
  std::cout << "(ShapeAwareFunctional) mu(J) for right tetrahedron: " << saf_qoi(t, *u) << std::endl;
  std::cout << "(          Functional) mu(J) for right tetrahedron: " << qoi(t, *u) << std::endl;

  // apply a displacement to make the domain into a regular tetrahedron
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      (*u)[i + j * 4] = displacement_to_regular_tetrahedron[i][j];
    }
  }

  std::cout << "(ShapeAwareFunctional) mu(J) for regular tetrahedron: " << saf_qoi(t, *u) << std::endl;
  std::cout << "(          Functional) mu(J) for regular tetrahedron: " << qoi(t, *u) << std::endl;
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
