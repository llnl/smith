// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/shape_aware_functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/state/finite_element_state.hpp"

using namespace smith;

double t = 0.0;

struct IdentityFunctor {
  template <typename Arg1, typename Arg2>
  SMITH_HOST_DEVICE auto operator()(Arg1, Arg2) const
  {
    return 1.0;
  }
};

int num_procs, my_rank;

TEST(BoundaryIntegralQOI, AttrBug)
{
  constexpr int ORDER = 1;

  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(10, 10, mfem::Element::QUADRILATERAL, false, 1.0, 1.0);

  auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);

  pmesh->EnsureNodes();
  pmesh->ExchangeFaceNbrData();

  using shapeFES = smith::H1<ORDER, 2>;
  auto [shape_fes, shape_fec] = smith::generateParFiniteElementSpace<shapeFES>(pmesh.get());

  Domain whole_boundary = EntireBoundary(*pmesh);

  smith::ShapeAwareFunctional<shapeFES, double()> totalSurfArea(shape_fes.get(), {});
  totalSurfArea.AddBoundaryIntegral(smith::Dimension<2 - 1>{}, smith::DependsOn<>{}, IdentityFunctor{}, whole_boundary);
  smith::FiniteElementState shape(*shape_fes);
  double totalSurfaceArea = totalSurfArea(0.0, shape);

  EXPECT_NEAR(totalSurfaceArea, 4.0, 1.0e-14);

  smith::Domain attr1 = smith::Domain::ofBoundaryElements(*pmesh, smith::by_attr<2>(1));
  smith::ShapeAwareFunctional<shapeFES, double()> attr1SurfArea(shape_fes.get(), {});
  attr1SurfArea.AddBoundaryIntegral(smith::Dimension<2 - 1>{}, smith::DependsOn<>{}, IdentityFunctor{}, attr1);
  double attr1SurfaceArea = attr1SurfArea(0.0, shape);

  EXPECT_NEAR(attr1SurfaceArea, 1.0, 1.0e-14);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
