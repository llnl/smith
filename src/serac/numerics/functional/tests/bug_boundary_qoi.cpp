// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/physics/state/finite_element_state.hpp"

using namespace serac;
using namespace serac::profiling;

double t = 0.0;

struct IdentityFunctor {
  template <typename Arg1, typename Arg2>
  SERAC_HOST_DEVICE auto operator()(Arg1, Arg2) const
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

  using shapeFES = serac::H1<ORDER, 2>;
  auto [shape_fes, shape_fec] = serac::generateParFiniteElementSpace<shapeFES>(pmesh.get());

  Domain whole_boundary = EntireBoundary(*pmesh);

  serac::ShapeAwareFunctional<shapeFES, double()> totalSurfArea(shape_fes.get(), {});
  totalSurfArea.AddBoundaryIntegral(serac::Dimension<2 - 1>{}, serac::DependsOn<>{}, IdentityFunctor{}, whole_boundary);
  serac::FiniteElementState shape(*shape_fes);
  double totalSurfaceArea = totalSurfArea(0.0, shape);

  EXPECT_NEAR(totalSurfaceArea, 4.0, 1.0e-14);

  serac::Domain attr1 = serac::Domain::ofBoundaryElements(*pmesh, serac::by_attr<2>(1));
  serac::ShapeAwareFunctional<shapeFES, double()> attr1SurfArea(shape_fes.get(), {});
  attr1SurfArea.AddBoundaryIntegral(serac::Dimension<2 - 1>{}, serac::DependsOn<>{}, IdentityFunctor{}, attr1);
  double attr1SurfaceArea = attr1SurfArea(0.0, shape);

  EXPECT_NEAR(attr1SurfaceArea, 1.0, 1.0e-14);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
