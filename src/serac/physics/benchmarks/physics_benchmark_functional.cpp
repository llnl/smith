// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/materials/thermal_material.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/heat_transfer.hpp"

template <int p, int dim, int components>
void functional_test(int parallel_refinement)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 1;

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for thermal functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SERAC_REPO_DIR "/data/meshes/star.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh =
      serac::mesh::refineAndDistribute(serac::buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  // Create standard MFEM bilinear and linear forms on H1
  using space = serac::H1<p, components>;
  auto [fespace, fec] = serac::generateParFiniteElementSpace<space>(mesh.get());

  serac::Domain whole_domain = serac::EntireDomain(*mesh);

  serac::Functional<space(space)> residual(fespace.get(), {fespace.get()});

  // Add the total domain residual term to the functional
  residual.AddDomainIntegral(
      serac::Dimension<dim>{}, serac::DependsOn<0>{},
      [](double /*t*/, auto /*x*/, auto phi) {
        // get the value and the gradient from the input tuple
        auto [u, du_dx] = phi;
        return serac::tuple{u, du_dx};
      },
      whole_domain);

  // Set a random state to evaluate the residual
  mfem::ParGridFunction u_global(fespace.get());
  u_global.Randomize();

  mfem::Vector U(fespace->TrueVSize());
  u_global.GetTrueDofs(U);

  // Compute the residual using functional
  double t = 0.0;

  SERAC_MARK_BEGIN("residual evaluation");
  mfem::Vector r1 = residual(t, U);
  SERAC_MARK_END("residual evaluation");

  SERAC_MARK_BEGIN("compute gradient");
  auto [r2, drdU] = residual(t, serac::differentiate_wrt(U));
  SERAC_MARK_END("compute gradient");

  SERAC_MARK_BEGIN("apply gradient");
  mfem::Vector g = drdU(U);
  SERAC_MARK_END("apply gradient");

  SERAC_MARK_BEGIN("assemble gradient");
  auto g_mat = assemble(drdU);
  SERAC_MARK_END("assemble gradient");
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int parallel_refinement = 3;

  axom::slic::SimpleLogger logger;

  // Initialize profiling
  serac::profiling::initialize();

  // Add metadata
  SERAC_SET_METADATA("test", "functional");

  SERAC_MARK_BEGIN("scalar H1");

  SERAC_MARK_BEGIN("dimension 2, order 1");
  functional_test<1, 2, 1>(parallel_refinement);
  SERAC_MARK_END("dimension 2, order 1");

  SERAC_MARK_BEGIN("dimension 2, order 2");
  functional_test<2, 2, 1>(parallel_refinement);
  SERAC_MARK_END("dimension 2, order 2");

  SERAC_MARK_BEGIN("dimension 3, order 1");
  functional_test<1, 3, 1>(parallel_refinement);
  SERAC_MARK_END("dimension 3, order 1");

  SERAC_MARK_BEGIN("dimension 3, order 2");
  functional_test<2, 3, 1>(parallel_refinement);
  SERAC_MARK_END("dimension 3, order 2");

  SERAC_MARK_END("scalar H1");

  SERAC_MARK_BEGIN("vector H1");

  SERAC_MARK_BEGIN("dimension 2, order 1");
  functional_test<1, 2, 2>(parallel_refinement);
  SERAC_MARK_END("dimension 2, order 1");

  SERAC_MARK_BEGIN("dimension 2, order 2");
  functional_test<2, 2, 2>(parallel_refinement);
  SERAC_MARK_END("dimension 2, order 2");

  SERAC_MARK_BEGIN("dimension 3, order 1");
  functional_test<1, 3, 3>(parallel_refinement);
  SERAC_MARK_END("dimension 3, order 1");

  SERAC_MARK_BEGIN("dimension 3, order 2");
  functional_test<2, 3, 3>(parallel_refinement);
  SERAC_MARK_END("dimension 3, order 2");

  SERAC_MARK_END("vector H1");

  // Finalize profiling
  serac::profiling::finalize();

  MPI_Finalize();

  return 0;
}
