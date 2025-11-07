// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/thermal_material.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/heat_transfer.hpp"

template <int p, int dim, int components>
void functional_test(int parallel_refinement)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 1;

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for thermal functional test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SMITH_REPO_DIR "/data/meshes/star.mesh" : SMITH_REPO_DIR "/data/meshes/beam-hex.mesh";

  ::axom::sidre::DataStore datastore;
  ::smith::StateManager::initialize(datastore, "sidreDataStore");

  auto pmesh = std::make_shared<smith::Mesh>(filename, "mesh_tag", serial_refinement, parallel_refinement);

  // Create standard MFEM bilinear and linear forms on H1
  using space = smith::H1<p, components>;
  auto [fespace, fec] = smith::generateParFiniteElementSpace<space>(&pmesh->mfemParMesh());

  smith::Functional<space(space)> residual(fespace.get(), {fespace.get()});

  // Add the total domain residual term to the functional
  residual.AddDomainIntegral(
      smith::Dimension<dim>{}, smith::DependsOn<0>{},
      [](double /*t*/, auto /*x*/, auto phi) {
        // get the value and the gradient from the input tuple
        auto [u, du_dx] = phi;
        return smith::tuple{u, du_dx};
      },
      pmesh->entireBody());

  // Set a random state to evaluate the residual
  mfem::ParGridFunction u_global(fespace.get());
  int seed = 1;
  u_global.Randomize(seed);

  mfem::Vector U(fespace->TrueVSize());
  u_global.GetTrueDofs(U);

  // Compute the residual using functional
  double t = 0.0;

  SMITH_MARK_BEGIN("residual evaluation");
  mfem::Vector r1 = residual(t, U);
  SMITH_MARK_END("residual evaluation");

  SMITH_MARK_BEGIN("compute gradient");
  auto [r2, drdU] = residual(t, smith::differentiate_wrt(U));
  SMITH_MARK_END("compute gradient");

  SMITH_MARK_BEGIN("apply gradient");
  mfem::Vector g = drdU(U);
  SMITH_MARK_END("apply gradient");

  SMITH_MARK_BEGIN("assemble gradient");
  auto g_mat = assemble(drdU);
  SMITH_MARK_END("assemble gradient");
}

int main(int argc, char* argv[])
{
  smith::ApplicationManager applicationManager(argc, argv);

  int parallel_refinement = 3;

  // Add metadata
  SMITH_SET_METADATA("test", "functional");

  SMITH_MARK_BEGIN("scalar H1");

  SMITH_MARK_BEGIN("dimension 2, order 1");
  functional_test<1, 2, 1>(parallel_refinement);
  SMITH_MARK_END("dimension 2, order 1");

  SMITH_MARK_BEGIN("dimension 2, order 2");
  functional_test<2, 2, 1>(parallel_refinement);
  SMITH_MARK_END("dimension 2, order 2");

  SMITH_MARK_BEGIN("dimension 3, order 1");
  functional_test<1, 3, 1>(parallel_refinement);
  SMITH_MARK_END("dimension 3, order 1");

  SMITH_MARK_BEGIN("dimension 3, order 2");
  functional_test<2, 3, 1>(parallel_refinement);
  SMITH_MARK_END("dimension 3, order 2");

  SMITH_MARK_END("scalar H1");

  SMITH_MARK_BEGIN("vector H1");

  SMITH_MARK_BEGIN("dimension 2, order 1");
  functional_test<1, 2, 2>(parallel_refinement);
  SMITH_MARK_END("dimension 2, order 1");

  SMITH_MARK_BEGIN("dimension 2, order 2");
  functional_test<2, 2, 2>(parallel_refinement);
  SMITH_MARK_END("dimension 2, order 2");

  SMITH_MARK_BEGIN("dimension 3, order 1");
  functional_test<1, 3, 3>(parallel_refinement);
  SMITH_MARK_END("dimension 3, order 1");

  SMITH_MARK_BEGIN("dimension 3, order 2");
  functional_test<2, 3, 3>(parallel_refinement);
  SMITH_MARK_END("dimension 3, order 2");

  SMITH_MARK_END("vector H1");

  return 0;
}
