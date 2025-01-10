#include <gtest/gtest.h>

#include <iostream>
#include <functional>

#include "serac/numerics/refactor/evaluate.hpp"
#include "serac/numerics/refactor/tests/common.hpp"

using namespace refactor;

template< int dim, int p >
void nodal_coordinates_test(std::string mesh_filename) {

  std::function< double(tensor<double, dim>) > f;

  mfem::ParMesh pmesh = load_parmesh(SERAC_MESH_DIR + mesh_filename);

  serac::FiniteElementState u(pmesh, H1<p>{});

  nd::array<double, 2> nodal_coords = get_nodal_coordinates(u);
  uint32_t num_nodes = nodal_coords.shape[0];

  //void FiniteElementState::project(mfem::Coefficient& coef, const Domain& domain)

  mfem::FunctionCoefficient mfem_func([](mfem::Vector x, double /*t*/) {
    return x[0] + 2 * x[1];
  });

  u.project(mfem_func);

  std::cout << u.Size() << std::endl;
  std::cout << num_nodes << std::endl;

  //for (uint32_t i = 0; i < num_nodes; i++) {
  //  std::cout << u[i]
  //}

}

TEST(nodal_coords, triangles) { nodal_coordinates_test<2, 1>("patch2D_tris.mesh"); }

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);

  int result = RUN_ALL_TESTS();

  serac::exitGracefully(result);
}
