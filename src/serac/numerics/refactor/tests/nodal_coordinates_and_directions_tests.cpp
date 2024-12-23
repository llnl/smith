#include <gtest/gtest.h>

#include <iostream>
#include <functional>

#include "serac/numerics/refactor/evaluate.hpp"
#include "serac/numerics/refactor/tests/common.hpp"

using namespace refactor;

template< int p >
void nodal_coordinates_test(std::string mesh_filename) {

    mfem::ParMesh pmesh = load_parmesh(SERAC_MESH_DIR + mesh_filename);

    serac::FiniteElementState u(pmesh, H1<p>{});

    nd::array<double, 2> nodal_coords = get_nodal_coordinates(u);

}

TEST(nodal_coords, triangles) { nodal_coordinates_test<1>("patch2D_tris.mesh"); }


int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);

  int result = RUN_ALL_TESTS();

  serac::exitGracefully(result);
}
