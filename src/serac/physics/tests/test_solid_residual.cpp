#include <gtest/gtest.h>
#include "mfem.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/materials/solid_material.hpp"

#include "serac/physics/solid_residual.hpp"

static constexpr serac::ELEMENT_SHAPE SHAPE = serac::ELEMENT_SHAPE::QUADRILATERAL; // TRIANGLE
const std::string MESHTAG = "mesh";

struct MeshFixture : public testing::Test
{
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using DensitySpace = serac::L2<disp_order-1, 1>;
  using VectorSpace = serac::H1<disp_order, dim>;

  using SolidMaterial = serac::solid_mechanics::NeoHookeanWithFieldDensity;

  template <typename T>
  std::vector<const T*> getPointers(const std::vector<T>& values)
  {
    std::vector<const T*> pointers;
    for (auto& t : values) {
      pointers.push_back(&t);
    }
    return pointers;
  }

  void SetUp() 
  {
    MPI_Barrier(MPI_COMM_WORLD);
    serac::StateManager::initialize(datastore, "solid_dynamics");

    // create mesh
    auto mfem_shape = SHAPE==serac::ELEMENT_SHAPE::TRIANGLE ? mfem::Element::TRIANGLE : mfem::Element::QUADRILATERAL;
    double length = 0.5; double width = 2.0;
    // MRT: ,make shared_ptr, have Mechanics hold it
    mesh = std::make_unique<serac::Mesh>(mfem::Mesh::MakeCartesian2D(6, 20, mfem_shape, true, length, width), MESHTAG, 0, 0);
    
    // create residual evaluator
    const double density_scalar = 1.0;
    std::string physics_name = "solid";

    serac::FiniteElementState disp = serac::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
    serac::FiniteElementState velo = serac::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
    serac::FiniteElementState accel = serac::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());

    serac::FiniteElementState density = serac::StateManager::newState(DensitySpace{}, "density", mesh->tag());
    density = density_scalar;

    states = {disp, velo, accel};
    std::vector<std::string> param_names{"density"};
    params = {density};
    auto param_ptrs = getPointers(params);
    auto solid_mechanics_residual = serac::create_solid_residual<disp_order, dim, DensitySpace>(physics_name, *mesh, param_names, param_ptrs);
    SolidMaterial mat;
    mat.K = 1.0;
    mat.G = 0.5;
    solid_mechanics_residual->setMaterial(serac::DependsOn<0>{}, mat, mesh->entireDomain());

    // specify dirichlet bcs
    //auto bc_manager = std::make_shared<serac::BoundaryConditionManager>(mesh->mfemParMesh());
    //auto zero_bcs = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return 0.0; });
    //bc_manager->addEssential({1}, zero_bcs, states[0].space());
  }

  std::string velo_name = "solid_velocity";

  axom::sidre::DataStore datastore;
  std::unique_ptr<serac::Mesh> mesh;

  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;

  const double dt = 0.01;
  const size_t num_steps = 2;

};


TEST_F(MeshFixture, TRANSIENT_DYNAMICS_GRETL)
{
  SERAC_MARK_FUNCTION;



}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  serac::exitGracefully(result);

  return result;
}
