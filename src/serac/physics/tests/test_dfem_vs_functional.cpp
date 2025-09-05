// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "mfem.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/mesh_utils/mesh_utils.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/state_manager.hpp"

#include "serac/physics/tests/physics_test_utils.hpp"
#include "serac/physics/solid_residual.hpp"
#include "serac/physics/solid_dfem_residual.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

namespace serac {

template <template <typename, int...> class TensorT>
struct NeoHookeanWithFieldDensityDfem {
  static constexpr int dim = 2;
  /**
   * @brief stress calculation for a NeoHookean material model
   * @tparam T type of float or dual in tensor
   * @tparam dim Dimensionality of space
   * @param du_dX displacement gradient with respect to the reference configuration
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   * @return The first Piola stress
   */
  template <typename T, int dim, typename Density>
  SERAC_HOST_DEVICE auto pkStress(double, const TensorT<T, dim, dim>& du_dX, const TensorT<T, dim, dim>&,
                                  const Density&) const
  {
    using std::log1p;
    auto I = mfem::future::IdentityMatrix<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = mfem::future::dot(du_dX, transpose(du_dX)) + mfem::future::transpose(du_dX) + du_dX;

    auto logJ = log1p(detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I
    auto TK = lambda * logJ * I + G * B_minus_I;

    // Pull back to Piola
    auto F = du_dX + I;
    return mfem::future::tuple{mfem::future::dot(TK, mfem::future::inv(mfem::future::transpose(F)))};
  }

  /// @brief interpolates density field
  template <typename Density>
  SERAC_HOST_DEVICE auto density(const Density& density) const
  {
    return density;
  }

  double K;  ///< bulk modulus
  double G;  ///< shear modulus
};

template <template <typename, int...> class TensorT>
struct NeoHookeanWithFieldWithRateFunctional {
  using State = Empty;  ///< this material has no internal variables

  template <typename T1, typename T2, int dim>
  SERAC_HOST_DEVICE auto pkStress(double /*dt*/, State& /* state */, const TensorT<T1, dim, dim>& du_dX,
                                  const TensorT<T2, dim, dim>& /*dv_dX*/) const
  {
    using std::log1p;
    constexpr auto I = Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = dot(du_dX, transpose(du_dX)) + transpose(du_dX) + du_dX;

    auto logJ = log1p(detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I
    auto TK = lambda * logJ * I + G * B_minus_I;

    // Pull back to Piola
    auto F = du_dX + I;
    return dot(TK, inv(transpose(F)));
  }

  SERAC_HOST_DEVICE auto density() const { return Rho; }

  double K;    ///< bulk modulus
  double G;    ///< shear modulus
  double Rho;  ///< density
};

template <template <typename, int...> class TensorT>
struct NeoHookeanWithFieldWithRateDfem {
  template <typename T1, typename T2, int dim>
  SERAC_HOST_DEVICE auto pkStress(double /*dt*/, const TensorT<T1, dim, dim>& du_dX,
                                  const TensorT<T2, dim, dim>& /*dv_dX*/) const
  {
    using std::log1p;
    constexpr auto I = mfem::future::IdentityMatrix<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = mfem::future::dot(du_dX, mfem::future::transpose(du_dX)) + mfem::future::transpose(du_dX) + du_dX;

    auto logJ = log1p(detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I
    auto TK = lambda * logJ * I + G * B_minus_I;

    // Pull back to Piola
    auto F = du_dX + I;
    return mfem::future::dot(TK, mfem::future::inv(mfem::future::transpose(F)));
  }

  SERAC_HOST_DEVICE auto density() const { return Rho; }

  double K;    ///< bulk modulus
  double G;    ///< shear modulus
  double Rho;  ///< density
};

}  // namespace serac

struct DfemVsFunctionalFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using VectorSpace = serac::H1<disp_order, dim>;
  using DensitySpace = serac::L2<disp_order - 1>;

  using SolidMaterialFunctional = serac::solid_mechanics::NeoHookeanWithFieldDensity;
  using SolidMaterialDfem = serac::NeoHookeanWithFieldDensityDfem<mfem::future::tensor>;
  using SolidRateMaterialFunctional = serac::NeoHookeanWithFieldWithRateFunctional<serac::tensor>;
  using SolidRateMaterialDfem = serac::NeoHookeanWithFieldWithRateDfem<mfem::future::tensor>;

  enum PARAMS
  {
    DENSITY
  };

  void SetUp()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    serac::StateManager::initialize(datastore, "solid_dynamics");

    constexpr double length = 1.0;
    constexpr double width = 1.0;
    constexpr int nel_x = 2;
    constexpr int nel_y = 1;
    mesh = std::make_shared<serac::Mesh>(mfem::Mesh::MakeCartesian2D(nel_x, nel_y, element_shape, true, length, width),
                                         "this_mesh_name", 0, 0);
    // shift one of the x coordinates so the mesh is not affine
    auto& coords = *mesh->mfemParMesh().GetNodes();
    coords[6] += 0.1;

    serac::FiniteElementState disp = serac::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
    serac::FiniteElementState velo = serac::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
    serac::FiniteElementState accel = serac::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
    serac::FiniteElementState shape_disp =
        serac::StateManager::newState(VectorSpace{}, "shape_displacement", mesh->tag());
    serac::FiniteElementState density = serac::StateManager::newState(DensitySpace{}, "density", mesh->tag());

    states = {shape_disp, disp, velo, accel};
    params = {density};

    for (auto s : states) {
      state_duals.push_back(serac::FiniteElementDual(s.space(), s.name() + "_dual"));
    }
    for (auto p : params) {
      state_params.push_back(serac::FiniteElementDual(p.space(), p.name() + "_dual"));
    }

    state_tangents = states;
    param_tangents = params;

    std::string physics_name = "solid";

    auto solid_mechanics_residual =
        std::make_shared<SolidResidualT>(physics_name, mesh, states[SolidResidualT::SHAPE_DISPLACEMENT].space(),
                                         states[SolidResidualT::DISPLACEMENT].space(), getSpaces(params));
    SolidMaterialFunctional mat;
    double E = 1.0e3;
    double nu = 0.3;
    mat.K = E / (3.0 * (1.0 - 2.0 * nu));  // bulk modulus
    mat.G = E / (2.0 * (1.0 + nu));        // shear modulus
    // SolidRateMaterialFunctional rate_mat;
    // rate_mat.K = 1.0;
    // rate_mat.G = 0.5;
    // rate_mat.Rho = 1.5;

    solid_mechanics_residual->setMaterial(serac::DependsOn<0>{}, mesh->entireBodyName(), mat);
    // solid_mechanics_residual->setRateMaterial(serac::DependsOn<>{}, mesh->entireBodyName(), rate_mat);

    auto solid_dfem_residual = std::make_shared<SolidDfemT>(
        physics_name, mesh, states[SolidResidualT::DISPLACEMENT].space(), getSpaces(params));
    SolidMaterialDfem dfem_mat;
    dfem_mat.K = mat.K;
    dfem_mat.G = mat.G;
    // SolidRateMaterialDfem dfem_rate_mat;
    // dfem_rate_mat.K = 1.0;
    // dfem_rate_mat.G = 0.5;
    // dfem_rate_mat.Rho = 1.5;

    int ir_order = 2;
    const mfem::IntegrationRule& displacement_ir = mfem::IntRules.Get(disp.space().GetFE(0)->GetGeomType(), ir_order);
    mfem::Array<int> solid_attrib({1});

    solid_dfem_residual->setMaterial<SolidMaterialDfem, serac::ScalarParameter<0>>(solid_attrib, dfem_mat,
                                                                                   displacement_ir);

    // // apply traction boundary conditions
    // std::string surface_name = "side";
    // mesh->addDomainOfBoundaryElements(surface_name, serac::by_attr<dim>(1));
    // solid_mechanics_residual->addBoundaryIntegral(surface_name, [](auto /*t*/, auto /*x*/, auto n) { return 1.0 * n;
    // }); solid_mechanics_residual->addPressure(surface_name, [](auto /*t*/, auto /*x*/) { return 0.6; });

    // initialize fields for testing
    for (auto& s : state_tangents) {
      pseudoRand(s);
    }
    for (auto& p : param_tangents) {
      pseudoRand(p);
    }

    state_duals[0] = 1.0;  // used to test that vjp acts via +=, add initial value to shape displacement dual

    // states[SolidResidualT::DISPLACEMENT] = 0.0;
    states[SolidResidualT::DISPLACEMENT].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = 0.1 * x;
      return u;
    });
    states[SolidResidualT::VELOCITY] = 0.0;
    // states[SolidResidualT::ACCELERATION] = 0.0;
    states[SolidResidualT::ACCELERATION].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = -0.01 * x;
      return u;
    });
    states[SolidResidualT::SHAPE_DISPLACEMENT] = 0.0;
    params[0] = 1.2;

    // residual is abstract Residual class to ensure usage only through BasePhysics interface
    functional_residual = solid_mechanics_residual;
    dfem_residual = solid_dfem_residual;
  }

  using SolidResidualT = serac::SolidResidual<disp_order, dim, serac::Parameters<DensitySpace>>;
  using SolidDfemT = serac::SolidDfemResidual<false, false>;

  const double time = 0.0;
  const double dt = 1.0;

  std::string velo_name = "solid_velocity";

  axom::sidre::DataStore datastore;
  std::shared_ptr<serac::Mesh> mesh;
  std::shared_ptr<serac::Residual> functional_residual;
  std::shared_ptr<serac::Residual> dfem_residual;

  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;

  std::vector<serac::FiniteElementDual> state_duals;
  std::vector<serac::FiniteElementDual> state_params;

  std::vector<serac::FiniteElementState> state_tangents;
  std::vector<serac::FiniteElementState> param_tangents;
};

TEST_F(DfemVsFunctionalFixture, CheckDfemVsFunctionalResidual)
{
  // initialize the displacement and acceleration to a non-trivial field
  auto functional_input_fields = getConstFieldPointers(states, params);
  serac::FiniteElementState coords(*static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes())->ParFESpace(),
                                   "coordinates");
  coords.setFromGridFunction(*static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes()));
  std::vector<const serac::FiniteElementState*> dfem_input_fields(
      {functional_input_fields[1], functional_input_fields[2], functional_input_fields[3], &coords,
       functional_input_fields[4]});

  serac::FiniteElementDual functional_res_vector(states[SolidResidualT::DISPLACEMENT].space(), "functional_residual");
  functional_res_vector = functional_residual->residual(time, dt, functional_input_fields);
  serac::FiniteElementDual dfem_vs_functional_vector(states[SolidResidualT::DISPLACEMENT].space(), "dfem_residual");
  // set nodes at current coords for dfem
  //(*mesh->mfemParMesh().GetNodes()) += states[1].gridFunction();
  dfem_vs_functional_vector = dfem_residual->residual(time, dt, dfem_input_fields);
  dfem_vs_functional_vector -= functional_res_vector;
  ASSERT_NEAR(0.0, dfem_vs_functional_vector.Norml2(), 1.0e-12) << "Functional and DFEM residuals do not match!";
}

// TEST_F(DfemVsFunctionalFixture, JvpConsistency)
// {
//   // initialize the displacement and acceleration to a non-trivial field
//   auto functional_input_fields = getConstFieldPointers(states, params);
//   serac::FiniteElementState
//   coords(*static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes())->ParFESpace(),
//                                    "coordinates");
//   std::vector<const serac::FiniteElementState*> dfem_input_fields(
//       {functional_input_fields[1], functional_input_fields[2], functional_input_fields[3], &coords,
//        functional_input_fields[4]});

//   auto jacobianWeights = [&](size_t i) {
//     std::vector<double> tangents(functional_input_fields.size());
//     tangents[i] = 1.0;
//     return tangents;
//   };

//   auto selectStates = [&](size_t i) {
//     auto field_tangents = getConstFieldPointers(state_tangents, param_tangents);
//     for (size_t j = 0; j < field_tangents.size(); ++j) {
//       if (i != j) {
//         field_tangents[j] = nullptr;
//       }
//     }
//     return field_tangents;
//   };

//   serac::FiniteElementDual jvp_functional(states[SolidResidualT::DISPLACEMENT].space(), "jvp_functional");
//   jvp_functional = 4.0;  // set to some value to test jvp resets these values
//   auto jvps_functional = getFieldPointers(jvp_functional);
//   serac::FiniteElementDual jvp_dfem(states[SolidResidualT::DISPLACEMENT].space(), "jvp_dfem");
//   jvp_dfem = 4.0;  // set to some value to test jvp resets these values
//   auto jvps_dfem = getFieldPointers(jvp_dfem);

//   auto functional_field_tangents = getConstFieldPointers(state_tangents, param_tangents);
//   std::vector<const serac::FiniteElementState*> dfem_field_tangents(
//       {dfem_field_tangents[1], dfem_field_tangents[2], dfem_field_tangents[3], &coords, dfem_field_tangents[4]});

// // TODO from here: figure out a better way to map fields between dfem and functional
// for (size_t i = 0; i < input_fields.size(); ++i) {
//   auto J = residual->jacobian(time, dt, input_fields, jacobianWeights(i));
//   J->Mult(*field_tangents[i], jvp_slow);
//   residual->jvp(time, dt, input_fields, selectStates(i), jvps);
//   EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
// }

// // test jacobians in weighted combinations
// {
//   field_tangents[SolidResidualT::SHAPE_DISPLACEMENT] = nullptr;
//   field_tangents[SolidResidualT::VELOCITY] = nullptr;
//   field_tangents[size_t(SolidResidualT::NUM_STATES) + size_t(DENSITY)] = nullptr;

//   double acceleration_factor = 0.2;
//   std::vector<double> jacobian_weights = {0.0, 1.0, 0.0, acceleration_factor, 0.0};

//   auto J = residual->jacobian(time, dt, input_fields, jacobian_weights);
//   J->Mult(*field_tangents[SolidResidualT::DISPLACEMENT], jvp_slow);

//   state_tangents[SolidResidualT::ACCELERATION] *= acceleration_factor;

//   residual->jvp(time, dt, input_fields, field_tangents, jvps);
//   EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
// }
// }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
