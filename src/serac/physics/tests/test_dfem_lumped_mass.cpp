// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "mfem.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/state_manager.hpp"

#include "serac/physics/tests/physics_test_utils.hpp"
#include "serac/physics/dfem_solid_weak_form.hpp"
#include "serac/physics/dfem_mass_weak_form.hpp"
#include "serac/physics/functional_weak_form.hpp"

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
    auto B_minus_I = mfem::future::dot(du_dX, mfem::future::transpose(du_dX)) + mfem::future::transpose(du_dX) + du_dX;

    auto logJ = log1p(mfem::future::detApIm1(du_dX));
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

}  // namespace serac

struct LumpedMassFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using VectorSpace = serac::H1<disp_order, dim>;
  using DensitySpace = serac::L2<disp_order - 1>;

  using SolidMaterialDfem = serac::NeoHookeanWithFieldDensityDfem<mfem::future::tensor>;

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
    double E = 1.0e3;
    double nu = 0.3;

    auto solid_dfem_residual =
        std::make_shared<SolidT>(physics_name, mesh, states[SolidT::DISPLACEMENT].space(), getSpaces(params));
    SolidMaterialDfem dfem_mat;
    dfem_mat.K = E / (3.0 * (1.0 - 2.0 * nu));  // bulk modulus
    dfem_mat.G = E / (2.0 * (1.0 + nu));        // shear modulus

    int ir_order = 3;
    const mfem::IntegrationRule& displacement_ir = mfem::IntRules.Get(disp.space().GetFE(0)->GetGeomType(), ir_order);
    mfem::Array<int> solid_attrib({1});

    solid_dfem_residual->setMaterial<SolidMaterialDfem, serac::ScalarParameter<0>>(solid_attrib, dfem_mat,
                                                                                   displacement_ir);

    mfem::future::tensor<mfem::real_t, dim> g({0.0, -9.81});  // gravity vector
    mfem::future::tuple<mfem::future::Value<SolidT::DISPLACEMENT>, mfem::future::Value<SolidT::VELOCITY>,
                        mfem::future::Value<SolidT::ACCELERATION>, mfem::future::Gradient<SolidT::COORDINATES>,
                        mfem::future::Weight, mfem::future::Value<SolidT::NUM_STATES>>
        g_inputs{};
    mfem::future::tuple<mfem::future::Value<SolidT::NUM_STATES + 1>> g_outputs{};
    solid_dfem_residual->addBodyIntegral(
        solid_attrib,
        [=] SERAC_HOST_DEVICE(const mfem::future::tensor<mfem::real_t, dim>&,
                              const mfem::future::tensor<mfem::real_t, dim>&,
                              const mfem::future::tensor<mfem::real_t, dim>&,
                              const mfem::future::tensor<mfem::real_t, dim, dim>& dX_dxi, mfem::real_t weight, double) {
          auto J = mfem::future::det(dX_dxi) * weight;
          return mfem::future::tuple{g * J};
        },
        g_inputs, g_outputs, displacement_ir, std::index_sequence<>{});

    // initialize fields for testing
    for (auto& s : state_tangents) {
      pseudoRand(s);
    }
    for (auto& p : param_tangents) {
      pseudoRand(p);
    }

    state_duals[0] = 1.0;  // used to test that vjp acts via +=, add initial value to shape displacement dual

    states[SolidT::DISPLACEMENT].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = 0.1 * x;
      return u;
    });
    states[SolidT::VELOCITY] = 0.0;
    states[SolidT::ACCELERATION].setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto u = -0.01 * x;
      return u;
    });
    params[0] = 1.0;

    dfem_residual = solid_dfem_residual;
    // lobatto rule at nodes
    // mfem::IntegrationRule rule_1d;
    // mfem::QuadratureFunctions1D::GaussLobatto(2, &rule_1d);
    // nodal_ir_2d = mfem::IntegrationRule(rule_1d, rule_1d);
    auto mass_dfem_residual = serac::create_solid_mass_weak_form<2, 2>(physics_name, mesh, states[SolidT::DISPLACEMENT],
                                                                       params[0], displacement_ir);  // nodal_ir_2d);
    mass_residual = mass_dfem_residual;
  }

  static constexpr bool quasi_static = false;
  static constexpr bool lumped_mass = false;

  using SolidT = serac::DfemSolidWeakForm<quasi_static, lumped_mass>;

  const double time = 0.0;
  const double dt = 1.0;

  std::string velo_name = "solid_velocity";

  axom::sidre::DataStore datastore;
  std::shared_ptr<serac::Mesh> mesh;
  std::shared_ptr<serac::DfemSolidWeakForm<quasi_static, lumped_mass>> dfem_residual;

  std::shared_ptr<serac::DfemWeakForm> mass_residual;

  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;

  std::vector<serac::FiniteElementDual> state_duals;
  std::vector<serac::FiniteElementDual> state_params;

  std::vector<serac::FiniteElementState> state_tangents;
  std::vector<serac::FiniteElementState> param_tangents;

  mfem::IntegrationRule nodal_ir_2d;
};

TEST_F(LumpedMassFixture, CheckDfemVsFunctionalResidual)
{
  // initialize the displacement and acceleration to a non-trivial field
  auto functional_input_fields = getConstFieldPointers(states, params);
  serac::FiniteElementState coords(*static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes())->ParFESpace(),
                                   "coordinates");
  coords.setFromGridFunction(*static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes()));
  std::vector<const serac::FiniteElementState*> dfem_input_fields(
      {functional_input_fields[1], functional_input_fields[2], functional_input_fields[3], &coords,
       functional_input_fields[4]});

  std::vector<const serac::FiniteElementState*> mass_input_fields = {&coords, &params[0]};
  auto lumped_mass_vector = mass_residual->residual(time, dt, nullptr, mass_input_fields);
  // std::cout << "lumped mass vector = " << std::endl;
  // lumped_mass_vector.Print();

  serac::FiniteElementDual ones_vector(states[SolidT::DISPLACEMENT].space(), "ones_vector");
  ones_vector = 1.0;
  serac::FiniteElementDual full_mass_vector(states[SolidT::DISPLACEMENT].space(), "full_mass_vector");
  full_mass_vector = 0.0;
  dfem_residual->massMatrix(dfem_input_fields, ones_vector, full_mass_vector);
  for (int i = 0; i < lumped_mass_vector.Size(); ++i) {
    EXPECT_NEAR(lumped_mass_vector[i], -full_mass_vector[i], 1.0e-14);
  }
  // std::cout << "full mass vector = " << std::endl;
  // full_mass_vector.Print();
  // ones_vector = 0.0;
  // for (int i = 0; i < ones_vector.Size(); ++i) {
  //   ones_vector[i] = 1.0;
  //   serac::FiniteElementDual mass_row(states[SolidResidualT::DISPLACEMENT].space(), "mass_row");
  //   mass_row = 0.0;
  //   dfem_residual->massMatrix(dfem_input_fields, ones_vector, mass_row);
  //   std::cout << "lumped mass vector = " << std::endl;
  //   mass_row.Print();
  //   ones_vector[i] = 0.0;
  // }
}

// TEST_F(ResidualFixture, JvpConsistency)
// {
//   // initialize the displacement and acceleration to a non-trivial field
//   auto input_fields = getConstFieldPointers(states, params);

//   serac::FiniteElementDual res_vector(states[SolidResidualT::DISPLACEMENT].space(), "residual");
//   res_vector = residual->residual(time, dt, input_fields);
//   ASSERT_NE(0.0, res_vector.Norml2());

//   auto jacobianWeights = [&](size_t i) {
//     std::vector<double> tangents(input_fields.size());
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

//   serac::FiniteElementDual jvp_slow(states[SolidResidualT::DISPLACEMENT].space(), "jvp_slow");
//   serac::FiniteElementDual jvp(states[SolidResidualT::DISPLACEMENT].space(), "jvp");
//   jvp = 4.0;  // set to some value to test jvp resets these values
//   auto jvps = getFieldPointers(jvp);

//   auto field_tangents = getConstFieldPointers(state_tangents, param_tangents);

//   for (size_t i = 0; i < input_fields.size(); ++i) {
//     auto J = residual->jacobian(time, dt, input_fields, jacobianWeights(i));
//     J->Mult(*field_tangents[i], jvp_slow);
//     residual->jvp(time, dt, input_fields, selectStates(i), jvps);
//     EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
//   }

//   // test jacobians in weighted combinations
//   {
//     field_tangents[SolidResidualT::SHAPE_DISPLACEMENT] = nullptr;
//     field_tangents[SolidResidualT::VELOCITY] = nullptr;
//     field_tangents[size_t(SolidResidualT::NUM_STATES) + size_t(DENSITY)] = nullptr;

//     double acceleration_factor = 0.2;
//     std::vector<double> jacobian_weights = {0.0, 1.0, 0.0, acceleration_factor, 0.0};

//     auto J = residual->jacobian(time, dt, input_fields, jacobian_weights);
//     J->Mult(*field_tangents[SolidResidualT::DISPLACEMENT], jvp_slow);

//     state_tangents[SolidResidualT::ACCELERATION] *= acceleration_factor;

//     residual->jvp(time, dt, input_fields, field_tangents, jvps);
//     EXPECT_NEAR(jvp_slow.Norml2(), jvp.Norml2(), 1e-12);
//   }
// }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
