// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "smith/physics/functional_weak_form.hpp"
#include "smith/physics/dfem_solid_weak_form.hpp"
#include "smith/physics/dfem_mass_weak_form.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

const std::string MESHTAG = "mesh";

namespace mfem {
namespace future {

/**
 * @brief Compute Green's strain from the displacement gradient
 */
template <typename T, int dim>
MFEM_HOST_DEVICE auto greenStrain(const tensor<T, dim, dim>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

}  // namespace future
}  // namespace mfem

namespace smith {

// NOTE (EBC): NeoHookean is not working with dfem on device with HIP, since some needed LLVM intrinsics are not
// implemented in Enzyme with the call to log1p()/log().
struct StVenantKirchhoffWithFieldDensityDfem {
  static constexpr int dim = 2;

  /**
   * @brief stress calculation for a St. Venant Kirchhoff material model
   *
   * @tparam T Type of the displacement gradient components (number-like)
   *
   * @param[in] grad_u Displacement gradient
   *
   * @return The first Piola stress
   */
  template <typename T, int dim, typename Density>
  SMITH_HOST_DEVICE auto pkStress(double, const mfem::future::tensor<T, dim, dim>& du_dX,
                                  const mfem::future::tensor<T, dim, dim>&, const Density&) const
  {
    auto I = mfem::future::IdentityMatrix<dim>();
    auto F = du_dX + I;
    const auto E = mfem::future::greenStrain(du_dX);

    // stress
    const auto S = K * mfem::future::tr(E) * I + 2.0 * G * mfem::future::dev(E);
    return mfem::future::tuple{mfem::future::dot(F, S)};
  }

  /// @brief interpolates density field
  template <typename Density>
  SMITH_HOST_DEVICE auto density(const Density& density) const
  {
    return density;
  }

  double K;  ///< Bulk modulus
  double G;  ///< Shear modulus
};

}  // namespace smith

struct ExplicitDynamicsFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using VectorSpace = smith::H1<disp_order, dim>;
  using DensitySpace = smith::L2<disp_order - 1>;

  using SolidMaterialDfem = smith::StVenantKirchhoffWithFieldDensityDfem;

  enum STATE
  {
    DISPLACEMENT,
    VELOCITY,
    ACCELERATION,
    COORDINATES
  };

  enum PARAMS
  {
    DENSITY
  };

  void SetUp()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    smith::StateManager::initialize(datastore, "solid_dynamics");

    // create mesh
    constexpr double length = 1.0;
    constexpr double width = 1.0;
    constexpr int nel_x = 4;
    constexpr int nel_y = 5;
    mesh = std::make_shared<smith::Mesh>(mfem::Mesh::MakeCartesian2D(nel_x, nel_y, element_shape, true, length, width),
                                         MESHTAG, 0, 0);
    // shift one of the x coordinates so the mesh is not affine
    auto* coords = mesh->mfemParMesh().GetNodes()->ReadWrite();
    coords[6] += 0.01;

    // create residual evaluator
    smith::FiniteElementState disp = smith::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
    smith::FiniteElementState velo = smith::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
    smith::FiniteElementState accel = smith::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
    smith::FiniteElementState density = smith::StateManager::newState(DensitySpace{}, "density", mesh->tag());
    disp.UseDevice(true);
    velo.UseDevice(true);
    accel.UseDevice(true);
    density.UseDevice(true);

    states = {disp, velo, accel};
    params = {density};

    std::string physics_name = "solid";
    double E = 1.0e3;
    double nu = 0.3;

    using SolidT = smith::DfemSolidWeakForm;
    auto solid_dfem_weak_form =
        std::make_shared<SolidT>(physics_name, mesh, states[DISPLACEMENT].space(), getSpaces(params));

    SolidMaterialDfem dfem_mat;
    dfem_mat.K = E / (3.0 * (1.0 - 2.0 * nu));  // bulk modulus
    dfem_mat.G = E / (2.0 * (1.0 + nu));        // shear modulus
    int ir_order = 2;
    const mfem::IntegrationRule& displacement_ir = mfem::IntRules.Get(disp.space().GetFE(0)->GetGeomType(), ir_order);
    mfem::Array<int> solid_attrib({1});
    solid_dfem_weak_form->setMaterial<SolidMaterialDfem, smith::ScalarParameter<0>>(solid_attrib, dfem_mat,
                                                                                    displacement_ir);

    mfem::future::tensor<mfem::real_t, dim> g({0.0, -9.81});  // gravity vector
    mfem::future::tuple<mfem::future::Value<SolidT::DISPLACEMENT>, mfem::future::Value<SolidT::VELOCITY>,
                        mfem::future::Value<SolidT::ACCELERATION>, mfem::future::Gradient<SolidT::COORDINATES>,
                        mfem::future::Weight, mfem::future::Value<SolidT::NUM_STATE_VARS>>
        g_inputs{};
    mfem::future::tuple<mfem::future::Value<SolidT::NUM_STATE_VARS + 1>> g_outputs{};
    solid_dfem_weak_form->addBodyIntegral(
        solid_attrib,
        [=] SMITH_HOST_DEVICE(const mfem::future::tensor<mfem::real_t, dim>&,
                              const mfem::future::tensor<mfem::real_t, dim>&,
                              const mfem::future::tensor<mfem::real_t, dim>&,
                              const mfem::future::tensor<mfem::real_t, dim, dim>& dX_dxi, mfem::real_t weight, double) {
          auto J = mfem::future::det(dX_dxi) * weight;
          return mfem::future::tuple{g * J};
        },
        g_inputs, g_outputs, displacement_ir, std::index_sequence<>{});

    states[DISPLACEMENT] = 0.0;
    states[VELOCITY] = 0.0;
    states[ACCELERATION].setFromFieldFunction([](smith::tensor<double, dim>) {
      smith::tensor<double, dim> u({0.0, -9.81});
      return u;
    });
    params[DENSITY] = 1.0;

    dfem_weak_form = solid_dfem_weak_form;

    auto mass_dfem_weak_form =
        smith::create_solid_mass_weak_form<dim, dim>(physics_name, mesh, states[DISPLACEMENT], params[0],
                                                     displacement_ir);  // nodal_ir_2d);
    mass_weak_form = mass_dfem_weak_form;

    // create time advancer
    advancer = std::make_shared<smith::LumpedMassExplicitNewmark>(dfem_weak_form, mass_weak_form, nullptr);
  }

  static constexpr bool quasi_static = true;
  static constexpr bool lumped_mass = false;

  const double dt = 0.001;
  const size_t num_steps = 3;

  // NOTE: max_error is driven by integration error on the perturbed element
  static constexpr double max_error = 1.0e-12;

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
  std::shared_ptr<smith::DfemSolidWeakForm> dfem_weak_form;

  std::shared_ptr<smith::DfemWeakForm> mass_weak_form;

  std::vector<smith::FiniteElementState> states;
  std::vector<smith::FiniteElementState> params;

  mfem::IntegrationRule nodal_ir_2d;
  std::shared_ptr<smith::LumpedMassExplicitNewmark> advancer;
};

TEST_F(ExplicitDynamicsFixture, RunDfemExplicitDynamicsSim)
{
  // acceleration (constant)
  mfem::Vector exact_accel_value({0.0, -9.81});
  smith::FiniteElementState exact_accel(states[ACCELERATION].space(), "exact_acceleration");
  exact_accel.UseDevice(true);
  mfem::VectorConstantCoefficient exact_accel_coeff(exact_accel_value);
  exact_accel.project(exact_accel_coeff);
  auto exact_accel_ptr = exact_accel.HostRead();

  // velocity
  mfem::Vector exact_velo_value({0.0, exact_accel_value[1] * dt});

  // displacement
  mfem::Vector exact_disp_value({0.0, 0.5 * exact_velo_value[1] * dt});

  double time = 0.0;
  for (size_t step = 0; step < num_steps; ++step) {
    for (auto& state : states) {
      state.gridFunction();
    }
    std::cout << "Step " << step << ", time = " << time << std::endl;
    auto state_ptrs = smith::getConstFieldPointers(states);
    smith::FiniteElementState coords(*static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes())->ParFESpace(),
                                     "coordinates");
    coords.UseDevice(true);
    coords.setFromGridFunction(*static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes()));
    state_ptrs.push_back(&coords);
    auto new_states_and_time = advancer->advanceState(state_ptrs, getConstFieldPointers(params), time, dt);

    time = std::get<1>(new_states_and_time);
    for (size_t i = 0; i < states.size(); ++i) {
      states[i] = std::get<0>(new_states_and_time)[i];
    }

    // check acceleration
    auto accel_ptr = states[ACCELERATION].HostRead();
    for (int i = 0; i < states[ACCELERATION].Size(); ++i) {
      EXPECT_NEAR(accel_ptr[i], exact_accel_ptr[i], max_error);
    }

    // update and check velocity
    smith::FiniteElementState exact_velo(states[VELOCITY].space(), "exact_velocity");
    exact_velo.UseDevice(true);
    mfem::VectorConstantCoefficient exact_velo_coeff(exact_velo_value);
    exact_velo.project(exact_velo_coeff);
    auto velo_ptr = states[VELOCITY].HostRead();
    auto exact_velo_ptr = exact_velo.HostRead();
    for (int i = 0; i < states[VELOCITY].Size(); ++i) {
      EXPECT_NEAR(velo_ptr[i], exact_velo_ptr[i], max_error);
    }
    exact_velo_value[1] += exact_accel_value[1] * dt;

    // update and check displacement
    smith::FiniteElementState exact_disp(states[DISPLACEMENT].space(), "exact_displacement");
    exact_disp.UseDevice(true);
    mfem::VectorConstantCoefficient exact_disp_coeff(exact_disp_value);
    exact_disp.project(exact_disp_coeff);
    auto disp_ptr = states[DISPLACEMENT].HostRead();
    auto exact_disp_ptr = exact_disp.HostRead();
    for (int i = 0; i < states[DISPLACEMENT].Size(); ++i) {
      EXPECT_NEAR(disp_ptr[i], exact_disp_ptr[i], max_error);
    }
    exact_disp_value[1] += exact_velo_value[1] * dt - 0.5 * exact_accel_value[1] * dt * dt;
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv, MPI_COMM_WORLD, true, smith::ExecutionSpace::GPU);
  return RUN_ALL_TESTS();
}
