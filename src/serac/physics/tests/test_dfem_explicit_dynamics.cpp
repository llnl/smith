// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "mfem.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/state_manager.hpp"

#include "serac/physics/tests/physics_test_utils.hpp"
#include "serac/physics/solid_residual.hpp"
#include "serac/physics/solid_dfem_residual.hpp"
#include "serac/physics/dfem_mass_residual.hpp"

#include "serac/numerics/equation_solver.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

const std::string MESHTAG = "mesh";

serac::LinearSolverOptions linear_options{.linear_solver = serac::LinearSolver::CG,
                                          .preconditioner = serac::Preconditioner::HypreJacobi,
                                          .relative_tol = 1e-9,
                                          .absolute_tol = 1e-9,
                                          .max_iterations = 100000,
                                          .print_level = 0};

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

class LumpedMassExplicitNewmark {
 public:
  LumpedMassExplicitNewmark(const std::shared_ptr<Residual>& residual, const std::shared_ptr<Residual>& mass_residual,
                            std::shared_ptr<BoundaryConditionManager> bc_manager)
      : residual_(residual), mass_residual_(mass_residual), bc_manager_(bc_manager)
  {
  }

  std::tuple<std::vector<FiniteElementState>, double> advanceState(const std::vector<ConstFieldPtr>& states,
                                                                   const std::vector<ConstFieldPtr>& params,
                                                                   double time, double dt)
  {
    SLIC_ERROR_ROOT_IF(states.size() != 4, "Expected 4 states: displacement, velocity, acceleration, and coordinates");

    enum States
    {
      DISP,
      VELO,
      ACCEL,
      COORD
    };

    enum Params
    {
      DENSITY
    };

    const auto& u = *states[DISP];
    const auto& v = *states[VELO];
    const auto& a = *states[ACCEL];

    auto v_pred = v;
    v_pred.Add(0.5 * dt, a);
    auto u_pred = u;
    u_pred.Add(dt, v_pred);

    if (bc_manager_) {
      u_pred.SetSubVector(bc_manager_->allEssentialTrueDofs(), 0.0);
    }

    auto m_inv = mass_residual_->residual(time, dt, {states[COORD], params[DENSITY]});
    m_inv.Reciprocal();

    std::vector<ConstFieldPtr> pred_states = {&u_pred, &v_pred, &a, states[COORD], params[DENSITY]};

    auto zero_mass_resid = residual_->residual(time, dt, pred_states);

    FiniteElementState a_pred(a.space(), "acceleration_pred");
    auto a_pred_ptr = a_pred.Write();
    auto m_inv_ptr = m_inv.Read();
    auto zero_mass_resid_ptr = zero_mass_resid.Read();
    mfem::forall_switch(a_pred.UseDevice(), a_pred.Size(),
                        [=] MFEM_HOST_DEVICE(int i) { a_pred_ptr[i] = m_inv_ptr[i] * zero_mass_resid_ptr[i]; });

    v_pred.Add(0.5 * dt, a_pred);

    return {{u_pred, v_pred, a_pred, *states[COORD]}, time + dt};
  }

 private:
  std::shared_ptr<Residual> residual_;
  std::shared_ptr<Residual> mass_residual_;
  std::shared_ptr<BoundaryConditionManager> bc_manager_;
};

}  // namespace serac

struct ExplicitDynamicsFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using VectorSpace = serac::H1<disp_order, dim>;
  using DensitySpace = serac::L2<disp_order - 1>;

  using SolidMaterialDfem = serac::NeoHookeanWithFieldDensityDfem<mfem::future::tensor>;

  enum STATE
  {
    DISP,
    VELO,
    ACCEL,
    COORD
  };

  enum PARAMS
  {
    DENSITY
  };

  void SetUp()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    serac::StateManager::initialize(datastore, "solid_dynamics");

    // create mesh
    constexpr double length = 1.0;
    constexpr double width = 1.0;
    constexpr int nel_x = 1;
    constexpr int nel_y = 1;
    mesh = std::make_shared<serac::Mesh>(mfem::Mesh::MakeCartesian2D(nel_x, nel_y, element_shape, true, length, width),
                                         MESHTAG, 0, 0);
    // shift one of the x coordinates so the mesh is not affine
    auto* coords = mesh->mfemParMesh().GetNodes()->ReadWrite();
    coords[6] += 0.1;

    // create solver
    auto [linear_solver, mech_precond] = serac::buildLinearSolverAndPreconditioner(linear_options, mesh->getComm());

    // create residual evaluator
    serac::FiniteElementState disp = serac::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
    serac::FiniteElementState velo = serac::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
    serac::FiniteElementState accel = serac::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
    serac::FiniteElementState density = serac::StateManager::newState(DensitySpace{}, "density", mesh->tag());

    states = {disp, velo, accel};
    params = {density};

    std::string physics_name = "solid";
    double E = 1.0e3;
    double nu = 0.3;

    using SolidT = serac::SolidDfemResidual<quasi_static, lumped_mass>;
    auto solid_dfem_residual = std::make_shared<SolidT>(physics_name, mesh, states[DISP].space(), getSpaces(params));

    SolidMaterialDfem dfem_mat;
    dfem_mat.K = E / (3.0 * (1.0 - 2.0 * nu));  // bulk modulus
    dfem_mat.G = E / (2.0 * (1.0 + nu));        // shear modulus
    int ir_order = 3;
    const mfem::IntegrationRule& displacement_ir = mfem::IntRules.Get(disp.space().GetFE(0)->GetGeomType(), ir_order);
    mfem::Array<int> solid_attrib({1});
    solid_dfem_residual->setMaterial<SolidMaterialDfem, serac::ScalarParameter<0>>(solid_attrib, dfem_mat,
                                                                                   displacement_ir);

    mfem::future::tensor<mfem::real_t, dim> g({0.0, -9.81});  // gravity vector
    mfem::future::tuple<mfem::future::Value<SolidT::DISP>, mfem::future::Value<SolidT::VELO>,
                        mfem::future::Value<SolidT::ACCEL>, mfem::future::Gradient<SolidT::COORD>, mfem::future::Weight,
                        mfem::future::Value<SolidT::NUM_STATE_VARS>>
        g_inputs{};
    mfem::future::tuple<mfem::future::Value<SolidT::NUM_STATE_VARS + 1>> g_outputs{};
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

    states[DISP] = 0.0;
    // states[DISP].setFromFieldFunction([](serac::tensor<double, dim> x) {
    //   auto u = 0.1 * x;
    //   return u;
    // });
    states[VELO] = 0.0;
    states[ACCEL] = 0.0;
    // states[ACCEL].setFromFieldFunction([](serac::tensor<double, dim> x) {
    //   auto u = -0.01 * x;
    //   return u;
    // });
    params[DENSITY] = 1.0;

    dfem_residual = solid_dfem_residual;

    auto mass_dfem_residual = serac::create_solid_mass_residual<dim, dim>(physics_name, mesh, states[DISP], params[0],
                                                                          displacement_ir);  // nodal_ir_2d);
    mass_residual = mass_dfem_residual;

    // create time advancer
    advancer = std::make_shared<serac::LumpedMassExplicitNewmark>(dfem_residual, mass_residual, nullptr);
  }

  static constexpr bool quasi_static = true;
  static constexpr bool lumped_mass = false;

  const double dt = 0.1;
  const size_t num_steps = 5;

  axom::sidre::DataStore datastore;
  std::shared_ptr<serac::Mesh> mesh;
  std::shared_ptr<serac::SolidDfemResidual<quasi_static, lumped_mass>> dfem_residual;

  std::shared_ptr<serac::DfemResidual> mass_residual;

  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;

  std::vector<serac::FiniteElementDual> state_duals;
  std::vector<serac::FiniteElementDual> state_params;

  std::vector<serac::FiniteElementState> state_tangents;
  std::vector<serac::FiniteElementState> param_tangents;

  mfem::IntegrationRule nodal_ir_2d;
  std::shared_ptr<serac::LumpedMassExplicitNewmark> advancer;

  std::unique_ptr<axom::sidre::MFEMSidreDataCollection> dc;
};

TEST_F(ExplicitDynamicsFixture, RunDfemExplicitDynamicsSim)
{
  mfem::VisItDataCollection dc("solid_explicit_dynamics", &mesh->mfemParMesh());
  dc.RegisterField("displacement", &states[DISP].gridFunction());
  dc.RegisterField("velocity", &states[VELO].gridFunction());
  dc.RegisterField("acceleration", &states[ACCEL].gridFunction());
  dc.RegisterField("density", &params[DENSITY].gridFunction());
  double time = 0.0;
  int cycle = 0;
  for (size_t step = 0; step < num_steps; ++step) {
    for (auto& state : states) {
      state.gridFunction();
    }
    dc.SetCycle(cycle);
    dc.SetTime(time);
    dc.Save();
    std::cout << "Step " << step << ", time = " << time << std::endl;
    auto state_ptrs = serac::getConstFieldPointers(states);
    serac::FiniteElementState coords(*static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes())->ParFESpace(),
                                     "coordinates");
    coords.setFromGridFunction(*static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes()));
    state_ptrs.push_back(&coords);
    auto new_states_and_time = advancer->advanceState(state_ptrs, getConstFieldPointers(params), time, dt);
    cycle++;
    time = std::get<1>(new_states_and_time);
    for (size_t i = 0; i < states.size(); ++i) {
      states[i] = std::get<0>(new_states_and_time)[i];
    }
  }
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
