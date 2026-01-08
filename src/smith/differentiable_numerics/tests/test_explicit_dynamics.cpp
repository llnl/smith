// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include "mfem.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"

#include "gretl/data_store.hpp"
#include "smith/physics/solid_weak_form.hpp"
#include "smith/physics/functional_objective.hpp"

#include "smith/differentiable_numerics/lumped_mass_explicit_newmark_state_advancer.hpp"
#include "smith/differentiable_numerics/lumped_mass_weak_form.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/timestep_estimator.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/evaluate_objective.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"

// This tests the interface between the new smith::WeakForm with gretl and its conformity to the existing base_physics
// interface

const std::string MESHTAG = "mesh";

/**
 * @brief Neo-Hookean material model
 * This struct differs in style relative to the older materials as it needs to evaluate both stress
 * and density.  As a result, we want to clearly name these functions.
 * This is likely going to be a new design going forward, at the moment it works with the
 * SolidResidual class.
 *
 */
struct NeoHookeanWithFixedDensity {
  using State = smith::Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a NeoHookean material model
   * @tparam T type of float or dual in tensor
   * @tparam dim Dimensionality of space
   * @param du_dX displacement gradient with respect to the reference configuration
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   * @return The first Piola stress
   */
  template <typename T, int dim>
  SMITH_HOST_DEVICE auto pkStress(State& /* state */, const smith::tensor<T, dim, dim>& du_dX) const
  {
    using std::log1p;
    constexpr auto I = smith::Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = dot(du_dX, smith::transpose(du_dX)) + smith::transpose(du_dX) + du_dX;

    auto logJ = log1p(smith::detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I
    auto TK = lambda * logJ * I + G * B_minus_I;

    // Pull back to Piola
    auto F = du_dX + I;
    return smith::dot(TK, smith::inv(smith::transpose(F)));
  }

  /// @brief interpolates density field
  SMITH_HOST_DEVICE auto density() const { return density0; }

  double K;  ///< bulk modulus
  double G;  ///< shear modulus
  double density0;
};

struct MeshFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  static constexpr int scalar_field_order = 1;
  using DensitySpace = smith::L2<scalar_field_order>;
  using VectorSpace = smith::H1<disp_order, dim>;

  using SolidMaterial = NeoHookeanWithFixedDensity;

  static constexpr double gravity = -9.0;

  enum STATE
  {
    DISP,
    VELO,
    ACCEL
  };

  enum PARAMS
  {
    DENSITY
  };

  enum FIELD
  {
    F_DISP,
    F_VELO,
    F_ACCEL,
    F_DENSITY
  };

  void SetUp()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    smith::StateManager::initialize(datastore_, "solid_dynamics");

    // create mesh
    auto mfem_shape = mfem::Element::QUADRILATERAL;  // mfem::Element::TRIANGLE;
    double length = 0.5;
    double width = 2.0;
    int num_elems_x = 5;
    int num_elems_y = 4;
    int num_refine_serial = 0;
    int num_refine_parallel = 0;
    mesh_ = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian2D(num_elems_x, num_elems_y, mfem_shape, true, length, width), MESHTAG,
        num_refine_serial, num_refine_parallel);
    // checkpointing graph
    checkpointer_ = std::make_shared<gretl::DataStore>(200);

    // create residual evaluator
    const double density = 1.0;
    std::string physics_name = "solid";

    shape_disp_ = std::make_unique<smith::FieldState>(
        createFieldState(*checkpointer_, VectorSpace{}, physics_name + "_shape_displacement", mesh_->tag()));
    auto disp = createFieldState(*checkpointer_, VectorSpace{}, physics_name + "_displacement", mesh_->tag());
    auto velo = createFieldState(*checkpointer_, VectorSpace{}, physics_name + "_velocity", mesh_->tag());
    auto accel = createFieldState(*checkpointer_, VectorSpace{}, physics_name + "_acceleration", mesh_->tag());
    auto density0 = createFieldState(*checkpointer_, DensitySpace{}, physics_name + "_density", mesh_->tag());

    *disp.get() = 0.0;
    *velo.get() = 0.0;
    *accel.get() = 0.0;
    *density0.get() = density;

    initial_states_ = {disp, velo, accel};
    params_ = {density0};
    std::vector<smith::FieldState> states{disp, velo, accel};

    auto solid_mechanics_residual = smith::create_solid_weak_form<disp_order, dim, DensitySpace>(
        physics_name, mesh_, getConstFieldPointers(states), getConstFieldPointers(params_));

    SolidMaterial mat;
    mat.density0 = density;
    mat.K = 1.0;
    mat.G = 0.5;

    solid_mechanics_residual->setMaterial(smith::DependsOn<>{}, mesh_->entireBodyName(), mat);

    solid_mechanics_residual->addBodySource(mesh_->entireBodyName(), [](auto /*time*/, auto X) {
      auto b = 0.0 * X;
      b[1] = gravity;
      return b;
    });

    // create mass evaluator and state in order to be able to create a diagonalized mass matrix
    std::string mass_residual_name = "mass";
    auto solid_mass_residual = smith::createSolidMassWeakForm<VectorSpace::components, VectorSpace, DensitySpace>(
        mass_residual_name, mesh_, *states[DISP].get(), *params_[DENSITY].get());

    // specify dirichlet bcs
    bc_manager_ = std::make_shared<smith::BoundaryConditionManager>(mesh_->mfemParMesh());

    auto dt_estimator =
        std::make_shared<smith::ConstantTimeStepEstimator>(dt / 10.0);  // reduce the timestep a bit, so it subcycles
    std::shared_ptr<smith::StateAdvancer> time_integrator =
        std::make_shared<smith::LumpedMassExplicitNewmarkStateAdvancer>(solid_mechanics_residual, solid_mass_residual,
                                                                        dt_estimator, bc_manager_);

    // construct mechanics
    mechanics_ = std::make_shared<smith::DifferentiablePhysics>(mesh_, checkpointer_, *shape_disp_, states, params_,
                                                                time_integrator, "mechanics");
    physics_ = mechanics_;

    auto ke_objective = std::make_shared<smith::FunctionalObjective<dim, smith::Parameters<VectorSpace, DensitySpace>>>(
        "integrated_squared_temperature", mesh_, smith::spaces({states[DISP], params_[DENSITY]}));

    ke_objective->addBodyIntegral(smith::DependsOn<0, 1>(), mesh_->entireBodyName(),
                                  [](auto /*t*/, auto /*X*/, auto U, auto Rho) {
                                    auto u = get<smith::VALUE>(U);
                                    return 0.5 * get<smith::VALUE>(Rho) * smith::inner(u, u);
                                  });
    objective_ = ke_objective;

    // kinetic energy integrator for qoi
    kinetic_energy_integrator_ = smith::createKineticEnergyIntegrator<VectorSpace, DensitySpace>(
        mesh_->entireBody(), shape_disp_->get()->space(), params_[DENSITY].get()->space());
  }

  void resetAndApplyInitialConditions()
  {
    mechanics_->resetStates();

    auto& velo_field = *initial_states_[VELO].get();
    velo_field.setFromFieldFunction([](smith::tensor<double, dim> x) {
      auto v = x;
      v[0] = 0.5 * x[0];
      v[1] = -0.1 * x[1];
      return v;
    });

    mechanics_->setState(velo_name, velo_field);
  }

  double integrateForward()
  {
    resetAndApplyInitialConditions();
    double base_physics_qoi = 0.0;
    for (size_t m = 0; m < num_steps; ++m) {
      physics_->advanceTimestep(dt);
      base_physics_qoi += (*kinetic_energy_integrator_)(physics_->time(), physics_->shapeDisplacement(),
                                                        physics_->state(velo_name), physics_->parameter(DENSITY));
    }

    return base_physics_qoi;
  }

  void adjointBackward(smith::FiniteElementDual& shape_sensitivity,
                       std::vector<smith::FiniteElementDual>& parameter_sensitivities)
  {
    smith::FiniteElementDual velo_adjoint_load(physics_->state(velo_name).space(),
                                               physics_->state(velo_name).name() + "_adjoint_load");
    physics_->resetAdjointStates();
    while (physics_->cycle() > 0) {
      auto shape_sensitivity_op = smith::get<smith::DERIVATIVE>(
          (*kinetic_energy_integrator_)(physics_->time(), differentiate_wrt(physics_->shapeDisplacement()),
                                        physics_->state(velo_name), physics_->parameter(DENSITY)));
      shape_sensitivity += *assemble(shape_sensitivity_op);

      auto density_sensitivity_op = smith::get<smith::DERIVATIVE>(
          (*kinetic_energy_integrator_)(physics_->time(), physics_->shapeDisplacement(), physics_->state(velo_name),
                                        differentiate_wrt(physics_->parameter(DENSITY))));
      parameter_sensitivities[DENSITY] += *assemble(density_sensitivity_op);

      auto velo_sensivitity_op = smith::get<smith::DERIVATIVE>((*kinetic_energy_integrator_)(
          physics_->time(), physics_->shapeDisplacement(), smith::differentiate_wrt(physics_->state(velo_name)),
          physics_->parameter(DENSITY)));
      velo_adjoint_load = *assemble(velo_sensivitity_op);

      physics_->setAdjointLoad({{velo_name, velo_adjoint_load}});
      physics_->reverseAdjointTimestep();
      shape_sensitivity += physics_->computeTimestepShapeSensitivity();
      for (size_t param_index = 0; param_index < parameter_sensitivities.size(); ++param_index) {
        parameter_sensitivities[param_index] += physics_->computeTimestepSensitivity(param_index);
      }
    }
  }

  std::string velo_name = "solid_velocity";

  axom::sidre::DataStore datastore_;
  std::shared_ptr<smith::Mesh> mesh_;
  std::shared_ptr<gretl::DataStore> checkpointer_;

  std::unique_ptr<smith::FieldState> shape_disp_;
  std::vector<smith::FieldState> initial_states_;
  std::vector<smith::FieldState> params_;

  std::shared_ptr<smith::DifferentiablePhysics> mechanics_;
  std::shared_ptr<smith::BasePhysics> physics_;

  std::shared_ptr<smith::ScalarObjective> objective_;
  std::shared_ptr<smith::Functional<double(VectorSpace, VectorSpace, DensitySpace)>> kinetic_energy_integrator_;

  std::shared_ptr<smith::BoundaryConditionManager> bc_manager_;

  static constexpr double total_simulation_time = 0.005;
  static constexpr size_t num_steps = 10;
  static constexpr double dt = total_simulation_time / num_steps;
};

TEST_F(MeshFixture, TransientDynamicsBasePhysics)
{
  SMITH_MARK_FUNCTION;

  auto zero_bcs = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return 0.0; });
  bc_manager_->addEssential(std::set<int>{1}, zero_bcs, initial_states_[DISP].get()->space());

  double qoi = integrateForward();
  std::cout << "qoi = " << qoi << std::endl;

  size_t num_params = physics_->parameterNames().size();

  smith::FiniteElementDual shape_sensitivity(*shape_disp_->get_dual());
  std::vector<smith::FiniteElementDual> parameter_sensitivities;
  for (size_t p = 0; p < num_params; ++p) {
    parameter_sensitivities.emplace_back(*params_[p].get_dual());
  }

  adjointBackward(shape_sensitivity, parameter_sensitivities);

  auto state_sensitivities = physics_->computeInitialConditionSensitivity();
  for (auto name_and_state_sensitivity : state_sensitivities) {
    std::cout << name_and_state_sensitivity.first << " " << name_and_state_sensitivity.second.Norml2() << std::endl;
  }

  std::cout << shape_sensitivity.name() << " " << shape_sensitivity.Norml2() << std::endl;

  for (size_t p = 0; p < num_params; ++p) {
    std::cout << parameter_sensitivities[p].name() << " " << parameter_sensitivities[p].Norml2() << std::endl;
  }
}

TEST_F(MeshFixture, TransientDynamicsGretl)
{
  SMITH_MARK_FUNCTION;

  auto zero_bcs = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return 0.0; });
  bc_manager_->addEssential(std::set<int>{1}, zero_bcs, initial_states_[DISP].get()->space());

  resetAndApplyInitialConditions();

  auto all_fields = mechanics_->getFieldStatesAndParamStates();

  gretl::State<double> gretl_qoi =
      0.0 * smith::evaluateObjective(*objective_, *shape_disp_, {all_fields[F_VELO], all_fields[F_DENSITY]});

  std::string pv_dir = std::string("paraview_") + mechanics_->name();
  auto pv_writer = smith::createParaviewWriter(*mesh_, all_fields, pv_dir);
  pv_writer.write(mechanics_->cycle(), mechanics_->time(), all_fields);
  for (size_t m = 0; m < num_steps; ++m) {
    mechanics_->advanceTimestep(dt);
    all_fields = mechanics_->getFieldStatesAndParamStates();
    gretl_qoi =
        gretl_qoi + smith::evaluateObjective(*objective_, *shape_disp_, {all_fields[F_VELO], all_fields[F_DENSITY]});
    pv_writer.write(mechanics_->cycle(), mechanics_->time(), all_fields);
  }

  gretl::set_as_objective(gretl_qoi);
  std::cout << "qoi = " << gretl_qoi.get() << std::endl;

  checkpointer_->back_prop();

  for (auto s : initial_states_) {
    std::cout << s.get()->name() << " " << s.get()->Norml2() << " " << s.get_dual()->Norml2() << std::endl;
  }

  std::cout << shape_disp_->get()->name() << " " << shape_disp_->get()->Norml2() << " "
            << shape_disp_->get_dual()->Norml2() << std::endl;

  for (size_t p = 0; p < params_.size(); ++p) {
    std::cout << params_[p].get()->name() << " " << params_[p].get()->Norml2() << " " << params_[p].get_dual()->Norml2()
              << std::endl;
  }

  EXPECT_GT(smith::checkGradWrt(gretl_qoi, *shape_disp_, 0.01, 4, true), 0.8);
  EXPECT_GT(smith::checkGradWrt(gretl_qoi, initial_states_[DISP], 0.01, 4, true), 0.8);
  EXPECT_GT(smith::checkGradWrt(gretl_qoi, initial_states_[VELO], 0.01, 4, true), 0.8);
  EXPECT_GT(smith::checkGradWrt(gretl_qoi, initial_states_[DENSITY], 0.01, 4, true), 0.8);
}

TEST_F(MeshFixture, TransientConstantGravity)
{
  SMITH_MARK_FUNCTION;

  mechanics_->resetStates();
  auto all_fields = mechanics_->getFieldStatesAndParamStates();

  std::string pv_dir = std::string("paraview_") + mechanics_->name();
  std::cout << "Writing output to " << pv_dir << std::endl;
  auto pv_writer = smith::createParaviewWriter(*mesh_, all_fields, pv_dir);
  pv_writer.write(mechanics_->cycle(), mechanics_->time(), all_fields);
  double time = 0.0;
  for (size_t m = 0; m < num_steps; ++m) {
    double timestep = dt / double(num_steps);
    mechanics_->advanceTimestep(timestep);
    all_fields = mechanics_->getFieldStatesAndParamStates();
    pv_writer.write(mechanics_->cycle(), mechanics_->time(), all_fields);
    time += timestep;
  }

  double a_exact = gravity;
  double v_exact = gravity * time;
  double u_exact = 0.5 * gravity * time * time;

  smith::FunctionalObjective<dim, smith::Parameters<VectorSpace>> accel_error("accel_error", mesh_,
                                                                              smith::spaces({all_fields[ACCEL]}));
  accel_error.addBodyIntegral(smith::DependsOn<0>{}, mesh_->entireBodyName(),
                              [a_exact](auto /*t*/, auto /*X*/, auto A) {
                                auto a = smith::get<smith::VALUE>(A);
                                auto da0 = a[0];
                                auto da1 = a[1] - a_exact;
                                return da0 * da0 + da1 * da1;
                              });
  double a_err = accel_error.evaluate(smith::TimeInfo(0.0, 1.0, 0), shape_disp_->get().get(),
                                      smith::getConstFieldPointers({all_fields[ACCEL]}));
  EXPECT_NEAR(0.0, a_err, 1e-14);

  smith::FunctionalObjective<dim, smith::Parameters<VectorSpace>> velo_error("velo_error", mesh_,
                                                                             smith::spaces({all_fields[VELO]}));
  velo_error.addBodyIntegral(smith::DependsOn<0>{}, mesh_->entireBodyName(), [v_exact](auto /*t*/, auto /*X*/, auto V) {
    auto v = smith::get<smith::VALUE>(V);
    auto dv0 = v[0];
    auto dv1 = v[1] - v_exact;
    return dv0 * dv0 + dv1 * dv1;
  });
  double v_err = velo_error.evaluate(smith::TimeInfo(0.0, 1.0, 0), shape_disp_->get().get(),
                                     smith::getConstFieldPointers({all_fields[VELO]}));
  EXPECT_NEAR(0.0, v_err, 1e-14);

  smith::FunctionalObjective<dim, smith::Parameters<VectorSpace>> disp_error("disp_error", mesh_,
                                                                             smith::spaces({all_fields[DISP]}));
  disp_error.addBodyIntegral(smith::DependsOn<0>{}, mesh_->entireBodyName(), [u_exact](auto /*t*/, auto /*X*/, auto U) {
    auto u = smith::get<smith::VALUE>(U);
    auto du0 = u[0];
    auto du1 = u[1] - u_exact;
    return du0 * du0 + du1 * du1;
  });
  double u_err = disp_error.evaluate(smith::TimeInfo(0.0, 1.0, 0), shape_disp_->get().get(),
                                     smith::getConstFieldPointers({all_fields[DISP]}));
  EXPECT_NEAR(0.0, u_err, 1e-14);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
