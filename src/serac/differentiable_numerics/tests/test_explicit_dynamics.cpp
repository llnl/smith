#include <gtest/gtest.h>
#include "mfem.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/mesh.hpp"

#include "serac/gretl/data_store.hpp"
#include "serac/physics/solid_weak_form.hpp"
#include "serac/physics/functional_objective.hpp"

#include "serac/differentiable_numerics/lumped_mass_explicit_newmark_state_advancer.hpp"
#include "serac/differentiable_numerics/lumped_mass_weak_form.hpp"
#include "serac/differentiable_numerics/tests/paraview_helper.hpp"
#include "serac/differentiable_numerics/differentiable_utils.hpp"
#include "serac/differentiable_numerics/timestep_estimator.hpp"
#include "serac/differentiable_numerics/differentiable_physics.hpp"

// This tests the interface between the new serac::WeakForm with gretl and its conformity to the existing base_physics
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
  using State = serac::Empty;  ///< this material has no internal variables

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
  SERAC_HOST_DEVICE auto pkStress(State& /* state */, const serac::tensor<T, dim, dim>& du_dX) const
  {
    using std::log1p;
    constexpr auto I = serac::Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = dot(du_dX, serac::transpose(du_dX)) + serac::transpose(du_dX) + du_dX;

    auto logJ = log1p(serac::detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I
    auto TK = lambda * logJ * I + G * B_minus_I;

    // Pull back to Piola
    auto F = du_dX + I;
    return serac::dot(TK, serac::inv(serac::transpose(F)));
  }

  /// @brief interpolates density field
  SERAC_HOST_DEVICE auto density() const { return density0; }

  double K;  ///< bulk modulus
  double G;  ///< shear modulus
  double density0;
};

struct MeshFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  static constexpr int scalar_field_order = 1;
  using DensitySpace = serac::L2<scalar_field_order>;
  using VectorSpace = serac::H1<disp_order, dim>;

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
    serac::StateManager::initialize(datastore_, "solid_dynamics");

    // create mesh
    auto mfem_shape = mfem::Element::QUADRILATERAL;  // mfem::Element::TRIANGLE;
    double length = 0.5;
    double width = 2.0;
    mesh = std::make_shared<serac::Mesh>(mfem::Mesh::MakeCartesian2D(5, 5, mfem_shape, true, length, width), MESHTAG, 0,
                                         0);
    // checkpointing graph
    checkpointer_ = std::make_shared<gretl::DataStore>(200);

    // create residual evaluator
    const double density = 1.0;
    std::string physics_name = "solid";

    shape_disp = std::make_unique<serac::FieldState>(
        create_field_state(*checkpointer_, VectorSpace{}, physics_name + "_shape_displacement", mesh->tag()));
    auto disp = create_field_state(*checkpointer_, VectorSpace{}, physics_name + "_displacement", mesh->tag());
    auto velo = create_field_state(*checkpointer_, VectorSpace{}, physics_name + "_velocity", mesh->tag());
    auto accel = create_field_state(*checkpointer_, VectorSpace{}, physics_name + "_acceleration", mesh->tag());
    auto density0 = create_field_state(*checkpointer_, DensitySpace{}, physics_name + "_density", mesh->tag());

    *disp.get() = 0.0;
    *velo.get() = 0.0;
    *accel.get() = 0.0;
    *density0.get() = density;

    initial_states = {disp, velo, accel};
    params = {density0};
    std::vector<serac::FieldState> states{disp, velo, accel};

    auto solid_mechanics_residual = serac::create_solid_weak_form<disp_order, dim, DensitySpace>(
        physics_name, mesh, getConstFieldPointers(states), getConstFieldPointers(params));

    SolidMaterial mat;
    mat.density0 = density;
    mat.K = 1.0;
    mat.G = 0.5;

    solid_mechanics_residual->setMaterial(serac::DependsOn<>{}, mesh->entireBodyName(), mat);

    solid_mechanics_residual->addBodySource(mesh->entireBodyName(), [](auto /*time*/, auto X) {
      auto b = 0.0 * X;
      b[1] = gravity;
      return b;
    });

    // create mass evaluator and state in order to be able to create a diagonalized mass matrix
    std::string mass_residual_name = "mass";
    auto solid_mass_residual = serac::createSolidMassWeakForm<VectorSpace::components, VectorSpace, DensitySpace>(
        mass_residual_name, mesh, *states[DISP].get(), *params[DENSITY].get());

    // specify dirichlet bcs
    bc_manager = std::make_shared<serac::BoundaryConditionManager>(mesh->mfemParMesh());

    auto dt_estimator = std::make_shared<serac::ConstantTimeStepEstimator>(dt / double(dt_reduction));
    std::shared_ptr<serac::StateAdvancer> time_integrator =
        std::make_shared<serac::LumpedMassExplicitNewmarkStateAdvancer>(solid_mechanics_residual, solid_mass_residual,
                                                                        dt_estimator, bc_manager);

    // construct mechanics
    mechanics = std::make_shared<serac::DifferentiablePhysics>(mesh, checkpointer_, *shape_disp, states, params,
                                                               time_integrator, "mechanics");
    physics = mechanics;

    auto ke_objective = std::make_shared<serac::FunctionalObjective<dim, serac::Parameters<VectorSpace, DensitySpace>>>(
        "integrated_squared_temperature", mesh, serac::spaces({states[DISP], params[DENSITY]}));

    ke_objective->addBodyIntegral(serac::DependsOn<0, 1>(), mesh->entireBodyName(),
                                  [](auto /*t*/, auto /*X*/, auto U, auto Rho) {
                                    auto u = get<serac::VALUE>(U);
                                    return 0.5 * get<serac::VALUE>(Rho) * serac::inner(u, u);
                                  });
    objective = ke_objective;

    // kinetic energy integrator for qoi
    kinetic_energy_integrator = serac::createKineticEnergyIntegrator<VectorSpace, DensitySpace>(
        mesh->entireBody(), shape_disp->get()->space(), params[DENSITY].get()->space());
  }

  void resetAndApplyInitialConditions()
  {
    mechanics->resetStates();

    auto& velo_field = *initial_states[VELO].get();
    velo_field.setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto v = x;
      v[0] = 0.5 * x[0];
      v[1] = -0.1 * x[1];
      return v;
    });

    mechanics->setState(velo_name, velo_field);
  }

  double integrateForward()
  {
    resetAndApplyInitialConditions();
    double lido_qoi = 0.0;
    for (size_t m = 0; m < num_steps; ++m) {
      physics->advanceTimestep(dt);
      lido_qoi += (*kinetic_energy_integrator)(physics->time(), physics->shapeDisplacement(), physics->state(velo_name),
                                               physics->parameter(DENSITY));
    }

    return lido_qoi;
  }

  void adjointBackward(serac::FiniteElementDual& shape_sensitivity,
                       std::vector<serac::FiniteElementDual>& parameter_sensitivities)
  {
    serac::FiniteElementDual velo_adjoint_load(physics->state(velo_name).space(),
                                               physics->state(velo_name).name() + "_adjoint_load");
    physics->resetAdjointStates();
    while (physics->cycle() > 0) {
      auto shape_sensitivity_op = serac::get<serac::DERIVATIVE>(
          (*kinetic_energy_integrator)(physics->time(), differentiate_wrt(physics->shapeDisplacement()),
                                       physics->state(velo_name), physics->parameter(DENSITY)));
      shape_sensitivity += *assemble(shape_sensitivity_op);

      auto density_sensitivity_op = serac::get<serac::DERIVATIVE>(
          (*kinetic_energy_integrator)(physics->time(), physics->shapeDisplacement(), physics->state(velo_name),
                                       differentiate_wrt(physics->parameter(DENSITY))));
      parameter_sensitivities[DENSITY] += *assemble(density_sensitivity_op);

      auto velo_sensivitity_op = serac::get<serac::DERIVATIVE>((*kinetic_energy_integrator)(
          physics->time(), physics->shapeDisplacement(), serac::differentiate_wrt(physics->state(velo_name)),
          physics->parameter(DENSITY)));
      velo_adjoint_load = *assemble(velo_sensivitity_op);

      physics->setAdjointLoad({{velo_name, velo_adjoint_load}});
      physics->reverseAdjointTimestep();
      shape_sensitivity += physics->computeTimestepShapeSensitivity();
      for (size_t param_index = 0; param_index < parameter_sensitivities.size(); ++param_index) {
        parameter_sensitivities[param_index] += physics->computeTimestepSensitivity(param_index);
      }
    }
  }

  std::string velo_name = "solid_velocity";

  axom::sidre::DataStore datastore_;
  std::shared_ptr<serac::Mesh> mesh;
  std::shared_ptr<gretl::DataStore> checkpointer_;

  std::unique_ptr<serac::FieldState> shape_disp;
  std::vector<serac::FieldState> initial_states;
  std::vector<serac::FieldState> params;

  std::shared_ptr<serac::DifferentiablePhysics> mechanics;
  std::shared_ptr<serac::BasePhysics> physics;

  std::shared_ptr<serac::ScalarObjective> objective;
  std::shared_ptr<serac::Functional<double(VectorSpace, VectorSpace, DensitySpace)>> kinetic_energy_integrator;

  std::shared_ptr<serac::BoundaryConditionManager> bc_manager;

  const double dt = 1e-2;
  const size_t dt_reduction = 10;
  const size_t num_steps = 4;
};

TEST_F(MeshFixture, TRANSIENT_DYNAMICS_LIDO)
{
  SERAC_MARK_FUNCTION;

  auto zero_bcs = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return 0.0; });
  bc_manager->addEssential(std::set<int>{1}, zero_bcs, initial_states[DISP].get()->space());

  double qoi = integrateForward();
  std::cout << "qoi = " << qoi << std::endl;

  size_t num_params = physics->parameterNames().size();

  serac::FiniteElementDual shape_sensitivity(*shape_disp->get_dual());
  std::vector<serac::FiniteElementDual> parameter_sensitivities;
  for (size_t p = 0; p < num_params; ++p) {
    parameter_sensitivities.emplace_back(*params[p].get_dual());
  }

  adjointBackward(shape_sensitivity, parameter_sensitivities);

  auto state_sensitivities = physics->computeInitialConditionSensitivity();
  for (auto name_and_state_sensitivity : state_sensitivities) {
    std::cout << name_and_state_sensitivity.first << " " << name_and_state_sensitivity.second.Norml2() << std::endl;
  }

  std::cout << shape_sensitivity.name() << " " << shape_sensitivity.Norml2() << std::endl;

  for (size_t p = 0; p < num_params; ++p) {
    std::cout << parameter_sensitivities[p].name() << " " << parameter_sensitivities[p].Norml2() << std::endl;
  }
}

TEST_F(MeshFixture, TRANSIENT_DYNAMICS_GRETL)
{
  SERAC_MARK_FUNCTION;

  auto zero_bcs = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return 0.0; });
  bc_manager->addEssential(std::set<int>{1}, zero_bcs, initial_states[DISP].get()->space());

  resetAndApplyInitialConditions();

  auto all_fields = mechanics->getAllFieldStates();

  gretl::State<double> gretl_qoi =
      0.0 * serac::evaluateObjective(objective, *shape_disp, {all_fields[F_VELO], all_fields[F_DENSITY]});

  std::string pv_dir = std::string("paraview_") + mechanics->name();
  auto pv_writer = serac::createParaviewOutput(*mesh, all_fields, pv_dir);
  pv_writer.write(mechanics->cycle(), mechanics->time(), all_fields);
  for (size_t m = 0; m < num_steps; ++m) {
    for (size_t n = 0; n < dt_reduction; ++n) {
      mechanics->advanceTimestep(dt / double(dt_reduction));
    }
    all_fields = mechanics->getAllFieldStates();
    gretl_qoi =
        gretl_qoi + serac::evaluateObjective(objective, *shape_disp, {all_fields[F_VELO], all_fields[F_DENSITY]});
    pv_writer.write(mechanics->cycle(), mechanics->time(), all_fields);
  }

  gretl::set_as_objective(gretl_qoi);
  std::cout << "qoi = " << gretl_qoi.get() << std::endl;

  checkpointer_->back_prop();

  for (auto s : initial_states) {
    std::cout << s.get()->name() << " " << s.get()->Norml2() << " " << s.get_dual()->Norml2() << std::endl;
  }

  std::cout << shape_disp->get()->name() << " " << shape_disp->get()->Norml2() << " "
            << shape_disp->get_dual()->Norml2() << std::endl;

  for (size_t p = 0; p < params.size(); ++p) {
    std::cout << params[p].get()->name() << " " << params[p].get()->Norml2() << " " << params[p].get_dual()->Norml2()
              << std::endl;
  }

  EXPECT_GT(serac::check_grad_wrt(gretl_qoi, *shape_disp, *checkpointer_, 0.01, 4, true), 0.8);
  EXPECT_GT(serac::check_grad_wrt(gretl_qoi, initial_states[DISP], *checkpointer_, 0.01, 4, true), 0.8);
  EXPECT_GT(serac::check_grad_wrt(gretl_qoi, initial_states[VELO], *checkpointer_, 0.01, 4, true), 0.8);
  // EXPECT_GT(serac::check_grad_wrt(gretl_qoi, initial_states[ACCEL], *checkpointer_, 1.0, 4, true), 0.8);
  EXPECT_GT(serac::check_grad_wrt(gretl_qoi, initial_states[DENSITY], *checkpointer_, 0.01, 4, true), 0.8);
}

TEST_F(MeshFixture, TRANSIENT_CONSTANT_GRAVITY)
{
  SERAC_MARK_FUNCTION;

  mechanics->resetStates();
  auto all_fields = mechanics->getAllFieldStates();

  std::string pv_dir = std::string("paraview_") + mechanics->name();
  std::cout << "Writing output to " << pv_dir << std::endl;
  auto pv_writer = serac::createParaviewOutput(*mesh, all_fields, pv_dir);
  pv_writer.write(mechanics->cycle(), mechanics->time(), all_fields);
  double time = 0.0;
  for (size_t m = 0; m < dt_reduction * num_steps; ++m) {
    double timestep = dt / double(dt_reduction);
    mechanics->advanceTimestep(timestep);
    all_fields = mechanics->getAllFieldStates();
    pv_writer.write(mechanics->cycle(), mechanics->time(), all_fields);
    time += timestep;
  }

  double a_exact = gravity;
  double v_exact = gravity * time;
  double u_exact = 0.5 * gravity * time * time;

  serac::FunctionalObjective<dim, serac::Parameters<VectorSpace>> accel_error("accel_error", mesh,
                                                                              serac::spaces({all_fields[ACCEL]}));
  accel_error.addBodyIntegral(serac::DependsOn<0>{}, mesh->entireBodyName(), [a_exact](auto /*t*/, auto /*X*/, auto A) {
    auto a = serac::get<serac::VALUE>(A);
    auto da0 = a[0];
    auto da1 = a[1] - a_exact;
    return da0 * da0 + da1 * da1;
  });
  double a_err = accel_error.evaluate(serac::TimeInfo(0.0, 1.0, 0), shape_disp->get().get(),
                                      serac::getConstFieldPointers({all_fields[ACCEL]}));
  EXPECT_NEAR(0.0, a_err, 1e-14);

  serac::FunctionalObjective<dim, serac::Parameters<VectorSpace>> velo_error("velo_error", mesh,
                                                                             serac::spaces({all_fields[VELO]}));
  velo_error.addBodyIntegral(serac::DependsOn<0>{}, mesh->entireBodyName(), [v_exact](auto /*t*/, auto /*X*/, auto V) {
    auto v = serac::get<serac::VALUE>(V);
    auto dv0 = v[0];
    auto dv1 = v[1] - v_exact;
    return dv0 * dv0 + dv1 * dv1;
  });
  double v_err = velo_error.evaluate(serac::TimeInfo(0.0, 1.0, 0), shape_disp->get().get(),
                                     serac::getConstFieldPointers({all_fields[VELO]}));
  EXPECT_NEAR(0.0, v_err, 1e-14);

  serac::FunctionalObjective<dim, serac::Parameters<VectorSpace>> disp_error("disp_error", mesh,
                                                                             serac::spaces({all_fields[DISP]}));
  disp_error.addBodyIntegral(serac::DependsOn<0>{}, mesh->entireBodyName(), [u_exact](auto /*t*/, auto /*X*/, auto U) {
    auto u = serac::get<serac::VALUE>(U);
    auto du0 = u[0];
    auto du1 = u[1] - u_exact;
    return du0 * du0 + du1 * du1;
  });
  double u_err = disp_error.evaluate(serac::TimeInfo(0.0, 1.0, 0), shape_disp->get().get(),
                                     serac::getConstFieldPointers({all_fields[DISP]}));
  EXPECT_NEAR(0.0, u_err, 1e-14);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
