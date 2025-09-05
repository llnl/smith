#include <gtest/gtest.h>
#include "mfem.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/mesh.hpp"

#include "serac/differentiable_numerics/mechanics.hpp"
#include "serac/gretl/data_store.hpp"
#include "serac/physics/solid_weak_form.hpp"

#include "serac/differentiable_numerics/state_advancer.hpp"
#include "serac/differentiable_numerics/lumped_mass_weak_form.hpp"
#include "serac/differentiable_numerics/tests/paraview_helper.hpp"
#include "serac/differentiable_numerics/differentiable_utils.hpp"

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

std::vector<serac::FiniteElementState*> getStatePtrs(const std::vector<serac::FieldState>& field_states)
{
  std::vector<serac::FiniteElementState*> pointers;
  for (const auto& s : field_states) {
    pointers.push_back(s.get().get());
  }
  return pointers;
}

struct MeshFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  static constexpr int scalar_field_order = 1;
  using DensitySpace = serac::L2<scalar_field_order>;
  using VectorSpace = serac::H1<disp_order, dim>;

  using SolidMaterial = NeoHookeanWithFixedDensity;

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
    mesh = std::make_shared<serac::Mesh>(mfem::Mesh::MakeCartesian2D(2, 2, mfem_shape, true, length, width), MESHTAG, 0,
                                         0);
    // checkpointing graph
    checkpointer = std::make_shared<gretl::DataStore>(200);

    // create residual evaluator
    const double density = 1.0;
    std::string physics_name = "solid";

    shape_disp = std::make_unique<serac::FieldState>(
        create_field_state(*checkpointer, VectorSpace{}, physics_name + "_shape_displacement", mesh->tag()));
    auto disp = create_field_state(*checkpointer, VectorSpace{}, physics_name + "_displacement", mesh->tag());
    auto velo = create_field_state(*checkpointer, VectorSpace{}, physics_name + "_velocity", mesh->tag());
    auto accel = create_field_state(*checkpointer, VectorSpace{}, physics_name + "_acceleration", mesh->tag());
    auto density0 = create_field_state(*checkpointer, DensitySpace{}, physics_name + "_density", mesh->tag());
    serac::DoubleState fixed_dt = checkpointer->create_state<double, double>(1e-3);

    *density0.get() = density;

    initial_states = {disp, velo, accel};
    params = {density0};

    std::vector<serac::FieldState> states{disp, velo, accel};

    auto solid_mechanics_residual = serac::create_solid_weak_form<disp_order, dim, DensitySpace>(
        physics_name, mesh, getStatePtrs(states), getStatePtrs(params));

    SolidMaterial mat;
    mat.density0 = density;
    mat.K = 1.0;
    mat.G = 0.5;

    solid_mechanics_residual->setMaterial(serac::DependsOn<>{}, mesh->entireBodyName(), mat);

    // create mass evaluator and state in order to be able to create a diagonalized mass matrix
    std::string mass_residual_name = "mass";
    auto solid_mass_residual = serac::create_solid_mass_weak_form<VectorSpace::components, VectorSpace, DensitySpace>(
        mass_residual_name, mesh, *states[DISP].get(), *params[DENSITY].get());

    // specify dirichlet bcs
    auto bc_manager = std::make_shared<serac::BoundaryConditionManager>(mesh->mfemParMesh());
    auto zero_bcs = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return 0.0; });
    bc_manager->addEssential(std::set<int>{1}, zero_bcs, states[DISP].get()->space());

    std::shared_ptr<serac::StateAdvancer> time_integrator =
        std::make_shared<serac::LumpedMassExplicitNewmark>(solid_mechanics_residual, solid_mass_residual, bc_manager);
    auto dt_estimator = std::make_shared<serac::ConstantTimeStepEstimator>(fixed_dt);

    // construct mechanics
    mechanics = std::make_shared<serac::Mechanics>(mesh, checkpointer, *shape_disp, states, params, time_integrator,
                                                   dt_estimator);
    physics = mechanics;

    // kinetic energy integrator for qoi
    kinetic_energy_integrator = serac::create_kinetic_energy_integrator<VectorSpace, DensitySpace>(
        mesh->entireBody(), shape_disp->get()->space(), params[DENSITY].get()->space());
  }

  void resetAndApplyInitialConditions()
  {
    mechanics->resetStates();

    auto& velo_field = *initial_states[VELO].get();
    velo_field.setFromFieldFunction([](serac::tensor<double, dim> x) {
      auto v = x;
      v[0] = 4.0 * x[0];
      v[1] = -0.1 * x[1];
      return v;
    });

    mechanics->setState(velo_name, velo_field);
  }

  double integrateForward()
  {
    resetAndApplyInitialConditions();
    double lido_qoi = (*kinetic_energy_integrator)(physics->time(), physics->shapeDisplacement(),
                                                   physics->state(velo_name), physics->parameter(DENSITY));
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

    auto shape_sensitivity_op = serac::get<serac::DERIVATIVE>(
        (*kinetic_energy_integrator)(physics->time(), differentiate_wrt(physics->shapeDisplacement()),
                                     physics->state(velo_name), physics->parameter(DENSITY)));
    shape_sensitivity += *assemble(shape_sensitivity_op);

    auto density_sensitivity_op = serac::get<serac::DERIVATIVE>(
        (*kinetic_energy_integrator)(physics->time(), physics->shapeDisplacement(), physics->state(velo_name),
                                     differentiate_wrt(physics->parameter(DENSITY))));
    parameter_sensitivities[DENSITY] += *assemble(density_sensitivity_op);
  }

  std::string velo_name = "solid_velocity";

  axom::sidre::DataStore datastore_;
  std::shared_ptr<serac::Mesh> mesh;
  std::shared_ptr<gretl::DataStore> checkpointer;

  std::unique_ptr<serac::FieldState> shape_disp;
  std::vector<serac::FieldState> initial_states;
  std::vector<serac::FieldState> params;

  std::shared_ptr<serac::Mechanics> mechanics;
  std::shared_ptr<serac::BasePhysics> physics;

  std::shared_ptr<serac::Functional<double(VectorSpace, VectorSpace, DensitySpace)>> kinetic_energy_integrator;

  const double dt = 0.001;
  const size_t num_steps = 4;
};

TEST_F(MeshFixture, TRANSIENT_DYNAMICS_LIDO)
{
  SERAC_MARK_FUNCTION;

  integrateForward();

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
    std::cout << parameter_sensitivities[p].Norml2() << std::endl;
  }
}

TEST_F(MeshFixture, TRANSIENT_DYNAMICS_GRETL)
{
  SERAC_MARK_FUNCTION;
  resetAndApplyInitialConditions();

  auto all_fields = mechanics->getAllFieldStates();
  gretl::State<double> gretl_qoi = serac::compute_kinetic_energy(kinetic_energy_integrator, *shape_disp,
                                                                 all_fields[F_VELO], all_fields[F_DENSITY], 1.0);
  std::string pv_dir = std::string("paraview_") + mechanics->name();
  auto pv_writer = serac::createParaviewOutput(*mesh, all_fields, pv_dir);
  pv_writer.write(mechanics->cycle(), mechanics->time(), all_fields);
  for (size_t m = 0; m < num_steps; ++m) {
    mechanics->advanceTimestep(dt);
    all_fields = mechanics->getAllFieldStates();
    gretl_qoi = gretl_qoi + serac::compute_kinetic_energy(kinetic_energy_integrator, *shape_disp, all_fields[F_VELO],
                                                          all_fields[F_DENSITY], 1.0);
    pv_writer.write(mechanics->cycle(), mechanics->time(), all_fields);
  }

  set_as_objective(gretl_qoi);
  checkpointer->back_prop();

  for (auto s : initial_states) {
    std::cout << s.get()->name() << " " << s.get()->Norml2() << " " << s.get_dual()->Norml2() << std::endl;
  }

  std::cout << shape_disp->get()->name() << " " << shape_disp->get()->Norml2() << " "
            << shape_disp->get_dual()->Norml2() << std::endl;

  for (size_t p = 0; p < params.size(); ++p) {
    std::cout << params[p].get()->name() << " " << params[p].get()->Norml2() << " " << params[p].get_dual()->Norml2()
              << std::endl;
  }

  EXPECT_GT(serac::check_grad_wrt(gretl_qoi, *shape_disp, *checkpointer, 0.01, 4, true), 0.8);
  EXPECT_GT(serac::check_grad_wrt(gretl_qoi, initial_states[DISP], *checkpointer, 0.01, 4, true), 0.8);
  EXPECT_GT(serac::check_grad_wrt(gretl_qoi, initial_states[VELO], *checkpointer, 0.01, 4, true), 0.8);
  EXPECT_GT(serac::check_grad_wrt(gretl_qoi, initial_states[ACCEL], *checkpointer, 1.0, 4, true), 0.8);
  EXPECT_GT(serac::check_grad_wrt(gretl_qoi, initial_states[DENSITY], *checkpointer, 0.01, 4, true), 0.8);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
