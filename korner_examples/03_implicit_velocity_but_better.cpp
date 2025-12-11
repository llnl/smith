#include <iostream>

#include "gretl/data_store.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"

#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/functional_objective.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"

#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
// #include "smith/differentiable_numerics/solid_mechanics_state_advancer.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/tests/paraview_helper.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"

namespace smith {

smith::LinearSolverOptions solid_linear_options{.linear_solver = smith::LinearSolver::CG,
                                                //
                                                .preconditioner = smith::Preconditioner::HypreJacobi,
                                                .relative_tol = 1e-8,
                                                .absolute_tol = 1e-11,
                                                .max_iterations = 10000,
                                                .print_level = 1};

smith::NonlinearSolverOptions solid_nonlinear_opts{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                   .relative_tol = 1.0e-8,
                                                   .absolute_tol = 1.0e-11,
                                                   .max_iterations = 500,
                                                   .print_level = 1};

static constexpr int dim = 3;
static constexpr int order = 1;

using ShapeDispSpace = H1<1, dim>;
using VectorSpace = H1<order, dim>;
using ScalarParameterSpace = L2<0>;

struct ParameterizedNeoHookeanWithViscosity {
  using State = Empty;

  template <int d, typename DispGradType, typename VelGradType, typename BulkType, typename ShearType>
  SMITH_HOST_DEVICE auto operator()(State& /*state*/, const smith::tensor<DispGradType, d, d>& du_dX,
                                    const smith::tensor<VelGradType, d, d>& dv_dX, const BulkType& DeltaK,
                                    const ShearType& DeltaG) const
  {
    using std::log1p;
    constexpr auto I = Identity<d>();

    auto K_eff = K0 + get<0>(DeltaK);
    auto G_eff = G0 + get<0>(DeltaG);
    auto lambda = K_eff - (2.0 / d) * G_eff;

    auto F = du_dX + I;

    auto grad_v = dv_dX * inv(F);
    auto B_minus_I = du_dX * transpose(du_dX) + transpose(du_dX) + du_dX;
    auto logJ = log1p(detApIm1(du_dX));
    auto TK_elastic = lambda * logJ * I + G_eff * B_minus_I;

    auto D = 0.5 * (grad_v + transpose(grad_v));
    auto TK_viscous = 2.0 * eta * det(F) * D;

    auto TK = TK_elastic + TK_viscous;
    return dot(TK, inv(transpose(F)));
  }

  static constexpr int numParameters() { return 2; }

  double density;
  double K0;
  double G0;
  double eta;
};

class SolidMechanicsStateAdvancer2 : public StateAdvancer {
 public:
  SolidMechanicsStateAdvancer2(std::shared_ptr<DifferentiableSolver> solid_solver,
                               std::shared_ptr<DirichletBoundaryConditions> vector_bcs,
                               std::shared_ptr<WeakForm> weak_form, SecondOrderTimeIntegrationRule time_rule)
      : solver_(solid_solver), vector_bcs_(vector_bcs), weak_form_(weak_form), time_rule_(time_rule)
  {
  }

  enum STATE
  {
    DISPLACEMENT,
    VELOCITY,
    ACCELERATION
  };

  template <typename FirstParamSpace, typename... ParamSpaces>
  static std::vector<FieldState> createParams(gretl::DataStore& graph, const std::string& name,
                                              const std::vector<std::string>& param_names, const std::string& tag,
                                              size_t index = 0)
  {
    FieldState newParam = create_field_state(graph, FirstParamSpace{}, name + "_" + param_names[index], tag);
    std::vector<FieldState> end_spaces{};
    if constexpr (sizeof...(ParamSpaces) > 0) {
      end_spaces = createParams<ParamSpaces...>(graph, name, param_names, tag, ++index);
    }
    end_spaces.insert(end_spaces.begin(), newParam);
    return end_spaces;
  }

  template <int spatial_dim, typename ShapeDispSpace, typename VectorSpace, typename... ParamSpaces>
  static auto buildWeakFormAndStates(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<gretl::DataStore>& graph,
                                     SecondOrderTimeIntegrationRule time_rule, std::string physics_name,
                                     const std::vector<std::string>& param_names, double initial_time = 0.0)
  {
    auto shape_disp = create_field_state(*graph, ShapeDispSpace{}, physics_name + "_shape_displacement", mesh->tag());
    auto disp = create_field_state(*graph, VectorSpace{}, physics_name + "_displacement", mesh->tag());
    auto velo = create_field_state(*graph, VectorSpace{}, physics_name + "_velocity", mesh->tag());
    auto acceleration = create_field_state(*graph, VectorSpace{}, physics_name + "_acceleration", mesh->tag());
    auto time = graph->create_state<double, double>(initial_time);
    std::vector<FieldState> params =
        createParams<ParamSpaces...>(*graph, physics_name + "_param", param_names, mesh->tag());
    std::vector<FieldState> states{disp, velo, acceleration};

    // weak form unknowns are disp, disp_old, velo_old, accel_old
    using SolidWeakFormT = SecondOrderTimeDiscretizedWeakForm<
        spatial_dim, VectorSpace, Parameters<VectorSpace, VectorSpace, VectorSpace, VectorSpace, ParamSpaces...>>;
    auto input_spaces = spaces({states[DISPLACEMENT], states[DISPLACEMENT], states[VELOCITY], states[ACCELERATION]});
    auto param_spaces = spaces(params);
    input_spaces.insert(input_spaces.end(), param_spaces.begin(), param_spaces.end());

    auto solid_mechanics_weak_form =
        std::make_shared<SolidWeakFormT>(physics_name, mesh, time_rule, space(states[DISPLACEMENT]), input_spaces);

    return std::make_tuple(shape_disp, states, params, time, solid_mechanics_weak_form);
  }

  std::vector<FieldState> advanceState(const FieldState& shape_disp, const std::vector<FieldState>& states_old,
                                       const std::vector<FieldState>& params, const TimeInfo& time_info) const override
  {
    double dt = time_info.dt();
    size_t cycle = time_info.cycle();
    double final_time = time_info.time() + dt;

    TimeInfo final_time_info(final_time, dt, cycle);

    // evaluate initial guesses
    FieldState displacement_guess = states_old[DISPLACEMENT] + dt * states_old[VELOCITY];

    // input fields for solid_weak_form
    std::vector<FieldState> solid_inputs{states_old[DISPLACEMENT], states_old[VELOCITY], states_old[ACCELERATION]};
    solid_inputs.insert(solid_inputs.end(), params.begin(), params.end());

    auto displacement =
        solve(displacement_guess, shape_disp, solid_inputs, final_time_info, *weak_form_, *solver_, *vector_bcs_);

    std::vector<FieldState> states = states_old;

    states[DISPLACEMENT] = displacement;
    states[VELOCITY] = time_rule_.derivative(final_time_info, displacement, states_old[DISPLACEMENT],
                                             states_old[VELOCITY], states_old[ACCELERATION]);
    // states[VELOCITY] = (1.0 / final_time_info.dt()) * (displacement - states_old[DISPLACEMENT]);
    states[ACCELERATION] = time_rule_.second_derivative(final_time_info, displacement, states_old[DISPLACEMENT],
                                                        states_old[VELOCITY], states_old[ACCELERATION]);

    return states;
  }

  std::vector<ResultantState> computeResultants(const FieldState& shape_disp, const std::vector<FieldState>& states,
                                                const std::vector<FieldState>& states_old,
                                                const std::vector<FieldState>& params,
                                                const TimeInfo& time_info) const override
  {
    std::vector<FieldState> solid_inputs{states[DISPLACEMENT], states_old[DISPLACEMENT], states_old[VELOCITY],
                                         states_old[ACCELERATION]};
    solid_inputs.insert(solid_inputs.end(), params.begin(), params.end());
    return {evaluateWeakForm(weak_form_, time_info, shape_disp, solid_inputs, states[DISPLACEMENT])};
  }

 private:
  std::shared_ptr<DifferentiableSolver> solver_;
  std::shared_ptr<DirichletBoundaryConditions> vector_bcs_;
  std::shared_ptr<WeakForm> weak_form_;
  SecondOrderTimeIntegrationRule time_rule_;
};

template <int dim, typename ShapeDispSpace, typename VectorSpace, typename... ParamSpaces>
auto buildSolidMechanics(std::shared_ptr<smith::Mesh> mesh,
                         std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver,
                         smith::SecondOrderTimeIntegrationRule time_rule, std::string physics_name,
                         const std::vector<std::string>& param_names = {})
{
  auto graph = std::make_shared<gretl::DataStore>(100);
  auto [shape_disp, states, params, time, solid_mechanics_weak_form] =
      SolidMechanicsStateAdvancer2::buildWeakFormAndStates<dim, ShapeDispSpace, VectorSpace, ParamSpaces...>(
          mesh, graph, time_rule, physics_name, param_names);

  auto vector_bcs = std::make_shared<DirichletBoundaryConditions>(
      mesh->mfemParMesh(), space(states[SolidMechanicsStateAdvancer2::DISPLACEMENT]));

  auto state_advancer = std::make_shared<SolidMechanicsStateAdvancer2>(d_solid_nonlinear_solver, vector_bcs,
                                                                       solid_mechanics_weak_form, time_rule);

  auto physics =
      std::make_shared<DifferentiablePhysics>(mesh, graph, shape_disp, states, params, state_advancer, physics_name);

  return std::make_tuple(physics, solid_mechanics_weak_form, vector_bcs);
}
};  // namespace smith

int main(int argc, char* argv[])
{
  using namespace smith;
  smith::ApplicationManager applicationManager(argc, argv);
  SMITH_MARK_FUNCTION;

  double length = 1.0;
  double width = 0.04;
  int num_elements_x = 21;
  int num_elements_y = 3;
  int num_elements_z = 3;
  [[maybe_unused]] double elem_size = length / num_elements_x;

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
  smith::StateManager::initialize(datastore, "solid");
  auto mfem_shape = mfem::Element::QUADRILATERAL;
  mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(num_elements_x, num_elements_y, num_elements_z, mfem_shape, length, width, width),
      "mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));

  std::string physics_name = "solid";

  std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver =
      buildDifferentiableNonlinearSolve(solid_nonlinear_opts, solid_linear_options, *mesh);

  smith::SecondOrderTimeIntegrationRule time_rule(1.0);

  // warm-start.
  // implicit Newmark.

  auto [physics, weak_form, bcs] =
      buildSolidMechanics<dim, ShapeDispSpace, VectorSpace, ScalarParameterSpace, ScalarParameterSpace>(
          mesh, d_solid_nonlinear_solver, time_rule, physics_name, {"bulk", "shear"});

  bcs->setFixedVectorBCs<dim>(mesh->domain("right"));
  bcs->setVectorBCs<dim>(mesh->domain("left"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    bc[0] = -0.1 * t;
    bc[1] = -0.1 * t;
    return bc;
  });

  double E = 100.0;
  double nu = 0.25;
  auto K = E / (3.0 * (1.0 - 2 * nu));
  auto G = E / (2.0 * (1.0 + nu));
  using MaterialType = ParameterizedNeoHookeanWithViscosity;
  MaterialType material{.density = 10.0, .K0 = K, .G0 = G, .eta = 1.0e0};

  weak_form->addBodyIntegral(
      smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
      [material](const auto& /*time_info*/, auto /*X*/, auto u, auto v, auto /*a*/, auto bulk, auto shear) {
        MaterialType::State state;
        auto du_dX = get<DERIVATIVE>(u);
        auto Grad_v = get<DERIVATIVE>(v);
        auto pk_stress = material(state, du_dX, Grad_v, bulk, shear);
        // return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
        return smith::tuple{smith::zero{}, pk_stress};
      });

  auto shape_disp = physics->getShapeDispFieldState();
  auto params = physics->getFieldParams();
  auto states = physics->getInitialFieldStates();

  params[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) {
    double scaling = 1.0;
    return scaling * material.K0;
  });

  params[1].get()->setFromFieldFunction([=](smith::tensor<double, dim>) {
    double scaling = 1.0;
    return scaling * material.G0;
  });

  physics->resetStates();

  double time_increment = 1.0e-2;
  auto pv_writer = smith::createParaviewOutput(*mesh, physics->getFieldStatesAndParamStates(), physics_name);
  pv_writer.write(0, physics->time(), physics->getFieldStatesAndParamStates());
  for (size_t m = 0; m < 50; ++m) {
    if (mfem::Mpi::Root()) {
      std::cout << "Time Step: " << m << std::endl;
    }
    physics->advanceTimestep(time_increment);
    pv_writer.write(m + 1, physics->time(), physics->getFieldStatesAndParamStates());
  }

  TimeInfo time_info(physics->time() - time_increment, time_increment);

  auto final_states = physics->getFieldStates();
  auto previous_to_final_states = physics->getFieldStatesOld();

  auto state_advancer = physics->getStateAdvancer();
  printf("a\n");
  auto reactions =
      state_advancer->computeResultants(shape_disp, final_states, previous_to_final_states, params, time_info);

  printf("b\n");
  auto disp_squared = innerProduct(reactions[0], reactions[0]);

  gretl::set_as_objective(disp_squared);
  std::cout << "final disp norm2 = " << disp_squared.get() << std::endl;
}
