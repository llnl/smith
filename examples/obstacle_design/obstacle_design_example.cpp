// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file obstacle_design_example.cpp
 *
 * @brief Obstacle Design example
 *
 * Intended to show how to solve a problem with the HomotopySolver.
 * The example problem solved is an inertia relief problem.
 */

#include <format>

#include "smith/smith.hpp"

#include "mfem.hpp"

// ContinuationSolver headers
#include "continuationsolvers/problems/ObstacleProblems.hpp"
#include "continuationsolvers/problems/OptProblems.hpp"
#include "continuationsolvers/problems/MPECProblems.hpp"
#include "continuationsolvers/solvers/MPECSolver.hpp"

#include "axom/sidre.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;
static constexpr int dim = 2;
static constexpr int order = 1;

using StateSpace = smith::H1<order>;
using ObstacleSpace = smith::H1<order>;
using TrialSpace = ObstacleSpace;
using ObjectiveT = smith::FunctionalObjective<dim, smith::Parameters<StateSpace, ObstacleSpace>>;
using WeakFormT = smith::FunctionalWeakForm<dim, TrialSpace, smith::Parameters<StateSpace, ObstacleSpace>>;

class ParaviewWriter {
 public:
  using StateVecs = std::vector<std::shared_ptr<smith::FiniteElementState>>;
  using DualVecs = std::vector<std::shared_ptr<smith::FiniteElementDual>>;

  ParaviewWriter(std::unique_ptr<mfem::ParaViewDataCollection> pv_, const StateVecs& states_)
      : pv(std::move(pv_)), states(states_)
  {
  }

  ParaviewWriter(std::unique_ptr<mfem::ParaViewDataCollection> pv_, const StateVecs& states_, const StateVecs& duals_)
      : pv(std::move(pv_)), states(states_), dual_states(duals_)
  {
  }

  void write(int step, double time, const std::vector<smith::FiniteElementState const*>& current_states)
  {
    SMITH_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(current_states.size() != states.size(), "wrong number of output states to write");

    for (size_t n = 0; n < states.size(); ++n) {
      auto& state = states[n];
      *state = *current_states[n];
      state->gridFunction();
    }

    pv->SetCycle(step);
    pv->SetTime(time);
    pv->Save();
  }

 private:
  std::unique_ptr<mfem::ParaViewDataCollection> pv;
  StateVecs states;
  StateVecs dual_states;
};

class SmithObstacleDesignProblem : public ObstacleDesignProblem {
 protected:
  std::vector<double> jacobian_weights_ = {0.0, 0.0};
  smith::TimeInfo time_info_;
  std::shared_ptr<smith::Mesh> mesh_;
  std::shared_ptr<ObjectiveT> design_objective_;
  std::shared_ptr<WeakFormT> du_objective_;
  std::shared_ptr<WeakFormT> dth_objective_;
  std::vector<smith::FiniteElementState*> states_;
  std::unique_ptr<smith::FiniteElementState> shape_disp_;  // shape displacement
  std::shared_ptr<mfem::HypreParMatrix> HuuE_;
  std::shared_ptr<mfem::HypreParMatrix> HuthE_;
  std::shared_ptr<mfem::HypreParMatrix> HththE_;

 public:
  SmithObstacleDesignProblem(ParamOptProblem* paramopt, std::vector<smith::FiniteElementState*> states,
                             std::shared_ptr<smith::Mesh> mesh, std::shared_ptr<ObjectiveT> design_objective,
                             std::shared_ptr<WeakFormT> du_objective, std::shared_ptr<WeakFormT> dth_objective);
  double E(const mfem::Vector& u, const mfem::Vector& p, const mfem::Vector& theta, int& eval_err) override;
  void DuE(const mfem::Vector& u, const mfem::Vector& p, const mfem::Vector& theta, mfem::Vector& gradE) override;
  void DthE(const mfem::Vector& u, const mfem::Vector& p, const mfem::Vector& theta, mfem::Vector& gradE) override;
  mfem::Operator* DththE(const mfem::Vector& u, const mfem::Vector& p, const mfem::Vector& theta) override;
  mfem::Operator* DuthE(const mfem::Vector& u, const mfem::Vector& p, const mfem::Vector& theta) override;
  mfem::Operator* DuuE(const mfem::Vector& u, const mfem::Vector& p, const mfem::Vector& theta) override;
};

double fRhs(const mfem::Vector& x);
double flat_obstacle(const mfem::Vector& x);

int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  // Command line arguments
  // Mesh options
  double xlength = 1.0;
  double ylength = 1.0;
  int nx = 10;
  int ny = 10;
  int serial_refinement = 0;
  int parallel_refinement = 0;
  int visualize = 0;

  // Solver options
  double nonlinear_solve_tol = 1e-5;
  int nonlinear_solve_maxiter = 30;
  // Handle command line arguments
  axom::CLI::App app{"Obstacle design."};
  // Mesh options
  app.add_option("--xlength", xlength, "extent along x-axis")
      ->default_val("1.0")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--ylength", ylength, "extent along y-axis")
      ->default_val("1.0")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--serial-refinement", serial_refinement, "Serial refinement steps")
      ->default_val("0")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--parallel-refinement", parallel_refinement, "Parallel refinement steps")
      ->default_val("0")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--visualize", visualize, "solution visualization")
      ->default_val("0")  // Matches value set above
      ->check(axom::CLI::Range(0, 1));
  app.set_help_flag("--help");

  CLI11_PARSE(app, argc, argv);

  int nprocs;
  int myid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::sidre::DataStore datastore;
  const std::string simulation_tag = "obstacle_design";
  const std::string mesh_tag = simulation_tag + "mesh";
  smith::StateManager::initialize(datastore, simulation_tag + "_data");

  std::shared_ptr<smith::Mesh> mesh;

  bool generate_edges = false;
  mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian2D(nx, ny, element_shape, generate_edges, xlength, ylength), mesh_tag, serial_refinement,
      parallel_refinement);

  /* TODO:
   * 1) utilize a weak form to define the objective of the "lower level" obstacle problem
   *    1/2 u^T K u - f^T u
   * 2) utilize FunctionalObjective + WeakForm to define the "upper level" design objective + gradient + Hessian
   * */
  smith::FiniteElementState disp = smith::StateManager::newState(StateSpace{}, "displacement", mesh->tag());
  smith::FiniteElementState obstacle = smith::StateManager::newState(ObstacleSpace{}, "obstacle", mesh->tag());
  std::unique_ptr<smith::FiniteElementState> shape_disp =
      std::make_unique<smith::FiniteElementState>(mesh->newShapeDisplacement());

  std::vector<smith::FiniteElementState> states;
  std::vector<smith::FiniteElementState> params;
  states = {disp};
  params = {obstacle};

  std::vector<smith::FiniteElementState*> nonconst_states = {&disp, &obstacle};

  //// weak_form to define the "lower level" obstacle problem
  // std::string physics_name = "elasticity";
  // auto elasticity_weak_form = std::make_shared<WeakFormT>(physics_name, mesh, states[0].space(), getSpaces(params));
  // elasticity_weak_form->addBodyIntegral(mesh->entireBodyName(), [](auto /*t*/, auto /*X*/, auto U) {
  //     auto gradu = smith::get<smith::DERIVATIVE>(U);
  ////  return smith::tuple{constant_force, 0.0 * smith::get<smith::DERIVATIVE>(x)};
  //    return smith::tuple{0.5 * smith::inner(gradu, gradu), gradu}; // complete me!
  //});
  // NOTE: not going to use this idea for now. It seems more questionable to me that this adds much value
  // also does the solid weak form reduce to what we want in the case of a H1 field. Is it assumed

  // define objective in terms of displacement/deformation field and the obstacle
  // challenge: how to incorporate the pressure as a design parameter into a functional objective
  // given that the pressure is a dual field?
  double time = 0.0;
  double dt = 1.0;
  smith::TimeInfo time_info(time, dt, 0);
  auto all_states = getConstFieldPointers(states, params);
  auto objective_states = {all_states[0], all_states[1]};
  ObjectiveT::SpacesT space_ptrs{&disp.space(), &obstacle.space()};

  auto design_objective = std::make_shared<ObjectiveT>("design_objective", mesh, space_ptrs);

  design_objective->addBodyIntegral(
      smith::DependsOn<1>{}, mesh->entireBodyName(), [](double /*t*/, auto /*X*/, auto OBSTACLE) {
        return 0.5 * smith::get<smith::VALUE>(OBSTACLE) * smith::get<smith::VALUE>(OBSTACLE);
      });
  params[0] = 1.0;

  // WeakFormT: name, mesh, trial space, const space pointers to those spaces that the weak form can depend on
  auto dEdth_weak_form = std::make_shared<WeakFormT>("design_obj_design_residual", mesh, obstacle.space(), space_ptrs);
  auto dEdu_weak_form = std::make_shared<WeakFormT>("design_obj_disp_residual", mesh, disp.space(), space_ptrs);
  dEdth_weak_form->addBodyIntegral(smith::DependsOn<1>{}, mesh->entireBodyName(),
                                   [](double /*t*/, auto /*X*/, auto OBSTACLE) {
                                     auto obstacle = get<smith::VALUE>(OBSTACLE);
                                     return smith::tuple{obstacle, smith::zero{}};
                                   });

  dEdu_weak_form->addBodyIntegral(smith::DependsOn<>{}, mesh->entireBodyName(),
                                  [](double /*t*/, auto /*X*/) { return smith::tuple{smith::zero{}, smith::zero{}}; });

  auto res_vector = dEdth_weak_form->residual(time_info, shape_disp.get(), objective_states);
  for (int i = 0; i < res_vector.Size(); i++) {
    std::cout << "weak_form_" << i << " = " << res_vector(i) << std::endl;
  }
  std::vector<double> jacobian_weights = {0.0, 1.0};
  auto HE_thth = dEdth_weak_form->jacobian(time_info, shape_disp.get(), objective_states, jacobian_weights);
  auto HE_thu = dEdu_weak_form->jacobian(time_info, shape_disp.get(), objective_states, jacobian_weights);
  jacobian_weights[0] = 1.0;
  jacobian_weights[1] = 0.0;
  auto HE_uth = dEdth_weak_form->jacobian(time_info, shape_disp.get(), objective_states, jacobian_weights);
  auto HE_uu = dEdu_weak_form->jacobian(time_info, shape_disp.get(), objective_states, jacobian_weights);

  auto [Vh, _] = smith::generateParFiniteElementSpace<StateSpace>(&mesh->mfemParMesh());
  mfem::Array<int> ess_tdof_list;
  int dimU = Vh->GetTrueVSize();
  mfem::Vector uDC(dimU);
  uDC = 0.0;
  ParamObstacleProblem problem(Vh.get(), &fRhs, &flat_obstacle, ess_tdof_list, uDC);
  SmithObstacleDesignProblem smithdesignproblem(&problem, nonconst_states, mesh, design_objective, dEdu_weak_form,
                                                dEdth_weak_form);
  int dimPrimal = smithdesignproblem.GetDimU();
  mfem::Vector X0(dimPrimal);
  X0 = 0.0;
  mfem::Vector Xf(dimPrimal);
  Xf = 0.0;
  MPECSolver designoptimizer(&smithdesignproblem);
  designoptimizer.SetTol(nonlinear_solve_tol);
  designoptimizer.SetBarrierParameter(1.e-3);
  designoptimizer.SetMaxIter(nonlinear_solve_maxiter);
  designoptimizer.CheckLinearSystemResiduals();
  designoptimizer.RegularizePrimalHessian(1.e-10);
  designoptimizer.Mult(X0, Xf);

  auto conststates = getConstFieldPointers(states, params);
  auto vis_states = {conststates[0]};
  auto writer = createParaviewWriter(mesh->mfemParMesh(), vis_states, "obstacledesign");

  if (visualize) {
    mfem::Vector uf(Xf, 0, dimU);
    states[0] = uf;
    writer.write(0, 0.0, vis_states);
  }
}

double fRhs(const mfem::Vector& x)
{
  double fx = 0.;
  fx = 0.2 - 2.0 * (std::pow(x(0), 3) - 1.5 * std::pow(x(0), 2.) - 6 * x(0) + 3.) +
       (1. + std::pow(2. * M_PI, 2)) * std::cos(2. * M_PI * x(0));
  return fx;
}

double flat_obstacle(const mfem::Vector& /*x*/) { return 0.0; }

SmithObstacleDesignProblem::SmithObstacleDesignProblem(ParamOptProblem* paramopt,
                                                       std::vector<smith::FiniteElementState*> states,
                                                       std::shared_ptr<smith::Mesh> mesh,
                                                       std::shared_ptr<ObjectiveT> design_objective,
                                                       std::shared_ptr<WeakFormT> du_objective,
                                                       std::shared_ptr<WeakFormT> dth_objective)
    : ObstacleDesignProblem(paramopt), time_info_(0.0, 0.0, 0)
{
  states_.resize(states.size());
  std::copy(states.begin(), states.end(), states_.begin());
  mesh_ = mesh;
  shape_disp_ = std::make_unique<smith::FiniteElementState>(mesh_->newShapeDisplacement());
  design_objective_ = design_objective;
  du_objective_ = du_objective;
  dth_objective_ = dth_objective;
}

double SmithObstacleDesignProblem::E(const mfem::Vector& u, const mfem::Vector& /*p*/, const mfem::Vector& theta,
                                     int& eval_err)
{
  states_[0]->Set(1.0, u);
  states_[1]->Set(1.0, theta);
  eval_err = 0;
  return design_objective_->evaluate(time_info_, shape_disp_.get(), smith::getConstFieldPointers(states_));
}

void SmithObstacleDesignProblem::DuE(const mfem::Vector& u, const mfem::Vector& /*p*/, const mfem::Vector& theta,
                                     mfem::Vector& gradE)
{
  states_[0]->Set(1.0, u);
  states_[1]->Set(1.0, theta);
  auto res_vector = du_objective_->residual(time_info_, shape_disp_.get(), smith::getConstFieldPointers(states_));
  gradE.Add(1.0, res_vector);
}

void SmithObstacleDesignProblem::DthE(const mfem::Vector& u, const mfem::Vector& /*p*/, const mfem::Vector& theta,
                                      mfem::Vector& gradE)
{
  states_[0]->Set(1.0, u);
  states_[1]->Set(1.0, theta);
  auto res_vector = dth_objective_->residual(time_info_, shape_disp_.get(), smith::getConstFieldPointers(states_));
  gradE.Add(1.0, res_vector);
  gradE = 0.0;
}

mfem::Operator* SmithObstacleDesignProblem::DththE(const mfem::Vector& u, const mfem::Vector& /*p*/,
                                                   const mfem::Vector& theta)
{
  states_[0]->Set(1.0, u);
  states_[1]->Set(1.0, theta);
  jacobian_weights_[0] = 0.0;
  jacobian_weights_[1] = 1.0;
  auto HE_thth_uniq =
      dth_objective_->jacobian(time_info_, shape_disp_.get(), smith::getConstFieldPointers(states_), jacobian_weights_);
  HththE_.reset(HE_thth_uniq.release());
  return HththE_.get();
}

mfem::Operator* SmithObstacleDesignProblem::DuthE(const mfem::Vector& u, const mfem::Vector& /*p*/,
                                                  const mfem::Vector& theta)
{
  states_[0]->Set(1.0, u);
  states_[1]->Set(1.0, theta);
  jacobian_weights_[0] = 0.0;
  jacobian_weights_[1] = 1.0;
  auto HE_uth_uniq =
      du_objective_->jacobian(time_info_, shape_disp_.get(), smith::getConstFieldPointers(states_), jacobian_weights_);
  HuthE_.reset(HE_uth_uniq.release());
  return HuthE_.get();
}

mfem::Operator* SmithObstacleDesignProblem::DuuE(const mfem::Vector& u, const mfem::Vector& /*p*/,
                                                 const mfem::Vector& theta)
{
  states_[0]->Set(1.0, u);
  states_[1]->Set(1.0, theta);
  jacobian_weights_[0] = 1.0;
  jacobian_weights_[1] = 0.0;
  auto HE_uu_uniq =
      du_objective_->jacobian(time_info_, shape_disp_.get(), smith::getConstFieldPointers(states_), jacobian_weights_);
  HuuE_.reset(HE_uu_uniq.release());
  return HuuE_.get();
}
