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
 * Intended to show how to solve an obstacle design with
 * ContinuationSolvers' MPECSolver.
 * The design problem contains regularized obstacle problem constraints.
 */

#include <cmath>
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

using FESpace = smith::H1<order>;
using VectorSpace = smith::H1<order, dim>;
using StateSpace = FESpace;
using PressureSpace = FESpace;
using ObstacleSpace = FESpace;
using TrialSpace = FESpace;
using ObjectiveT = smith::FunctionalObjective<dim, smith::Parameters<StateSpace, PressureSpace, ObstacleSpace>>;
using WeakFormT =
    smith::FunctionalWeakForm<dim, TrialSpace, smith::Parameters<StateSpace, PressureSpace, ObstacleSpace>>;
using WeakFormT2 =
    smith::FunctionalWeakForm<dim, VectorSpace, smith::Parameters<StateSpace, PressureSpace, ObstacleSpace>>;

class SmithObstacleDesignProblem : public ObstacleDesignProblem {
 protected:
  std::vector<double> jacobian_weights_ = {0.0, 0.0, 0.0};  // weights for weak_form_->jacobian calls
  smith::TimeInfo time_info_;                               // time info for constraint and weak_form function calls
  std::shared_ptr<smith::Mesh> mesh_;
  std::shared_ptr<ObjectiveT> design_objective_;  // obstacle design objective function
  std::shared_ptr<WeakFormT>
      objective_grad_displacement_;  // obstacle design objective gradient with respect to displacement
  std::shared_ptr<WeakFormT> objective_grad_pressure_;  // obstacle design objective gradient with respect to pressure
  std::shared_ptr<WeakFormT>
      objective_grad_obstacle_;          // obstacle design objective gradient with respect to obstacle design field
  std::vector<smith::FieldPtr> states_;  // optimization state variables
  std::unique_ptr<smith::FiniteElementState> shape_disp_;  // shape displacement
  std::shared_ptr<mfem::HypreParMatrix> HuuE_;  // Hessian (u,u) of obstacle design objective, u = "displacement"
  std::shared_ptr<mfem::HypreParMatrix>
      HupE_;  // Hessian (u,p) of obstacle design objective, u = "displacement", p = "pressure"
  std::shared_ptr<mfem::HypreParMatrix>
      HuthE_;  // Hessian (u, th) of obstacle design objective, th = "theta" = "obstacle design"
  std::shared_ptr<mfem::HypreParMatrix> HppE_;  // Hessian (p,p) of obstacle design objective, p = "pressure"
  std::shared_ptr<mfem::HypreParMatrix>
      HpthE_;  // Hessian (p,th) of obstacle design objective, p = "pressure", th = "obstacle design"
  std::shared_ptr<mfem::HypreParMatrix> HththE_;  // Hessian (th,th) of obstacle design objective
 public:
  SmithObstacleDesignProblem(ParamObstacleProblem* parametrized_obst_problem,
                             std::vector<smith::FiniteElementState*> states, std::shared_ptr<smith::Mesh> mesh,
                             std::shared_ptr<ObjectiveT> design_objective,
                             std::shared_ptr<WeakFormT> objective_grad_displacement,
                             std::shared_ptr<WeakFormT> objective_grad_pressure,
                             std::shared_ptr<WeakFormT> objective_grad_obstacle);
  double E(const mfem::Vector& displacement, const mfem::Vector& pressure, const mfem::Vector& obstacle,
           int& eval_err) override;
  void DuE(const mfem::Vector& displacement, const mfem::Vector& pressure, const mfem::Vector& obstacle,
           mfem::Vector& displacementGradE) override;
  void DpE(const mfem::Vector& displacement, const mfem::Vector& pressure, const mfem::Vector& obstacle,
           mfem::Vector& pressureGradE) override;
  void DthE(const mfem::Vector& displacement, const mfem::Vector& pressure, const mfem::Vector& obstacle,
            mfem::Vector& obstacleGradE) override;
  mfem::Operator* DuuE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                       const mfem::Vector& obstacle) override;
  mfem::Operator* DupE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                       const mfem::Vector& obstacle) override;
  mfem::Operator* DuthE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                        const mfem::Vector& obstacle) override;
  mfem::Operator* DppE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                       const mfem::Vector& obstacle) override;
  mfem::Operator* DpthE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                        const mfem::Vector& obstacle) override;
  mfem::Operator* DththE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                         const mfem::Vector& obstacle) override;
};

double obstacle_forcing(const mfem::Vector& x);
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
  double gamma0 = 1.e0;  // scaling for design objective component: \gamma_0 / 2 * \int_{\Omega} \theta * \theta dV
  double gamma1 =
      1.e-4;  // scaling for design objective component: \gamma_1 / 2 * \int_{\Omega} ||\nabla \theta||_{2}^2 dV
  double gamma2 = 1.e-6;  // scaling for design objective component: \gamma_2 / 2 * \int_{\Omega} p * p dV
  double gamma3 = 1.e-3;  // scaling for design objective component: \gamma_3 / 2 * \int_{\Omega} ||\nabla p||_{2}^2 dV

  // Nonlinear solver options
  double tol = 1e-6;
  int maxiter = 100;
  // Handle command line arguments
  axom::CLI::App app{"Obstacle design."};
  // Mesh options
  app.add_option("--xlength", xlength, "extent of problem domain along x-axis")
      ->default_val("1.0")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--ylength", ylength, "extent of problem domain along y-axis")
      ->default_val("1.0")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--serial-refinement", serial_refinement, "Serial mesh refinement steps")
      ->default_val("0")  // Matches value set above
      ->check(axom::CLI::NonNegativeNumber);
  app.add_option("--parallel-refinement", parallel_refinement, "Parallel mesh refinement steps")
      ->default_val("0")  // Matches value set above
      ->check(axom::CLI::NonNegativeNumber);
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

  smith::FiniteElementState displacement = smith::StateManager::newState(StateSpace{}, "displacement", mesh->tag());
  smith::FiniteElementState pressure = smith::StateManager::newState(PressureSpace{}, "pressure", mesh->tag());
  smith::FiniteElementState obstacle = smith::StateManager::newState(ObstacleSpace{}, "obstacle", mesh->tag());

  std::vector<smith::FiniteElementState> states;
  states = {displacement, pressure, obstacle};
  obstacle = 1.0;
  std::vector<smith::FiniteElementState*> state_ptrs = {&displacement, &pressure, &obstacle};

  auto const_state_ptrs = getConstFieldPointers(states);
  ObjectiveT::SpacesT space_ptrs{&displacement.space(), &pressure.space(), &obstacle.space()};

  auto design_objective = std::make_shared<ObjectiveT>("design_objective", mesh, space_ptrs);

  design_objective->addBodyIntegral(
      smith::DependsOn<1, 2>{}, mesh->entireBodyName(),
      [gamma0, gamma1, gamma2, gamma3](double /*t*/, auto /*X*/, auto PRESSURE, auto OBSTACLE) {
        auto fo1 = 0.5 * gamma0 * smith::get<smith::VALUE>(OBSTACLE) * smith::get<smith::VALUE>(OBSTACLE);
        auto fo2 = 0.5 * gamma1 *
                   smith::inner(smith::get<smith::DERIVATIVE>(OBSTACLE), smith::get<smith::DERIVATIVE>(OBSTACLE));
        auto fp1 = 0.5 * gamma2 * smith::get<smith::VALUE>(PRESSURE) * smith::get<smith::VALUE>(PRESSURE);
        auto fp2 = 0.5 * gamma3 *
                   smith::inner(smith::get<smith::DERIVATIVE>(PRESSURE), smith::get<smith::DERIVATIVE>(PRESSURE));
        return fo1 + fo2 + fp1 + fp2;
      });

  // weak form for gradient (w.r.t. displacement) and subsequent Hessian callbacks of design objective
  auto weak_form_objective_grad_displacement =
      std::make_shared<WeakFormT>("design_obj_disp_residual", mesh, displacement.space(), space_ptrs);
  weak_form_objective_grad_displacement->addBodyIntegral(
      smith::DependsOn<>{}, mesh->entireBodyName(),
      [](double /*t*/, auto /*X*/) { return smith::tuple{smith::zero{}, smith::zero{}}; });

  // weak form for gradient (w.r.t. pressure) and subsequent Hessian callbacks of design objective
  auto weak_form_objective_grad_pressure =
      std::make_shared<WeakFormT>("design_obj_pres_residual", mesh, pressure.space(), space_ptrs);
  weak_form_objective_grad_pressure->addBodyIntegral(smith::DependsOn<1>{}, mesh->entireBodyName(),
                                                     [gamma2, gamma3](double /*t*/, auto /*X*/, auto PRESSURE) {
                                                       auto resp1 = gamma2 * smith::get<smith::VALUE>(PRESSURE);
                                                       auto resp2 = gamma3 * smith::get<smith::DERIVATIVE>(PRESSURE);
                                                       return smith::tuple{resp1, resp2};
                                                     });

  // weak form for gradient (w.r.t. obstacle design) and subsequent Hessian callbacks of design objective
  auto weak_form_objective_grad_design =
      std::make_shared<WeakFormT>("design_obj_design_residual", mesh, obstacle.space(), space_ptrs);
  weak_form_objective_grad_design->addBodyIntegral(smith::DependsOn<2>{}, mesh->entireBodyName(),
                                                   [gamma0, gamma1](double /*t*/, auto /*X*/, auto OBSTACLE) {
                                                     auto reso1 = gamma0 * smith::get<smith::VALUE>(OBSTACLE);
                                                     auto reso2 = gamma1 * smith::get<smith::DERIVATIVE>(OBSTACLE);
                                                     return smith::tuple{reso1, reso2};
                                                   });

  auto [Vh, unused] = smith::generateParFiniteElementSpace<StateSpace>(&mesh->mfemParMesh());
  (void)unused;
  int dimU = Vh->GetTrueVSize();
  ParamObstacleProblem problem(Vh.get(), &obstacle_forcing, &flat_obstacle);
  SmithObstacleDesignProblem smithdesignproblem(&problem, state_ptrs, mesh, design_objective,
                                                weak_form_objective_grad_displacement,
                                                weak_form_objective_grad_pressure, weak_form_objective_grad_design);
  int dimPrimal = smithdesignproblem.GetDimU();  // dim(displacement) + dim(pressure) + dim(obstacle) + ...
  mfem::Vector X0(dimPrimal);
  X0 = 0.0;
  mfem::Vector Xf(dimPrimal);
  Xf = 0.0;
  MPECSolver designoptimizer(&smithdesignproblem);
  designoptimizer.SetTol(tol);
  designoptimizer.SetMaxIter(maxiter);
  designoptimizer.SetPrintLevel(2);
  designoptimizer.EnableMassWeightedNorms();
  designoptimizer.Mult(X0, Xf);

  std::vector<const smith::FiniteElementState*> vis_states{const_state_ptrs[0], const_state_ptrs[1],
                                                           const_state_ptrs[2]};
  auto writer = createParaviewWriter(mesh->mfemParMesh(), vis_states, "obstacledesign");

  if (visualize) {
    mfem::Vector displacementf(Xf, 0, dimU);     // displacement
    mfem::Vector pressuref(Xf, dimU, dimU);      // pressure
    mfem::Vector obstaclef(Xf, 2 * dimU, dimU);  // obstacle
    states[0] = displacementf;
    states[1] = pressuref;
    states[2] = obstaclef;
    writer.write(0, 0.0, vis_states);
  }
}

double obstacle_forcing(const mfem::Vector& x)
{
  double fx = 0.;
  fx = 0.2 - 2.0 * (std::pow(x(0), 3) - 1.5 * std::pow(x(0), 2.) - 6 * x(0) + 3.) +
       (1. + std::pow(2. * M_PI, 2)) * std::cos(2. * M_PI * x(0));
  return fx;
}

double flat_obstacle(const mfem::Vector& /*x*/) { return 0.0; }

SmithObstacleDesignProblem::SmithObstacleDesignProblem(ParamObstacleProblem* parametrized_obst_problem,
                                                       std::vector<smith::FiniteElementState*> states,
                                                       std::shared_ptr<smith::Mesh> mesh,
                                                       std::shared_ptr<ObjectiveT> design_objective,
                                                       std::shared_ptr<WeakFormT> objective_grad_displacement,
                                                       std::shared_ptr<WeakFormT> objective_grad_pressure,
                                                       std::shared_ptr<WeakFormT> objective_grad_obstacle)
    : ObstacleDesignProblem(parametrized_obst_problem), time_info_(0.0, 0.0, 0)
{
  mesh_ = mesh;
  shape_disp_ = std::make_unique<smith::FiniteElementState>(mesh_->newShapeDisplacement());
  design_objective_ = design_objective;
  objective_grad_displacement_ = objective_grad_displacement;
  objective_grad_pressure_ = objective_grad_pressure;
  objective_grad_obstacle_ = objective_grad_obstacle;
  std::copy(states.begin(), states.end(), std::back_inserter(states_));
  // lumped mass matrices
  mfem::Vector Mlump;
  mfem::Vector Mduallump;
  parametrized_obst_problem->GetConstraintMassLump(Mlump);
  Mduallump.SetSize(Mlump.Size());
  Mduallump = 1.0;
  Mduallump /= Mlump;
  mfem::BlockVector MUlump_blk(primal_blockoffsets);
  mfem::BlockVector MClump_blk(constraint_blockoffsets);
  // U = (u, p, theta, s, z)
  MUlump_blk.GetBlock(0).Set(1.0, Mlump);      // u
  MUlump_blk.GetBlock(1).Set(1.0, Mlump);      // p
  MUlump_blk.GetBlock(2).Set(1.0, Mlump);      // theta
  MUlump_blk.GetBlock(3).Set(1.0, Mduallump);  // s
  MUlump_blk.GetBlock(4).Set(1.0, Mlump);      // z

  // C = \nabla_u L, g - s, p - z, \Phi_{FB}
  MClump_blk.GetBlock(0).Set(1.0, Mduallump);  // \nabla_u L
  MClump_blk.GetBlock(1).Set(1.0, Mduallump);  // g - s
  MClump_blk.GetBlock(2).Set(1.0, Mlump);      // p - z
  MClump_blk.GetBlock(3).Set(1.0, Mlump);      // \Phi_{FB}

  Mulump.Set(1.0, MUlump_blk);
  Mclump.Set(1.0, MClump_blk);
}

// design objective (E)
double SmithObstacleDesignProblem::E(const mfem::Vector& displacement, const mfem::Vector& pressure,
                                     const mfem::Vector& obstacle, int& eval_err)
{
  states_[0]->Set(1.0, displacement);
  states_[1]->Set(1.0, pressure);
  states_[2]->Set(1.0, obstacle);
  eval_err = 0;
  return design_objective_->evaluate(time_info_, shape_disp_.get(), smith::getConstFieldPointers(states_));
}

// gradient of design objective with respect to displacement (u)
void SmithObstacleDesignProblem::DuE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                                     const mfem::Vector& obstacle, mfem::Vector& displacementGradE)
{
  states_[0]->Set(1.0, displacement);
  states_[1]->Set(1.0, pressure);
  states_[2]->Set(1.0, obstacle);
  auto res_vector =
      objective_grad_displacement_->residual(time_info_, shape_disp_.get(), smith::getConstFieldPointers(states_));
  displacementGradE.Set(1.0, res_vector);
}

// gradient of design objective with respect to pressure (p)
void SmithObstacleDesignProblem::DpE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                                     const mfem::Vector& obstacle, mfem::Vector& pressureGradE)
{
  states_[0]->Set(1.0, displacement);
  states_[1]->Set(1.0, pressure);
  states_[2]->Set(1.0, obstacle);
  auto res_vector =
      objective_grad_pressure_->residual(time_info_, shape_disp_.get(), smith::getConstFieldPointers(states_));
  pressureGradE.Set(1.0, res_vector);
}

// gradient of design objective with respect to obstacle design (th)
void SmithObstacleDesignProblem::DthE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                                      const mfem::Vector& obstacle, mfem::Vector& obstacleGradE)
{
  states_[0]->Set(1.0, displacement);
  states_[1]->Set(1.0, pressure);
  states_[2]->Set(1.0, obstacle);
  auto res_vector =
      objective_grad_obstacle_->residual(time_info_, shape_disp_.get(), smith::getConstFieldPointers(states_));
  obstacleGradE.Set(1.0, res_vector);
}

// return non-owning Hessian of the design objective (u, u)
mfem::Operator* SmithObstacleDesignProblem::DuuE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                                                 const mfem::Vector& obstacle)
{
  states_[0]->Set(1.0, displacement);
  states_[1]->Set(1.0, pressure);
  states_[2]->Set(1.0, obstacle);
  std::fill(jacobian_weights_.begin(), jacobian_weights_.end(), 0.0);
  jacobian_weights_[0] = 1.0;
  auto HE_uu_uniq = objective_grad_displacement_->jacobian(time_info_, shape_disp_.get(),
                                                           smith::getConstFieldPointers(states_), jacobian_weights_);
  HuuE_.reset(HE_uu_uniq.release());
  return HuuE_.get();
}

// return non-owning Hessian of the design objective (u, p)
mfem::Operator* SmithObstacleDesignProblem::DupE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                                                 const mfem::Vector& obstacle)
{
  states_[0]->Set(1.0, displacement);
  states_[1]->Set(1.0, pressure);
  states_[2]->Set(1.0, obstacle);
  std::fill(jacobian_weights_.begin(), jacobian_weights_.end(), 0.0);
  jacobian_weights_[1] = 1.0;
  auto HE_up_uniq = objective_grad_displacement_->jacobian(time_info_, shape_disp_.get(),
                                                           smith::getConstFieldPointers(states_), jacobian_weights_);
  HupE_.reset(HE_up_uniq.release());
  return HupE_.get();
}

// return non-owning Hessian of the design objective (u, th)
mfem::Operator* SmithObstacleDesignProblem::DuthE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                                                  const mfem::Vector& obstacle)
{
  states_[0]->Set(1.0, displacement);
  states_[1]->Set(1.0, pressure);
  states_[2]->Set(1.0, obstacle);
  std::fill(jacobian_weights_.begin(), jacobian_weights_.end(), 0.0);
  jacobian_weights_[2] = 1.0;
  auto HE_uth_uniq = objective_grad_displacement_->jacobian(time_info_, shape_disp_.get(),
                                                            smith::getConstFieldPointers(states_), jacobian_weights_);
  HuthE_.reset(HE_uth_uniq.release());
  return HuthE_.get();
}

// return non-owning Hessian of the design objective (p, p)
mfem::Operator* SmithObstacleDesignProblem::DppE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                                                 const mfem::Vector& obstacle)
{
  states_[0]->Set(1.0, displacement);
  states_[1]->Set(1.0, pressure);
  states_[2]->Set(1.0, obstacle);
  std::fill(jacobian_weights_.begin(), jacobian_weights_.end(), 0.0);
  jacobian_weights_[1] = 1.0;
  auto HE_pp_uniq = objective_grad_pressure_->jacobian(time_info_, shape_disp_.get(),
                                                       smith::getConstFieldPointers(states_), jacobian_weights_);
  HppE_.reset(HE_pp_uniq.release());
  return HppE_.get();
}

// return non-owning Hessian of the design objective (p, th)
mfem::Operator* SmithObstacleDesignProblem::DpthE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                                                  const mfem::Vector& obstacle)
{
  states_[0]->Set(1.0, displacement);
  states_[1]->Set(1.0, pressure);
  states_[2]->Set(1.0, obstacle);
  std::fill(jacobian_weights_.begin(), jacobian_weights_.end(), 0.0);
  jacobian_weights_[2] = 1.0;
  auto HE_pth_uniq = objective_grad_pressure_->jacobian(time_info_, shape_disp_.get(),
                                                        smith::getConstFieldPointers(states_), jacobian_weights_);
  HpthE_.reset(HE_pth_uniq.release());
  return HpthE_.get();
}

// return non-owning Hessian of the design objective (th, th)
mfem::Operator* SmithObstacleDesignProblem::DththE(const mfem::Vector& displacement, const mfem::Vector& pressure,
                                                   const mfem::Vector& obstacle)
{
  states_[0]->Set(1.0, displacement);
  states_[1]->Set(1.0, pressure);
  states_[2]->Set(1.0, obstacle);
  std::fill(jacobian_weights_.begin(), jacobian_weights_.end(), 0.0);
  jacobian_weights_[2] = 1.0;
  auto HE_thth_uniq = objective_grad_obstacle_->jacobian(time_info_, shape_disp_.get(),
                                                         smith::getConstFieldPointers(states_), jacobian_weights_);
  HththE_.reset(HE_thth_uniq.release());
  return HththE_.get();
}
