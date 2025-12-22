// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file inertia_relief_example.cpp
 *
 * @brief Inertia Relief example
 *
 * Intended to show how to solve a problem with the HomotopySolver.
 * The example problem solved is an inertia relief problem.
 */

#include "smith/smith.hpp"

#include "mfem.hpp"

// ContinuationSolver headers
#include "problems/Problems.hpp"
#include "solvers/HomotopySolver.hpp"
#include "utilities.hpp"

#include "axom/sidre.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;
static constexpr int dim = 3;
static constexpr int disp_order = 1;

using VectorSpace = smith::H1<disp_order, dim>;

using DensitySpace = smith::L2<disp_order - 1>;

using SolidMaterial = smith::solid_mechanics::NeoHookeanWithFieldDensity;
using SolidWeakFormT = smith::SolidWeakForm<disp_order, dim, smith::Parameters<DensitySpace>>;

enum FIELD
{
  DISP = SolidWeakFormT::DISPLACEMENT,
  VELO = SolidWeakFormT::VELOCITY,
  ACCEL = SolidWeakFormT::ACCELERATION,
  DENSITY = SolidWeakFormT::NUM_STATES
};

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

/* Nonlinear problem of the form
 * F(X) = [ r(u) + (dc/du)^T l ] = [ 0 ]
 *        [ -c(u)              ]   [ 0 ]
 *   X  = [ u ]
 *        [ l ]
 *
 * wherein r(u) is the elasticity nonlinear residual
 *         c(u) are the tied gap contacts
 *            u are the displacement dofs
 *            l are the Lagrange multipliers
 *
 * This problem inherits from EqualityConstrainedHomotopyProblem
 * for compatibility with the HomotopySolver.
 */
class InertialReliefProblem : public EqualityConstrainedHomotopyProblem {
  InertialReliefProblem() : time_info_(0.0, 0.0, 0) {}

 protected:
  std::unique_ptr<mfem::HypreParMatrix> drdu_;             // Jacobian of residual
  std::unique_ptr<mfem::HypreParMatrix> dcdu_;             // Jacobian of constraint
  std::vector<smith::FieldPtr> obj_states_;                // states for objective evaluation
  std::vector<smith::FieldPtr> all_states_;                // states for weak_form evaluation
  std::shared_ptr<SolidWeakFormT> weak_form_;              // weak_form
  std::unique_ptr<smith::FiniteElementState> shape_disp_;  // shape displacement
  std::shared_ptr<smith::Mesh> mesh_;
  std::vector<std::shared_ptr<smith::ScalarObjective>> constraints_;  // vector of constraints
  smith::TimeInfo time_info_;  // time info for constraint and weak_form function calls
  std::vector<double> jacobian_weights_ = {1.0, 0.0, 0.0, 0.0};  // weights for weak_form_->jacobian calls
  mutable mfem::Vector constraint_cached_;
  mutable mfem::Vector residual_cached_;
  mutable mfem::Vector JTvp_cached_;

 public:
  InertialReliefProblem(std::vector<smith::FieldPtr> obj_states, std::vector<smith::FieldPtr> all_states,
                        std::shared_ptr<smith::Mesh> mesh, std::shared_ptr<SolidWeakFormT> weak_form,
                        std::vector<std::shared_ptr<smith::ScalarObjective>> constraints);
  mfem::Vector residual(const mfem::Vector& u, bool fresh_evaluation) const;
  mfem::Vector constraintJacobianTvp(const mfem::Vector& u, const mfem::Vector& l, bool fresh_evaluation) const;
  mfem::Vector constraint(const mfem::Vector& u, bool fresh_evaluation) const;
  mfem::HypreParMatrix* constraintJacobian(const mfem::Vector& u, bool fresh_evaluation);
  mfem::HypreParMatrix* residualJacobian(const mfem::Vector& u, bool fresh_evaluation);
  virtual ~InertialReliefProblem();
};

int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  // Command line arguments
  // Mesh options
  double xlength = 0.5;
  double ylength = 0.7;
  double zlength = 0.3;
  int nx = 6;
  int ny = 4;
  int nz = 4;
  int visualize = 0;

  // Solver options
  double nonlinear_absolute_tol = 1e-6;
  int nonlinear_max_iterations = 50;
  // Handle command line arguments
  axom::CLI::App app{"Inertial relief."};
  // Mesh options
  app.add_option("--xlength", xlength, "extent along x-axis")
      ->default_val("0.5")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--ylength", ylength, "extent along y-axis")
      ->default_val("0.7")  // Matches value set above
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--zlength", zlength, "extent along z-axis")
      ->default_val("0.3")  // Matches value set above
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
  smith::StateManager::initialize(datastore, "solid_dynamics");

  std::shared_ptr<smith::Mesh> mesh;
  std::vector<smith::FiniteElementState> states;
  std::vector<smith::FiniteElementState> params;
  std::vector<std::shared_ptr<smith::ScalarObjective>> constraints;

  mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(nx, ny, nz, element_shape, xlength, ylength, zlength), "this_mesh_name", 0, 0);

  smith::FiniteElementState disp = smith::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
  smith::FiniteElementState velo = smith::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
  smith::FiniteElementState accel = smith::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
  smith::FiniteElementState density = smith::StateManager::newState(DensitySpace{}, "density", mesh->tag());
  std::unique_ptr<smith::FiniteElementState> shape_disp =
      std::make_unique<smith::FiniteElementState>(mesh->newShapeDisplacement());

  velo = 0.0;
  accel = 0.0;

  states = {disp, velo, accel};
  params = {density};

  std::string physics_name = "solid";

  // construct residual
  auto solid_mechanics_weak_form =
      std::make_shared<SolidWeakFormT>(physics_name, mesh, states[FIELD::DISP].space(), getSpaces(params));

  SolidMaterial mat;
  mat.K = 1.0;
  mat.G = 0.5;
  solid_mechanics_weak_form->setMaterial(smith::DependsOn<0>{}, mesh->entireBodyName(), mat);

  // apply some traction boundary conditions
  std::string surface_name = "side";
  mesh->addDomainOfBoundaryElements(surface_name, smith::by_attr<dim>(1));
  solid_mechanics_weak_form->addBoundaryFlux(surface_name, [](auto /*x*/, auto n, auto /*t*/) { return 1.0 * n; });

  smith::tensor<double, dim> constant_force{};
  for (int i = 0; i < dim; i++) {
    constant_force[i] = 1.e0;
  }

  solid_mechanics_weak_form->addBodyIntegral(mesh->entireBodyName(), [constant_force](double /* t */, auto x) {
    return smith::tuple{constant_force, 0.0 * smith::get<smith::DERIVATIVE>(x)};
  });

  // construct constraints
  params[0] = 1.;

  using ObjectiveT =
      smith::FunctionalObjective<dim, smith::Parameters<VectorSpace, DensitySpace>>;  // functional objective on
                                                                                      // displacement/density

  double time = 0.0;
  double dt = 1.0;
  smith::TimeInfo time_info(time, dt, 0);
  auto all_states = getConstFieldPointers(states, params);
  auto objective_states = {all_states[FIELD::DISP], all_states[FIELD::DENSITY]};

  ObjectiveT::SpacesT param_space_ptrs{&all_states[FIELD::DISP]->space(), &all_states[FIELD::DENSITY]->space()};

  ObjectiveT mass_objective("mass constraining", mesh, param_space_ptrs);

  mass_objective.addBodyIntegral(smith::DependsOn<1>{}, mesh->entireBodyName(),
                                 [](double /*t*/, auto /*X*/, auto RHO) { return get<smith::VALUE>(RHO); });
  double mass = mass_objective.evaluate(time_info, shape_disp.get(), objective_states);

  smith::tensor<double, dim> initial_cg;  // center of gravity

  for (int i = 0; i < dim; ++i) {
    auto cg_objective = std::make_shared<ObjectiveT>("translation " + std::to_string(i), mesh, param_space_ptrs);
    cg_objective->addBodyIntegral(smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
                                  [i](double
                                      /*time*/,
                                      auto X, auto U, auto RHO) {
                                    return (get<smith::VALUE>(X)[i] + get<smith::VALUE>(U)[i]) * get<smith::VALUE>(RHO);
                                  });
    initial_cg[i] = cg_objective->evaluate(time_info, shape_disp.get(), objective_states) / mass;

    constraints.push_back(cg_objective);
  }

  for (int i = 0; i < dim; ++i) {
    auto center_rotation_objective =
        std::make_shared<ObjectiveT>("rotation" + std::to_string(i), mesh, param_space_ptrs);
    center_rotation_objective->addBodyIntegral(smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
                                               [i, initial_cg](double /*time*/, auto X, auto U, auto RHO) {
                                                 auto u = get<smith::VALUE>(U);
                                                 auto x = get<smith::VALUE>(X) + u;
                                                 auto dx = x - initial_cg;
                                                 auto x_cross_u = smith::cross(dx, u);
                                                 return x_cross_u[i] * get<smith::VALUE>(RHO);
                                               });
    constraints.push_back(center_rotation_objective);
  }

  // initialize displacement
  states[FIELD::DISP].setFromFieldFunction([](smith::tensor<double, dim> x) {
    auto u = 0.0 * x;
    return u;
  });

  auto writer = createParaviewWriter(mesh->mfemParMesh(), objective_states, "inertia_relief");
  if (visualize) {
    writer.write(0, 0.0, objective_states);
  }
  auto non_const_states = getFieldPointers(states, params);
  // create an inertial relief problem
  InertialReliefProblem problem({non_const_states[FIELD::DISP], non_const_states[FIELD::DENSITY]}, non_const_states,
                                mesh, solid_mechanics_weak_form, constraints);

  // optimization variables
  auto X0 = problem.GetOptimizationVariable();
  auto Xf = problem.GetOptimizationVariable();

  // define a homotopy solver for the inertia relief problem
  HomotopySolver solver(&problem);
  // set solver options
  solver.SetTol(nonlinear_absolute_tol);
  solver.SetMaxIter(nonlinear_max_iterations);
  solver.EnableRegularizedNewtonMode();
  // solve the inertia relief problem
  solver.SetPrintLevel(2);
  solver.Mult(X0, Xf);
  // extract displacement and Lagrange multipliers
  mfem::Vector displacement_sol = problem.GetDisplacement(Xf);
  mfem::Vector multiplier_sol = problem.GetLagrangeMultiplier(Xf);
  bool converged = solver.GetConverged();
  SLIC_ERROR_ROOT_IF(!converged, "Homotopy solver did not converge");
  double displacement_norm = mfem::GlobalLpNorm(2, displacement_sol.Norml2(), MPI_COMM_WORLD);
  double multiplier_norm = mfem::GlobalLpNorm(2, multiplier_sol.Norml2(), MPI_COMM_WORLD);
  SLIC_INFO_ROOT(axom::fmt::format("||displacement|| = {}", displacement_norm));
  SLIC_INFO_ROOT(axom::fmt::format("||multiplier|| = {}", multiplier_norm));
  auto adjoint = problem.GetOptimizationVariable();
  auto adjoint_load = problem.GetOptimizationVariable();
  adjoint = 0.0;
  adjoint_load = 1.0;
  problem.AdjointSolve(displacement_sol, adjoint_load, adjoint);

  if (visualize) {
    writer.write(1, 1.0, objective_states);
  }
}

InertialReliefProblem::InertialReliefProblem(std::vector<smith::FiniteElementState*> obj_states,
                                             std::vector<smith::FiniteElementState*> all_states,
                                             std::shared_ptr<smith::Mesh> mesh,
                                             std::shared_ptr<SolidWeakFormT> weak_form,
                                             std::vector<std::shared_ptr<smith::ScalarObjective>> constraints)
    : EqualityConstrainedHomotopyProblem(), time_info_(0.0, 0.0, 0)
{
  weak_form_ = weak_form;
  mesh_ = mesh;
  shape_disp_ = std::make_unique<smith::FiniteElementState>(mesh_->newShapeDisplacement());

  constraints_.resize(constraints.size());
  std::copy(constraints.begin(), constraints.end(), constraints_.begin());

  all_states_.resize(all_states.size());
  std::copy(all_states.begin(), all_states.end(), all_states_.begin());

  obj_states_.resize(obj_states.size());
  std::copy(obj_states.begin(), obj_states.end(), obj_states_.begin());

  int dim_displacement = all_states_[FIELD::DISP]->space().GetTrueVSize();
  int dim_constraints = static_cast<int>(constraints_.size());
  int myid = mfem::Mpi::WorldRank();
  if (myid > 0) {
    dim_constraints = 0;
  }
  SetSizes(dim_displacement, dim_constraints);

  constraint_cached_.SetSize(dim_constraints);
  constraint_cached_ = 0.0;
  residual_cached_.SetSize(dim_displacement);
  residual_cached_ = 0.0;
  JTvp_cached_.SetSize(dim_displacement);
  JTvp_cached_ = 0.0;
}

// residual callback
mfem::Vector InertialReliefProblem::residual(const mfem::Vector& u, bool fresh_evaluation) const
{
  if (fresh_evaluation) {
    obj_states_[FIELD::DISP]->Set(1.0, u);
    residual_cached_.Set(
        1.0, weak_form_->residual(time_info_, shape_disp_.get(), smith::getConstFieldPointers(all_states_)));
  }
  return residual_cached_;
}

// constraint Jacobian transpose vector product
mfem::Vector InertialReliefProblem::constraintJacobianTvp(const mfem::Vector& u, const mfem::Vector& l,
                                                          bool fresh_evaluation) const
{
  if (fresh_evaluation) {
    int dim_constraints = GetMultiplierDim();
    int dim_displacement = GetDisplacementDim();
    obj_states_[FIELD::DISP]->Set(1.0, u);
    std::vector<double> multipliers(constraints_.size());
    for (int i = 0; i < dim_constraints; i++) {
      multipliers[static_cast<size_t>(i)] = l(i);
    }
    const int nconstraints = static_cast<int>(constraints_.size());
    MPI_Bcast(multipliers.data(), nconstraints, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    mfem::Vector constraint_gradient(dim_displacement);
    constraint_gradient = 0.0;
    JTvp_cached_ = 0.0;
    for (size_t i = 0; i < constraints_.size(); i++) {
      mfem::Vector grad_temp = constraints_[i]->gradient(time_info_, shape_disp_.get(),
                                                         smith::getConstFieldPointers(obj_states_), FIELD::DISP);
      constraint_gradient.Set(1.0, grad_temp);
      JTvp_cached_.Add(multipliers[i], constraint_gradient);
    }
  }
  return JTvp_cached_;
}

// Jacobian of the residual
mfem::HypreParMatrix* InertialReliefProblem::residualJacobian(const mfem::Vector& u, bool fresh_evaluation)
{
  if (fresh_evaluation) {
    obj_states_[FIELD::DISP]->Set(1.0, u);
    drdu_.reset();
    drdu_ = weak_form_->jacobian(time_info_, shape_disp_.get(), getConstFieldPointers(all_states_), jacobian_weights_);
  }
  int dim_displacement = GetDisplacementDim();
  SLIC_ERROR_ROOT_IF(drdu_->Height() != dim_displacement || drdu_->Width() != dim_displacement,
                     "residual Jacobian of an unexpected shape");
  return drdu_.get();
}

// constraint callback
mfem::Vector InertialReliefProblem::constraint(const mfem::Vector& u, bool fresh_evaluation) const
{
  if (fresh_evaluation) {
    int dim_constraints = GetMultiplierDim();
    obj_states_[FIELD::DISP]->Set(1.0, u);

    for (size_t i = 0; i < constraints_.size(); i++) {
      const int idx = static_cast<int>(i);
      const size_t i2 = static_cast<size_t>(idx);
      SLIC_ERROR_ROOT_IF(i2 != i, "Constraint index is out of range, bad cast from size_t to int");

      double constraint_i =
          constraints_[i]->evaluate(time_info_, shape_disp_.get(), smith::getConstFieldPointers(obj_states_));
      if (dim_constraints > 0) {
        constraint_cached_(idx) = constraint_i;
      }
    }
  }
  return constraint_cached_;
}

// Jacobian of the constraint
mfem::HypreParMatrix* InertialReliefProblem::constraintJacobian(const mfem::Vector& u, bool fresh_evaluation)
{
  int dim_constraints = GetMultiplierDim();
  int glbdim_displacement = GetGlobalDisplacementDim();
  if (fresh_evaluation) {
    obj_states_[FIELD::DISP]->Set(1.0, u);
    // dense rows
    int nentries = glbdim_displacement;
    if (dimc_ == 0) {
      nentries = 0;
    }
    mfem::SparseMatrix dcdumat(dim_constraints, glbdim_displacement, nentries);
    mfem::Array<int> cols;
    cols.SetSize(glbdim_displacement);
    for (int i = 0; i < glbdim_displacement; i++) {
      cols[i] = i;
    }
    std::unique_ptr<mfem::Vector> globalGradVector;
    for (size_t i = 0; i < constraints_.size(); i++) {
      const int idx = static_cast<int>(i);
      const size_t i2 = static_cast<size_t>(idx);
      SLIC_ERROR_ROOT_IF(i2 != i, "Constraint index is out of range, bad cast from size_t to int");
      mfem::HypreParVector gradVector(MPI_COMM_WORLD, glbdim_displacement, uOffsets_);
      gradVector.Set(1.0, constraints_[i]->gradient(time_info_, shape_disp_.get(),
                                                    smith::getConstFieldPointers(obj_states_), FIELD::DISP));
      globalGradVector.reset(gradVector.GlobalVector());
      if (dim_constraints > 0) {
        dcdumat.SetRow(idx, cols, *globalGradVector.get());
      }
    }
    dcdumat.Finalize();
    dcdu_.reset(GenerateHypreParMatrixFromSparseMatrix(cOffsets_, uOffsets_, &dcdumat));
  }
  return dcdu_.get();
}

InertialReliefProblem::~InertialReliefProblem() {}
