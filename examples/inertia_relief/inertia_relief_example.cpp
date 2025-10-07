// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
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

#include "serac/serac.hpp"

#include "mfem.hpp"

// ContinuationSolver headers
#include "problems/Problems.hpp"
#include "solvers/HomotopySolver.hpp"
#include "utilities.hpp"

#include "axom/sidre.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;
static constexpr int dim = 3;
static constexpr int disp_order = 1;

using VectorSpace = serac::H1<disp_order, dim>;

using DensitySpace = serac::L2<disp_order - 1>;

using SolidMaterial = serac::solid_mechanics::NeoHookeanWithFieldDensity;
using SolidWeakFormT = serac::SolidWeakForm<disp_order, dim, serac::Parameters<DensitySpace>>;

enum FIELD
{
  DISP = SolidWeakFormT::DISPLACEMENT,
  VELO = SolidWeakFormT::VELOCITY,
  ACCEL = SolidWeakFormT::ACCELERATION,
  DENSITY = SolidWeakFormT::NUM_STATES
};

class ParaviewWriter {
 public:
  using StateVecs = std::vector<std::shared_ptr<serac::FiniteElementState>>;
  using DualVecs = std::vector<std::shared_ptr<serac::FiniteElementDual>>;

  ParaviewWriter(std::unique_ptr<mfem::ParaViewDataCollection> pv_, const StateVecs& states_)
      : pv(std::move(pv_)), states(states_)
  {
  }

  ParaviewWriter(std::unique_ptr<mfem::ParaViewDataCollection> pv_, const StateVecs& states_, const StateVecs& duals_)
      : pv(std::move(pv_)), states(states_), dual_states(duals_)
  {
  }

  void write(int step, double time, const std::vector<serac::FiniteElementState const*>& current_states)
  {
    SERAC_MARK_FUNCTION;
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

auto createParaviewOutput(const mfem::ParMesh& mesh, const std::vector<serac::FiniteElementState const*>& states,
                          std::string output_name)
{
  if (output_name == "") {
    output_name = "default";
  }

  ParaviewWriter::StateVecs output_states;
  for (const auto& s : states) {
    output_states.push_back(std::make_shared<serac::FiniteElementState>(s->space(), s->name()));
  }

  auto non_const_mesh = const_cast<mfem::ParMesh*>(&mesh);
  auto paraview_dc = std::make_unique<mfem::ParaViewDataCollection>(output_name, non_const_mesh);
  int max_order_in_fields = 0;

  // Find the maximum polynomial order in the physics module's states
  for (const auto& state : output_states) {
    paraview_dc->RegisterField(state->name(), &state->gridFunction());
    max_order_in_fields = std::max(max_order_in_fields, state->space().GetOrder(0));
  }

  // Set the options for the paraview output files
  paraview_dc->SetLevelsOfDetail(max_order_in_fields);
  paraview_dc->SetHighOrderOutput(true);
  paraview_dc->SetDataFormat(mfem::VTKFormat::BINARY);
  paraview_dc->SetCompression(true);

  return ParaviewWriter(std::move(paraview_dc), output_states, {});
}

/* NLMCP of the form
 * 0 <= x \perp F(x, y) >= 0
 *              Q(x, y)  = 0
 * Here, F and x are both 0-dimensional
 * and   Q(x, y) = [ r(u) + (dc/du)^T l]
 *                 [-c(u)]
 *            y  = [ u ]
 *                 [ l ]
 * we use the approximate Jacobian
 *       dQ/dy \approx [ dr/du     (dc/du)^T]
 *                     [-dc/du        0 ]
 * we note that the sign-convention with regard to "c" is important
 * as  the approximate Jacobian is positive semi-definite when dr/du is
 * and thus the NLMC problem is guaranteed to be semi-monotone.
 */
class InertialReliefProblem : public EqualityConstrainedHomotopyProblem {
  InertialReliefProblem() : time_info_(0.0, 0.0, 0) {}

 protected:
  mfem::HypreParMatrix* drdu_ = nullptr;
  mfem::HypreParMatrix* dcdu_ = nullptr;
  int dimu_;
  int dimc_;
  int dimuglb_;
  int dimcglb_;
  mfem::Array<int> y_partition_;
  std::vector<serac::FieldPtr> obj_states_;
  std::vector<serac::FieldPtr> all_states_;
  std::shared_ptr<SolidWeakFormT> weak_form_;
  std::unique_ptr<serac::FiniteElementState> shape_disp_;
  std::shared_ptr<serac::Mesh> mesh_;
  std::vector<std::shared_ptr<serac::ScalarObjective>> constraints_;
  serac::TimeInfo time_info_;
  std::vector<double> jacobian_weights_ = {1.0, 0.0, 0.0, 0.0};

 public:
  InertialReliefProblem(std::vector<serac::FieldPtr> obj_states, std::vector<serac::FieldPtr> all_states,
                        std::shared_ptr<serac::Mesh> mesh, std::shared_ptr<SolidWeakFormT> weak_form,
                        std::vector<std::shared_ptr<serac::ScalarObjective>> constraints);
  mfem::Vector residual(const mfem::Vector& u) const;
  mfem::Vector constraintJacobianTvp(const mfem::Vector& u, const mfem::Vector& l) const;
  mfem::Vector constraint(const mfem::Vector& u) const;
  mfem::HypreParMatrix* constraintJacobian(const mfem::Vector& u);
  mfem::HypreParMatrix* residualJacobian(const mfem::Vector& u);
  virtual ~InertialReliefProblem();
};

int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  serac::ApplicationManager applicationManager(argc, argv);

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
  serac::StateManager::initialize(datastore, "solid_dynamics");

  std::shared_ptr<serac::Mesh> mesh;
  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;
  std::vector<std::shared_ptr<serac::ScalarObjective>> constraints;

  mesh = std::make_shared<serac::Mesh>(
      mfem::Mesh::MakeCartesian3D(nx, ny, nz, element_shape, xlength, ylength, zlength), "this_mesh_name", 0, 0);

  serac::FiniteElementState disp = serac::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
  serac::FiniteElementState velo = serac::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
  serac::FiniteElementState accel = serac::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
  serac::FiniteElementState density = serac::StateManager::newState(DensitySpace{}, "density", mesh->tag());
  std::unique_ptr<serac::FiniteElementState> shape_disp =
      std::make_unique<serac::FiniteElementState>(mesh->newShapeDisplacement());

  velo = 0.0;
  accel = 0.0;

  states = {disp, velo, accel};
  params = {density};

  std::string physics_name = "solid";

  // construct residual
  auto solid_mechanics_weak_form =
      std::make_shared<SolidWeakFormT>(physics_name, mesh, states[DISP].space(), getSpaces(params));

  SolidMaterial mat;
  mat.K = 1.0;
  mat.G = 0.5;
  solid_mechanics_weak_form->setMaterial(serac::DependsOn<0>{}, mesh->entireBodyName(), mat);

  // apply some traction boundary conditions
  std::string surface_name = "side";
  mesh->addDomainOfBoundaryElements(surface_name, serac::by_attr<dim>(1));
  solid_mechanics_weak_form->addBoundaryFlux(surface_name, [](auto /*x*/, auto n, auto /*t*/) { return 1.0 * n; });

  serac::tensor<double, dim> constant_force{};
  for (int i = 0; i < dim; i++) {
    constant_force[i] = 1.e0;
  }

  solid_mechanics_weak_form->addBodyIntegral(mesh->entireBodyName(), [constant_force](double /* t */, auto x) {
    return serac::tuple{constant_force, 0.0 * serac::get<serac::DERIVATIVE>(x)};
  });

  // construct constraints
  params[0] = 1.;

  using ObjectiveT =
      serac::FunctionalObjective<dim, serac::Parameters<VectorSpace, DensitySpace>>;  // functional objective on
                                                                                      // displacement/density

  double time = 0.0;
  double dt = 1.0;
  serac::TimeInfo time_info(time, dt, 0);
  auto all_states = getConstFieldPointers(states, params);
  auto objective_states = {all_states[DISP], all_states[DENSITY]};

  ObjectiveT::SpacesT param_space_ptrs{&all_states[DISP]->space(), &all_states[DENSITY]->space()};

  ObjectiveT mass_objective("mass constraining", mesh, param_space_ptrs);

  mass_objective.addBodyIntegral(serac::DependsOn<1>{}, mesh->entireBodyName(),
                                 [](double /*t*/, auto /*X*/, auto RHO) { return get<serac::VALUE>(RHO); });
  double mass = mass_objective.evaluate(time_info, shape_disp.get(), objective_states);

  serac::tensor<double, dim> initial_cg;

  for (int i = 0; i < dim; ++i) {
    auto cg_objective = std::make_shared<ObjectiveT>("translation " + std::to_string(i), mesh, param_space_ptrs);
    cg_objective->addBodyIntegral(serac::DependsOn<0, 1>{}, mesh->entireBodyName(),
                                  [i](double
                                      /*time*/,
                                      auto X, auto U, auto RHO) {
                                    return (get<serac::VALUE>(X)[i] + get<serac::VALUE>(U)[i]) * get<serac::VALUE>(RHO);
                                  });
    initial_cg[i] = cg_objective->evaluate(time_info, shape_disp.get(), objective_states) / mass;

    constraints.push_back(cg_objective);
  }

  for (int i = 0; i < dim; ++i) {
    auto center_rotation_objective =
        std::make_shared<ObjectiveT>("rotation" + std::to_string(i), mesh, param_space_ptrs);
    center_rotation_objective->addBodyIntegral(serac::DependsOn<0, 1>{}, mesh->entireBodyName(),
                                               [i, initial_cg](double /*time*/, auto X, auto U, auto RHO) {
                                                 auto u = get<serac::VALUE>(U);
                                                 auto x = get<serac::VALUE>(X) + u;
                                                 auto dx = x - initial_cg;
                                                 auto x_cross_u = serac::cross(dx, u);
                                                 return x_cross_u[i] * get<serac::VALUE>(RHO);
                                               });
    constraints.push_back(center_rotation_objective);
  }

  // initialize displacement
  states[FIELD::DISP].setFromFieldFunction([](serac::tensor<double, dim> x) {
    auto u = 0.0 * x;
    return u;
  });

  auto writer = createParaviewOutput(mesh->mfemParMesh(), objective_states, "inertia_relief");
  if (visualize) {
    writer.write(0, 0.0, objective_states);
  }
  auto non_const_states = getFieldPointers(states, params);
  // create an inertial relief problem
  InertialReliefProblem problem({non_const_states[DISP], non_const_states[DENSITY]}, non_const_states, mesh,
                                solid_mechanics_weak_form, constraints);

  // optimization variables
  auto X0 = problem.GetOptimizationVariable();
  auto Xf = problem.GetOptimizationVariable();

  // define a homotopy solver for the inertia relief problem
  HomotopySolver solver(&problem);
  // set solver options
  solver.SetTol(nonlinear_absolute_tol);
  solver.SetMaxIter(nonlinear_max_iterations);

  // solve the inertia relief problem
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
  if (visualize) {
    writer.write(1, 1.0, objective_states);
  }
}

InertialReliefProblem::InertialReliefProblem(std::vector<serac::FiniteElementState*> obj_states,
                                             std::vector<serac::FiniteElementState*> all_states,
                                             std::shared_ptr<serac::Mesh> mesh,
                                             std::shared_ptr<SolidWeakFormT> weak_form,
                                             std::vector<std::shared_ptr<serac::ScalarObjective>> constraints)
    : EqualityConstrainedHomotopyProblem(), time_info_(0.0, 0.0, 0)
{
  weak_form_ = weak_form;
  mesh_ = mesh;
  shape_disp_ = std::make_unique<serac::FiniteElementState>(mesh_->newShapeDisplacement());

  constraints_.resize(constraints.size());
  std::copy(constraints.begin(), constraints.end(), constraints_.begin());

  all_states_.resize(all_states.size());
  std::copy(all_states.begin(), all_states.end(), all_states_.begin());

  obj_states_.resize(obj_states.size());
  std::copy(obj_states.begin(), obj_states.end(), obj_states_.begin());

  HYPRE_BigInt cOffsets[2];
  HYPRE_BigInt* uOffsets = all_states[FIELD::DISP]->space().GetTrueDofOffsets();
  dimc_ = static_cast<int>(constraints_.size());
  int myid = mfem::Mpi::WorldRank();
  cOffsets[0] = 0;
  cOffsets[1] = dimc_;
  if (myid > 0) {
    dimc_ = 0;
    cOffsets[0] = cOffsets[1];
  }

  dimu_ = all_states_[FIELD::DISP]->space().GetTrueVSize();
  MPI_Allreduce(&dimc_, &dimcglb_, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimu_, &dimuglb_, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  SetSizes(uOffsets, cOffsets);
}

// residual callback
mfem::Vector InertialReliefProblem::residual(const mfem::Vector& u) const
{
  obj_states_[DISP]->Set(1.0, u);
  auto res_vector = weak_form_->residual(time_info_, shape_disp_.get(), serac::getConstFieldPointers(all_states_));
  return res_vector;
}

// constraint Jacobian transpose vector product
mfem::Vector InertialReliefProblem::constraintJacobianTvp(const mfem::Vector& u, const mfem::Vector& l) const
{
  obj_states_[DISP]->Set(1.0, u);
  std::vector<double> multipliers(constraints_.size());
  for (int i = 0; i < dimc_; i++) {
    multipliers[static_cast<size_t>(i)] = l(i);
  }
  const int nconstraints = static_cast<int>(constraints_.size());
  MPI_Bcast(multipliers.data(), nconstraints, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  mfem::Vector constraint_gradient(dimu_);
  constraint_gradient = 0.0;
  mfem::Vector output_vec(dimu_);
  output_vec = 0.0;

  for (size_t i = 0; i < constraints_.size(); i++) {
    mfem::Vector grad_temp =
        constraints_[i]->gradient(time_info_, shape_disp_.get(), serac::getConstFieldPointers(obj_states_), DISP);
    constraint_gradient.Set(1.0, grad_temp);
    output_vec.Add(multipliers[i], constraint_gradient);
  }
  return output_vec;
}

// Jacobian of the residual
mfem::HypreParMatrix* InertialReliefProblem::residualJacobian(const mfem::Vector& u)
{
  obj_states_[DISP]->Set(1.0, u);
  auto drdu_unique =
      weak_form_->jacobian(time_info_, shape_disp_.get(), getConstFieldPointers(all_states_), jacobian_weights_);

  if (drdu_) {
    delete drdu_;
  }
  drdu_ = drdu_unique.release();
  SLIC_ERROR_ROOT_IF(drdu_->Height() != dimu_ || drdu_->Width() != dimu_, "residual Jacobian of an unexpected shape");
  return drdu_;
}

// constraint callback
mfem::Vector InertialReliefProblem::constraint(const mfem::Vector& u) const
{
  obj_states_[DISP]->Set(1.0, u);
  mfem::Vector output_vec(dimc_);
  output_vec = 0.0;

  for (size_t i = 0; i < constraints_.size(); i++) {
    const int idx = static_cast<int>(i);
    const size_t i2 = static_cast<size_t>(idx);
    SLIC_ERROR_ROOT_IF(i2 != i, "Constraint index is out of range, bad cast from size_t to int");

    double constraint_i =
        constraints_[i]->evaluate(time_info_, shape_disp_.get(), serac::getConstFieldPointers(obj_states_));
    if (dimc_ > 0) {
      output_vec(idx) = constraint_i;
    }
  }
  return output_vec;
}

// Jacobian of the constraint
mfem::HypreParMatrix* InertialReliefProblem::constraintJacobian(const mfem::Vector& u)
{
  obj_states_[DISP]->Set(1.0, u);
  int myid = mfem::Mpi::WorldRank();
  int nentries = dimuglb_;
  if (myid > 0) {
    nentries = 0;
  }
  mfem::SparseMatrix dcdumat(dimc_, dimuglb_, nentries);
  mfem::Array<int> cols;
  cols.SetSize(dimuglb_);
  for (int i = 0; i < dimuglb_; i++) {
    cols[i] = i;
  }
  for (size_t i = 0; i < constraints_.size(); i++) {
    const int idx = static_cast<int>(i);
    const size_t i2 = static_cast<size_t>(idx);
    SLIC_ERROR_ROOT_IF(i2 != i, "Constraint index is out of range, bad cast from size_t to int");
    mfem::HypreParVector gradVector(MPI_COMM_WORLD, dimuglb_, uOffsets_);
    gradVector.Set(
        1.0, constraints_[i]->gradient(time_info_, shape_disp_.get(), serac::getConstFieldPointers(obj_states_), DISP));
    mfem::Vector* globalGradVector = gradVector.GlobalVector();
    if (myid == 0) {
      dcdumat.SetRow(idx, cols, *globalGradVector);
    }
    delete globalGradVector;
  }
  dcdumat.Threshold(1.e-20);
  dcdumat.Finalize();
  if (dcdu_) {
    delete dcdu_;
  }
  dcdu_ = GenerateHypreParMatrixFromSparseMatrix(uOffsets_, cOffsets_, &dcdumat);
  return dcdu_;
}

InertialReliefProblem::~InertialReliefProblem()
{
  if (drdu_) {
    delete drdu_;
  }
  if (dcdu_) {
    delete dcdu_;
  }
}
