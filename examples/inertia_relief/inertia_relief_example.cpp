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

enum FIELD
{
  DISP,
  VELO,
  ACCEL,
  DENSITY
};

auto element_shape = mfem::Element::QUADRILATERAL;
static constexpr int dim = 3;
static constexpr int disp_order = 1;

using VectorSpace = serac::H1<disp_order, dim>;

using DensitySpace = serac::L2<disp_order - 1>;

using SolidMaterial = serac::solid_mechanics::NeoHookeanWithFieldDensity;

using SolidResidualT = serac::SolidResidual<disp_order, dim, serac::Parameters<DensitySpace>>;

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
class InertialReliefProblem : public GeneralNLMCProblem {
 protected:
  mfem::HypreParMatrix* dFdx;
  mfem::HypreParMatrix* dFdy;
  mfem::HypreParMatrix* dQdx;
  mfem::HypreParMatrix* dQdy;
  HYPRE_BigInt* uOffsets;
  HYPRE_BigInt* cOffsets;
  int dimu;
  int dimc;
  mfem::Array<int> y_partition;
  std::vector<serac::FieldPtr> obj_states;
  std::vector<serac::FieldPtr> all_states;
  std::shared_ptr<serac::Residual> residual;
  std::vector<std::shared_ptr<serac::ScalarObjective>> constraints;
  double time = 0.0;
  double dt = 0.0;
  std::vector<double> jacobian_weights = {1.0, 0.0, 0.0, 0.0};

 public:
  InertialReliefProblem(std::vector<serac::FieldPtr> obj_states_, std::vector<serac::FieldPtr> all_states_,
                        std::shared_ptr<serac::Residual> residual_,
                        std::vector<std::shared_ptr<serac::ScalarObjective>> constraints_);
  void F(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& feval, int& Feval_err) const;
  void Q(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& qeval, int& Qeval_err) const;
  mfem::HypreParMatrix* DxF(const mfem::Vector& x, const mfem::Vector& y);
  mfem::HypreParMatrix* DyF(const mfem::Vector& x, const mfem::Vector& y);
  mfem::HypreParMatrix* DxQ(const mfem::Vector& x, const mfem::Vector& y);
  mfem::HypreParMatrix* DyQ(const mfem::Vector& x, const mfem::Vector& y);
  virtual ~InertialReliefProblem();
};

int main(int argc, char* argv[])
{
  // Command line arguments
  // Mesh options
  double xlength = 0.5;
  double ylength = 0.7;
  double zlength = 0.3;
  int nx = 6;
  int ny = 4;
  int nz = 4;
  bool visualize = false;

  // Solver options
  [[maybe_unused]] double nonlinear_absolute_tol = 1e-6;
  [[maybe_unused]] int nonlinear_max_iterations = 100;
  
  // Initialize and automatically finalize MPI and other libraries
  serac::ApplicationManager applicationManager(argc, argv);
  
  //// Handle command line arguments
  //axom::CLI::App app{"Inertia relief example"};
  //// Mesh options
  //app.add_option("--nx", nx, "Elements in x-direction")->check(axom::CLI::PositiveNumber);
  //app.add_option("--ny", ny, "Elements in y-direction")->check(axom::CLI::PositiveNumber);
  //app.add_option("--nz", nz, "Elements in z-direction")->check(axom::CLI::PositiveNumber);
  //app.add_option("--Lx", xlength, "Domain extent in x-direction")->check(axom::CLI::PositiveNumber);
  //app.add_option("--Ly", ylength, "Domain extent in y-direction")->check(axom::CLI::PositiveNumber);
  //app.add_option("--Lz", zlength, "Domain extent in z-direction")->check(axom::CLI::PositiveNumber);
  //app.add_option("--visualize", visualize,
  //               "Visaulize solution of inertia relief problem, 0 for no visualization penalty or 1 for visualization")
  //    ->expected(0, 1);
  //// Solver options
  //app.add_option("--nonlinear-solver-tolerance", nonlinear_absolute_tol,
  //               "Nonlinear solver absolute tolerance")->check(axom::CLI::PositiveNumber);
  //app.add_option("--nonlinear-solver-max-iterations", nonlinear_max_iterations,
  //               "Nonlinear solver maximum iterations")->check(axom::CLI::PositiveNumber);
  
  int nprocs;
  int myid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  //SLIC_ERROR_ROOT_IF(nprocs > 1, "serial example (for now)");

  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_dynamics");


  std::shared_ptr<serac::Mesh> mesh;
  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;
  std::shared_ptr<serac::Residual> residual;
  std::vector<std::shared_ptr<serac::ScalarObjective>> constraints;

  mesh = std::make_shared<serac::Mesh>(
      mfem::Mesh::MakeCartesian3D(nx, ny, nz, element_shape, xlength, ylength, zlength), "this_mesh_name", 0, 0);

  serac::FiniteElementState disp = serac::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
  serac::FiniteElementState velo = serac::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
  serac::FiniteElementState accel = serac::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
  serac::FiniteElementState density = serac::StateManager::newState(DensitySpace{}, "density", mesh->tag());

  velo = 0.0;
  accel = 0.0;

  states = {disp, velo, accel};
  params = {density};

  std::string physics_name = "solid";

  // construct residual
  auto solid_mechanics_residual =
      std::make_shared<SolidResidualT>(physics_name, mesh, states[DISP].space(), getSpaces(params));

  SolidMaterial mat;
  mat.K = 1.0;
  mat.G = 0.5;
  solid_mechanics_residual->setMaterial(serac::DependsOn<0>{}, mesh->entireBodyName(), mat);

  // apply some traction boundary conditions
  std::string surface_name = "side";
  mesh->addDomainOfBoundaryElements(surface_name, serac::by_attr<dim>(1));
  solid_mechanics_residual->addBoundaryIntegral(surface_name, [](auto /*x*/, auto n, auto /*t*/) { return 1.0 * n; });

  serac::tensor<double, dim> constant_force{};
  for (int i = 0; i < dim; i++) {
    constant_force[i] = -1.e0;
  }

  solid_mechanics_residual->addBodyIntegral(mesh->entireBodyName(), [constant_force](double /* t */, auto x) {
    return serac::tuple{constant_force, 0.0 * serac::get<serac::DERIVATIVE>(x)};
  });
  residual = solid_mechanics_residual;

  // construct constraints
  params[0] = 1.0;

  using ObjectiveT =
      serac::FunctionalObjective<dim, serac::Parameters<VectorSpace, DensitySpace>>;  // functional objective on
                                                                                      // displacement/density

  double time = 0.0;
  double dt = 1.0;
  auto all_states = getConstFieldPointers(states, params);
  auto objective_states = {all_states[DISP], all_states[DENSITY]};

  ObjectiveT::SpacesT param_space_ptrs{&all_states[DISP]->space(), &all_states[DENSITY]->space()};

  ObjectiveT mass_objective("mass constraining", mesh, param_space_ptrs);

  mass_objective.addBodyIntegral(serac::DependsOn<1>{}, mesh->entireBodyName(),
                                 [](double /*t*/, auto /*X*/, auto RHO) { return get<serac::VALUE>(RHO); });
  double mass = mass_objective.evaluate(time, dt, objective_states);

  serac::tensor<double, dim> initial_cg;

  for (int i = 0; i < dim; ++i) {
    auto cg_objective = std::make_shared<ObjectiveT>("translation " + std::to_string(i), mesh, param_space_ptrs);
    cg_objective->addBodyIntegral(serac::DependsOn<0, 1>{}, mesh->entireBodyName(),
                                  [i](double
                                      /*time*/,
                                      auto X, auto U, auto RHO) {
                                    return (get<serac::VALUE>(X)[i] + get<serac::VALUE>(U)[i]) * get<serac::VALUE>(RHO);
                                  });
    initial_cg[i] = cg_objective->evaluate(time, dt, objective_states) / mass;

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

  auto writer = createParaviewOutput(mesh->mfemParMesh(), objective_states, "");
  if (visualize)
  {
    writer.write(0, 0.0, objective_states);
  }
  auto non_const_states = getFieldPointers(states, params);
  InertialReliefProblem problem({non_const_states[DISP], non_const_states[DENSITY]}, non_const_states,
                                solid_mechanics_residual, constraints);
  int dimx = problem.GetDimx();
  int dimy = problem.GetDimy();

  mfem::Vector x0(dimx);
  x0 = 0.0;
  mfem::Vector y0(dimy);
  y0 = 0.0;
  mfem::Vector xf(dimx);
  xf = 0.0;
  mfem::Vector yf(dimy);
  yf = 0.0;

  mfem::Vector q0(dimy); q0 = 0.0;
  mfem::Vector qf(dimy); qf = 0.0;
  [[maybe_unused]] mfem::HypreParMatrix * dQy;
  int eval_err = 0;
  problem.Q(x0, y0, q0, eval_err);
  std::cout << "queried Q once\n";
  //dQy = problem.DyQ(x0, y0);
  problem.Q(xf, yf, qf, eval_err);
  std::cout << "queried Q twice\n";
  //problem.Q(x0, y0, q0, eval_err);
  //std::cout << "queried Q thrice\n";
  //dQy = problem.DyQ(x0, y0);
  //problem.Q(xf, yf, qf, eval_err);
  
  //HomotopySolver solver(&problem);
  //mfem::MINRESSolver linSolver(MPI_COMM_WORLD);
  //linSolver.SetPrintLevel(1);
  //linSolver.SetMaxIter(400);
  //linSolver.SetRelTol(1.e-10);
  //solver.SetLinearSolver(linSolver);
  //solver.SetTol(nonlinear_absolute_tol);
  //solver.SetMaxIter(nonlinear_max_iterations);

  //solver.Mult(x0, y0, xf, yf);
  //bool converged = solver.GetConverged();
  //if (myid == 0) {
  //  if (converged) {
  //    std::cout << "converged!\n";
  //  } else {
  //    std::cout << "homotopy solver did not converge\n";
  //  }
  //}
  //if (visualize)
  //{
  //   writer.write(1, 1.0, objective_states);
  //}
}

InertialReliefProblem::InertialReliefProblem(std::vector<serac::FiniteElementState*> obj_states_,
                                             std::vector<serac::FiniteElementState*> all_states_,
                                             std::shared_ptr<serac::Residual> residual_,
                                             std::vector<std::shared_ptr<serac::ScalarObjective>> constraints_)
    : GeneralNLMCProblem(),
      dFdx(nullptr),
      dFdy(nullptr),
      dQdx(nullptr),
      dQdy(nullptr),
      uOffsets(nullptr),
      cOffsets(nullptr)
{
  residual = residual_;
  constraints.resize(constraints_.size());
  std::copy(constraints_.begin(), constraints_.end(), constraints.begin());

  all_states.resize(all_states_.size());
  std::copy(all_states_.begin(), all_states_.end(), all_states.begin());

  obj_states.resize(obj_states_.size());
  std::copy(obj_states_.begin(), obj_states_.end(), obj_states.begin());

  int numConstraints = static_cast<int>(constraints.size());
  uOffsets = new HYPRE_BigInt[2];
  cOffsets = new HYPRE_BigInt[2];
  for (int i = 0; i < 2; i++) {
    uOffsets[i] = all_states[FIELD::DISP]->space().GetTrueDofOffsets()[i];
  }

  int myid = mfem::Mpi::WorldRank();
  for (int i = 0; i < 2; i++) {
    std::cout << "uOffsets_" << i << " = " << uOffsets[i] << " (rank " << myid << ")\n";
  }
  if (myid == 0) {
    cOffsets[0] = 0;
  } else {
    cOffsets[0] = numConstraints;
  }
  cOffsets[1] = numConstraints;

  dimu = uOffsets[1] - uOffsets[0];
  dimc = cOffsets[1] - cOffsets[0];
  y_partition.SetSize(3);
  y_partition[0] = 0;
  y_partition[1] = dimu;
  y_partition[2] = dimc;
  y_partition.PartialSum();

  {
    HYPRE_BigInt dofOffsets[2];
    HYPRE_BigInt complementarityOffsets[2];
    for (int i = 0; i < 2; i++) {
      dofOffsets[i] = uOffsets[i] + cOffsets[i];
    }
    complementarityOffsets[0] = 0;
    complementarityOffsets[1] = 0;
    Init(complementarityOffsets, dofOffsets);
  }

  // dF / dx 0 x 0 matrix
  {
    int nentries = 0;
    auto temp = new mfem::SparseMatrix(dimx, dimxglb, nentries);
    dFdx = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsx, dofOffsetsx, temp);
    delete temp;
  }

  // dF / dy 0 x dimy matrix
  {
    int nentries = 0;
    auto temp = new mfem::SparseMatrix(dimx, dimyglb, nentries);
    dFdy = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsy, dofOffsetsx, temp);
    delete temp;
  }

  // dQ / dx dimy x 0 matrix
  {
    int nentries = 0;
    auto temp = new mfem::SparseMatrix(dimy, dimxglb, nentries);
    dQdx = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsx, dofOffsetsy, temp);
    delete temp;
  }
}

void InertialReliefProblem::F(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& feval, int& Feval_err) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && feval.Size() == dimx,
              "InertialReliefProblem::F -- Inconsistent dimensions");
  feval = 0.0;
  Feval_err = 0;
}

// Q = [ r + J^T l]
//     [ c ]
// dQ / dy = [K  J^T]
//           [J   0 ]
// Q(x, y) = 0 (equalities)
void InertialReliefProblem::Q(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& qeval, int& Qeval_err) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimy,
              "InertialReliefProblem::Q -- Inconsistent dimensions");
  std::cout << "in Q, (rank " << mfem::Mpi::WorldRank() << ")\n";
  qeval = 0.0;
  mfem::BlockVector yblock(y_partition);
  yblock.Set(1.0, y);
  mfem::BlockVector qblock(y_partition);
  qblock = 0.0;

  obj_states[DISP]->Set(1.0, yblock.GetBlock(0));
  //auto const_all_states = serac::getConstFieldPointers(all_states);
  //auto const_obj_states = serac::getConstFieldPointers(obj_states);
  serac::FiniteElementDual res_vector(all_states[DISP]->space(), "tempresidual");
  res_vector = residual->residual(time, dt, serac::getConstFieldPointers(all_states));
  //res_vector = residual->residual(time, dt, const_all_states);
  qblock.GetBlock(0).Set(-1.0, res_vector);

  double * multipliers = new double[6];
  double * send_multipliers = new double[static_cast<size_t>(dimc)];
  for (int i = 0; i < dimc; i++)
  {
     send_multipliers[i] = yblock.GetBlock(1)(i);
  }
  MPI_Allgather(send_multipliers, dimc, MPI_DOUBLE, multipliers, 6, MPI_DOUBLE, MPI_COMM_WORLD);

  mfem::Vector gradc(dimu);
  gradc = 0.0;

  for (size_t i = 0; i < constraints.size(); i++) {
    const int idx = static_cast<int>(i);
    const size_t i2 = static_cast<size_t>(idx);
    SLIC_ERROR_ROOT_IF(i2 != i, axom::fmt::format("Constraint index is out of range, bad cast from size_t to int"));
    gradc = 0.0;
    std::cout << "about to compute constraint gradient\n";
    mfem::Vector grad_temp = constraints[i]->gradient(time, dt, serac::getConstFieldPointers(obj_states), DISP);
    //mfem::Vector grad_temp = constraints[i]->gradient(time, dt, const_obj_states, DISP);
    std::cout << "computed the gradient\n";
    gradc.Set(1.0, grad_temp);
    //gradc.Set(1.0, constraints[i]->gradient(time, dt, serac::getConstFieldPointers(obj_states), DISP));
    
    
    qblock.GetBlock(0).Add(multipliers[idx], gradc);

    double constraint_i = constraints[i]->evaluate(time, dt, serac::getConstFieldPointers(obj_states));
    //double constraint_i = constraints[i]->evaluate(time, dt, const_obj_states);


    if (dimc > 0)
    {
      qblock.GetBlock(1)(idx) = -1.0 * constraint_i;
      std::cout << "c_" << i << " = " << constraint_i << std::endl;
    }
  }

  qeval.Set(1.0, qblock);

  Qeval_err = 0;
  int Qeval_err_loc = 0;
  for (int i = 0; i < qeval.Size(); i++) {
    if (std::isnan(qeval(i))) {
      Qeval_err_loc = 1;
      break;
    }
  }
  MPI_Allreduce(&Qeval_err_loc, &Qeval_err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (Qeval_err > 0)
  {
    Qeval_err = 1;
  }
  if (Qeval_err > 0 && mfem::Mpi::WorldRank() == 0) {
    std::cout << "at least one nan entry\n";
  }
  std::cout << "end Q, (rank " << mfem::Mpi::WorldRank() << ")\n";
}

mfem::HypreParMatrix* InertialReliefProblem::DxF(const mfem::Vector& /*x*/, const mfem::Vector& /*y*/) { return dFdx; }

mfem::HypreParMatrix* InertialReliefProblem::DyF(const mfem::Vector& /*x*/, const mfem::Vector& /*y*/) { return dFdy; }

mfem::HypreParMatrix* InertialReliefProblem::DxQ(const mfem::Vector& /*x*/, const mfem::Vector& /*y*/) { return dQdx; }

mfem::HypreParMatrix* InertialReliefProblem::DyQ(const mfem::Vector& /*x*/, const mfem::Vector& y)
{
  MFEM_VERIFY(y.Size() == dimy,
              "InertialReliefProblem::DyQ -- Inconsistent dimensions");
  std::cout << "in DyQ, (rank " << mfem::Mpi::WorldRank() << ")\n";
  // dQdy = [dr/du   dc/du^T]
  //        [dc/du   0  ]
  // note we are neglecting Hessian constraint terms
  mfem::BlockVector yblock(y_partition);
  yblock.Set(1.0, y);
  mfem::BlockVector qblock(y_partition);
  qblock = 0.0;
  std::cout << "settign obj_states[DISP]\n";
  obj_states[DISP]->Set(1.0, yblock.GetBlock(0));
  std::cout << "set obj_states[DISP]\n";
  if (dQdy) {
    delete dQdy;
  }
  {
    mfem::HypreParMatrix* drdu = nullptr;
    std::cout << "computing Jacobian\n";
    auto drdu_unique = residual->jacobian(time, dt, getConstFieldPointers(all_states), jacobian_weights);
    std::cout << "computed Jacobian\n";
    MFEM_VERIFY(drdu_unique->Height() == dimu, "size error");

    drdu = drdu_unique.release();
    *drdu *= -1.0;

    mfem::HypreParMatrix* dcdu = nullptr;
    mfem::SparseMatrix* dcdumat = nullptr;
    int myid = mfem::Mpi::WorldRank();
    int dimuglb = drdu->GetGlobalNumCols();
    int nentries = dimuglb;
    if (myid > 0)
    {
       nentries = 0;
    }
    dcdumat = new mfem::SparseMatrix(dimc, dimuglb, nentries);

    mfem::Array<int> cols;
    cols.SetSize(dimuglb);
    for (int i = 0; i < dimuglb; i++)
    {
       cols[i] = i;
    }
    for (size_t i = 0; i < constraints.size(); i++)
    {
      const int idx = static_cast<int>(i);
      const size_t i2 = static_cast<size_t>(idx);
      SLIC_ERROR_ROOT_IF(i2 != i, axom::fmt::format("Constraint index is out of range, bad cast from size_t to int"));
       // gradient --> HypreParVec --> Global Vec --> passed to row on process 0
       mfem::HypreParVector gradVector(MPI_COMM_WORLD, dimuglb, uOffsets);
       gradVector.Set(1.0, constraints[i]->gradient(time, dt, serac::getConstFieldPointers(obj_states), DISP));
       mfem::Vector globalGradVector(dimuglb);
       globalGradVector = 0.0;
       globalGradVector.Add(1.0, *gradVector.GlobalVector());
       if (myid == 0)
       {
          dcdumat->SetRow(idx, cols, globalGradVector);
       }
    }

    dcdumat->Threshold(1.e-20);
    dcdumat->Finalize();
    dcdu = GenerateHypreParMatrixFromSparseMatrix(uOffsets, cOffsets, dcdumat);
    mfem::HypreParMatrix* dcduT = dcdu->Transpose();
    mfem::Vector scale(dcdu->Height());
    scale = -1.0;
    dcdu->ScaleRows(scale);

    mfem::Array2D<const mfem::HypreParMatrix*> BlockMat(2, 2);
    BlockMat(0, 0) = drdu;
    BlockMat(0, 1) = dcduT;
    BlockMat(1, 0) = dcdu;
    BlockMat(1, 1) = nullptr;
    dQdy = HypreParMatrixFromBlocks(BlockMat);
    //delete drdu;
    //delete dcduT;
    //delete dcdu;
  }
  std::cout << "end DyQ, (rank " << mfem::Mpi::WorldRank() << ")\n";
  dQdy->Print("dQdy");
  std::cout << "Printed dQdy\n";
  return dQdy;
}

InertialReliefProblem::~InertialReliefProblem()
{
  delete[] uOffsets;
  delete[] cOffsets;
  delete dFdx;
  delete dFdy;
  delete dQdx;
  delete dQdy;
}
