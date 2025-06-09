// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file smith_example.cpp
 *
 * @brief Basic Smith example
 *
 * Intended to verify that external projects can include Smith
 */

#include "serac/physics/solid_residual.hpp"
#include "serac/physics/functional_objective.hpp"

#include "serac/infrastructure/application_manager.hpp"
#include <serac/physics/state/state_manager.hpp>
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/common.hpp"
#include "mfem.hpp"
#include "serac/physics/tests/physics_test_utils.hpp"
#include "serac/numerics/functional/tensor.hpp"

// WORKS WITH CODEVELOP OFF
#include "problems/Problems.hpp"
#include "solvers/HomotopySolver.hpp"
#include "utilities.hpp"

// WORKS WITH CODEVELOP ON
// #include "ContinuationSolvers/problems/Problems.hpp"
// #include "ContinuationSolvers/solvers/HomotopySolver.hpp"
// #include "ContinuationSolvers/utilities.hpp"

#include <axom/sidre.hpp>

enum FIELD
{
  SHAPE_DISP,
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

  void write(int step, double time, const std::vector<serac::FiniteElementState*>& current_states)
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

auto createParaviewOutput(const mfem::ParMesh& mesh, const std::vector<serac::FiniteElementState*>& states,
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
  std::vector<serac::FiniteElementState*> obj_states;
  std::vector<serac::FiniteElementState*> all_states;
  std::shared_ptr<serac::Residual> residual;
  std::vector<std::shared_ptr<serac::ScalarObjective>> constraints;
  double time = 0.0;
  double dt = 0.0;
  std::vector<double> jacobian_weights = {0.0, 1.0, 0.0, 0.0, 0.0};

 public:
  InertialReliefProblem(std::vector<serac::FiniteElementState*> obj_states_,
                        std::vector<serac::FiniteElementState*> all_states_, std::shared_ptr<serac::Residual> residual_,
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
  serac::ApplicationManager applicationManager(argc, argv);

  int myid = mfem::Mpi::WorldRank();

  MPI_Barrier(MPI_COMM_WORLD);

  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_dynamics");

  double xlength = 0.5;
  double ylength = 0.7;
  double zlength = 0.3;

  std::shared_ptr<serac::Mesh> mesh;
  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;
  std::shared_ptr<serac::Residual> residual;
  std::vector<std::shared_ptr<serac::ScalarObjective>> constraints;

  int nx = 6;
  int ny = 4;
  int nz = 4;
  mesh = std::make_shared<serac::Mesh>(
      mfem::Mesh::MakeCartesian3D(nx, ny, nz, element_shape, xlength, ylength, zlength), "this_mesh_name", 0, 0);

  serac::FiniteElementState disp = serac::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
  serac::FiniteElementState shape_disp = serac::StateManager::newState(VectorSpace{}, "shape_disp", mesh->tag());
  serac::FiniteElementState velo = serac::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
  serac::FiniteElementState accel = serac::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
  serac::FiniteElementState density = serac::StateManager::newState(DensitySpace{}, "density", mesh->tag());

  states = {shape_disp, disp, velo, accel};
  params = {density};

  std::string physics_name = "solid";

  // construct residual
  auto solid_mechanics_residual = std::make_shared<SolidResidualT>(physics_name, mesh, states[SHAPE_DISP].space(),
                                                                   states[DISP].space(), getSpaces(params));

  SolidMaterial mat;
  mat.K = 1.0;
  mat.G = 0.5;
  solid_mechanics_residual->setMaterial(serac::DependsOn<0>{}, mesh->entireBodyName(), mat);

  // apply some traction boundary conditions
  std::string surface_name = "side";
  mesh->addDomainOfBoundaryElements(surface_name, serac::by_attr<dim>(1));
  solid_mechanics_residual->addBoundaryIntegral(surface_name, [](auto /*x*/, auto n, auto /*t*/) { return -1.0 * n; });

  serac::tensor<double, dim> constant_force{};
  for (int i = 0; i < dim; i++) {
    constant_force[i] = 1.e0;
  }

  solid_mechanics_residual->addBodyIntegral(mesh->entireBodyName(), [constant_force](double /* t */, auto x) {
    return serac::tuple{constant_force, 0.0 * serac::get<serac::DERIVATIVE>(x)};
  });
  residual = solid_mechanics_residual;

  // construct constraints
  params[0] = 1.0;

  using ObjectiveT = serac::FunctionalObjective<
      dim, VectorSpace, serac::Parameters<VectorSpace, DensitySpace>>;  // functional objective on displacement/density

  double time = 0.0;
  double dt = 1.0;
  auto all_states = getPointers(states, params);
  auto objective_states = {all_states[SHAPE_DISP], all_states[DISP], all_states[DENSITY]};

  ObjectiveT::SpacesT param_space_ptrs{&all_states[DISP]->space(), &all_states[DENSITY]->space()};

  ObjectiveT mass_objective("mass constraining", mesh, all_states[SHAPE_DISP]->space(), param_space_ptrs);

  mass_objective.addBodyIntegral(serac::DependsOn<1>{}, mesh->entireBodyName(),
                                 [](double /*t*/, auto /*X*/, auto RHO) { return get<serac::VALUE>(RHO); });
  double mass = mass_objective.evaluate(time, dt, objective_states);

  serac::tensor<double, dim> initial_cg;

  for (int i = 0; i < dim; ++i) {
    auto cg_objective = std::make_shared<ObjectiveT>("translation " + std::to_string(i), mesh,
                                                     all_states[SHAPE_DISP]->space(), param_space_ptrs);
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
    auto center_rotation_objective = std::make_shared<ObjectiveT>("rotation" + std::to_string(i), mesh,
                                                                  all_states[SHAPE_DISP]->space(), param_space_ptrs);
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

  serac::FiniteElementDual res_vector(states[DISP].space(), "residual");
  res_vector = residual->residual(time, dt, all_states);

  std::vector<double> jacobian_weights = {0.0, 1.0, 0.0, 0.0, 0.0};
  auto drdu_unique = residual->jacobian(time, dt, all_states, jacobian_weights);

  // initialize displacement
  states[FIELD::DISP].setFromFieldFunction([](serac::tensor<double, dim> x) {
    auto u = 0.1 * x;
    return u;
  });

  auto writer = createParaviewOutput(mesh->mfemParMesh(), objective_states, "");
  writer.write(0, 0.0, objective_states);
  InertialReliefProblem problem(objective_states, all_states, solid_mechanics_residual, constraints);
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

  HomotopySolver solver(&problem);
  double tol = 1.e-6;
  int maxIter = 100;
  solver.SetTol(tol);
  solver.SetMaxIter(maxIter);

  solver.Mult(x0, y0, xf, yf);
  bool converged = solver.GetConverged();
  if (myid == 0) {
    if (converged) {
      std::cout << "converged !\n";
    } else {
      std::cout << "homotopy solver did not converge\n";
    }
  }
  writer.write(1, 1.0, objective_states);
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
    HYPRE_BigInt* dofOffsets = new HYPRE_BigInt[2];
    HYPRE_BigInt* complementarityOffsets = new HYPRE_BigInt[2];
    for (int i = 0; i < 2; i++) {
      dofOffsets[i] = uOffsets[i] + cOffsets[i];
    }
    complementarityOffsets[0] = 0;
    complementarityOffsets[1] = 0;
    Init(complementarityOffsets, dofOffsets);
    delete[] dofOffsets;
    delete[] complementarityOffsets;
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

#if 1
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
  qeval = 0.0;
  mfem::BlockVector yblock(y_partition);
  yblock.Set(1.0, y);
  mfem::BlockVector qblock(y_partition);
  qblock = 0.0;

  obj_states[DISP]->Set(1.0, yblock.GetBlock(0));

  serac::FiniteElementDual res_vector(all_states[DISP]->space(), "tempresidual");
  res_vector = residual->residual(time, dt, all_states);
  qblock.GetBlock(0).Set(1.0, res_vector);

  Vector gradc(dimu);
  gradc = 0.0;
  for (size_t i = 0; i < constraints.size(); i++) {
    const int idx = static_cast<int>(i);
    const size_t i2 = static_cast<size_t>(idx);
    SLIC_ERROR_ROOT_IF(i2 != i, axom::fmt::format("Constraint index is out of range, bad cast from size_t to int"));
    gradc = 0.0;
    gradc.Set(1.0, constraints[i]->gradient(time, dt, obj_states, DISP));
    qblock.GetBlock(0).Add(yblock.GetBlock(1)(idx), gradc);
    qblock.GetBlock(1)(idx) = -1.0 * constraints[i]->evaluate(time, dt, obj_states);
  }
  qeval.Set(1.0, qblock);
  Qeval_err = 0;
  for (int i = 0; i < qeval.Size(); i++) {
    if (std::isnan(qeval(i))) {
      Qeval_err = 1;
    }
  }
  if (Qeval_err > 0 && mfem::Mpi::WorldRank() == 0) {
    cout << "at least one nan entry\n";
  }
}

mfem::HypreParMatrix* InertialReliefProblem::DxF([[maybe_unused]] const mfem::Vector& x,
                                                 [[maybe_unused]] const mfem::Vector& y)
{
  return dFdx;
}

mfem::HypreParMatrix* InertialReliefProblem::DyF([[maybe_unused]] const mfem::Vector& x,
                                                 [[maybe_unused]] const mfem::Vector& y)
{
  return dFdy;
}

mfem::HypreParMatrix* InertialReliefProblem::DxQ([[maybe_unused]] const mfem::Vector& x,
                                                 [[maybe_unused]] const mfem::Vector& y)
{
  return dQdx;
}

mfem::HypreParMatrix* InertialReliefProblem::DyQ([[maybe_unused]] const mfem::Vector& x,
                                                 [[maybe_unused]] const mfem::Vector& y)
{
  // see Homotopy Example5
  // dQdy = [dr/du   dc/du^T]
  //        [dc/du   0  ]
  // note we are neglecting Hessian constraint terms
  mfem::BlockVector yblock(y_partition);
  yblock.Set(1.0, y);
  mfem::BlockVector qblock(y_partition);
  qblock = 0.0;
  obj_states[DISP]->Set(1.0, yblock.GetBlock(0));

  if (dQdy) {
    delete dQdy;
  }
  {
    mfem::HypreParMatrix* drdu = nullptr;
    auto drdu_unique = residual->jacobian(time, dt, all_states, jacobian_weights);
    MFEM_VERIFY(drdu_unique->Height() == dimu, "size error");

    drdu = drdu_unique.release();

    mfem::HypreParMatrix* dcdu = nullptr;
    mfem::SparseMatrix* dcdumat = nullptr;
    // TODO: the following won't work with more than 1 MPI process
    if (dimc > 0) {
      dcdumat = new mfem::SparseMatrix(dimc, drdu->GetGlobalNumCols(), dimu);

      mfem::Array<int> cols;
      cols.SetSize(dimu);
      mfem::Vector entries(dimu);
      entries = 0.;
      for (int i = 0; i < dimu; i++) {
        cols[i] = i;
      }
      for (int i = 0; i < dimc; i++) {
        entries = 0.;
        entries.Add(
            1.0, constraints[static_cast<size_t>(i)]->gradient(time, dt, obj_states, DISP));  // j = 0 shape displacement, u displacemtn j =1
        dcdumat->SetRow(i, cols, entries);
      }
    } else {
      dcdumat = new mfem::SparseMatrix(dimc, drdu->GetGlobalNumCols(), dimu);
    }
    dcdumat->Threshold(1.e-20);
    dcdumat->Finalize();
    dcdu = GenerateHypreParMatrixFromSparseMatrix(uOffsets, cOffsets, dcdumat);  // utility function call
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

    delete drdu;
    delete dcduT;
    delete dcdu;
  }
  return dQdy;
}
#endif

InertialReliefProblem::~InertialReliefProblem()
{
  delete[] uOffsets;
  delete[] cOffsets;
  delete dFdx;
  delete dFdy;
  delete dQdx;
  delete dQdy;
}
