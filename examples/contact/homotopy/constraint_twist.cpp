// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>

#include <set>
#include <string>

#include "axom/slic.hpp"

#include "mfem.hpp"

// ContinuationSolver headers
#include "problems/Problems.hpp"
#include "solvers/HomotopySolver.hpp"
#include "utilities.hpp"

#include "serac/serac.hpp"

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
template <typename SolidWeakFormType>
class TiedContactProblem : public GeneralNLMCProblem {
 protected:
  mfem::HypreParMatrix* dFdx_ = nullptr;
  mfem::HypreParMatrix* dFdy_ = nullptr;
  mfem::HypreParMatrix* dQdx_ = nullptr;
  mfem::HypreParMatrix* dQdy_ = nullptr;
  HYPRE_BigInt* uOffsets_ = nullptr;
  HYPRE_BigInt* cOffsets_ = nullptr;
  int dimu_;
  int dimc_;
  mfem::Array<int> y_partition_;
  std::vector<serac::FieldPtr> contact_states_;
  std::vector<serac::FieldPtr> all_states_;
  std::shared_ptr<SolidWeakFormType> weak_form_;
  std::unique_ptr<serac::FiniteElementState> shape_disp_;
  std::shared_ptr<serac::Mesh> mesh_;
  std::shared_ptr<serac::ContactConstraint> constraints_;
  double time_ = 0.0;
  double dt_ = 0.0;
  std::vector<double> jacobian_weights_ = {1.0, 0.0, 0.0, 0.0};

 public:
  TiedContactProblem(std::vector<serac::FieldPtr> contact_states, std::vector<serac::FieldPtr> all_states,
                     std::shared_ptr<serac::Mesh> mesh, std::shared_ptr<SolidWeakFormType> weak_form,
                     std::shared_ptr<serac::ContactConstraint> constraints);
  void F(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& feval, int& Feval_err) const;
  void Q(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& qeval, int& Qeval_err) const;
  mfem::HypreParMatrix* DxF(const mfem::Vector& x, const mfem::Vector& y);
  mfem::HypreParMatrix* DyF(const mfem::Vector& x, const mfem::Vector& y);
  mfem::HypreParMatrix* DxQ(const mfem::Vector& x, const mfem::Vector& y);
  mfem::HypreParMatrix* DyQ(const mfem::Vector& x, const mfem::Vector& y);
  virtual ~TiedContactProblem();
};

// this example is intended to eventually replace twist.cpp
int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  serac::ApplicationManager applicationManager(argc, argv);

  //// NOTE: p must be equal to 1 to work with Tribol's mortar method
  // constexpr int p = 1;
  //// NOTE: dim must be equal to 3
  // constexpr int dim = 3;

  // using VectorSpace = serac::H1<p, dim>;

  // Create DataStore
  std::string name = "contact_twist_example";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/twohex_for_contact.mesh";
  auto mesh = std::make_shared<serac::Mesh>(serac::buildMeshFromFile(filename), "twist_mesh", 3, 0);

  mesh->addDomainOfBoundaryElements("fixed_surface", serac::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("driven_surface", serac::by_attr<dim>(6));

  serac::ContactOptions contact_options{.method = serac::ContactMethod::SingleMortar,
                                        .enforcement = serac::ContactEnforcement::LagrangeMultiplier,
                                        .type = serac::ContactType::Frictionless,
                                        .jacobian = serac::ContactJacobian::Exact};

  std::string contact_constraint_name = "default_contact";

  // Specify the contact interaction
  auto contact_interaction_id = 0;
  std::set<int> surface_1_boundary_attributes({4});
  std::set<int> surface_2_boundary_attributes({5});
  auto contact_constraint = std::make_shared<serac::ContactConstraint>(
      contact_interaction_id, mesh->mfemParMesh(), surface_1_boundary_attributes, surface_2_boundary_attributes,
      contact_options, contact_constraint_name);

  serac::FiniteElementState shape = serac::StateManager::newState(VectorSpace{}, "shape", mesh->tag());
  serac::FiniteElementState disp = serac::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
  serac::FiniteElementState velo = serac::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
  serac::FiniteElementState accel = serac::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
  serac::FiniteElementState density = serac::StateManager::newState(DensitySpace{}, "density", mesh->tag());

  std::vector<serac::FiniteElementState> contact_states;
  std::vector<serac::FiniteElementState> states;
  std::vector<serac::FiniteElementState> params;
  contact_states = {shape, disp};
  states = {disp, velo, accel};
  params = {density};

  // initialize displacement
  contact_states[serac::ContactFields::DISP].setFromFieldFunction([](serac::tensor<double, dim> x) {
    auto u = 0.1 * x;
    return u;
  });

  contact_states[serac::ContactFields::SHAPE] = 0.0;
  states[FIELD::VELO] = 0.0;
  states[FIELD::ACCEL] = 0.0;
  params[0] = 1.0;

  std::string physics_name = "solid";

  // construct residual
  auto solid_mechanics_weak_form =
      std::make_shared<SolidWeakFormT>(physics_name, mesh, states[FIELD::DISP].space(), getSpaces(params));

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

  auto all_states = serac::getConstFieldPointers(states, params);
  auto non_const_states = serac::getFieldPointers(states, params);
  auto contact_state_ptrs = serac::getFieldPointers(contact_states);
  TiedContactProblem<SolidWeakFormT> problem(contact_state_ptrs, non_const_states, mesh, solid_mechanics_weak_form,
                                             contact_constraint);
  
  //int dimx = problem.GetDimx();
  //int dimy = problem.GetDimy();

  //double nonlinear_absolute_tol = 1.e-4;
  //int nonlinear_max_iterations = 30;
  //mfem::Vector x0(dimx);
  //x0 = 0.0;
  //mfem::Vector y0(dimy);
  //y0 = 0.0;
  //mfem::Vector xf(dimx);
  //xf = 0.0;
  //mfem::Vector yf(dimy);
  //yf = 0.0;

  //mfem::Vector q0(dimy);
  //mfem::Vector f0(dimx); 
  //int qeval_err, feval_err;
  //problem.Q(x0, y0, q0, qeval_err);
  //if (qeval_err)
  //{
  //   std::cout << "qeval_err\n";
  //   exit(1);
  //}
  //problem.F(x0, y0, f0, feval_err);
  //if (feval_err)
  //{
  //   std::cout << "feval_err\n";
  //   exit(1);
  //}
  //problem.Q(x0, y0, q0, qeval_err);
  //if (qeval_err)
  //{
  //   std::cout << "qeval_err\n";
  //   exit(1);
  //}
  //problem.F(x0, y0, f0, feval_err);
  //if (feval_err)
  //{
  //   std::cout << "feval_err\n";
  //   exit(1);
  //}

  //HomotopySolver solver(&problem);
  //solver.SetTol(nonlinear_absolute_tol);
  //solver.SetMaxIter(nonlinear_max_iterations);

  //solver.Mult(x0, y0, xf, yf);
  //bool converged = solver.GetConverged();
  //int myid = mfem::Mpi::WorldRank();
  //if (myid == 0) {
  //  if (converged) {
  //    std::cout << "converged!\n";
  //  } else {
  //    std::cout << "homotopy solver did not converge\n";
  //  }
  //}


  double time = 0.0, dt = 1.0;
  int direction = serac::ContactFields::DISP;
  auto input_states = getConstFieldPointers(contact_states);
  int nPressureDofs = contact_constraint->numPressureDofs();
  mfem::Vector multipliers(nPressureDofs);
  multipliers = 0.0;
  auto residual = contact_constraint->residual_contribution(time, dt, input_states, multipliers, direction);
  auto gap = contact_constraint->evaluate(time, dt, input_states);
  auto gap_Jacobian = contact_constraint->jacobian(time, dt, input_states, direction);
  auto gap_Jacobian_tilde = contact_constraint->jacobian_tilde(time, dt, input_states, direction);


  //auto residual_Jacobian = contact_constraint->residual_contribution_jacobian(time, dt,
  //       	                                                             input_states, multipliers,
  //       								     direction);

  return 0;
}

template <typename SolidWeakFormType>
TiedContactProblem<SolidWeakFormType>::TiedContactProblem(std::vector<serac::FiniteElementState*> contact_states,
                                                          std::vector<serac::FiniteElementState*> all_states,
                                                          std::shared_ptr<serac::Mesh> mesh,
                                                          std::shared_ptr<SolidWeakFormType> weak_form,
                                                          std::shared_ptr<serac::ContactConstraint> constraints)
    : GeneralNLMCProblem()
{
  weak_form_ = weak_form;
  mesh_ = mesh;
  shape_disp_ = std::make_unique<serac::FiniteElementState>(mesh->newShapeDisplacement());

  constraints_ = constraints;

  all_states_.resize(all_states.size());
  std::copy(all_states.begin(), all_states.end(), all_states_.begin());

  contact_states_.resize(contact_states.size());
  std::copy(contact_states.begin(), contact_states.end(), contact_states_.begin());

  int numConstraints = constraints_->numPressureDofs();

  uOffsets_ = new HYPRE_BigInt[2];
  cOffsets_ = new HYPRE_BigInt[2];
  for (int i = 0; i < 2; i++) {
    uOffsets_[i] = all_states_[FIELD::DISP]->space().GetTrueDofOffsets()[i];
  }

  int cumulativeConstraints = 0;
  MPI_Scan(&numConstraints, &cumulativeConstraints, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  cOffsets_[0] = cumulativeConstraints - numConstraints;
  cOffsets_[1] = cumulativeConstraints;

  dimu_ = uOffsets_[1] - uOffsets_[0];
  dimc_ = cOffsets_[1] - cOffsets_[0];
  y_partition_.SetSize(3);
  y_partition_[0] = 0;
  y_partition_[1] = dimu_;
  y_partition_[2] = dimc_;
  y_partition_.PartialSum();

  {
    HYPRE_BigInt dofOffsets[2];
    HYPRE_BigInt complementarityOffsets[2];
    for (int i = 0; i < 2; i++) {
      dofOffsets[i] = uOffsets_[i] + cOffsets_[i];
    }
    complementarityOffsets[0] = 0;
    complementarityOffsets[1] = 0;
    Init(complementarityOffsets, dofOffsets);
  }

  // dF / dx 0 x 0 matrix
  {
    int nentries = 0;
    auto temp = new mfem::SparseMatrix(dimx, dimxglb, nentries);
    dFdx_ = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsx, dofOffsetsx, temp);
    delete temp;
  }

  // dF / dy 0 x dimy matrix
  {
    int nentries = 0;
    auto temp = new mfem::SparseMatrix(dimx, dimyglb, nentries);
    dFdy_ = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsy, dofOffsetsx, temp);
    delete temp;
  }

  // dQ / dx dimy x 0 matrix
  {
    int nentries = 0;
    auto temp = new mfem::SparseMatrix(dimy, dimxglb, nentries);
    dQdx_ = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsx, dofOffsetsy, temp);
    delete temp;
  }
}

template <typename SolidWeakFormType>
void TiedContactProblem<SolidWeakFormType>::F(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& feval,
                                              int& Feval_err) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && feval.Size() == dimx,
              "TiedContactProblem::F -- Inconsistent dimensions");
  feval = 0.0;
  Feval_err = 0;
}

// Q = [  r + J^T l]
//     [ -c ]
// dQ / dy = [ K  J^T]
//           [-J   0 ]
template <typename SolidWeakFormType>
void TiedContactProblem<SolidWeakFormType>::Q(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& qeval,
                                              int& Qeval_err) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimy,
              "TiedContactProblem::Q -- Inconsistent dimensions");
  qeval = 0.0;
  mfem::BlockVector yblock(y_partition_);
  yblock.Set(1.0, y);
  mfem::BlockVector qblock(y_partition_);
  qblock = 0.0;

  contact_states_[serac::ContactFields::DISP]->Set(1.0, yblock.GetBlock(0));
  std::cout << "updated contact state\n";
  auto res_vector = weak_form_->residual(time_, dt_, shape_disp_.get(), serac::getConstFieldPointers(all_states_));
  std::cout << "residual computed\n";

  // TODO: why does gap need to be computed prior to residaul contributio
  auto gap = constraints_->evaluate(time_, dt_, serac::getConstFieldPointers(contact_states_));

  auto res_contribution = constraints_->residual_contribution(time_, dt_, serac::getConstFieldPointers(contact_states_),
                                                              yblock.GetBlock(1), serac::ContactFields::DISP);
  std::cout << "residual contribution computed\n";
  std::cout << "gap computed\n";

  qblock.GetBlock(0).Set(1.0, res_vector);
  qblock.GetBlock(0).Add(1.0, res_contribution);
  qblock.GetBlock(1).Set(-1.0, gap);

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
  if (Qeval_err > 0) {
    Qeval_err = 1;
  }
  if (Qeval_err > 0 && mfem::Mpi::WorldRank() == 0) {
    std::cout << "at least one nan entry\n";
  }
}

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::DxF(const mfem::Vector& /*x*/, const mfem::Vector& /*y*/)
{
  return dFdx_;
}

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::DyF(const mfem::Vector& /*x*/, const mfem::Vector& /*y*/)
{
  return dFdy_;
}

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::DxQ(const mfem::Vector& /*x*/, const mfem::Vector& /*y*/)
{
  return dQdx_;
}

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::DyQ(const mfem::Vector& /*x*/, const mfem::Vector& y)
{
  MFEM_VERIFY(y.Size() == dimy, "TiedContactProblem::DyQ -- Inconsistent dimensions");
  // dQdy = [dr/du   dc/du^T]
  //        [dc/du   0  ]
  // note we are neglecting Hessian constraint terms
  mfem::BlockVector yblock(y_partition_);
  yblock.Set(1.0, y);
  mfem::BlockVector qblock(y_partition_);
  qblock = 0.0;
  contact_states_[serac::ContactFields::DISP]->Set(1.0, yblock.GetBlock(0));
  if (dQdy_) {
    delete dQdy_;
  }
  {
    mfem::HypreParMatrix* drdu = nullptr;
    auto drdu_unique =
        weak_form_->jacobian(time_, dt_, shape_disp_.get(), getConstFieldPointers(all_states_), jacobian_weights_);
    MFEM_VERIFY(drdu_unique->Height() == dimu_, "size error");

    drdu = drdu_unique.release();

    auto negdcdu_unique =
        constraints_->jacobian(time_, dt_, serac::getConstFieldPointers(contact_states_), serac::ContactFields::DISP);
    auto negdcdu = negdcdu_unique.release();
    mfem::Vector scale(dimc_);
    scale = -1.0;
    negdcdu->ScaleRows(scale);

    auto dcdutilde_unique = constraints_->jacobian_tilde(time_, dt_, serac::getConstFieldPointers(contact_states_),
                                                         serac::ContactFields::DISP);
    auto dcdutildeT = dcdutilde_unique->Transpose();

    mfem::Array2D<const mfem::HypreParMatrix*> BlockMat(2, 2);
    BlockMat(0, 0) = drdu;
    BlockMat(0, 1) = dcdutildeT;
    BlockMat(1, 0) = negdcdu;
    BlockMat(1, 1) = nullptr;
    dQdy_ = HypreParMatrixFromBlocks(BlockMat);
    delete drdu;
    delete dcdutildeT;
    delete negdcdu;
  }
  return dQdy_;
}

template <typename SolidWeakFormType>
TiedContactProblem<SolidWeakFormType>::~TiedContactProblem()
{
  delete[] uOffsets_;
  delete[] cOffsets_;
  delete dFdx_;
  delete dFdy_;
  delete dQdx_;
  delete dQdy_;
}
