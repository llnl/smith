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
  mfem::HypreParMatrix* dFdx;
  mfem::HypreParMatrix* dFdy;
  mfem::HypreParMatrix* dQdx;
  mfem::HypreParMatrix* dQdy;
  HYPRE_BigInt* uOffsets;
  HYPRE_BigInt* cOffsets;
  int dimu;
  int dimc;
  int dimcglb;
  mfem::Array<int> y_partition;
  std::vector<serac::FieldPtr> contact_states;
  std::vector<serac::FieldPtr> all_states;
  std::shared_ptr<SolidWeakFormType> weak_form;
  std::unique_ptr<serac::FiniteElementState> shape_disp;
  std::shared_ptr<serac::Mesh> mesh;
  std::shared_ptr<serac::ContactConstraint> constraints;
  double time = 0.0;
  double dt = 0.0;
  std::vector<double> jacobian_weights = {1.0, 0.0, 0.0, 0.0};

 public:
  TiedContactProblem(std::vector<serac::FieldPtr> contact_states_, std::vector<serac::FieldPtr> all_states_,
                        std::shared_ptr<serac::Mesh> mesh_, std::shared_ptr<SolidWeakFormType> weak_form_,
                        std::shared_ptr<serac::ContactConstraint> constraints_);
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
  //constexpr int p = 1;
  //// NOTE: dim must be equal to 3
  //constexpr int dim = 3;

  //using VectorSpace = serac::H1<p, dim>;

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
  auto contact_constraint = std::make_shared<serac::ContactConstraint>(contact_interaction_id, mesh->mfemParMesh(),
                                              surface_1_boundary_attributes, surface_2_boundary_attributes,
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
  TiedContactProblem<SolidWeakFormT> problem(contact_state_ptrs, non_const_states, mesh, solid_mechanics_weak_form, contact_constraint);


  //double time = 0.0, dt = 1.0;
  //int direction = serac::ContactFields::DISP;
  //auto input_states = getConstFieldPointers(contact_states);
  //auto gap = contact_constraint.evaluate(time, dt, input_states);
  //auto gap_Jacobian = contact_constraint.jacobian(time, dt, input_states, direction);
  //auto gap_Jacobian_tilde = contact_constraint.jacobian_tilde(time, dt, input_states, direction);

  //int nPressureDofs = contact_constraint.numPressureDofs();
  //mfem::Vector multipliers(nPressureDofs);
  //multipliers = 0.0;
  //auto residual = contact_constraint.residual_contribution(time, dt, input_states, multipliers, direction);
  
  
  
  // auto residual_Jacobian = contact_constraint.residual_contribution_jacobian(time, dt,
  //        	                                                             input_states, multipliers,
  //        								     direction);

  return 0;
}

template <typename SolidWeakFormType>
TiedContactProblem<SolidWeakFormType>::TiedContactProblem(std::vector<serac::FiniteElementState*> contact_states_,
                                             std::vector<serac::FiniteElementState*> all_states_,
                                             std::shared_ptr<serac::Mesh> mesh_,
                                             std::shared_ptr<SolidWeakFormType> weak_form_,
                                             std::shared_ptr<serac::ContactConstraint> constraints_)
    : GeneralNLMCProblem(),
      dFdx(nullptr),
      dFdy(nullptr),
      dQdx(nullptr),
      dQdy(nullptr),
      uOffsets(nullptr),
      cOffsets(nullptr)
{
  weak_form = weak_form_;
  mesh = mesh_;
  shape_disp = std::make_unique<serac::FiniteElementState>(mesh->newShapeDisplacement());

  constraints = constraints_;

  all_states.resize(all_states_.size());
  std::copy(all_states_.begin(), all_states_.end(), all_states.begin());

  contact_states.resize(contact_states_.size());
  std::copy(contact_states_.begin(), contact_states_.end(), contact_states.begin());

  int numConstraints = constraints->numPressureDofs();

  uOffsets = new HYPRE_BigInt[2];
  cOffsets = new HYPRE_BigInt[2];
  for (int i = 0; i < 2; i++) {
    uOffsets[i] = all_states[FIELD::DISP]->space().GetTrueDofOffsets()[i];
  }

  int cumulativeConstraints = 0;
  MPI_Scan(&numConstraints, &cumulativeConstraints, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  cOffsets[0] = cumulativeConstraints - numConstraints;
  cOffsets[1] = cumulativeConstraints;
  
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

template <typename SolidWeakFormType>
void TiedContactProblem<SolidWeakFormType>::F(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& feval, int& Feval_err) const
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
void TiedContactProblem<SolidWeakFormType>::Q(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& qeval, int& Qeval_err) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimy,
              "TiedContactProblem::Q -- Inconsistent dimensions");
  qeval = 0.0;
  mfem::BlockVector yblock(y_partition);
  yblock.Set(1.0, y);
  mfem::BlockVector qblock(y_partition);
  qblock = 0.0;

  contact_states[serac::ContactFields::DISP]->Set(1.0, yblock.GetBlock(0));
  serac::FiniteElementDual res_vector(all_states[FIELD::DISP]->space(), "tempresidual");
  res_vector = weak_form->residual(time, dt, shape_disp.get(), serac::getConstFieldPointers(all_states));

  // TODO: is this the right field pointer to pass?
  auto res_contribution = constraints->residual_contribution(time, dt, serac::getConstFieldPointers(contact_states), yblock.GetBlock(1), serac::ContactFields::DISP);
  auto gap = constraints->evaluate(time, dt, serac::getConstFieldPointers(contact_states));

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
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::DxF(const mfem::Vector& /*x*/, const mfem::Vector& /*y*/) { return dFdx; }

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::DyF(const mfem::Vector& /*x*/, const mfem::Vector& /*y*/) { return dFdy; }

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::DxQ(const mfem::Vector& /*x*/, const mfem::Vector& /*y*/) { return dQdx; }

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::DyQ(const mfem::Vector& /*x*/, const mfem::Vector& y)
{
  MFEM_VERIFY(y.Size() == dimy, "TiedContactProblem::DyQ -- Inconsistent dimensions");
  // dQdy = [dr/du   dc/du^T]
  //        [dc/du   0  ]
  // note we are neglecting Hessian constraint terms
  mfem::BlockVector yblock(y_partition);
  yblock.Set(1.0, y);
  mfem::BlockVector qblock(y_partition);
  qblock = 0.0;
  contact_states[serac::ContactFields::DISP]->Set(1.0, yblock.GetBlock(0));
  if (dQdy) {
    delete dQdy;
  }
  {
    mfem::HypreParMatrix* drdu = nullptr;
    auto drdu_unique =
        weak_form->jacobian(time, dt, shape_disp.get(), getConstFieldPointers(all_states), jacobian_weights);
    MFEM_VERIFY(drdu_unique->Height() == dimu, "size error");

    drdu = drdu_unique.release();

    auto negdcdu_unique = constraints->jacobian(time, dt, serac::getConstFieldPointers(contact_states), serac::ContactFields::DISP);
    auto negdcdu = negdcdu_unique.release();
    mfem::Vector scale(dimc); 
    scale = -1.0;
    negdcdu->ScaleRows(scale);

    auto dcdutilde_unique = constraints->jacobian_tilde(time, dt, serac::getConstFieldPointers(contact_states), serac::ContactFields::DISP);
    auto dcdutildeT = dcdutilde_unique->Transpose();

    mfem::Array2D<const mfem::HypreParMatrix*> BlockMat(2, 2);
    BlockMat(0, 0) = drdu;
    BlockMat(0, 1) = dcdutildeT;
    BlockMat(1, 0) = negdcdu;
    BlockMat(1, 1) = nullptr;
    dQdy = HypreParMatrixFromBlocks(BlockMat);
    delete drdu;
    delete dcdutildeT;
    delete negdcdu;
  }
  return dQdy;
}

template <typename SolidWeakFormType>
TiedContactProblem<SolidWeakFormType>::~TiedContactProblem()
{
  delete[] uOffsets;
  delete[] cOffsets;
  delete dFdx;
  delete dFdy;
  delete dQdx;
  delete dQdy;
}

