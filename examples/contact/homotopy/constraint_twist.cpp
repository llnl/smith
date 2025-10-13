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

/* Nonlinear mixed complementarity problem (NLMCP)
 * of the form
 * 0 <= x \perp F(x, y) >= 0
 *              Q(x, y)  = 0
 * Here, F and x are both 0-dimensional
 * and   Q(x, y) = [ r(u) + (dc/du)^T l]
 *                 [-c(u)]
 *            y  = [ u ]
 *                 [ l ]
 *
 * wherein r(u) is the elasticity nonlinear residual
 *         c(u) are the tied gap contacts
 *            u are the displacement dofs
 *            l are the Lagrange multipliers
 *
 * we use the approximate Jacobian
 *       dQ/dy \approx [ dr/du     (dc/du)^T]
 *                     [-dc/du        0 ]
 */
template <typename SolidWeakFormType>
class TiedContactProblem : public EqualityConstrainedHomotopyProblem {
 protected:
  std::unique_ptr<mfem::HypreParMatrix> drdu_;
  std::unique_ptr<mfem::HypreParMatrix> dcdu_;
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
  mfem::Vector residual(const mfem::Vector& u) const;
  mfem::HypreParMatrix* residualJacobian(const mfem::Vector& u);
  mfem::Vector constraint(const mfem::Vector& u) const;
  mfem::HypreParMatrix* constraintJacobian(const mfem::Vector& u);
  mfem::Vector constraintJacobianTvp(const mfem::Vector& u, const mfem::Vector& l) const;
  virtual ~TiedContactProblem();
};

// this example is intended to eventually replace twist.cpp
int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  serac::ApplicationManager applicationManager(argc, argv);

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
    auto u = 0.0 * x;  // 0.1 * x;
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
  if (true) {
    double nonlinear_absolute_tol = 1.e-4;
    int nonlinear_max_iterations = 2;
    // optimization variables
    auto X0 = problem.GetOptimizationVariable();
    auto Xf = problem.GetOptimizationVariable();

    HomotopySolver solver(&problem);
    solver.SetTol(nonlinear_absolute_tol);
    solver.SetMaxIter(nonlinear_max_iterations);
    solver.Mult(X0, Xf);
    bool converged = solver.GetConverged();
    SLIC_WARNING_ROOT_IF(!converged, "Homotopy solver did not converge");
  } else {
    // double time = 0.0, dt = 1.0;
    // int direction = serac::ContactFields::DISP;
    // auto input_states = serac::getConstFieldPointers(contact_states);
    // auto gap = contact_constraint->evaluate(time, dt, input_states);
    // for (int i = 0; i < gap.Size(); i++) {
    //   if (std::isnan(gap(i))) {
    //     std::cout << "nan entry at " << i << std::endl;
    //   }
    // }
    // auto gap_Jacobian = contact_constraint->jacobian(time, dt, input_states, direction);
    // auto gap_Jacobian_tilde = contact_constraint->jacobian_tilde(time, dt, input_states, direction);
    // int nPressureDofs = contact_constraint->numPressureDofs();
    // mfem::Vector multipliers(nPressureDofs);
    // multipliers = 0.0;
    // auto residual = contact_constraint->residual_contribution(time, dt, input_states, multipliers, direction);
    // double gnorm = mfem::GlobalLpNorm(2, gap.Norml2(), MPI_COMM_WORLD);
    // std::cout << "||g||_2 = " << gnorm << std::endl;
    // std::cout << "||dg / du||_F = " << gap_Jacobian->FNorm() << std::endl;
    //  std::cout << "||tilde(dg / du)||_F = " << gap_Jacobian_tilde->FNorm() << std::endl;
    //  auto residual_Jacobian = contact_constraint->residual_contribution_jacobian(time, dt,
    //         	                                                             input_states, multipliers,
    //         								     direction);
  }
  return 0;
}

template <typename SolidWeakFormType>
TiedContactProblem<SolidWeakFormType>::TiedContactProblem(std::vector<serac::FiniteElementState*> contact_states,
                                                          std::vector<serac::FiniteElementState*> all_states,
                                                          std::shared_ptr<serac::Mesh> mesh,
                                                          std::shared_ptr<SolidWeakFormType> weak_form,
                                                          std::shared_ptr<serac::ContactConstraint> constraints)
    : EqualityConstrainedHomotopyProblem(), weak_form_(weak_form), mesh_(mesh), constraints_(constraints)
{
  // copy states
  all_states_.resize(all_states.size());
  std::copy(all_states.begin(), all_states.end(), all_states_.begin());

  // copy contact states
  contact_states_.resize(contact_states.size());
  std::copy(contact_states.begin(), contact_states.end(), contact_states_.begin());

  // obtain displacement dof information
  HYPRE_BigInt* uOffsets = all_states[FIELD::DISP]->space().GetTrueDofOffsets();
  dimu_ = all_states[FIELD::DISP]->space().GetTrueVSize();

  // obtain pressure dof information
  dimc_ = constraints_->numPressureDofs();
  HYPRE_BigInt pressure_offset = 0;
  MPI_Scan(&dimc_, &pressure_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  HYPRE_BigInt cOffsets[2];
  cOffsets[1] = pressure_offset;
  cOffsets[0] = pressure_offset - dimc_;

  // set pressure and displacement dof information
  SetSizes(uOffsets, cOffsets);

  // shape_disp field
  shape_disp_ = std::make_unique<serac::FiniteElementState>(mesh->newShapeDisplacement());
}

template <typename SolidWeakFormType>
mfem::Vector TiedContactProblem<SolidWeakFormType>::residual(const mfem::Vector& u) const
{
  contact_states_[serac::ContactFields::DISP]->Set(1.0, u);
  auto res_vector = weak_form_->residual(time_, dt_, shape_disp_.get(), serac::getConstFieldPointers(all_states_));
  return res_vector;
}

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::residualJacobian(const mfem::Vector& u)
{
  contact_states_[serac::ContactFields::DISP]->Set(1.0, u);
  drdu_.reset();
  drdu_ = weak_form_->jacobian(time_, dt_, shape_disp_.get(), getConstFieldPointers(all_states_), jacobian_weights_);
  MFEM_VERIFY(drdu_->Height() == dimu_, "weak form Jacobian/TiedContactProblem displacement dofs inconsistent sizes");
  return drdu_.get();
}

template <typename SolidWeakFormType>
mfem::Vector TiedContactProblem<SolidWeakFormType>::constraint(const mfem::Vector& u) const
{
  contact_states_[serac::ContactFields::DISP]->Set(1.0, u);
  auto gap = constraints_->evaluate(time_, dt_, serac::getConstFieldPointers(contact_states_));
  return gap;
}

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::constraintJacobian(const mfem::Vector& u)
{
  contact_states_[serac::ContactFields::DISP]->Set(1.0, u);
  dcdu_.reset();
  dcdu_ = std::move(
      constraints_->jacobian(time_, dt_, serac::getConstFieldPointers(contact_states_), serac::ContactFields::DISP));
  return dcdu_.get();
}

template <typename SolidWeakFormType>
mfem::Vector TiedContactProblem<SolidWeakFormType>::constraintJacobianTvp(const mfem::Vector& u,
                                                                          const mfem::Vector& l) const
{
  contact_states_[serac::ContactFields::DISP]->Set(1.0, u);
  auto res_contribution = constraints_->residual_contribution(time_, dt_, serac::getConstFieldPointers(contact_states_),
                                                              l, serac::ContactFields::DISP);
  return res_contribution;
}

template <typename SolidWeakFormType>
TiedContactProblem<SolidWeakFormType>::~TiedContactProblem()
{
}
