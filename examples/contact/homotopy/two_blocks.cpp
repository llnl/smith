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
#include "smith/physics/contact/contact_config.hpp"
#include "shared/mesh/MeshBuilder.hpp"

// ContinuationSolver headers
#include "problems/Problems.hpp"
#include "solvers/HomotopySolver.hpp"
#include "utilities.hpp"

#include "smith/smith.hpp"

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

/* Nonlinear problem of the form
 * F(X) = [  r(u) + (dc/du)^T l ] = [ 0 ]
 *        [ -c(u)               ]   [ 0 ]
 *   X  = [ u ]
 *        [ l ]
 *
 * wherein r(u) is the elasticity nonlinear residual
 *         c(u) are the tied gap contacts
 *           u  are the displacement dofs
 *           l  are the Lagrange multipliers
 *
 * This problem inherits from EqualityConstrainedHomotopyProblem
 * for compatibility with the HomotopySolver.
 */
template <typename SolidWeakFormType>
class TiedContactProblem : public EqualityConstrainedHomotopyProblem {
 protected:
  std::unique_ptr<mfem::HypreParMatrix> drdu_;
  std::unique_ptr<mfem::HypreParMatrix> dcdu_;
  std::vector<smith::FieldPtr> contact_states_;
  std::vector<smith::FieldPtr> residual_states_;
  std::shared_ptr<SolidWeakFormType> weak_form_;
  std::unique_ptr<smith::FiniteElementState> shape_disp_;
  std::shared_ptr<smith::Mesh> mesh_;
  std::shared_ptr<smith::ContactConstraint> constraints_;
  smith::TimeInfo time_info_;
  std::vector<double> jacobian_weights_ = {1.0, 0.0, 0.0, 0.0};

 public:
  TiedContactProblem(std::vector<smith::FieldPtr> contact_states, std::vector<smith::FieldPtr> residual_states,
                     std::shared_ptr<smith::Mesh> mesh, std::shared_ptr<SolidWeakFormType> weak_form,
                     std::shared_ptr<smith::ContactConstraint> constraints, mfem::Array<int> fixed_tdof_list,
                     mfem::Array<int> disp_tdof_list, mfem::Vector uDC);
  mfem::Vector residual(const mfem::Vector& u, bool fresh_evaluation) const;
  mfem::HypreParMatrix* residualJacobian(const mfem::Vector& u, bool fresh_evaluation);
  mfem::Vector constraint(const mfem::Vector& u, bool fresh_evaluation) const;
  mfem::HypreParMatrix* constraintJacobian(const mfem::Vector& u, bool fresh_evaluation);
  mfem::Vector constraintJacobianTvp(const mfem::Vector& u, const mfem::Vector& l, bool fresh_evaluation) const;
  void fullDisplacement(const mfem::Vector& x, mfem::Vector& u);
  virtual ~TiedContactProblem();
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

auto createParaviewOutput(const mfem::ParMesh& mesh, const std::vector<const smith::FiniteElementState*>& states,
                          std::string output_name)
{
  if (output_name == "") {
    output_name = "default";
  }

  ParaviewWriter::StateVecs output_states;
  for (const auto& s : states) {
    output_states.push_back(std::make_shared<smith::FiniteElementState>(s->space(), s->name()));
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

int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  int visualize = 0;
  int visualize_all_iterates = 0;
  // command line arguments
  axom::CLI::App app{"Two block contact."};
  app.add_option("--visualize", visualize, "solution visualization")
      ->default_val("0")  // Matches value set above
      ->check(axom::CLI::Range(0, 1));
  app.set_help_flag("--help");

  CLI11_PARSE(app, argc, argv);

  // Create DataStore
  std::string name = "two_block_example";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/twohex_for_contact.mesh";
  auto mesh = std::make_shared<smith::Mesh>(smith::buildMeshFromFile(filename), "two_block_mesh", 3, 0);

  mesh->addDomainOfBoundaryElements("fixed_surface", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("driven_surface", smith::by_attr<dim>(6));

  smith::ContactOptions contact_options{.method = smith::ContactMethod::SingleMortar,
                                        .enforcement = smith::ContactEnforcement::LagrangeMultiplier,
                                        .type = smith::ContactType::TiedNormal,
                                        .jacobian = smith::ContactJacobian::Exact};

  std::string contact_constraint_name = "default_contact";

  // Specify the contact interaction
  auto contact_interaction_id = 0;
  std::set<int> surface_1_boundary_attributes({4});
  std::set<int> surface_2_boundary_attributes({5});
  auto contact_constraint = std::make_shared<smith::ContactConstraint>(
      contact_interaction_id, mesh->mfemParMesh(), surface_1_boundary_attributes, surface_2_boundary_attributes,
      contact_options, contact_constraint_name);

  smith::FiniteElementState shape = smith::StateManager::newState(VectorSpace{}, "shape", mesh->tag());
  smith::FiniteElementState disp = smith::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
  smith::FiniteElementState velo = smith::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
  smith::FiniteElementState accel = smith::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
  smith::FiniteElementState density = smith::StateManager::newState(DensitySpace{}, "density", mesh->tag());

  std::vector<smith::FiniteElementState> contact_states;
  std::vector<smith::FiniteElementState> states;
  std::vector<smith::FiniteElementState> params;
  contact_states = {shape, disp};
  states = {disp, velo, accel};
  params = {density};

  // initialize displacement
  contact_states[smith::ContactFields::DISP].setFromFieldFunction([](smith::tensor<double, dim> x) {
    auto u = 0.0 * x;
    return u;
  });

  contact_states[smith::ContactFields::SHAPE] = 0.0;
  states[FIELD::VELO] = 0.0;
  states[FIELD::ACCEL] = 0.0;
  params[0] = 1.0;  // density

  std::string physics_name = "solid";

  // construct residual
  auto solid_mechanics_weak_form =
      std::make_shared<SolidWeakFormT>(physics_name, mesh, states[FIELD::DISP].space(), getSpaces(params));

  // set material parameters
  SolidMaterial mat;
  mat.K = 1.0;
  mat.G = 0.5;
  solid_mechanics_weak_form->setMaterial(smith::DependsOn<0>{}, mesh->entireBodyName(), mat);

  // constant body force
  smith::tensor<double, dim> constant_force{};
  for (int i = 0; i < dim; i++) {
    constant_force[i] = 0.0;
  }
  constant_force[dim - 1] = -1.e-4;

  solid_mechanics_weak_form->addBodyIntegral(mesh->entireBodyName(), [constant_force](double /* t */, auto x) {
    return smith::tuple{constant_force, 0.0 * smith::get<smith::DERIVATIVE>(x)};
  });

  auto residual_state_ptrs = smith::getFieldPointers(states, params);
  auto contact_state_ptrs = smith::getFieldPointers(contact_states);

  // Dirichlet boundary conditions
  mfem::Array<int> ess_fixed_tdof_list;
  mfem::Array<int> ess_disp_tdof_list;
  mfem::Array<int> ess_bdr_marker(mesh->mfemParMesh().bdr_attributes.Max());
  ess_bdr_marker = 0;
  ess_bdr_marker[2] = 1;
  ess_bdr_marker[5] = 0;
  states[FIELD::DISP].space().GetEssentialTrueDofs(ess_bdr_marker, ess_fixed_tdof_list);
  ess_bdr_marker = 0;
  ess_bdr_marker[5] = 1;
  states[FIELD::DISP].space().GetEssentialTrueDofs(ess_bdr_marker, ess_disp_tdof_list);
  auto applied_displacement = [](smith::tensor<double, dim> /*x*/) {
    smith::tensor<double, dim> u{};
    u[0] = 0.0;
    u[1] = 0.0;
    u[2] = -0.06;
    return u;
  };
  states[FIELD::DISP].setFromFieldFunction(applied_displacement);
  mfem::Vector uDC(states[FIELD::DISP].space().GetTrueVSize());
  uDC = 0.0;
  uDC.Set(1.0, states[FIELD::DISP]);

  // define tied contact problem
  TiedContactProblem<SolidWeakFormT> problem(contact_state_ptrs, residual_state_ptrs, mesh, solid_mechanics_weak_form,
                                             contact_constraint, ess_fixed_tdof_list, ess_disp_tdof_list, uDC);
  // optimization variables
  auto X0 = problem.GetOptimizationVariable();
  auto Xf = problem.GetOptimizationVariable();

  // set optimization parameters
  double nonlinear_absolute_tol = 1.e-6;
  int nonlinear_max_iterations = 30;
  int nonlinear_print_level = 1;

  // setup Homotopy solver
  HomotopySolver solver(&problem);
  solver.SetTol(nonlinear_absolute_tol);
  solver.SetMaxIter(nonlinear_max_iterations);
  solver.SetPrintLevel(nonlinear_print_level);
  solver.EnableRegularizedNewtonMode();
  solver.EnableSaveIterates();
  // solve tied contact problem via Homotopy solver
  solver.Mult(X0, Xf);
  bool converged = solver.GetConverged();
  SLIC_WARNING_ROOT_IF(!converged, "Homotopy solver did not converge");

  // visualize
  auto writer = createParaviewOutput(mesh->mfemParMesh(), smith::getConstFieldPointers(states), "contact");
  if (visualize) {
    mfem::Vector u(states[FIELD::DISP].space().GetTrueVSize());
    u = problem.GetDisplacement(X0);
    states[FIELD::DISP].Set(1.0, u);
    writer.write(0, 0.0, smith::getConstFieldPointers(states));
    if (!visualize_all_iterates) {
      u = problem.GetDisplacement(Xf);
      states[FIELD::DISP].Set(1.0, u);
      writer.write(1, 1.0, smith::getConstFieldPointers(states));
    } else {
      auto iterates = solver.GetIterates();
      for (int i = 0; i < iterates.Size(); i++) {
        u = problem.GetDisplacement(*iterates[i]);
        states[FIELD::DISP].Set(1.0, u);
        writer.write((i + 1), static_cast<double>(i + 1), smith::getConstFieldPointers(states));
      }
    }
  }
  return 0;
}

template <typename SolidWeakFormType>
TiedContactProblem<SolidWeakFormType>::TiedContactProblem(std::vector<smith::FiniteElementState*> contact_states,
                                                          std::vector<smith::FiniteElementState*> residual_states,
                                                          std::shared_ptr<smith::Mesh> mesh,
                                                          std::shared_ptr<SolidWeakFormType> weak_form,
                                                          std::shared_ptr<smith::ContactConstraint> constraints,
                                                          mfem::Array<int> fixed_tdof_list,
                                                          mfem::Array<int> disp_tdof_list, mfem::Vector uDC)
    : EqualityConstrainedHomotopyProblem(fixed_tdof_list, disp_tdof_list, uDC),
      weak_form_(weak_form),
      mesh_(mesh),
      constraints_(constraints),
      time_info_(0.0, 0.0, 0)
{
  // copy residual states
  residual_states_.resize(residual_states.size());
  std::copy(residual_states.begin(), residual_states.end(), residual_states_.begin());

  // copy contact states
  contact_states_.resize(contact_states.size());
  std::copy(contact_states.begin(), contact_states.end(), contact_states_.begin());

  // number of "internal" non-essential dofs
  const int dimu =
      residual_states[FIELD::DISP]->space().GetTrueVSize() - fixed_tdof_list.Size() - disp_tdof_list.Size();
  // number of contact constraints
  const int dimc = constraints->numPressureDofs();
  // EqualityConstrainedHomotopyProblem utility function
  SetSizes(dimu, dimc);

  // shape_disp field
  shape_disp_ = std::make_unique<smith::FiniteElementState>(mesh->newShapeDisplacement());
}

template <typename SolidWeakFormType>
mfem::Vector TiedContactProblem<SolidWeakFormType>::residual(const mfem::Vector& u, bool /*fresh_evaluation*/) const
{
  residual_states_[FIELD::DISP]->Set(1.0, u);
  auto res = weak_form_->residual(time_info_, shape_disp_.get(), smith::getConstFieldPointers(residual_states_));
  return res;
};

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::residualJacobian(const mfem::Vector& u,
                                                                              bool /*fresh_evaluation*/)
{
  residual_states_[FIELD::DISP]->Set(1.0, u);
  drdu_ = weak_form_->jacobian(time_info_, shape_disp_.get(), smith::getConstFieldPointers(residual_states_),
                               jacobian_weights_);
  return drdu_.get();
}

template <typename SolidWeakFormType>
mfem::Vector TiedContactProblem<SolidWeakFormType>::constraint(const mfem::Vector& u, bool /*fresh_evaluation*/) const
{
  bool fresh_evaluation = true;
  contact_states_[smith::ContactFields::DISP]->Set(1.0, u);
  auto gap = constraints_->evaluate(time_info_.time(), time_info_.dt(), smith::getConstFieldPointers(contact_states_),
                                    fresh_evaluation);
  return gap;
}

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::constraintJacobian(const mfem::Vector& u,
                                                                                bool /*fresh_evaluation*/)
{
  bool fresh_evaluation = true;
  contact_states_[smith::ContactFields::DISP]->Set(1.0, u);
  dcdu_ = constraints_->jacobian(time_info_.time(), time_info_.dt(), smith::getConstFieldPointers(contact_states_),
                                 smith::ContactFields::DISP, fresh_evaluation);
  return dcdu_.get();
}

template <typename SolidWeakFormType>
mfem::Vector TiedContactProblem<SolidWeakFormType>::constraintJacobianTvp(const mfem::Vector& u, const mfem::Vector& l,
                                                                          bool /*fresh_evaluation*/) const
{
  bool fresh_evaluation = true;
  contact_states_[smith::ContactFields::DISP]->Set(1.0, u);
  auto res_contribution = constraints_->residual_contribution(time_info_.time(), time_info_.dt(),
                                                              smith::getConstFieldPointers(contact_states_), l,
                                                              smith::ContactFields::DISP, fresh_evaluation);
  return res_contribution;
}

template <typename SolidWeakFormType>
TiedContactProblem<SolidWeakFormType>::~TiedContactProblem()
{
}
