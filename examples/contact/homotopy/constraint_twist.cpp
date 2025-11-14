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
#include "serac/physics/contact/contact_config.hpp"
#include "shared/mesh/MeshBuilder.hpp"

// ContinuationSolver headers
#include "problems/Problems.hpp"
#include "solvers/HomotopySolver.hpp"
#include "utilities.hpp"

#include "serac/serac.hpp"

static constexpr int dim = 3;
static constexpr int disp_order = 1;

bool linearized_contact = true;
bool contact = true;

using VectorSpace = serac::H1<disp_order, dim>;

using DensitySpace = serac::L2<disp_order - 1>;

struct MyLinearIsotropic {
  using State = serac::Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a linear isotropic material model
   *
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   *
   * @tparam T Number-like type for the displacement gradient components
   * @tparam dim Dimensionality of space
   * @param du_dX Displacement gradient with respect to the reference configuration
   * @return The stress
   */
  template <typename T, int dim>
  SERAC_HOST_DEVICE auto operator()(State& /* state */, const serac::tensor<T, dim, dim>& du_dX) const
  {
    auto I = serac::Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto epsilon = 0.5 * (transpose(du_dX) + du_dX);
    return lambda * tr(epsilon) * I + 2.0 * G * epsilon;
  }
  template <typename T, int dim, typename Density>
  SERAC_HOST_DEVICE auto pkStress(State& /* state */, const serac::tensor<T, dim, dim>& du_dX, const Density&) const
  {
    auto I = serac::Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto epsilon = 0.5 * (transpose(du_dX) + du_dX);
    return lambda * tr(epsilon) * I + 2.0 * G * epsilon;
    // using std::log1p;
    // constexpr auto I = Identity<dim>();
    // auto lambda = K - (2.0 / 3.0) * G;
    // auto B_minus_I = dot(du_dX, transpose(du_dX)) + transpose(du_dX) + du_dX;

    // auto logJ = log1p(detApIm1(du_dX));
    //// Kirchoff stress, in form that avoids cancellation error when F is near I
    // auto TK = lambda * logJ * I + G * B_minus_I;

    //// Pull back to Piola
    // auto F = du_dX + I;
    // return dot(TK, inv(transpose(F)));
  }

  /// @brief interpolates density field
  template <typename Density>
  SERAC_HOST_DEVICE auto density(const Density& density) const
  {
    return get<serac::VALUE>(density);
  }

  // double density;  ///< mass density
  double K;  ///< bulk modulus
  double G;  ///< shear modulus
};

using SolidMaterial = MyLinearIsotropic;
// using SolidMaterial = serac::solid_mechanics::NeoHookeanWithFieldDensity;

using SolidWeakFormT = serac::SolidWeakForm<disp_order, dim, serac::Parameters<DensitySpace>>;

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
  int dimu_;
  int dimc_;
  std::vector<serac::FieldPtr> contact_states_;
  std::vector<serac::FieldPtr> residual_states_;
  std::shared_ptr<SolidWeakFormType> weak_form_;
  std::unique_ptr<serac::FiniteElementState> shape_disp_;
  std::shared_ptr<serac::Mesh> mesh_;
  std::shared_ptr<serac::ContactConstraint> constraints_;
  double time_ = 0.0;
  double dt_ = 0.0;
  std::vector<double> jacobian_weights_ = {1.0, 0.0, 0.0, 0.0};
  std::unique_ptr<mfem::HypreParMatrix> restriction_;
  std::unique_ptr<mfem::HypreParMatrix> prolongation_;
  mutable mfem::Vector ufull_;
  mfem::Vector g0_;

 public:
  TiedContactProblem(std::vector<serac::FieldPtr> contact_states, std::vector<serac::FieldPtr> residual_states,
                     std::shared_ptr<serac::Mesh> mesh, std::shared_ptr<SolidWeakFormType> weak_form,
                     std::shared_ptr<serac::ContactConstraint> constraints, mfem::Array<int> ess_tdof_list);
  mfem::Vector residual(const mfem::Vector& u, bool new_point) const;
  mfem::HypreParMatrix* residualJacobian(const mfem::Vector& u, bool new_point);
  mfem::Vector constraint(const mfem::Vector& u, bool new_point) const;
  mfem::HypreParMatrix* constraintJacobian(const mfem::Vector& u, bool new_point);
  mfem::Vector constraintJacobianTvp(const mfem::Vector& u, const mfem::Vector& l, bool new_point) const;
  void fullDisplacement(const mfem::Vector& x, mfem::Vector& u);
  virtual ~TiedContactProblem();
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

auto createParaviewOutput(const mfem::ParMesh& mesh, const std::vector<const serac::FiniteElementState*>& states,
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
    auto u = 0.0 * x;
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

  SolidMaterial mat{10.0, 10.0};
  solid_mechanics_weak_form->setMaterial(serac::DependsOn<0>{}, mesh->entireBodyName(), mat);

  // apply some traction boundary conditions
  std::string surface_name = "side";
  mesh->addDomainOfBoundaryElements(surface_name, serac::by_attr<dim>(6));
  solid_mechanics_weak_form->addBoundaryFlux(surface_name, [](auto /*x*/, auto n, auto /*t*/) { return 1.e-2 * n; });

  serac::tensor<double, dim> constant_force{};
  constant_force[0] = 1.0;
  constant_force[1] = 0.0;
  constant_force[2] = 0.0;

  solid_mechanics_weak_form->addBodyIntegral(mesh->entireBodyName(), [constant_force](double /* t */, auto x) {
    return serac::tuple{constant_force, 0.0 * serac::get<serac::DERIVATIVE>(x)};
  });

  auto residual_state_ptrs = serac::getFieldPointers(states, params);
  auto contact_state_ptrs = serac::getFieldPointers(contact_states);

  // boundary conditions
  mfem::Array<int> ess_bdr_marker(mesh->mfemParMesh().bdr_attributes.Max());
  ess_bdr_marker = 0;
  ess_bdr_marker[2] = 1;
  ess_bdr_marker[5] = 1;
  if (!contact) {
    ess_bdr_marker[3] = 1;
    ess_bdr_marker[4] = 1;
  }
  mfem::Array<int> ess_fixed_tdof_list;
  states[FIELD::DISP].space().GetEssentialTrueDofs(ess_bdr_marker, ess_fixed_tdof_list);

  TiedContactProblem<SolidWeakFormT> problem(contact_state_ptrs, residual_state_ptrs, mesh, solid_mechanics_weak_form,
                                             contact_constraint, ess_fixed_tdof_list);
  if (true) {
    double nonlinear_absolute_tol = 1.e-6;
    // double beta0 = 1.0;
    // double delta_MAX = 5.0;
    int nonlinear_max_iterations = 30;
    int nonlinear_print_level = 1;
    // optimization variables
    auto X0 = problem.GetOptimizationVariable();
    auto Xf = problem.GetOptimizationVariable();

    HomotopySolver solver(&problem);
    solver.SetTol(nonlinear_absolute_tol);
    solver.SetMaxIter(nonlinear_max_iterations);
    solver.SetPrintLevel(nonlinear_print_level);
    // solver.SetNeighborhoodParameter(beta0);
    // solver.SetDeltaMax(delta_MAX);

    auto writer = createParaviewOutput(mesh->mfemParMesh(), serac::getConstFieldPointers(states), "contact");
    writer.write(0, 0.0, serac::getConstFieldPointers(states));
    solver.Mult(X0, Xf);
    bool converged = solver.GetConverged();
    SLIC_WARNING_ROOT_IF(!converged, "Homotopy solver did not converge");
    mfem::Vector u(states[FIELD::DISP].space().GetTrueVSize());
    problem.fullDisplacement(Xf, u);
    states[FIELD::DISP].Set(1.0, u);
    writer.write(1, 1.0, serac::getConstFieldPointers(states));
  } else {
    // finite difference check on residual
    // check r(u0 + eps * udir) - r(u0) / eps - dr/dr(u0) * udir = O(eps)

    auto dimu = problem.GetDisplacementDim();
    mfem::Vector u0(dimu);
    u0 = 1.0;
    mfem::Vector u1(dimu);
    u1 = 0.0;
    mfem::Vector udir(dimu);
    udir.Randomize();
    udir *= 1.e-1;
    bool new_point = true;
    auto res0 = problem.residual(u0, new_point);

    auto resJacobian = problem.residualJacobian(u0, new_point);

    mfem::Vector resJacobianudir(resJacobian->Height());
    resJacobianudir = 0.0;
    mfem::Vector error(resJacobian->Height());
    error = 0.0;
    resJacobian->Mult(udir, resJacobianudir);
    double eps = 1.0;
    for (int i = 0; i < 30; i++) {
      u1.Set(1.0, u0);
      u1.Add(eps, udir);
      auto res1 = problem.residual(u1, new_point);
      error.Set(1.0, res1);
      error.Add(-1.0, res0);
      error /= eps;
      error.Add(-1.0, resJacobianudir);
      double err = mfem::GlobalLpNorm(2, error.Norml2(), MPI_COMM_WORLD);
      double res_norm = mfem::GlobalLpNorm(2, res1.Norml2(), MPI_COMM_WORLD);
      SLIC_INFO_ROOT(axom::fmt::format("|| (res(u0 + eps * udir)||_2 = {}, eps = {}", res_norm, eps));
      SLIC_INFO_ROOT(
          axom::fmt::format("|| (res(u0 + eps * udir) - res(u0)) / eps - J(u0) * udir||_2 = {}, eps = {}", err, eps));
      eps /= 2.0;
    }
    SLIC_INFO_ROOT(axom::fmt::format("|| res Jacobian ||_F = {}", resJacobian->FNorm()));
    // if (gapJacobianFNorm > 1.e-12) {
    //   auto adjoint = problem.GetOptimizationVariable();
    //   auto adjoint_load = problem.GetOptimizationVariable();
    //   adjoint_load = 1.0;
    //   problem.AdjointSolve(u, adjoint_load, adjoint);
    // }
  }
  return 0;
}

template <typename SolidWeakFormType>
TiedContactProblem<SolidWeakFormType>::TiedContactProblem(std::vector<serac::FiniteElementState*> contact_states,
                                                          std::vector<serac::FiniteElementState*> residual_states,
                                                          std::shared_ptr<serac::Mesh> mesh,
                                                          std::shared_ptr<SolidWeakFormType> weak_form,
                                                          std::shared_ptr<serac::ContactConstraint> constraints,
                                                          mfem::Array<int> ess_tdof_list)
    : EqualityConstrainedHomotopyProblem(), weak_form_(weak_form), mesh_(mesh), constraints_(constraints)
{
  // copy residual states
  residual_states_.resize(residual_states.size());
  std::copy(residual_states.begin(), residual_states.end(), residual_states_.begin());

  // copy contact states
  contact_states_.resize(contact_states.size());
  std::copy(contact_states.begin(), contact_states.end(), contact_states_.begin());

  // obtain displacement dof information
  // degrees of freedom with respect to solver
  // are the internal non essential dofs
  const int dimufull_ = residual_states[FIELD::DISP]->space().GetTrueVSize();
  ufull_.SetSize(dimufull_);
  ufull_ = 0.0;
  dimu_ = dimufull_ - ess_tdof_list.Size();

  std::unique_ptr<HYPRE_BigInt> uOffsets;
  uOffsets.reset(offsetsFromLocalSizes(dimu_, MPI_COMM_WORLD));
  std::unique_ptr<HYPRE_BigInt[]> cOffsets = std::make_unique<HYPRE_BigInt[]>(2);

  // obtain pressure dof information
  if (contact) {
    dimc_ = constraints_->numPressureDofs();
    HYPRE_BigInt pressure_offset = 0;
    MPI_Scan(&dimc_, &pressure_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    cOffsets[0] = pressure_offset - dimc_;
    cOffsets[1] = pressure_offset;
  } else {
    dimc_ = 0;
    cOffsets[0] = 0;
    cOffsets[1] = 0;
  }
  // set pressure and displacement dof information
  SetSizes(uOffsets.get(), cOffsets.get());

  // mask out essential tdofs
  // mask[i] = 1 ==> ith dof will be made visible to solver
  // these dofs will be made visiible via explicit use
  // of prolongation/restriction operators in residual/constraint  member functions
  std::unique_ptr<HYPRE_Int[]> mask = std::make_unique<HYPRE_Int[]>(static_cast<size_t>(dimufull_));
  for (int i = 0; i < dimufull_; i++) {
    mask[static_cast<size_t>(i)] = 1;
  }
  for (int i = 0; i < ess_tdof_list.Size(); i++) {
    mask[static_cast<size_t>(ess_tdof_list[i])] = 0;
  }
  restriction_.reset(
      GenerateProjector(uOffsets.get(), residual_states_[FIELD::DISP]->space().GetTrueDofOffsets(), mask.get()));

  prolongation_.reset(restriction_->Transpose());

  // shape_disp field
  shape_disp_ = std::make_unique<serac::FiniteElementState>(mesh->newShapeDisplacement());

  if (contact) {
    // Linearize gap about zero displacement state
    if (linearized_contact) {
      mfem::Vector u0(dimu_);
      u0 = 0.0;
      prolongation_->Mult(u0, ufull_);
      contact_states_[serac::ContactFields::DISP]->Set(1.0, ufull_);
      g0_.SetSize(dimc_);
      g0_.Set(1.0,
              constraints_->evaluate(time_, dt_, serac::getConstFieldPointers(contact_states_), true));  // new_point);

      auto dcdufull_ =
          constraints_->jacobian(time_, dt_, serac::getConstFieldPointers(contact_states_), serac::ContactFields::DISP,
                                 true);  // new_point);
      dcdu_.reset(mfem::ParMult(dcdufull_.get(), prolongation_.get(), true));
    }
  } else {
    int nentries = 0;
    int dimuglb_;
    dimuglb_ = 0;
    MPI_Allreduce(&dimu_, &dimuglb_, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    auto temp = std::make_unique<mfem::SparseMatrix>(dimc_, dimuglb_, nentries);
    dcdu_.reset(GenerateHypreParMatrixFromSparseMatrix(cOffsets.get(), uOffsets.get(), temp.get()));
  }
}

template <typename SolidWeakFormType>
mfem::Vector TiedContactProblem<SolidWeakFormType>::residual(const mfem::Vector& u, bool /*new_point*/) const
{
  // 1. prolongate u --> u_prolongated
  // 2. obtain full residual via u_prolongated
  // 3. restrict residual
  prolongation_->Mult(u, ufull_);
  residual_states_[FIELD::DISP]->Set(1.0, ufull_);
  auto resfull = weak_form_->residual(time_, dt_, shape_disp_.get(), serac::getConstFieldPointers(residual_states_));
  mfem::Vector res(dimu_);
  res = 0.0;
  restriction_->Mult(resfull, res);
  return res;
};

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::residualJacobian(const mfem::Vector& u, bool /*new_point*/)
{
  prolongation_->Mult(u, ufull_);
  residual_states_[FIELD::DISP]->Set(1.0, ufull_);
  auto drdufull_ =
      weak_form_->jacobian(time_, dt_, shape_disp_.get(), getConstFieldPointers(residual_states_), jacobian_weights_);
  drdu_.reset(RAP(drdufull_.get(), prolongation_.get()));
  return drdu_.get();
}

template <typename SolidWeakFormType>
mfem::Vector TiedContactProblem<SolidWeakFormType>::constraint(const mfem::Vector& u, bool /*new_point*/) const
{
  bool new_point = true;
  mfem::Vector gap(dimc_);
  gap = 0.0;
  if (contact) {
    if (linearized_contact) {
      dcdu_->Mult(u, gap);
      gap.Add(1.0, g0_);
    } else {
      prolongation_->Mult(u, ufull_);
      contact_states_[serac::ContactFields::DISP]->Set(1.0, ufull_);
      gap = constraints_->evaluate(time_, dt_, serac::getConstFieldPointers(contact_states_), new_point);
    }
  }
  return gap;
}

template <typename SolidWeakFormType>
mfem::HypreParMatrix* TiedContactProblem<SolidWeakFormType>::constraintJacobian(const mfem::Vector& u,
                                                                                bool /*new_point*/)
{
  bool new_point = true;
  if (contact) {
    if (!linearized_contact) {
      prolongation_->Mult(u, ufull_);
      contact_states_[serac::ContactFields::DISP]->Set(1.0, ufull_);
      auto dcdufull_ = constraints_->jacobian(time_, dt_, serac::getConstFieldPointers(contact_states_),
                                              serac::ContactFields::DISP, new_point);
      dcdu_.reset(mfem::ParMult(dcdufull_.get(), prolongation_.get(), new_point));
    }
  }
  return dcdu_.get();
}

template <typename SolidWeakFormType>
mfem::Vector TiedContactProblem<SolidWeakFormType>::constraintJacobianTvp(const mfem::Vector& u, const mfem::Vector& l,
                                                                          bool /*new_point*/) const
{
  bool new_point = true;
  mfem::Vector res(dimu_);
  res = 0.0;
  if (contact) {
    if (linearized_contact) {
      dcdu_->MultTranspose(l, res);
    } else {
      prolongation_->Mult(u, ufull_);
      contact_states_[serac::ContactFields::DISP]->Set(1.0, ufull_);
      auto res_contribution = constraints_->residual_contribution(
          time_, dt_, serac::getConstFieldPointers(contact_states_), l, serac::ContactFields::DISP, new_point);
      restriction_->Mult(res_contribution, res);
    }
  }
  return res;
}

template <typename SolidWeakFormType>
void TiedContactProblem<SolidWeakFormType>::fullDisplacement(const mfem::Vector& X, mfem::Vector& u)
{
  mfem::BlockVector Xblock(y_partition);
  Xblock.Set(1.0, X);
  prolongation_->Mult(Xblock.GetBlock(0), u);
}

template <typename SolidWeakFormType>
TiedContactProblem<SolidWeakFormType>::~TiedContactProblem()
{
}
