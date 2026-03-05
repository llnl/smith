// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics_contact.hpp
 *
 * @brief An object containing the solver for total Lagrangian finite deformation solid mechanics with contact
 */

#pragma once

#include <memory>
#include <unordered_map>

#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/contact/contact_data.hpp"

namespace smith {

template <int order, int dim, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class SolidMechanicsContact;

/**
 * @brief The nonlinear solid with contact solver class
 *
 * The nonlinear total Lagrangian quasi-static with contact solver object. This uses Functional to compute
 * the tangent stiffness matrices.
 *
 * @tparam order The order of the discretization of the displacement field
 * @tparam dim The spatial dimension of the mesh
 */
template <int order, int dim, typename... parameter_space, int... parameter_indices>
class SolidMechanicsContact<order, dim, Parameters<parameter_space...>,
                            std::integer_sequence<int, parameter_indices...>>
    : public SolidMechanics<order, dim, Parameters<parameter_space...>,
                            std::integer_sequence<int, parameter_indices...>> {
  using SolidMechanicsBase =
      SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>;

 public:
  /**
   * @brief Construct a new SolidMechanicsContact object
   *
   * @param nonlinear_opts The nonlinear solver options for solving the nonlinear residual equations
   * @param lin_opts The linear solver options for solving the linearized Jacobian equations
   * @param timestepping_opts The timestepping options for the solid mechanics time evolution operator
   * @param physics_name A name for the physics module instance
   * @param smith_mesh Smith mesh for physics
   * @param parameter_names A vector of the names of the requested parameter fields
   * @param cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param time The simulation time to initialize the physics module to
   * @param checkpoint_to_disk Flag to save the transient states on disk instead of memory for transient adjoint solver
   * @param use_warm_start Flag to turn on or off the displacement warm start predictor which helps robustness for
   * large deformation problems
   */
  SolidMechanicsContact(const NonlinearSolverOptions nonlinear_opts, const LinearSolverOptions lin_opts,
                        const smith::TimesteppingOptions timestepping_opts, const std::string& physics_name,
                        std::shared_ptr<smith::Mesh> smith_mesh, std::vector<std::string> parameter_names = {},
                        int cycle = 0, double time = 0.0, bool checkpoint_to_disk = false, bool use_warm_start = true)
      : SolidMechanicsContact(std::make_unique<EquationSolver>(nonlinear_opts, lin_opts, smith_mesh->getComm()),
                              timestepping_opts, physics_name, smith_mesh, parameter_names, cycle, time,
                              checkpoint_to_disk, use_warm_start)
  {
  }

  /**
   * @brief Construct a new SolidMechanicsContact object
   *
   * @param solver The nonlinear equation solver for the implicit solid mechanics equations
   * @param timestepping_opts The timestepping options for the solid mechanics time evolution operator
   * @param physics_name A name for the physics module instance
   * @param smith_mesh Smith mesh for physics
   * @param parameter_names A vector of the names of the requested parameter fields
   * @param cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param time The simulation time to initialize the physics module to
   * @param checkpoint_to_disk Flag to save the transient states on disk instead of memory for transient adjoint solver
   * @param use_warm_start Flag to turn on or off the displacement warm start predictor which helps robustness for
   * large deformation problems
   */
  SolidMechanicsContact(std::unique_ptr<smith::EquationSolver> solver,
                        const smith::TimesteppingOptions timestepping_opts, const std::string& physics_name,
                        std::shared_ptr<smith::Mesh> smith_mesh, std::vector<std::string> parameter_names = {},
                        int cycle = 0, double time = 0.0, bool checkpoint_to_disk = false, bool use_warm_start = true)
      : SolidMechanicsBase(std::move(solver), timestepping_opts, physics_name, smith_mesh, parameter_names, cycle, time,
                           checkpoint_to_disk, use_warm_start),
        contact_(BasePhysics::mfemParMesh()),
        forces_(StateManager::newDual(displacement_.space(), detail::addPrefix(physics_name, "contact_forces")))
  {
    forces_ = 0;
    duals_.push_back(&forces_);
  }

  /**
   * @brief Construct a new Nonlinear SolidMechanicsContact Solver object
   *
   * @param[in] input_options The solver information parsed from the input file
   * @param[in] physics_name A name for the physics module instance
   * @param smith_mesh Smith mesh for physics
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   */
  SolidMechanicsContact(const SolidMechanicsInputOptions& input_options, const std::string& physics_name,
                        std::shared_ptr<smith::Mesh> smith_mesh, int cycle = 0, double time = 0.0)
      : SolidMechanicsBase(input_options, physics_name, smith_mesh, cycle, time),
        contact_(BasePhysics::mfemParMesh()),
        forces_(StateManager::newDual(displacement_.space(), detail::addPrefix(physics_name, "contact_forces")))
  {
    forces_ = 0;
    duals_.push_back(&forces_);
  }

  /// @overload
  void resetStates(int cycle = 0, double time = 0.0) override
  {
    SolidMechanicsBase::resetStates(cycle, time);
    forces_ = 0.0;
    contact_.setDisplacements(BasePhysics::shapeDisplacement(), displacement_);
    contact_.reset();
    double dt = 0.0;
    contact_.update(cycle, time, dt);
  }

  /// @brief Build the quasi-static operator corresponding to the total Lagrangian formulation
  std::unique_ptr<mfem_ext::StdFunctionOperator> buildQuasistaticOperator() override
  {
    auto residual_fn = [this](const mfem::Vector& u, mfem::Vector& r) {
      const mfem::Vector u_blk(const_cast<mfem::Vector&>(u), 0, displacement_.Size());
      const mfem::Vector res = (*residual_)(time_, BasePhysics::shapeDisplacement(), u_blk, acceleration_,
                                            *parameters_[parameter_indices].state...);

      // NOTE this copy is required as the sundials solvers do not allow move assignments because of their memory
      // tracking strategy
      // See https://github.com/mfem/mfem/issues/3531
      mfem::Vector r_blk(r, 0, displacement_.Size());
      r_blk = res;

      contact_.residualFunction(BasePhysics::shapeDisplacement(), u, r);
      r_blk.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
    };
    // This if-block below breaks up building the Jacobian operator depending if there is Lagrange multiplier
    // enforcement or not
    if (contact_.haveLagrangeMultipliers()) {
      // The quasistatic operator has blocks if any of the contact interactions are enforced using Lagrange multipliers.
      // Jacobian operator is an mfem::BlockOperator
      J_offsets_ = mfem::Array<int>({0, displacement_.Size(), displacement_.Size() + contact_.numPressureDofs()});
      return std::make_unique<mfem_ext::StdFunctionOperator>(
          displacement_.space().TrueVSize() + contact_.numPressureDofs(), residual_fn,
          // gradient of residual function
          [this](const mfem::Vector& u) -> mfem::Operator& {
            const mfem::Vector u_blk(const_cast<mfem::Vector&>(u), 0, displacement_.Size());
            auto [r, drdu] = (*residual_)(time_, BasePhysics::shapeDisplacement(), differentiate_wrt(u_blk),
                                          acceleration_, *parameters_[parameter_indices].state...);

            // create block operator holding jacobian contributions
            J_constraint_ = contact_.jacobianFunction(assemble(drdu));

            // take ownership of blocks
            J_constraint_->owns_blocks = false;
            J_ = std::unique_ptr<mfem::HypreParMatrix>(
                static_cast<mfem::HypreParMatrix*>(&J_constraint_->GetBlock(0, 0)));
            J_12_ = std::unique_ptr<mfem::HypreParMatrix>(
                static_cast<mfem::HypreParMatrix*>(&J_constraint_->GetBlock(0, 1)));
            J_21_ = std::unique_ptr<mfem::HypreParMatrix>(
                static_cast<mfem::HypreParMatrix*>(&J_constraint_->GetBlock(1, 0)));
            J_22_ = std::unique_ptr<mfem::HypreParMatrix>(
                static_cast<mfem::HypreParMatrix*>(&J_constraint_->GetBlock(1, 1)));

            // eliminate bcs and compute eliminated blocks
            J_e_ = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
            J_e_21_ = std::unique_ptr<mfem::HypreParMatrix>(J_21_->EliminateCols(bcs_.allEssentialTrueDofs()));
            J_12_->EliminateRows(bcs_.allEssentialTrueDofs());

            // create block operator for constraints
            J_constraint_e_ = std::make_unique<mfem::BlockOperator>(J_offsets_);
            J_constraint_e_->SetBlock(0, 0, J_e_.get());
            J_constraint_e_->SetBlock(1, 0, J_e_21_.get());

            J_operator_ = J_constraint_.get();
            return *J_constraint_;
          });
    } else {
      // If all of the contact interactions are penalty, then there will be no blocks.  Jacobian operator is a single
      // mfem::HypreParMatrix
      return std::make_unique<mfem_ext::StdFunctionOperator>(
          displacement_.space().TrueVSize(), residual_fn, [this](const mfem::Vector& u) -> mfem::Operator& {
            auto [r, drdu] = (*residual_)(time_, BasePhysics::shapeDisplacement(), differentiate_wrt(u), acceleration_,
                                          *parameters_[parameter_indices].state...);

            // get 11-block holding jacobian contributions
            auto block_J = contact_.jacobianFunction(assemble(drdu));
            block_J->owns_blocks = false;
            J_ = std::unique_ptr<mfem::HypreParMatrix>(static_cast<mfem::HypreParMatrix*>(&block_J->GetBlock(0, 0)));

            J_e_ = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);

            J_operator_ = J_.get();
            return *J_;
          });
    }
  }

  /**
   * @brief Add a mortar contact boundary condition
   *
   * @param interaction_id Unique identifier for the ContactInteraction
   * @param bdry_attr_surf1 MFEM boundary attributes for the first surface
   * @param bdry_attr_surf2 MFEM boundary attributes for the second surface
   * @param contact_opts Defines contact method, enforcement, type, and penalty
   */
  void addContactInteraction(int interaction_id, const std::set<int>& bdry_attr_surf1,
                             const std::set<int>& bdry_attr_surf2, ContactOptions contact_opts)
  {
    SLIC_ERROR_ROOT_IF(!is_quasistatic_, "Contact can only be applied to quasistatic problems.");
    SLIC_ERROR_ROOT_IF(order > 1, "Contact can only be applied to linear (order = 1) meshes.");

    const auto interaction_force_name = detail::addPrefix(name_, axom::fmt::format("contact_force_{}", interaction_id));
    SLIC_ERROR_ROOT_IF(StateManager::hasDual(interaction_force_name),
                       axom::fmt::format("StateManager already contains a dual named '{}'", interaction_force_name));
    SLIC_ERROR_ROOT_IF(
        contact_interaction_forces_.find(interaction_id) != contact_interaction_forces_.end(),
        axom::fmt::format("Contact interaction force dual already registered for interaction_id={}", interaction_id));

    auto interaction_force_dual =
        std::make_unique<FiniteElementDual>(StateManager::newDual(displacement_.space(), interaction_force_name));
    *interaction_force_dual = 0.0;
    duals_.push_back(interaction_force_dual.get());
    contact_interaction_forces_.emplace(interaction_id, std::move(interaction_force_dual));

    contact_.addContactInteraction(interaction_id, bdry_attr_surf1, bdry_attr_surf2, contact_opts);
  }

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * @note This must be called before AdvanceTimestep().
   */
  void completeSetup() override
  {
    double dt = 0.0;
    contact_.update(cycle_, time_, dt);

    SolidMechanicsBase::completeSetup();
  }

  /**
   " @brief Get the contact pressures from all contact interactions, merged into a single HypreParVector
   *
   * @return The merged contact pressures
   */
  mfem::HypreParVector pressure() const { return contact_.mergedPressures(); }

  /**
   * @brief Computes the shape sensitivity of the residual contribution from a single contact interaction
   *
   * @param interaction_id The unique identifier for the contact interaction
   * @return Contact-only shape sensitivity (as a dual on the shape displacement space)
   *
   * @pre The contact interaction must be penalty-enforced.
   * @pre ContactData/Tribol must have been updated for the current configuration so the interaction Jacobian is
   * up-to-date. For penalty enforcement this implies the same two-stage update sequence used by
   * ContactData::residualFunction() (update gaps -> set penalty pressures -> update).
   */
  const FiniteElementDual& computeTimestepContactShapeSensitivity(int interaction_id)
  {
#ifdef SMITH_USE_TRIBOL
    const ContactInteraction* interaction_ptr = nullptr;
    for (const auto& interaction : contact_.getContactInteractions()) {
      if (interaction.getInteractionId() == interaction_id) {
        interaction_ptr = &interaction;
        break;
      }
    }
    SLIC_ERROR_ROOT_IF(!interaction_ptr,
                       axom::fmt::format("No contact interaction found with interaction_id={}", interaction_id));

    const auto& interaction = *interaction_ptr;
    SLIC_ERROR_ROOT_IF(
        interaction.getContactOptions().enforcement != ContactEnforcement::Penalty,
        "computeTimestepContactShapeSensitivity currently only supports penalty-enforced contact interactions.");

    auto interaction_J = interaction.jacobianContribution();
    interaction_J->owns_blocks = false;  // this method manages ownership of blocks

    std::unique_ptr<mfem::HypreParMatrix> dfdx;
    if (!interaction_J->IsZeroBlock(0, 0)) {
      auto* block_0_0 = dynamic_cast<mfem::HypreParMatrix*>(&interaction_J->GetBlock(0, 0));
      SLIC_ERROR_ROOT_IF(!block_0_0, "Only HypreParMatrix constraint matrix blocks are currently supported.");
      dfdx.reset(block_0_0);
    }

    SLIC_ERROR_ROOT_IF(!dfdx,
                       axom::fmt::format("Contact interaction {} produced a zero df/dx contribution.", interaction_id));

    auto it = contact_interaction_shape_sensitivities_.find(interaction_id);
    if (it == contact_interaction_shape_sensitivities_.end()) {
      const auto sens_name =
          detail::addPrefix(name_, axom::fmt::format("contact_shape_sensitivity_{}", interaction_id));
      auto sens = std::make_unique<FiniteElementDual>(BasePhysics::shapeDisplacementSensitivity().space(), sens_name);
      *sens = 0.0;
      it = contact_interaction_shape_sensitivities_.emplace(interaction_id, std::move(sens)).first;
    }

    auto& contact_shape_sensitivity = *it->second;
    contact_shape_sensitivity = 0.0;
    dfdx->MultTranspose(adjoint_displacement_, contact_shape_sensitivity);
    return contact_shape_sensitivity;
#else
    SLIC_ERROR_ROOT("Smith built without Tribol support. Cannot compute contact shape sensitivities.");
    return BasePhysics::shapeDisplacementSensitivity();
#endif
  }

  /**
   * @brief Evaluates the scalar quantity \f$\lambda^T f_{\mathrm{contact},i}\f$ for a single contact interaction
   *
   * This is primarily useful for finite-difference checks of
   * computeTimestepContactShapeSensitivity(interaction_id).
   *
   * @param interaction_id The unique identifier for the contact interaction
   * @param shape_u Shape displacement true DOF vector
   * @param u Displacement true DOF vector
   * @param adjoint_u Displacement adjoint true DOF vector
   * @param dt Timestep size to pass to Tribol update
   * @return The scalar \f$\lambda^T f_{\mathrm{contact},i}\f$
   *
   * @pre The contact interaction must be penalty-enforced.
   */
  double evalContactInteractionForceAdjointProduct(int interaction_id, const mfem::Vector& shape_u,
                                                   const mfem::Vector& u, const mfem::Vector& adjoint_u,
                                                   double dt = 1.0)
  {
#ifdef SMITH_USE_TRIBOL
    const ContactInteraction* interaction_ptr = nullptr;
    for (const auto& interaction : contact_.getContactInteractions()) {
      if (interaction.getInteractionId() == interaction_id) {
        interaction_ptr = &interaction;
        break;
      }
    }
    SLIC_ERROR_ROOT_IF(!interaction_ptr,
                       axom::fmt::format("No contact interaction found with interaction_id={}", interaction_id));
    SLIC_ERROR_ROOT_IF(
        interaction_ptr->getContactOptions().enforcement != ContactEnforcement::Penalty,
        "evalContactInteractionForceAdjointProduct currently only supports penalty-enforced contact interactions.");

    // Mirror the update sequence in ContactData::residualFunction() for penalty enforcement.
    contact_.setDisplacements(shape_u, u);
    for (const auto& interaction : contact_.getContactInteractions()) {
      interaction.evalJacobian(false);
    }
    contact_.update(cycle_, time_end_step_, dt);

    mfem::Vector merged_pressures(contact_.numPressureDofs());
    merged_pressures = 0.0;
    contact_.setPressures(merged_pressures);

    for (const auto& interaction : contact_.getContactInteractions()) {
      interaction.evalJacobian(true);
    }
    contact_.update(cycle_, time_end_step_, dt);

    const FiniteElementDual f = interaction_ptr->forces();
    FiniteElementState adjoint_state(displacement_.space(), "adjoint_tmp");
    adjoint_state = adjoint_u;
    return innerProduct(f, adjoint_state);
#else
    (void)interaction_id;
    (void)shape_u;
    (void)u;
    (void)adjoint_u;
    (void)dt;
    SLIC_ERROR_ROOT("Smith built without Tribol support. Cannot evaluate contact interaction forces.");
    return 0.0;
#endif
  }

 protected:
  /// @brief Solve the Quasi-static Newton system
  void quasiStaticSolve(double dt) override
  {
    // warm start must be called prior to the time update so that the previous Jacobians can be used consistently
    // throughout. warm start for contact needs to include the previous stiffness terms associated with contact
    // otherwise the system will interpenetrate instantly on warm-starting.
    warmStartDisplacementContact(dt);
    time_ += dt;

    // In general, the solution vector is a stacked (block) vector:
    //  | displacement     |
    //  | contact pressure |
    // Contact pressure is only active when solving a contact problem with Lagrange multipliers.
    mfem::Vector augmented_solution(displacement_.Size() + contact_.numPressureDofs());
    augmented_solution.SetVector(displacement_, 0);
    augmented_solution.SetVector(contact_.mergedPressures(), displacement_.Size());

    // solve the non-linear system resid = 0 and pressure * gap = 0
    nonlin_solver_->solve(augmented_solution);
    displacement_.Set(1.0, mfem::Vector(augmented_solution, 0, displacement_.Size()));
    contact_.setPressures(mfem::Vector(augmented_solution, displacement_.Size(), contact_.numPressureDofs()));
    contact_.update(cycle_, time_, dt);
    forces_.SetVector(contact_.forces(), 0);

#ifdef SMITH_USE_TRIBOL
    for (const auto& interaction : contact_.getContactInteractions()) {
      auto it = contact_interaction_forces_.find(interaction.getInteractionId());
      if (it != contact_interaction_forces_.end()) {
        it->second->SetVector(interaction.forces(), 0);
      }
    }
#endif
  }

  /**
   * @brief Sets the Dirichlet BCs for the current time and computes an initial guess for parameters and displacement
   *
   * @note
   * We want to solve
   *\f$
   *r(u_{n+1}, p_{n+1}, U_{n+1}, t_{n+1}) = 0
   *\f$
   *for $u_{n+1}$, given new values of parameters, essential b.c.s and time. The problem is that if we use $u_n$ as the
   initial guess for this new solve, most nonlinear solver algorithms will start off by linearizing at (or near) the
   initial guess. But, if the essential boundary conditions change by an amount on the order of the mesh size, then it's
   possible to invert elements and make that linearization point inadmissible (either because it converges slowly or
   that the inverted elements crash the program). *So, we need a better initial guess. This "warm start" generates a
   guess by linear extrapolation from the previous known solution:

   *\f$
   *0 = r(u_{n+1}, p_{n+1}, U_{n+1}, t_{n+1}) \approx {r(u_n, p_n, U_n, t_n)} +  \frac{dr_n}{du} \Delta u +
   \frac{dr_n}{dp} \Delta p + \frac{dr_n}{dU} \Delta U + \frac{dr_n}{dt} \Delta t
   *\f$
   *If we assume that suddenly changing p and t will not lead to inverted elements, we can simplify the approximation to
   *\f$
   *0 = r(u_{n+1}, p_{n+1}, U_{n+1}, t_{n+1}) \approx r(u_n, p_{n+1}, U_n, t_{n+1}) +  \frac{dr_n}{du} \Delta u +
   \frac{dr_n}{dU} \Delta U
   *\f$
   *Move all the known terms to the rhs and solve for \f$\Delta u\f$,
   *\f$
   *\Delta u = - \bigg(  \frac{dr_n}{du} \bigg)^{-1} \bigg( r(u_n, p_{n+1}, U_n, t_{n+1}) + \frac{dr_n}{dU} \Delta U
   \bigg)
   *\f$
   *It is especially important to use the previously solved Jacobian in problems with material instabilities, as good
   nonlinear solvers will ensure positive definiteness at equilibrium. *Once any parameter is changed, it is no longer
   certain to be positive definite, which will cause issues for many types linear solvers.
   */
  void warmStartDisplacementContact(double dt)
  {
    SMITH_MARK_FUNCTION;

    du_ = 0.0;
    for (auto& bc : bcs_.essentials()) {
      // apply the future boundary conditions, but use the most recent Jacobians stiffness.
      bc.setDofs(du_, time_ + dt);
    }

    auto& constrained_dofs = bcs_.allEssentialTrueDofs();
    for (int i = 0; i < constrained_dofs.Size(); i++) {
      int j = constrained_dofs[i];
      du_[j] -= displacement_(j);
    }

    if (use_warm_start_) {
      // Update the system residual
      mfem::Vector augmented_residual(displacement_.Size() + contact_.numPressureDofs());
      augmented_residual = 0.0;
      const mfem::Vector res = (*residual_)(time_ + dt, BasePhysics::shapeDisplacement(), displacement_, acceleration_,
                                            *parameters_[parameter_indices].state...);

      contact_.setPressures(mfem::Vector(augmented_residual, displacement_.Size(), contact_.numPressureDofs()));
      contact_.update(cycle_, time_, dt);
      mfem::Vector r_blk(augmented_residual, 0, displacement_.space().TrueVSize());
      r_blk = res;

      mfem::Vector augmented_solution(displacement_.space().TrueVSize() + contact_.numPressureDofs());
      augmented_solution = 0.0;
      mfem::Vector du(augmented_solution, 0, displacement_.space().TrueVSize());
      du = displacement_;

      contact_.residualFunction(BasePhysics::shapeDisplacement(), augmented_solution, augmented_residual);
      r_blk.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);

      // use the most recently evaluated Jacobian
      auto [_, drdu] = (*residual_)(time_, BasePhysics::shapeDisplacement(), differentiate_wrt(displacement_),
                                    acceleration_, *parameters_[parameter_indices].previous_state...);

      contact_.update(cycle_, time_, dt);
      if (contact_.haveLagrangeMultipliers()) {
        J_offsets_ = mfem::Array<int>({0, displacement_.Size(), displacement_.Size() + contact_.numPressureDofs()});
        J_constraint_ = contact_.jacobianFunction(assemble(drdu));

        // take ownership of blocks
        J_constraint_->owns_blocks = false;
        J_ = std::unique_ptr<mfem::HypreParMatrix>(static_cast<mfem::HypreParMatrix*>(&J_constraint_->GetBlock(0, 0)));
        J_12_ =
            std::unique_ptr<mfem::HypreParMatrix>(static_cast<mfem::HypreParMatrix*>(&J_constraint_->GetBlock(0, 1)));
        J_21_ =
            std::unique_ptr<mfem::HypreParMatrix>(static_cast<mfem::HypreParMatrix*>(&J_constraint_->GetBlock(1, 0)));
        J_22_ =
            std::unique_ptr<mfem::HypreParMatrix>(static_cast<mfem::HypreParMatrix*>(&J_constraint_->GetBlock(1, 1)));

        J_e_ = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
        J_e_21_ = std::unique_ptr<mfem::HypreParMatrix>(J_21_->EliminateCols(bcs_.allEssentialTrueDofs()));
        J_12_->EliminateRows(bcs_.allEssentialTrueDofs());

        J_operator_ = J_constraint_.get();
      } else {
        // get 11-block holding jacobian contributions
        auto block_J = contact_.jacobianFunction(assemble(drdu));
        block_J->owns_blocks = false;
        J_ = std::unique_ptr<mfem::HypreParMatrix>(static_cast<mfem::HypreParMatrix*>(&block_J->GetBlock(0, 0)));

        J_e_ = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);

        J_operator_ = J_.get();
      }

      augmented_residual *= -1.0;

      du = du_;
      mfem::EliminateBC(*J_, *J_e_, constrained_dofs, du, r_blk);
      for (int i = 0; i < constrained_dofs.Size(); i++) {
        int j = constrained_dofs[i];
        r_blk[j] = du[j];
      }

      auto& lin_solver = nonlin_solver_->linearSolver();

      lin_solver.SetOperator(*J_operator_);

      lin_solver.Mult(augmented_residual, augmented_solution);

      du_ = du;
    }

    displacement_ += du_;
  }

  /// @brief Solve the Quasi-static Newton system
  void quasiStaticAdjointSolve(double /*dt*/) override
  {
    SLIC_ERROR_ROOT_IF(contact_.haveLagrangeMultipliers(),
                       "Lagrange multiplier contact does not currently support sensitivities/adjoints.");

    auto [_, drdu] = (*residual_)(time_, BasePhysics::shapeDisplacement(), differentiate_wrt(displacement_),
                                  acceleration_, *parameters_[parameter_indices].state...);

    auto block_J = contact_.jacobianFunction(assemble(drdu));
    block_J->owns_blocks = false;
    auto jacobian = std::unique_ptr<mfem::HypreParMatrix>(static_cast<mfem::HypreParMatrix*>(&block_J->GetBlock(0, 0)));
    auto J_T = std::unique_ptr<mfem::HypreParMatrix>(jacobian->Transpose());

    for (const auto& bc : bcs_.essentials()) {
      bc.apply(*J_T, displacement_adjoint_load_, reactions_adjoint_bcs_);
    }

    auto& lin_solver = nonlin_solver_->linearSolver();
    lin_solver.SetOperator(*J_T);
    lin_solver.Mult(displacement_adjoint_load_, adjoint_displacement_);
  }

  /// @overload
  const FiniteElementDual& computeTimestepShapeSensitivity() override
  {
    auto drdshape =
        smith::get<DERIVATIVE>((*residual_)(time_end_step_, differentiate_wrt(BasePhysics::shapeDisplacement()),
                                            displacement_, acceleration_, *parameters_[parameter_indices].state...));

    auto block_J = contact_.jacobianFunction(assemble(drdshape));
    block_J->owns_blocks = false;
    auto drdshape_mat =
        std::unique_ptr<mfem::HypreParMatrix>(static_cast<mfem::HypreParMatrix*>(&block_J->GetBlock(0, 0)));

    drdshape_mat->MultTranspose(adjoint_displacement_, shape_displacement_dual_);

    return BasePhysics::shapeDisplacementSensitivity();
  }

  using BasePhysics::bcs_;
  using BasePhysics::cycle_;
  using BasePhysics::duals_;
  using BasePhysics::is_quasistatic_;
  using BasePhysics::mesh_;
  using BasePhysics::name_;
  using BasePhysics::parameters_;
  using BasePhysics::states_;
  using BasePhysics::time_;
  using SolidMechanicsBase::acceleration_;
  using SolidMechanicsBase::adjoint_displacement_;
  using SolidMechanicsBase::d_residual_d_;
  using SolidMechanicsBase::DERIVATIVE;
  using SolidMechanicsBase::displacement_;
  using SolidMechanicsBase::displacement_adjoint_load_;
  using SolidMechanicsBase::du_;
  using SolidMechanicsBase::J_;
  using SolidMechanicsBase::J_e_;
  using SolidMechanicsBase::nonlin_solver_;
  using SolidMechanicsBase::reactions_adjoint_bcs_;
  using SolidMechanicsBase::residual_;
  using SolidMechanicsBase::residual_with_bcs_;
  using SolidMechanicsBase::shape_displacement_dual_;
  using SolidMechanicsBase::time_end_step_;
  using SolidMechanicsBase::use_warm_start_;
  using SolidMechanicsBase::warmStartDisplacement;

  /// Pointer to the Jacobian operator (J_ if no Lagrange multiplier contact, J_constraint_ otherwise)
  mfem::Operator* J_operator_;

  /// 21 Jacobian block if using Lagrange multiplier contact (dg/dx)
  std::unique_ptr<mfem::HypreParMatrix> J_21_;

  /// 12 Jacobian block if using Lagrange multiplier contact (df/dp)
  std::unique_ptr<mfem::HypreParMatrix> J_12_;

  /// 22 Jacobian block if using Lagrange multiplier contact (ones on diagonal for inactive t-dofs)
  std::unique_ptr<mfem::HypreParMatrix> J_22_;

  /// Block offsets for the J_constraint_ BlockOperator (must be owned outside J_constraint_)
  mfem::Array<int> J_offsets_;

  /// Assembled sparse matrix for the Jacobian with constraint blocks
  std::unique_ptr<mfem::BlockOperator> J_constraint_;

  /// Columns of J_21_ that have been separated out because they are associated with essential boundary conditions
  std::unique_ptr<mfem::HypreParMatrix> J_e_21_;

  /// rows and columns of J_constraint_ that have been separated out
  /// because are associated with essential boundary conditions
  std::unique_ptr<mfem::BlockOperator> J_constraint_e_;

  /// @brief Class holding contact constraint data
  ContactData contact_;

  /// forces for output
  FiniteElementDual forces_;

  /// per-interaction contact forces for output
  std::unordered_map<int, std::unique_ptr<FiniteElementDual>> contact_interaction_forces_;

  /// contact-only shape sensitivities (not stored in StateManager)
  std::unordered_map<int, std::unique_ptr<FiniteElementDual>> contact_interaction_shape_sensitivities_;
};

}  // namespace smith
