// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file differentiable_utils.hpp
 *
 * @brief Utility functions for testing.
 */

#pragma once

#include "serac/gretl/double_state.hpp"
#include "serac/differentiable_numerics/field_state.hpp"
#include "serac/physics/scalar_objective.hpp"

namespace serac {

DoubleState evaluate_objective(const TimeInfo& time_info, const FieldState& shape_disp,
                               const std::vector<FieldState>& inputs, const ScalarObjective* objective);

/// @brief Utility function to construct a serac::functional which evaluates the total kinetic energy
template <typename DispSpace, typename DensitySpace>
auto create_kinetic_energy_integrator(serac::Domain& domain, const mfem::ParFiniteElementSpace& velocity_space,
                                      const mfem::ParFiniteElementSpace& density_space)
{
  static constexpr int dim = DispSpace::components;
  auto ke_integrator = std::make_shared<serac::Functional<double(DispSpace, DispSpace, DensitySpace)>>(
      std::array<const mfem::ParFiniteElementSpace*, 3>{&velocity_space, &velocity_space, &density_space});
  ke_integrator->AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0, 1, 2>{},
      [&](auto /*t*/, auto /*X*/, auto U, auto V, auto Rho) {
        auto rho = get<VALUE>(Rho);
        auto v = get<VALUE>(V);
        auto ke = 0.5 * rho * inner(v, v);
        auto dx_dX = get<DERIVATIVE>(U) + Identity<dim>();
        auto J = det(dx_dX);
        return ke * J;
      },
      domain);
  return ke_integrator;
}

/// @brief Utility function which computes the kinetic energy and returns it as a gretl state (with its vjp defined)
template <typename DispSpace, typename DensitySpace>
gretl::State<double> compute_kinetic_energy(
    const std::shared_ptr<serac::Functional<double(DispSpace, DispSpace, DensitySpace)>>& energy_func,
    serac::FieldState disp, serac::FieldState velo, serac::FieldState density, double scaling)
{
  return gretl::create_state<double, double>(
      // specify how to zero the dual
      [](double forwardVal) { return 0 * forwardVal; },
      // define how to (re)evaluate the output
      [=](const serac::FEFieldPtr& Disp, const serac::FEFieldPtr& Velo, const serac::FEFieldPtr& Density) -> double {
        return (*energy_func)(0.0, *Disp, *Velo, *Density) * scaling;
      },
      // define how to backpropagate the vjp
      [=](const serac::FEFieldPtr& Disp, const serac::FEFieldPtr& Velo, const serac::FEFieldPtr& Density,
          const double& /*ke*/, serac::FEDualPtr& Disp_dual, serac::FEDualPtr& Velo_dual,
          serac::FEDualPtr& Density_dual, const double& ke_dual) -> void {
        auto ddisp = (*energy_func)(0.0, serac::differentiate_wrt(*Disp), *Velo, *Density);
        auto de_ddisp = assemble(serac::get<serac::DERIVATIVE>(ddisp));

        auto dvelo = (*energy_func)(0.0, *Disp, serac::differentiate_wrt(*Velo), *Density);
        auto de_dvelo = assemble(serac::get<serac::DERIVATIVE>(dvelo));

        auto ddens = (*energy_func)(0.0, *Disp, *Velo, serac::differentiate_wrt(*Density));
        auto de_ddensity = assemble(serac::get<serac::DERIVATIVE>(ddens));

        Disp_dual->Add(scaling * ke_dual, *de_ddisp);
        Velo_dual->Add(scaling * ke_dual, *de_dvelo);
        Density_dual->Add(scaling * ke_dual, *de_ddensity);
      },
      // give the input values
      disp, velo, density);
}

/// testing utility to confirm order of convergence of the finite differences relative to the backprop gradient
inline auto check_gradients(const gretl::State<double>& objectiveState, FieldState& inputState,
                            FiniteElementDual& inputDual, double objectiveBase, gretl::DataStore& dataStore, double eps)
{
  serac::FiniteElementState inputSave(*inputState.get());
  dataStore.reset();
  serac::FiniteElementState& input = *inputState.get();
  serac::FiniteElementState pert(input.space(), input.name() + "_pert");

  int sz = pert.Size();
  for (int i = 0; i < sz; ++i) {
    pert[i] = -1.2 + 2.02 * (double(i) / sz);
    input[i] += eps * pert[i];
  }

  double objectivePlus = objectiveState.get();

  double directionDeriv = 0.0;
  for (int i = 0; i < sz; ++i) {
    directionDeriv += pert[i] * inputDual[i];
  }

  *inputState.get() = inputSave;

  return std::make_pair(directionDeriv, (objectivePlus - objectiveBase) / eps);
}

/// testing utility to confirm order of convergence of the finite differences relative to the backprop gradient
inline auto check_gradients(const gretl::State<double>& objectiveState, gretl::State<double, double>& inputState,
                            double& inputDual, double objectiveBase, gretl::DataStore& dataStore, double eps)
{
  double inputSave = inputState.get();
  dataStore.reset();
  inputState.set(inputSave + eps);
  double objectivePlus = objectiveState.get();
  inputState.set(inputSave);
  return std::make_pair(inputDual, (objectivePlus - objectiveBase) / eps);
}

/// @brief Testing utility function which runs a gretl graph num_fd_steps (with increasingly smaller finite difference
/// steps) to check if the computed graph gradients are converging to the finite differenced gradients at the expected
/// rate
inline double check_grad_wrt(const gretl::State<double>& objective, serac::FieldState& input, gretl::DataStore& graph,
                             double eps, size_t num_fd_steps = 4, bool printmore = false)
{
  // reset each time, just to be sure
  graph.reset();

  // re-evaluate the final objective value
  double objectiveBase = objective.get();

  // back-propagate to get sensitivity wrt input states
  gretl::set_as_objective(objective);
  graph.back_prop();

  auto dual_vec = *input.get_dual();

  std::vector<double> grad_errors;
  auto [grad, grad_fd] = check_gradients(objective, input, dual_vec, objectiveBase, graph, eps);
  grad_errors.push_back(std::abs(grad - grad_fd));

  for (size_t step = 1; step < num_fd_steps; ++step) {
    eps /= 2;
    std::tie(grad, grad_fd) = check_gradients(objective, input, dual_vec, objectiveBase, graph, eps);
    if (printmore) std::cout << "grad    = " << grad << "\ngrad fd = " << grad_fd << std::endl;
    grad_errors.push_back(std::abs(grad - grad_fd));
  }

  for (size_t step = 0; step < num_fd_steps; ++step) {
    std::cout << "grad error " << step << " = " << grad_errors[step] << std::endl;
  }

  if (num_fd_steps >= 2) {
    return std::log2(grad_errors[0] / grad_errors[num_fd_steps - 1]) / static_cast<double>(num_fd_steps - 1);
  }

  return 0;
};

/// @brief Testing utility function which runs a gretl graph num_fd_steps (with increasingly smaller finite difference
/// steps) to check if the computed graph gradients are converging to the finite differenced gradients at the expected
/// rate
inline double check_grad_wrt(const gretl::State<double>& objective, gretl::State<double, double>& input,
                             gretl::DataStore& graph, double eps, size_t num_fd_steps = 4, bool printmore = false)
{
  // reset each time, just to be sure
  graph.reset();

  // re-evaluate the final objective value
  double objectiveBase = objective.get();

  // back-propagate to get sensitivity wrt input states
  gretl::set_as_objective(objective);
  graph.back_prop();

  auto dual = input.get_dual();

  std::vector<double> grad_errors;
  auto [grad, grad_fd] = check_gradients(objective, input, dual, objectiveBase, graph, eps);
  grad_errors.push_back(std::abs(grad - grad_fd));

  for (size_t step = 1; step < num_fd_steps; ++step) {
    eps /= 2;
    std::tie(grad, grad_fd) = check_gradients(objective, input, dual, objectiveBase, graph, eps);
    if (printmore) std::cout << "grad    = " << grad << "\ngrad fd = " << grad_fd << std::endl;
    grad_errors.push_back(std::abs(grad - grad_fd));
  }

  for (size_t step = 0; step < num_fd_steps; ++step) {
    std::cout << "grad error " << step << " = " << grad_errors[step] << std::endl;
  }

  if (num_fd_steps >= 2) {
    return std::log2(grad_errors[0] / grad_errors[num_fd_steps - 1]) / static_cast<double>(num_fd_steps - 1);
  }

  return 0;
};

}  // namespace serac