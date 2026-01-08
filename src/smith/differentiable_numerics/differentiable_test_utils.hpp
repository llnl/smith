// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file differentiable_test_utils.hpp
 *
 * @brief Utility functions for testing.
 */

#pragma once

#include "gretl/double_state.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/physics/scalar_objective.hpp"

namespace smith {

/// @brief Utility function to construct a smith::functional which evaluates the total kinetic energy
template <typename DispSpace, typename DensitySpace>
auto createKineticEnergyIntegrator(smith::Domain& domain, const mfem::ParFiniteElementSpace& velocity_space,
                                   const mfem::ParFiniteElementSpace& density_space)
{
  static constexpr int dim = DispSpace::components;
  auto ke_integrator = std::make_shared<smith::Functional<double(DispSpace, DispSpace, DensitySpace)>>(
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
gretl::State<double> computeKineticEnergy(
    const std::shared_ptr<smith::Functional<double(DispSpace, DispSpace, DensitySpace)>>& energy_func,
    smith::FieldState disp, smith::FieldState velo, smith::FieldState density, double scaling)
{
  return gretl::create_state<double, double>(
      // specify how to zero the dual
      [](double forwardVal) { return 0 * forwardVal; },
      // define how to (re)evaluate the output
      [=](const smith::FEFieldPtr& Disp, const smith::FEFieldPtr& Velo, const smith::FEFieldPtr& Density) -> double {
        return (*energy_func)(0.0, *Disp, *Velo, *Density) * scaling;
      },
      // define how to backpropagate the vjp
      [=](const smith::FEFieldPtr& Disp, const smith::FEFieldPtr& Velo, const smith::FEFieldPtr& Density,
          const double& /*ke*/, smith::FEDualPtr& Disp_dual, smith::FEDualPtr& Velo_dual,
          smith::FEDualPtr& Density_dual, const double& ke_dual) -> void {
        auto ddisp = (*energy_func)(0.0, smith::differentiate_wrt(*Disp), *Velo, *Density);
        auto de_ddisp = assemble(smith::get<smith::DERIVATIVE>(ddisp));

        auto dvelo = (*energy_func)(0.0, *Disp, smith::differentiate_wrt(*Velo), *Density);
        auto de_dvelo = assemble(smith::get<smith::DERIVATIVE>(dvelo));

        auto ddens = (*energy_func)(0.0, *Disp, *Velo, smith::differentiate_wrt(*Density));
        auto de_ddensity = assemble(smith::get<smith::DERIVATIVE>(ddens));

        Disp_dual->Add(scaling * ke_dual, *de_ddisp);
        Velo_dual->Add(scaling * ke_dual, *de_dvelo);
        Density_dual->Add(scaling * ke_dual, *de_ddensity);
      },
      // give the input values
      disp, velo, density);
}

/// @brief testing utility to confirm order of convergence of the finite differences relative to the backprop gradient
inline auto checkGradients(const gretl::State<double>& objectiveState, FieldState& inputState,
                           FiniteElementDual& inputDual, double objectiveBase, gretl::DataStore& dataStore, double eps)
{
  smith::FiniteElementState inputSave(*inputState.get());
  dataStore.reset();
  smith::FiniteElementState& input = *inputState.get();
  smith::FiniteElementState pert(input.space(), input.name() + "_pert");

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

/// @brief testing utility to confirm order of convergence of the finite differences relative to the backprop gradient
inline auto checkGradients(const gretl::State<double>& objectiveState, gretl::State<double, double>& inputState,
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
inline double checkGradWrt(const gretl::State<double>& objective, smith::FieldState& input, double eps,
                           size_t num_fd_steps = 4, bool printmore = false)
{
  auto& graph = objective.data_store();

  // reset each time, just to be sure
  graph.reset();

  // re-evaluate the final objective value
  double objectiveBase = objective.get();

  // back-propagate to get sensitivity wrt input states
  gretl::set_as_objective(objective);
  graph.back_prop();

  auto dual_vec = *input.get_dual();

  std::vector<double> grad_errors;
  auto [grad, grad_fd] = checkGradients(objective, input, dual_vec, objectiveBase, graph, eps);
  grad_errors.push_back(std::abs(grad - grad_fd));

  for (size_t step = 1; step < num_fd_steps; ++step) {
    eps /= 2;
    std::tie(grad, grad_fd) = checkGradients(objective, input, dual_vec, objectiveBase, graph, eps);
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
inline double checkGradWrt(const gretl::State<double>& objective, gretl::State<double, double>& input, double eps,
                           size_t num_fd_steps = 4, bool printmore = false)
{
  auto& graph = objective.data_store();

  // reset each time, just to be sure
  graph.reset();

  // re-evaluate the final objective value
  double objectiveBase = objective.get();

  // back-propagate to get sensitivity wrt input states
  gretl::set_as_objective(objective);
  graph.back_prop();

  auto dual = input.get_dual();

  std::vector<double> grad_errors;
  auto [grad, grad_fd] = checkGradients(objective, input, dual, objectiveBase, graph, eps);
  grad_errors.push_back(std::abs(grad - grad_fd));

  for (size_t step = 1; step < num_fd_steps; ++step) {
    eps /= 2;
    std::tie(grad, grad_fd) = checkGradients(objective, input, dual, objectiveBase, graph, eps);
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

}  // namespace smith
