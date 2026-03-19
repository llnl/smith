// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file paraview_writer.hpp
 *
 * @brief Simple paraview field output methods
 */

#pragma once

#include <string>
#include "mfem.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/physics/mesh.hpp"
#include <variant>

namespace smith {

/// @brief Class which interactions with ParaViewDataCollection to write arbitrary field results to disk.  This allows
/// output independent of a particular BasePhysics.
class ParaviewWriter {
 public:
  using StateVecs = std::vector<std::shared_ptr<FiniteElementState> >;  ///< using

  /// Construct from ParaViewDataCollection, a vector of shared_ptr to FiniteElementState, and vector of shared_ptr to
  /// FiniteElementState which dual fields will be copied into.
  ParaviewWriter(std::unique_ptr<mfem::ParaViewDataCollection> pv_, const StateVecs& states_, const StateVecs& duals_)
      : pv(std::move(pv_)), states(states_), dual_states(duals_)
  {
  }

  /// @brief write paraview output from vector of finite element states. states must be passed in with a consistent
  /// order as how the ParaviewWriter was constructed (consistent order of spaces)
  void write(size_t step, double time, const std::vector<const FiniteElementState*>& current_states)
  {
    SMITH_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(current_states.size() != states.size(), "wrong number of output states to write");

    for (size_t n = 0; n < states.size(); ++n) {
      auto& state = states[n];
      *state = *current_states[n];
      state->gridFunction();
    }

    pv->SetCycle(static_cast<int>(step));
    pv->SetTime(time);
    pv->Save();
  }

  /// @brief write paraview output from vector of finite element duals. duals must be passed in with a consistent order
  /// as how the ParaviewWriter was constructed (consistent order of spaces)
  void write(size_t step, double time, const std::vector<const FiniteElementDual*>& current_duals)
  {
    SMITH_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(current_duals.size() != dual_states.size(), "wrong number of output states to write");

    for (size_t n = 0; n < dual_states.size(); ++n) {
      auto& dual = dual_states[n];
      current_duals[n]->linearForm().ParallelAssemble(*dual);
      dual->gridFunction();
    }

    pv->SetCycle(static_cast<int>(step));
    pv->SetTime(time);
    pv->Save();
  }

  /// @brief write paraview output from vector of FieldState. These must be passed in with a consistent order as how the
  /// ParaviewWriter was constructed (consistent order of spaces).  Both the field, and its dual/reaction will be
  /// written.
  void write(int step, double time, const std::vector<FieldState>& current_fields)
  {
    SMITH_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(current_fields.size() != states.size(), "wrong number of output states to write");

    for (size_t n = 0; n < states.size(); ++n) {
      auto& state = states[n];
      *state = *current_fields[n].get();
      state->gridFunction();

      auto& dual = dual_states[n];
      current_fields[n].get_dual()->linearForm().ParallelAssemble(*dual);
      dual->gridFunction();
    }

    pv->SetCycle(step);
    pv->SetTime(time);
    pv->Save();
  }

  /// overload
  void write(size_t step, double time, const std::vector<FieldState>& current_fields)
  {
    write(static_cast<int>(step), time, current_fields);
  }

 private:
  std::unique_ptr<mfem::ParaViewDataCollection> pv;
  StateVecs states;
  StateVecs dual_states;
};

/// @brief Creates a ParaviewWriter from a mesh, vector of FieldState, and the name of the output paraview file.  File
/// will be in directory filename/filename.pvd.
inline auto createParaviewWriter(const smith::Mesh& mesh, const std::vector<FieldState>& states,
                                 std::string output_name)
{
  if (output_name == "") {
    output_name = "default";
  }

  ParaviewWriter::StateVecs output_states;
  ParaviewWriter::StateVecs output_duals;

  auto non_const_mesh = const_cast<mfem::ParMesh*>(&mesh.mfemParMesh());
  auto paraview_dc = std::make_unique<mfem::ParaViewDataCollection>(output_name, non_const_mesh);
  // visualization order has to be at least 1 for paraview (because there is no zero order mesh)
  int max_order_in_fields = 1;

  // Find the maximum polynomial order in the physics module's states
  for (const auto& fstate : states) {
    const auto& state = fstate.get();
    output_states.push_back(std::make_shared<smith::FiniteElementState>(state->space(), state->name()));
    paraview_dc->RegisterField(state->name(), &output_states.back()->gridFunction());
    max_order_in_fields = std::max(max_order_in_fields, state->space().GetOrder(0));

    const auto& dual = fstate.get_dual();
    output_duals.push_back(std::make_shared<smith::FiniteElementState>(dual->space(), dual->name()));
    paraview_dc->RegisterField(dual->name(), &output_duals.back()->gridFunction());
    max_order_in_fields = std::max(max_order_in_fields, dual->space().GetOrder(0));
  }

  // Set the options for the paraview output files
  paraview_dc->SetLevelsOfDetail(max_order_in_fields);
  paraview_dc->SetHighOrderOutput(true);
  paraview_dc->SetDataFormat(mfem::VTKFormat::BINARY);
  paraview_dc->SetCompression(true);

  return ParaviewWriter(std::move(paraview_dc), output_states, output_duals);
}

/// @brief Creates a ParaviewWriter from an mfem::ParMesh, vector of FiniteElementState pointers, and the name of the
/// output paraview file.  File will be in directory filename/filename.pvd.
inline auto createParaviewWriter(const mfem::ParMesh& mesh, const std::vector<const FiniteElementState*>& states,
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
  // visualization order has to be at least 1 for paraview (because there is no zero order mesh)
  int max_order_in_fields = 1;

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

}  // namespace smith
