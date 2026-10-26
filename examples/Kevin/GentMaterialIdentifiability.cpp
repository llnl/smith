// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file GentMaterialIdentifiability.cpp
 * @brief Gent-material identifiability from planar and dumbbell tension experiments.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "axom/sidre.hpp"
#include "mfem.hpp"
#include "mpi.h"

#include "smith/smith_config.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/pseudohessian_identifiability/pseudohessian_identifiability.hpp"

namespace {

constexpr int DIM = 3;
constexpr int ORDER = 1;
constexpr int NUM_PARAMETERS = 3;
constexpr int NUM_OBJECTIVES_PER_EXPERIMENT = 10;
constexpr double PLANAR_FINAL_DISPLACEMENT = 10.0;
constexpr double DUMBBELL_FINAL_DISPLACEMENT = 40.0;
constexpr double MAX_DISPLACEMENT_INCREMENT = 1.0;
// Reference stresses at the nominal material parameters.  Each load-point objective is scaled by its own fixed
// reference value, producing relative sensitivities while preserving the physical stress output.
constexpr std::array<double, NUM_OBJECTIVES_PER_EXPERIMENT> PLANAR_OBJECTIVE_SCALES = {
    0.41323014530291324, 0.7502037786810993, 1.0526600914197637, 1.3472440497004092, 1.6536901758528502,
    1.989618326933723,   2.3741326536495264, 2.831607455928018,  3.3972649112483047, 4.127360951471438};
constexpr std::array<double, NUM_OBJECTIVES_PER_EXPERIMENT> DUMBBELL_OBJECTIVE_SCALES = {
    0.08170249463625331, 0.15562487187942037, 0.22475754224842232, 0.291465096996205,  0.35763878983220043,
    0.4248215718949054,  0.4943073981107651,  0.5672179319534774,  0.6445619323233485, 0.7272838718021845};

const std::array<const char*, NUM_PARAMETERS> PARAMETER_NAMES = {"bulk_modulus", "shear_modulus",
                                                                 "limiting_chain_extensibility"};
// A normalized value of one represents the nominal physical value of each material parameter.
constexpr std::array<double, NUM_PARAMETERS> PARAMETER_SCALES = {100.0, 1.0, 5.0};

using ParameterSpace = smith::L2<0>;
using Solid = smith::SolidMechanics<ORDER, DIM, smith::Parameters<ParameterSpace, ParameterSpace, ParameterSpace>>;

struct ExperimentConfig {
  std::string name;
  std::string mesh_file;
  double y_min;
  double y_max;
  double final_displacement;
  double loaded_face_area;
  std::array<double, NUM_OBJECTIVES_PER_EXPERIMENT> objective_scales;
};

struct ObjectiveResult {
  double displacement = 0.0;
  double strain = 0.0;
  double stress = 0.0;
  double normalized_objective = 0.0;
  mfem::Vector gradient;
};

mfem::Mesh readMeshWithExternalBoundary(const std::string& mesh_file)
{
  auto mesh = smith::buildMeshFromFile(mesh_file);
  if (!mesh.HasBoundaryElements()) {
    mesh.GenerateBoundaryElements();
    mesh.SetAttributes(false, true);
  }
  return mesh;
}

int globalElementCount(const smith::Domain& domain, MPI_Comm comm)
{
  int local_count = domain.total_elements();
  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, comm);
  return global_count;
}

/**
 * @brief Compressible, isochoric-volumetric split Gent material.
 *
 * The strain energy density is
 *
 *   W = -mu J_m/2 log(1 - (I1_bar - 3)/J_m) + K/2 (J - 1)^2,
 *
 * where K, mu, and J_m are independent differentiable L2 fields.
 */
struct ParameterizedGent {
  using State = smith::Empty;

  template <int dim, typename GradType, typename BulkType, typename ShearType, typename ExtensibilityType>
  SMITH_HOST_DEVICE auto operator()(State&, const smith::tensor<GradType, dim, dim>& displacement_gradient,
                                    const BulkType& bulk_field, const ShearType& shear_field,
                                    const ExtensibilityType& extensibility_field) const
  {
    using std::pow;
    constexpr auto I = smith::Identity<dim>();
    const auto F = I + displacement_gradient;
    const auto J = smith::det(F);
    const auto J_minus_one = J - 1.0;
    const auto J_to_minus_one_third = pow(J, -1.0 / 3.0);
    const auto F_bar = J_to_minus_one_third * F;
    const auto I1_bar = smith::inner(F_bar, F_bar);

    const auto bulk_modulus = smith::get<0>(bulk_field);
    const auto shear_modulus = smith::get<0>(shear_field);
    const auto limiting_chain_extensibility = smith::get<0>(extensibility_field);
    const auto gent_denominator = 1.0 - (I1_bar - 3.0) / limiting_chain_extensibility;

    const auto inverse_F_transpose = smith::inv(smith::transpose(F));
    const auto inverse_F_bar_transpose = smith::inv(smith::transpose(F_bar));
    const auto deviatoric_stress =
        shear_modulus / gent_denominator * J_to_minus_one_third * (F_bar - (I1_bar / 3.0) * inverse_F_bar_transpose);
    const auto volumetric_stress = bulk_modulus * J_minus_one * J * inverse_F_transpose;
    return deviatoric_stress + volumetric_stress;
  }

  double density = 1.0;
};

/** @brief One displacement-controlled Gent tension experiment on an imported specimen mesh. */
class GentTensionExperiment {
 public:
  GentTensionExperiment(ExperimentConfig config, const std::vector<double>& initial_normalized_parameters)
      : config_(std::move(config))
  {
    mesh_ =
        std::make_shared<smith::Mesh>(readMeshWithExternalBoundary(config_.mesh_file), config_.name + "_mesh", 0, 0);

    const double coordinate_tolerance = 1.0e-10 * std::max(1.0, std::abs(config_.y_max - config_.y_min));
    mesh_->addDomainOfBoundaryElements("y_min", [this, coordinate_tolerance](std::vector<smith::vec3> vertices, int) {
      return std::abs(smith::average(vertices)[1] - config_.y_min) < coordinate_tolerance;
    });
    mesh_->addDomainOfBoundaryElements("y_max", [this, coordinate_tolerance](std::vector<smith::vec3> vertices, int) {
      return std::abs(smith::average(vertices)[1] - config_.y_max) < coordinate_tolerance;
    });

    const int fixed_face_count = globalElementCount(mesh_->domain("y_min"), mesh_->getComm());
    const int pulled_face_count = globalElementCount(mesh_->domain("y_max"), mesh_->getComm());
    SLIC_ERROR_ROOT_IF(fixed_face_count == 0, std::format("No boundary faces were found at y = {} in '{}'.",
                                                          config_.y_min, config_.mesh_file));
    SLIC_ERROR_ROOT_IF(pulled_face_count == 0, std::format("No boundary faces were found at y = {} in '{}'.",
                                                           config_.y_max, config_.mesh_file));
    SLIC_INFO_ROOT(
        std::format("{}: found {} fixed faces and {} pulled faces", config_.name, fixed_face_count, pulled_face_count));
    const bool has_invalid_objective_scale = std::any_of(
        config_.objective_scales.begin(), config_.objective_scales.end(), [](double scale) { return scale <= 0.0; });
    SLIC_ERROR_ROOT_IF(has_invalid_objective_scale,
                       std::format("{}: all objective scales must be positive.", config_.name));
    SLIC_INFO_ROOT(std::format("{}: normalizing each stress objective by its nominal value", config_.name));

    if (axom::slic::isRoot()) {
      const std::string output_filename = config_.name + "_stress_strain.csv";
      stress_strain_output_.open(output_filename);
      SLIC_ERROR_ROOT_IF(!stress_strain_output_, std::format("Unable to open '{}'.", output_filename));
      stress_strain_output_ << "engineering_strain,nominal_stress,displacement\n";
      stress_strain_output_ << std::setprecision(17) << "0,0,0\n";
      stress_strain_output_.flush();
      SLIC_INFO_ROOT(std::format("{}: writing stress-strain points to '{}'", config_.name, output_filename));
    }

    smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::Newton,
                                                    .relative_tol = 1.0e-10,
                                                    .absolute_tol = 1.0e-12,
                                                    .max_iterations = 100,
                                                    .print_level = 0};

    solid_ = std::make_unique<Solid>(nonlinear_options, smith::solid_mechanics::direct_linear_options,
                                     smith::solid_mechanics::default_quasistatic_options, config_.name, mesh_,
                                     std::vector<std::string>{PARAMETER_NAMES.begin(), PARAMETER_NAMES.end()});

    setNormalizedMaterialParameters(initial_normalized_parameters);
    solid_->setMaterial(smith::DependsOn<0, 1, 2>{}, ParameterizedGent{}, mesh_->entireBody());

    solid_->setFixedBCs(mesh_->domain("y_min"));
    solid_->setFixedBCs(mesh_->domain("y_max"), smith::Component::X + smith::Component::Z);
    solid_->setDisplacementBCs(
        [this](smith::vec3, double time) { return smith::vec3{{0.0, config_.final_displacement * time, 0.0}}; },
        mesh_->domain("y_max"), smith::Component::Y);

    solid_->completeSetup();

    paraview_states_.reserve(NUM_PARAMETERS + 1);
    paraview_states_.push_back(&solid_->displacement());
    for (int parameter_index = 0; parameter_index < NUM_PARAMETERS; ++parameter_index) {
      paraview_states_.push_back(&solid_->parameter(static_cast<std::size_t>(parameter_index)));
    }
    SLIC_INFO_ROOT(
        std::format("{}: writing each forward solve to '{}_objective_##_paraview'", config_.name, config_.name));
  }

  /** @brief Return the prescribed displacement at the final load fraction. */
  double finalDisplacement() const { return config_.final_displacement; }

  /** @brief Convert a prescribed displacement to engineering strain based on the initial grip separation. */
  double engineeringStrain(double displacement) const { return displacement / (config_.y_max - config_.y_min); }

  /** @brief Return the fixed reference stress used to normalize one load-point objective. */
  double objectiveScale(std::size_t objective_index) const { return config_.objective_scales[objective_index]; }

  /** @brief Evaluate nominal pulled-face stress and its normalized-parameter gradient at one load fraction. */
  void evaluate(const std::vector<double>& normalized_parameters, std::size_t objective_index, double load_fraction,
                double& objective, mfem::Vector& gradient)
  {
    SLIC_ERROR_ROOT_IF(
        normalized_parameters.size() != NUM_PARAMETERS,
        std::format("Expected {} material parameters, but received {}.", NUM_PARAMETERS, normalized_parameters.size()));
    SLIC_ERROR_ROOT_IF(load_fraction <= 0.0 || load_fraction > 1.0,
                       std::format("Load fraction {} is outside (0, 1].", load_fraction));
    SLIC_ERROR_ROOT_IF(
        objective_index >= NUM_OBJECTIVES_PER_EXPERIMENT,
        std::format("Objective index {} is outside [0, {}).", objective_index, NUM_OBJECTIVES_PER_EXPERIMENT));

    setNormalizedMaterialParameters(normalized_parameters);
    solid_->resetStates();

    const std::string paraview_output_directory =
        std::format("{}_objective_{:02}_paraview", config_.name, objective_index + 1);
    auto paraview_writer =
        smith::createParaviewWriter(mesh_->mfemParMesh(), paraview_states_, paraview_output_directory);
    paraview_writer.write(0, 0.0, paraview_states_);

    const double displacement = config_.final_displacement * load_fraction;
    const int load_step_count = static_cast<int>(std::ceil(displacement / MAX_DISPLACEMENT_INCREMENT - 1.0e-12));
    const double load_step = load_fraction / load_step_count;
    SLIC_INFO_ROOT(std::format("{}: solving displacement objective {} in {} load steps; ParaView output is in '{}'",
                               config_.name, displacement, load_step_count, paraview_output_directory));
    for (int load_step_index = 0; load_step_index < load_step_count; ++load_step_index) {
      solid_->advanceTimestep(load_step);
      const double completed_load_fraction = (load_step_index + 1) * load_step;
      const double completed_displacement = config_.final_displacement * completed_load_fraction;
      paraview_writer.write(static_cast<std::size_t>(load_step_index + 1), engineeringStrain(completed_displacement),
                            paraview_states_);
    }

    auto reaction_direction = createReactionDirection();
    objective = smith::innerProduct(solid_->dual("reactions"), reaction_direction);
    outputStressStrainPoint(displacement, objective);

    smith::FiniteElementDual displacement_adjoint_load(solid_->displacement().space(),
                                                       config_.name + "_displacement_adjoint_load");
    displacement_adjoint_load = 0.0;
    solid_->setAdjointLoad({{"displacement", displacement_adjoint_load}});
    solid_->setDualAdjointBcs({{"reactions", reaction_direction}});
    solid_->reverseAdjointTimestep();

    gradient.SetSize(NUM_PARAMETERS);
    for (int parameter_index = 0; parameter_index < NUM_PARAMETERS; ++parameter_index) {
      const auto index = static_cast<std::size_t>(parameter_index);
      const auto field_sensitivity = solid_->computeTimestepSensitivity(index);
      // Apply the chain rule for physical_parameter = parameter_scale * normalized_parameter.
      gradient[parameter_index] = PARAMETER_SCALES[index] * globalConstantFieldDerivative(field_sensitivity);
    }
  }

 private:
  void outputStressStrainPoint(double displacement, double stress)
  {
    const double strain = engineeringStrain(displacement);
    if (axom::slic::isRoot()) {
      stress_strain_output_ << strain << ',' << stress << ',' << displacement << '\n';
      stress_strain_output_.flush();
    }
    SLIC_INFO_ROOT(std::format("{}: engineering strain = {}, nominal stress = {}", config_.name, strain, stress));
    smith::logger::flush();
  }

  void setNormalizedMaterialParameters(const std::vector<double>& normalized_parameters)
  {
    SLIC_ERROR_ROOT_IF(
        normalized_parameters.size() != NUM_PARAMETERS,
        std::format("Expected {} material parameters, but received {}.", NUM_PARAMETERS, normalized_parameters.size()));

    for (int parameter_index = 0; parameter_index < NUM_PARAMETERS; ++parameter_index) {
      const auto index = static_cast<std::size_t>(parameter_index);
      smith::FiniteElementState parameter_field(solid_->parameter(index));
      mfem::ConstantCoefficient constant_parameter(PARAMETER_SCALES[index] * normalized_parameters[index]);
      parameter_field.project(constant_parameter);
      solid_->setParameter(index, parameter_field);
    }
  }

  smith::FiniteElementState createReactionDirection() const
  {
    smith::FiniteElementState reaction_direction(solid_->dual("reactions").space(),
                                                 config_.name + "_reaction_direction");
    reaction_direction = 0.0;

    const double inverse_loaded_area = 1.0 / config_.loaded_face_area;
    mfem::VectorFunctionCoefficient direction(DIM, [inverse_loaded_area](const mfem::Vector&, mfem::Vector& value) {
      value = 0.0;
      value[1] = inverse_loaded_area;
    });
    reaction_direction.project(direction, mesh_->domain("y_max"));
    return reaction_direction;
  }

  static double globalConstantFieldDerivative(const smith::FiniteElementDual& field_sensitivity)
  {
    double local_derivative = field_sensitivity.Sum();
    double global_derivative = 0.0;
    MPI_Allreduce(&local_derivative, &global_derivative, 1, MPI_DOUBLE, MPI_SUM, field_sensitivity.comm());
    return global_derivative;
  }

  ExperimentConfig config_;
  std::shared_ptr<smith::Mesh> mesh_;
  std::unique_ptr<Solid> solid_;
  std::vector<const smith::FiniteElementState*> paraview_states_;
  std::ofstream stress_strain_output_;
};

std::vector<smith::pseudohessian_identifiability::ObjectiveFunction> makeObjectives(
    GentTensionExperiment& experiment, std::vector<ObjectiveResult>& results)
{
  results.resize(NUM_OBJECTIVES_PER_EXPERIMENT);
  std::vector<smith::pseudohessian_identifiability::ObjectiveFunction> objectives;
  objectives.reserve(NUM_OBJECTIVES_PER_EXPERIMENT);

  for (int objective_index = 0; objective_index < NUM_OBJECTIVES_PER_EXPERIMENT; ++objective_index) {
    const auto index = static_cast<std::size_t>(objective_index);
    const double load_fraction = static_cast<double>(objective_index + 1) / NUM_OBJECTIVES_PER_EXPERIMENT;
    results[index].displacement = experiment.finalDisplacement() * load_fraction;
    results[index].strain = experiment.engineeringStrain(results[index].displacement);
    objectives.emplace_back([&experiment, &results, index, load_fraction](const std::vector<double>& parameters,
                                                                          double& objective, mfem::Vector& gradient) {
      experiment.evaluate(parameters, index, load_fraction, objective, gradient);
      results[index].stress = objective;
      const double inverse_objective_scale = 1.0 / experiment.objectiveScale(index);
      objective *= inverse_objective_scale;
      gradient *= inverse_objective_scale;
      results[index].normalized_objective = objective;
      results[index].gradient = gradient;
    });
  }
  return objectives;
}

void printResults(const std::string& experiment_name, const std::vector<ObjectiveResult>& results)
{
  for (const auto& result : results) {
    SLIC_INFO_ROOT(
        std::format("{}: engineering strain = {}, displacement = {}, nominal stress = {}, normalized objective = {}",
                    experiment_name, result.strain, result.displacement, result.stress, result.normalized_objective));
    for (int parameter_index = 0; parameter_index < NUM_PARAMETERS; ++parameter_index) {
      SLIC_INFO_ROOT(std::format("  d(normalized objective)/d(normalized {}) = {}",
                                 PARAMETER_NAMES[static_cast<std::size_t>(parameter_index)],
                                 result.gradient[parameter_index]));
    }
  }
  smith::logger::flush();
}

}  // namespace

int main(int argc, char* argv[])
{
  smith::ApplicationManager application_manager(argc, argv);
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "gent_material_identifiability_data");

  const std::vector<double> normalized_material_parameters = {1.0, 1.0, 1.0};
  GentTensionExperiment planar_tension({.name = "gent_planar_tension",
                                        .mesh_file = SMITH_REPO_DIR "/examples/Kevin/planar_tension_geometry.g",
                                        .y_min = 0.0,
                                        .y_max = 10.0,
                                        .final_displacement = PLANAR_FINAL_DISPLACEMENT,
                                        .loaded_face_area = 200.0,
                                        .objective_scales = PLANAR_OBJECTIVE_SCALES},
                                       normalized_material_parameters);
  GentTensionExperiment dumbbell_tension({.name = "gent_dumbbell_tension",
                                          .mesh_file = SMITH_REPO_DIR "/examples/Kevin/dumbbell_geometry.g",
                                          .y_min = 0.0,
                                          .y_max = 75.0,
                                          .final_displacement = DUMBBELL_FINAL_DISPLACEMENT,
                                          .loaded_face_area = 24.0,
                                          .objective_scales = DUMBBELL_OBJECTIVE_SCALES},
                                         normalized_material_parameters);

  std::vector<ObjectiveResult> planar_results;
  std::vector<ObjectiveResult> dumbbell_results;
  auto objectives = makeObjectives(planar_tension, planar_results);
  auto dumbbell_objectives = makeObjectives(dumbbell_tension, dumbbell_results);
  objectives.insert(objectives.end(), dumbbell_objectives.begin(), dumbbell_objectives.end());

  const auto [eigenvalues, eigenvectors] =
      smith::pseudohessian_identifiability::calcEigensystem(normalized_material_parameters, objectives);

  printResults("Planar tension", planar_results);
  printResults("Dumbbell tension", dumbbell_results);
  for (int eigenpair = 0; eigenpair < eigenvalues.Size(); ++eigenpair) {
    SLIC_INFO_ROOT(std::format("Combined pseudo-Hessian eigenvalue {}: {}", eigenpair, eigenvalues[eigenpair]));
    const auto& eigenvector = eigenvectors[static_cast<std::size_t>(eigenpair)];
    for (int component = 0; component < eigenvector.Size(); ++component) {
      SLIC_INFO_ROOT(
          std::format("  {}: {}", PARAMETER_NAMES[static_cast<std::size_t>(component)], eigenvector[component]));
    }
  }

  return 0;
}
