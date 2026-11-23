// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file differentiable_operator_weak_form.hpp
 *
 * @brief Implements the WeakForm interface with mfem::future::DifferentiableOperator.
 */

#pragma once

#include "smith/smith_config.hpp"

#ifdef SMITH_USE_ENZYME

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "fem/dfem/doperator.hpp"

#include "smith/infrastructure/logger.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/physics/field_types.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/weak_form.hpp"

namespace smith {

namespace detail {

template <typename Space>
using DfemFieldValueArgument =
    std::conditional_t<Space::components == 1, double, mfem::future::tensor<double, Space::components>>;

template <int spatial_dim, typename Space>
using DfemFieldGradientArgument =
    std::conditional_t<Space::components == 1, mfem::future::tensor<double, spatial_dim>,
                       mfem::future::tensor<double, Space::components, spatial_dim>>;

template <typename LoadFunction>
struct BodySourceIntegrand {
  LoadFunction load_function;

  template <typename Position, typename... Inputs>
  MFEM_HOST_DEVICE auto operator()(const TimeInfo& time_info, const Position& position, const Inputs&... inputs) const
  {
    return mfem::future::tuple{-load_function(time_info, mfem::future::get<VALUE>(position),
                                               mfem::future::get<VALUE>(inputs)...),
                               mfem::future::zero{}};
  }
};

template <int spatial_dim, typename OutputSpace, typename Integrand, typename ActiveParameters,
          typename FieldArgumentList, typename... InputSpaces>
struct DfemBodyQFunction;

template <int spatial_dim, typename OutputSpace, typename Integrand, int... active_parameters,
          typename... FieldArguments, typename... InputSpaces>
struct DfemBodyQFunction<spatial_dim, OutputSpace, Integrand, std::integer_sequence<int, active_parameters...>,
                         std::tuple<FieldArguments...>, InputSpaces...> {
  using Coordinate = mfem::future::tensor<double, spatial_dim>;
  using CoordinateGradient = mfem::future::tensor<double, spatial_dim, spatial_dim>;
  using OutputValue = std::conditional_t<OutputSpace::components == 1, double,
                                         mfem::future::tensor<double, OutputSpace::components>>;
  using OutputGradient =
      std::conditional_t<OutputSpace::components == 1, mfem::future::tensor<double, spatial_dim>,
                         mfem::future::tensor<double, OutputSpace::components, spatial_dim>>;

  Integrand integrand;
  const double* time_data;

  MFEM_HOST_DEVICE void operator()(const Coordinate& deformed_coordinates,
                                   const CoordinateGradient& deformed_jacobian, const double& weight,
                                   const FieldArguments&... field_arguments, OutputValue& output_value,
                                   OutputGradient& output_gradient) const
  {
    const auto inverse_jacobian = mfem::future::inv(deformed_jacobian);
    const auto jacobian_weight = mfem::future::det(deformed_jacobian) * weight;
    const mfem::future::tuple<const FieldArguments&...> arguments{field_arguments...};
    const TimeInfo time_info(time_data[0], time_data[1], static_cast<size_t>(time_data[2]),
                             time_data[3] == 0.0 ? TimeInfo::EvaluationMode::Regular
                                                 : TimeInfo::EvaluationMode::CycleZero);
    const auto position = mfem::future::tuple{deformed_coordinates, deformed_jacobian};
    const auto result = evaluateIntegrand(time_info, position, arguments, inverse_jacobian,
                                          std::integer_sequence<int, active_parameters...>{},
                                          std::make_index_sequence<sizeof...(active_parameters)>{});

    using Source = std::decay_t<decltype(mfem::future::get<VALUE>(result))>;
    using Flux = std::decay_t<decltype(mfem::future::get<DERIVATIVE>(result))>;
    if constexpr (mfem::future::is_zero<Source>::value) {
      output_value = OutputValue{};
    } else {
      output_value = mfem::future::get<VALUE>(result) * jacobian_weight;
    }
    if constexpr (mfem::future::is_zero<Flux>::value) {
      output_gradient = OutputGradient{};
    } else {
      output_gradient = mfem::future::dot(mfem::future::get<DERIVATIVE>(result),
                                          mfem::future::transpose(inverse_jacobian)) *
                        jacobian_weight;
    }
  }

 private:
  template <typename Position, typename Arguments, int... field, size_t... i>
  MFEM_HOST_DEVICE auto evaluateIntegrand(const TimeInfo& time_info, const Position& position,
                                          const Arguments& arguments, const CoordinateGradient& inverse_jacobian,
                                          std::integer_sequence<int, field...>,
                                          std::index_sequence<i...>) const
  {
    return integrand(time_info, position, fieldArgument<field, i>(arguments, inverse_jacobian)...);
  }

  template <int field, size_t active_index, typename Arguments>
  MFEM_HOST_DEVICE static auto fieldArgument(const Arguments& arguments, const CoordinateGradient& inverse_jacobian)
  {
    const auto& value = mfem::future::get<field>(arguments);
    const auto& gradient = mfem::future::get<sizeof...(InputSpaces) + active_index>(arguments);
    return mfem::future::tuple{value, mfem::future::dot(gradient, inverse_jacobian)};
  }
};

}  // namespace detail

template <int spatial_dim, typename OutputSpace, typename inputs = Parameters<>,
          typename input_indices = std::make_integer_sequence<int, inputs::n>>
class DifferentiableOperatorWeakForm;

/**
 * @brief A WeakForm implemented with mfem::future::DifferentiableOperator.
 *
 * Body integrands use MFEM-native quadrature-function types. The position argument is an `mfem::future::tuple` of
 * coordinates and the deformed Jacobian. Each selected input is an `mfem::future::tuple` of its value and physical
 * gradient. Scalar values are `double`; all other values and gradients are `mfem::future::tensor` objects. Integrands
 * return an `mfem::future::tuple` containing the source and flux and may use `mfem::future::zero` for either term.
 */
template <int spatial_dim, typename OutputSpace, typename... InputSpaces, int... input_indices>
class DifferentiableOperatorWeakForm<spatial_dim, OutputSpace, Parameters<InputSpaces...>,
                                     std::integer_sequence<int, input_indices...>> : public WeakForm {
 public:
  using SpacesT = std::vector<const mfem::ParFiniteElementSpace*>;
  using ShapeDispSpace = H1<SHAPE_ORDER, spatial_dim>;

  /**
   * @brief Construct a DifferentiableOperatorWeakForm.
   * @param[in] physics_name Name of the weak form.
   * @param[in] mesh Smith mesh.
   * @param[in] output_mfem_space Test space.
   * @param[in] input_mfem_spaces Spaces for the residual arguments.
   * @param[in] integration_order Optional MFEM integration order. The default matches FunctionalWeakForm.
   */
  DifferentiableOperatorWeakForm(std::string physics_name, std::shared_ptr<Mesh> mesh,
                                 const mfem::ParFiniteElementSpace& output_mfem_space,
                                 const SpacesT& input_mfem_spaces,
                                 std::optional<int> integration_order = std::nullopt)
      : WeakForm(std::move(physics_name)),
        mesh_(std::move(mesh)),
        output_mfem_space_(output_mfem_space),
        input_mfem_spaces_(input_mfem_spaces),
        integration_order_(integration_order.value_or(DEFAULT_INTEGRATION_ORDER)),
        weak_form_(makeInputDescriptors(mesh_, input_mfem_spaces_),
                   {{OUTPUT_FIELD_ID, &output_mfem_space_}}, mesh_->mfemParMesh()),
        residual_vector_(output_mfem_space_.GetTrueVSize())
  {
    static_assert(OutputSpace::family == Family::H1 || OutputSpace::family == Family::L2,
                  "DifferentiableOperatorWeakForm supports only H1 and L2 output spaces");
    static_assert(((InputSpaces::family == Family::H1 || InputSpaces::family == Family::L2) && ...),
                  "DifferentiableOperatorWeakForm supports only H1 and L2 input spaces");
    SLIC_ERROR_ROOT_IF(input_mfem_spaces_.size() != sizeof...(InputSpaces),
                       std::format("{} parameter spaces were specified but {} spaces were supplied",
                                   sizeof...(InputSpaces), input_mfem_spaces_.size()));
    SLIC_ERROR_ROOT_IF(integration_order_ < 0, "The integration order must be nonnegative");
    validateSpace<OutputSpace>(output_mfem_space_, "output");
    validateInputSpaces<0>();

    auto* nodes = dynamic_cast<mfem::ParGridFunction*>(mesh_->mfemParMesh().GetNodes());
    SLIC_ERROR_ROOT_IF(nodes == nullptr, "DifferentiableOperatorWeakForm requires mesh nodes");
    nodes->GetTrueDofs(base_coordinates_);
    base_coordinates_.UseDevice(true);
    deformed_coordinates_.SetSize(base_coordinates_.Size());
    deformed_coordinates_.UseDevice(true);
    time_data_.SetSize(4);
    time_data_.UseDevice(true);
    time_data_ = 0.0;
    time_data_.Read(true);
    residual_vector_.UseDevice(true);
  }

  /// @brief Add a body integral depending on selected input fields.
  template <int... active_parameters, typename BodyIntegralType>
  void addBodyIntegral(DependsOn<active_parameters...>, std::string body_name, BodyIntegralType integrand)
  {
    static_assert(((active_parameters >= 0 && active_parameters < static_cast<int>(sizeof...(InputSpaces))) && ...),
                  "DependsOn contains an invalid input index");
    addBodyIntegralImpl(body_name, std::move(integrand), std::integer_sequence<int, active_parameters...>{});
  }

  /// @brief Add a body integral depending on all input fields.
  template <typename BodyIntegralType>
  void addBodyIntegral(std::string body_name, BodyIntegralType integrand)
  {
    addBodyIntegralImpl(body_name, std::move(integrand),
                        std::make_integer_sequence<int, sizeof...(InputSpaces)>{});
  }

  /// @brief Add a body source depending on selected input fields.
  template <int... active_parameters, typename BodyLoadType>
  void addBodySource(DependsOn<active_parameters...> depends_on, std::string body_name, BodyLoadType load_function)
  {
    addBodyIntegral(depends_on, std::move(body_name),
                    detail::BodySourceIntegrand<BodyLoadType>{std::move(load_function)});
  }

  /// @brief Add a body source depending on all input fields.
  template <typename BodyLoadType>
  void addBodySource(std::string body_name, BodyLoadType load_function)
  {
    addBodyIntegral(std::move(body_name), detail::BodySourceIntegrand<BodyLoadType>{std::move(load_function)});
  }

  mfem::Vector residual(const TimeInfo& time_info, ConstFieldPtr shape_disp,
                        const std::vector<ConstFieldPtr>& fields,
                        const std::vector<ConstQuadratureFieldPtr>& quad_fields = {}) const override
  {
    validateEvaluationInputs("residual", shape_disp, fields, quad_fields);
    setTimeInfo(time_info);
    residual_vector_ = 0.0;
    if (has_integrals_) {
      auto state = makeState(shape_disp, fields);
      mfem::MultiVector output(residual_vector_);
      weak_form_.Mult(state, output);
    }
    return residual_vector_;
  }

  std::unique_ptr<mfem::HypreParMatrix> jacobian(
      const TimeInfo& time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
      const std::vector<double>& field_argument_tangents,
      const std::vector<ConstQuadratureFieldPtr>& quad_fields = {}) const override
  {
    validateEvaluationInputs("jacobian", shape_disp, fields, quad_fields);
    SLIC_ERROR_ROOT_IF(field_argument_tangents.size() != fields.size(),
                       "jacobian(): expected exactly one weight per input field");
    setTimeInfo(time_info);
    auto state = makeState(shape_disp, fields);
    std::unique_ptr<mfem::HypreParMatrix> result;
    const mfem::ParFiniteElementSpace* zero_space = nullptr;

    for (size_t i = 0; i < fields.size(); ++i) {
      const double weight = field_argument_tangents[i];
      if (weight == 0.0) continue;
      if (!active_derivatives_[i + 1]) {
        if (!result) zero_space = input_mfem_spaces_[i];
        continue;
      }
      SLIC_ERROR_ROOT_IF(
          derivative_integral_counts_[i] > 1,
          "jacobian(): MFEM currently cannot assemble multiple integrators contributing to the same derivative "
          "block; combine those terms in one body integrand");
      SLIC_ERROR_ROOT_IF(OutputSpace::family != Family::H1 || !INPUT_IS_H1[i],
                         "jacobian(): MFEM's LocalQF assembled-matrix path currently requires H1 input and output "
                         "spaces; use jvp() or vjp() for active L2 derivatives");
      auto derivative = weak_form_.GetDerivative(INPUT_FIELD_OFFSET + i, state);
      mfem::HypreParMatrix* assembled = nullptr;
      derivative->Assemble(assembled);
      std::unique_ptr<mfem::HypreParMatrix> contribution(assembled);
      SLIC_ERROR_ROOT_IF(contribution == nullptr, "dFEM did not assemble the requested Jacobian block");
      if (!result) {
        result = std::move(contribution);
        if (weight != 1.0) *result *= weight;
      } else {
        SLIC_ERROR_ROOT_IF(result->M() != contribution->M() || result->N() != contribution->N(),
                           "Cannot combine weighted Jacobian blocks with incompatible dimensions");
        result->Add(weight, *contribution);
      }
    }
    if (!result && zero_space != nullptr) result = makeZeroMatrix(*zero_space);
    return result;
  }

  void jvp(const TimeInfo& time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
           const std::vector<ConstQuadratureFieldPtr>& quad_fields, ConstFieldPtr v_shape_disp,
           const std::vector<ConstFieldPtr>& v_fields,
           const std::vector<ConstQuadratureFieldPtr>& v_quad_fields, DualFieldPtr jvp_reaction) const override
  {
    validateEvaluationInputs("jvp", shape_disp, fields, quad_fields);
    SLIC_ERROR_ROOT_IF(jvp_reaction == nullptr, "jvp(): output reaction must not be null");
    SLIC_ERROR_ROOT_IF(v_fields.size() != fields.size(), "jvp(): expected one tangent pointer per input field");
    SLIC_ERROR_ROOT_IF(!v_quad_fields.empty(), "jvp(): quadrature-field tangents are not supported");
    setTimeInfo(time_info);
    auto state = makeState(shape_disp, fields);
    *jvp_reaction = 0.0;
    if (v_shape_disp != nullptr && active_derivatives_[0]) {
      addDerivativeAction(SHAPE_FIELD_ID, state, *v_shape_disp, *jvp_reaction);
    }
    for (size_t i = 0; i < fields.size(); ++i) {
      if (v_fields[i] != nullptr && active_derivatives_[i + 1]) {
        addDerivativeAction(INPUT_FIELD_OFFSET + i, state, *v_fields[i], *jvp_reaction);
      }
    }
  }

  void vjp(const TimeInfo& time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
           const std::vector<ConstQuadratureFieldPtr>& quad_fields, ConstFieldPtr v_field,
           DualFieldPtr vjp_shape_disp_sensitivity, const std::vector<DualFieldPtr>& vjp_sensitivities,
           const std::vector<QuadratureFieldPtr>& vjp_quadrature_sensitivities) const override
  {
    validateEvaluationInputs("vjp", shape_disp, fields, quad_fields);
    SLIC_ERROR_ROOT_IF(v_field == nullptr, "vjp(): seed field must not be null");
    SLIC_ERROR_ROOT_IF(vjp_sensitivities.size() != fields.size(),
                       "vjp(): expected one sensitivity pointer per input field");
    SLIC_ERROR_ROOT_IF(!vjp_quadrature_sensitivities.empty(),
                       "vjp(): quadrature-field sensitivities are not supported");
    setTimeInfo(time_info);
    auto state = makeState(shape_disp, fields);
    if (vjp_shape_disp_sensitivity != nullptr && active_derivatives_[0]) {
      addDerivativeTranspose(SHAPE_FIELD_ID, state, *v_field, *vjp_shape_disp_sensitivity);
    }
    for (size_t i = 0; i < fields.size(); ++i) {
      if (vjp_sensitivities[i] != nullptr && active_derivatives_[i + 1]) {
        addDerivativeTranspose(INPUT_FIELD_OFFSET + i, state, *v_field, *vjp_sensitivities[i]);
      }
    }
  }

 private:
  static constexpr size_t SHAPE_FIELD_ID = 0;
  static constexpr size_t INPUT_FIELD_OFFSET = 1;
  static constexpr size_t OUTPUT_FIELD_ID = INPUT_FIELD_OFFSET + sizeof...(InputSpaces);
  static constexpr std::array<bool, sizeof...(InputSpaces)> INPUT_IS_H1{
      (InputSpaces::family == Family::H1)...};
  static constexpr int Q = std::max({OutputSpace::order, ShapeDispSpace::order, InputSpaces::order...}) + 1;
  static constexpr int DEFAULT_INTEGRATION_ORDER = 2 * Q - 1;
  static constexpr auto DERIVATIVE_KERNELS = mfem::future::DerivativeKernels::Action |
                                             mfem::future::DerivativeKernels::ApplyTranspose |
                                             mfem::future::DerivativeKernels::AssembleMatrix;

  static std::vector<mfem::future::FieldDescriptor> makeInputDescriptors(const std::shared_ptr<Mesh>& mesh,
                                                                          const SpacesT& input_spaces)
  {
    std::vector<mfem::future::FieldDescriptor> descriptors;
    descriptors.reserve(input_spaces.size() + 1);
    descriptors.emplace_back(SHAPE_FIELD_ID, &mesh->shapeDisplacementSpace());
    for (size_t i = 0; i < input_spaces.size(); ++i) {
      descriptors.emplace_back(INPUT_FIELD_OFFSET + i, input_spaces[i]);
    }
    return descriptors;
  }

  template <typename BodyIntegralType, int... active_parameters>
  void addBodyIntegralImpl(const std::string& body_name, BodyIntegralType integrand,
                           std::integer_sequence<int, active_parameters...>)
  {
    static_assert(sizeof...(InputSpaces) + sizeof...(active_parameters) <= 4,
                  "MFEM's LocalQF backend currently limits DifferentiableOperatorWeakForm to four total field "
                  "value/active-gradient slots; split the integral into terms with smaller DependsOn lists");
    auto [attributes, geometry, globally_empty] = domainAttributes(mesh_->domain(body_name));
    if (globally_empty) return;
    const auto& integration_rule = mfem::IntRules.Get(geometry, integration_order_);
    using QFunction = detail::DfemBodyQFunction<
        spatial_dim, OutputSpace, BodyIntegralType, std::integer_sequence<int, active_parameters...>,
        std::tuple<detail::DfemFieldValueArgument<InputSpaces>...,
                   detail::DfemFieldGradientArgument<
                       spatial_dim, std::tuple_element_t<active_parameters, std::tuple<InputSpaces...>>>...>,
        InputSpaces...>;
    QFunction q_function{std::move(integrand), time_data_.Read(true)};
    auto input_operators = makeInputOperators(std::integer_sequence<int, active_parameters...>{});
    auto output_operators = mfem::future::tuple{mfem::future::Value<OUTPUT_FIELD_ID>{},
                                                mfem::future::Gradient<OUTPUT_FIELD_ID>{}};
    auto derivatives =
        std::integer_sequence<size_t, SHAPE_FIELD_ID, INPUT_FIELD_OFFSET + active_parameters...>{};
    weak_form_.AddDomainIntegrator<mfem::future::LocalQFBackend, DERIVATIVE_KERNELS>(
        q_function, input_operators, output_operators, integration_rule, attributes, derivatives);
    active_derivatives_[0] = true;
    ((active_derivatives_[active_parameters + 1] = true), ...);
    ((++derivative_integral_counts_[active_parameters]), ...);
    has_integrals_ = true;
  }

  template <int... i>
  static auto makeInputOperators(std::integer_sequence<int, i...>)
  {
    return makeInputOperatorsImpl(std::integer_sequence<int, i...>{},
                                  std::make_index_sequence<sizeof...(InputSpaces)>{});
  }

  template <int... active, size_t... field>
  static auto makeInputOperatorsImpl(std::integer_sequence<int, active...>, std::index_sequence<field...>)
  {
    return mfem::future::tuple{mfem::future::Value<SHAPE_FIELD_ID>{},
                               mfem::future::Gradient<SHAPE_FIELD_ID>{}, mfem::future::Weight{},
                               mfem::future::Value<INPUT_FIELD_OFFSET + field>{}...,
                               mfem::future::Gradient<INPUT_FIELD_OFFSET + active>{}...};
  }

  struct DomainInfo {
    mfem::Array<int> attributes;
    mfem::Geometry::Type geometry;
    bool globally_empty;
  };

  DomainInfo domainAttributes(const Domain& domain) const
  {
    SLIC_ERROR_ROOT_IF(domain.type_ != Domain::Elements,
                       "DifferentiableOperatorWeakForm body integrals require an element domain");
    const auto& par_mesh = mesh_->mfemParMesh();
    const int max_attribute = par_mesh.attributes.Size() == 0 ? 0 : par_mesh.attributes.Max();
    std::vector<int> selected(static_cast<size_t>(par_mesh.GetNE()), 0);
    for (auto geometry : {mfem::Geometry::SEGMENT, mfem::Geometry::TRIANGLE, mfem::Geometry::SQUARE,
                          mfem::Geometry::TETRAHEDRON, mfem::Geometry::CUBE}) {
      for (int element : domain.get_mfem_ids(geometry)) selected[static_cast<size_t>(element)] = 1;
    }

    std::vector<int> local_selected(static_cast<size_t>(max_attribute), 0);
    std::vector<int> local_unselected(static_cast<size_t>(max_attribute), 0);
    for (int element = 0; element < par_mesh.GetNE(); ++element) {
      const int attribute = par_mesh.GetAttribute(element) - 1;
      if (selected[static_cast<size_t>(element)] != 0) {
        local_selected[static_cast<size_t>(attribute)] = 1;
      } else {
        local_unselected[static_cast<size_t>(attribute)] = 1;
      }
    }
    std::vector<int> global_selected(static_cast<size_t>(max_attribute), 0);
    std::vector<int> global_unselected(static_cast<size_t>(max_attribute), 0);
    MPI_Allreduce(local_selected.data(), global_selected.data(), max_attribute, MPI_INT, MPI_MAX, par_mesh.GetComm());
    MPI_Allreduce(local_unselected.data(), global_unselected.data(), max_attribute, MPI_INT, MPI_MAX,
                  par_mesh.GetComm());
    mfem::Array<int> marker(max_attribute);
    marker = 0;
    bool globally_empty = true;
    for (int attribute = 0; attribute < max_attribute; ++attribute) {
      SLIC_ERROR_ROOT_IF(global_selected[static_cast<size_t>(attribute)] != 0 &&
                             global_unselected[static_cast<size_t>(attribute)] != 0,
                         "DifferentiableOperatorWeakForm currently supports only complete attribute-based domains; "
                         "arbitrary domain subsets will be supported in the near future");
      marker[attribute] = global_selected[static_cast<size_t>(attribute)];
      globally_empty = globally_empty && marker[attribute] == 0;
    }

    std::array<int, mfem::Geometry::NUM_GEOMETRIES> local_geometry{};
    std::array<int, mfem::Geometry::NUM_GEOMETRIES> global_geometry{};
    for (auto geometry : {mfem::Geometry::SEGMENT, mfem::Geometry::TRIANGLE, mfem::Geometry::SQUARE,
                          mfem::Geometry::TETRAHEDRON, mfem::Geometry::CUBE}) {
      local_geometry[static_cast<size_t>(geometry)] = domain.get_mfem_ids(geometry).empty() ? 0 : 1;
    }
    MPI_Allreduce(local_geometry.data(), global_geometry.data(), mfem::Geometry::NUM_GEOMETRIES, MPI_INT, MPI_MAX,
                  par_mesh.GetComm());
    mfem::Geometry::Type selected_geometry = mfem::Geometry::INVALID;
    int geometry_count = 0;
    for (int geometry = 0; geometry < mfem::Geometry::NUM_GEOMETRIES; ++geometry) {
      if (global_geometry[static_cast<size_t>(geometry)] != 0) {
        selected_geometry = static_cast<mfem::Geometry::Type>(geometry);
        ++geometry_count;
      }
    }
    SLIC_ERROR_ROOT_IF(geometry_count > 1,
                       "DifferentiableOperatorWeakForm requires each integration domain to use one element geometry");
    return {std::move(marker), selected_geometry, globally_empty};
  }

  mfem::MultiVector makeState(ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields) const
  {
    deformed_coordinates_ = base_coordinates_;
    deformed_coordinates_ += *shape_disp;
    mfem::MultiVector state(static_cast<int>(fields.size() + 1));
    state.MakeRef(0, deformed_coordinates_);
    for (size_t i = 0; i < fields.size(); ++i) state.MakeRef(static_cast<int>(i + 1), *fields[i]);
    return state;
  }

  void setTimeInfo(const TimeInfo& time_info) const
  {
    auto* data = time_data_.Write(true);
    const double time = time_info.time() - time_info.dt();
    const double dt = time_info.dt();
    const double cycle = static_cast<double>(time_info.cycle());
    const double evaluation_mode = time_info.isCycleZeroEvaluation() ? 1.0 : 0.0;
    mfem::forall(1, [=] MFEM_HOST_DEVICE(int) {
      data[0] = time;
      data[1] = dt;
      data[2] = cycle;
      data[3] = evaluation_mode;
    });
  }

  void validateEvaluationInputs(const std::string& method, ConstFieldPtr shape_disp,
                                const std::vector<ConstFieldPtr>& fields,
                                const std::vector<ConstQuadratureFieldPtr>& quad_fields) const
  {
    SLIC_ERROR_ROOT_IF(shape_disp == nullptr, std::format("{}(): shape displacement must not be null", method));
    SLIC_ERROR_ROOT_IF(!sameSpace(shape_disp->space(), mesh_->shapeDisplacementSpace()),
                       std::format("{}(): shape displacement uses an incompatible finite element space", method));
    SLIC_ERROR_ROOT_IF(fields.size() != sizeof...(InputSpaces),
                       std::format("{}(): expected {} fields but received {}", method, sizeof...(InputSpaces),
                                   fields.size()));
    validateFieldsRecursive<0>(fields, method);
    SLIC_ERROR_ROOT_IF(!quad_fields.empty(),
                       std::format("{}(): quadrature fields are not supported by DifferentiableOperatorWeakForm",
                                   method));
  }

  template <size_t i>
  void validateInputSpaces() const
  {
    if constexpr (i < sizeof...(InputSpaces)) {
      using Space = std::tuple_element_t<i, std::tuple<InputSpaces...>>;
      validateSpace<Space>(*input_mfem_spaces_[i], std::format("input[{}]", i));
      validateInputSpaces<i + 1>();
    }
  }

  template <size_t i>
  void validateFieldsRecursive(const std::vector<ConstFieldPtr>& fields, const std::string& method) const
  {
    if constexpr (i < sizeof...(InputSpaces)) {
      using Space = std::tuple_element_t<i, std::tuple<InputSpaces...>>;
      SLIC_ERROR_ROOT_IF(fields[i] == nullptr, std::format("{}(): field[{}] must not be null", method, i));
      validateSpace<Space>(fields[i]->space(), std::format("{}(): field[{}]", method, i));
      validateFieldsRecursive<i + 1>(fields, method);
    }
  }

  template <typename Space>
  static void validateSpace(const mfem::ParFiniteElementSpace& space, const std::string& name)
  {
    const std::string collection_name = space.FEColl()->Name();
    if constexpr (Space::family == Family::H1) {
      SLIC_ERROR_ROOT_IF(collection_name.find("H1") == std::string::npos &&
                             collection_name.find("Linear") == std::string::npos,
                         std::format("Space '{}': expected H1 but received '{}'", name, collection_name));
    } else if constexpr (Space::family == Family::L2) {
      SLIC_ERROR_ROOT_IF(collection_name.find("L2") == std::string::npos &&
                             collection_name.find("DG") == std::string::npos &&
                             collection_name.find("Const") == std::string::npos,
                         std::format("Space '{}': expected L2 but received '{}'", name, collection_name));
    }
    SLIC_ERROR_ROOT_IF(space.FEColl()->GetOrder() != Space::order,
                       std::format("Space '{}': expected order {} but received {}", name, Space::order,
                                   space.FEColl()->GetOrder()));
    SLIC_ERROR_ROOT_IF(space.GetVDim() != Space::components,
                       std::format("Space '{}': expected {} components but received {}", name, Space::components,
                                   space.GetVDim()));
  }

  static bool sameSpace(const mfem::ParFiniteElementSpace& left, const mfem::ParFiniteElementSpace& right)
  {
    return left.GetParMesh() == right.GetParMesh() && std::strcmp(left.FEColl()->Name(), right.FEColl()->Name()) == 0 &&
           left.GetVDim() == right.GetVDim() && left.GetOrdering() == right.GetOrdering();
  }

  void addDerivativeAction(size_t derivative_id, const mfem::MultiVector& state, const mfem::Vector& direction,
                           mfem::Vector& output) const
  {
    auto derivative = weak_form_.GetDerivative(derivative_id, state);
    mfem::Vector contribution(output.Size());
    contribution.UseDevice(true);
    mfem::MultiVector result(contribution);
    derivative->Mult(direction, result);
    output.Add(1.0, contribution);
  }

  void addDerivativeTranspose(size_t derivative_id, const mfem::MultiVector& state, const mfem::Vector& seed,
                              mfem::Vector& output) const
  {
    auto derivative = weak_form_.GetDerivative(derivative_id, state);
    mfem::Vector contribution(output.Size());
    contribution.UseDevice(true);
    mfem::MultiVector direction(seed);
    derivative->MultTranspose(direction, contribution);
    output.Add(1.0, contribution);
  }

  std::unique_ptr<mfem::HypreParMatrix> makeZeroMatrix(const mfem::ParFiniteElementSpace& input_space) const
  {
    mfem::SparseMatrix local(output_mfem_space_.GetVSize(), input_space.GetVSize());
    local.Finalize();
    mfem::HypreParMatrix local_parallel(output_mfem_space_.GetComm(), output_mfem_space_.GlobalVSize(),
                                        input_space.GlobalVSize(), output_mfem_space_.GetDofOffsets(),
                                        input_space.GetDofOffsets(), &local);
    return std::unique_ptr<mfem::HypreParMatrix>(
        mfem::RAP(output_mfem_space_.Dof_TrueDof_Matrix(), &local_parallel, input_space.Dof_TrueDof_Matrix()));
  }

  std::shared_ptr<Mesh> mesh_;
  const mfem::ParFiniteElementSpace& output_mfem_space_;
  SpacesT input_mfem_spaces_;
  int integration_order_;
  mutable mfem::future::DifferentiableOperator weak_form_;
  mfem::Vector base_coordinates_;
  mutable mfem::Vector deformed_coordinates_;
  mutable mfem::Vector time_data_;
  mutable mfem::Vector residual_vector_;
  std::array<bool, sizeof...(InputSpaces) + 1> active_derivatives_{};
  std::array<size_t, sizeof...(InputSpaces)> derivative_integral_counts_{};
  bool has_integrals_ = false;
};

}  // namespace smith

#endif
