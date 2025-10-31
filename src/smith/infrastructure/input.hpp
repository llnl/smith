// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file input.hpp
 *
 * @brief This file contains the all the necessary functions for reading input files
 */

#pragma once

#include <string>
#include <variant>
#include <optional>
#include <functional>
#include <memory>
#include <set>
#include <unordered_map>

#include "mfem.hpp"
#include "axom/inlet.hpp"
#include "axom/sidre.hpp"

/**
 * @brief The input related helper functions and objects
 *
 */
namespace smith::input {

/**
 * @brief The input file languages supported by Inlet
 */
enum class Language
{
  Lua,
  JSON,
  YAML
};

/**
 * @brief Initializes Inlet with the given datastore and input file.
 *
 * @param[in] datastore Root of the Sidre datastore
 * @param[in] input_file_path Path to user given input file
 * @param[in] language The language of the file at @a input_file_path
 * @param[in] sidre_path The path within the datastore to use as the root of the Inlet hierarchy
 * @return initialized Inlet instance
 */
axom::inlet::Inlet initialize(axom::sidre::DataStore& datastore, const std::string& input_file_path,
                              const Language language = Language::Lua, const std::string& sidre_path = "input_file");

/**
 * @brief Returns the absolute path of the given mesh either relative
 * to CWD or the input file
 *
 * @param[in] mesh_path Given mesh path from input file
 * @param[in] input_file_path Path to user given input file
 * @return Absolute path of found file
 */
std::string findMeshFilePath(const std::string& mesh_path, const std::string& input_file_path);

/**
 * @brief Returns the absolute directory of the given file path.
 *
 * @param[in] file_path path to a file
 * @return Absolute directory of given file path
 */
std::string fullDirectoryFromPath(const std::string& file_path);

/**
 * @brief Returns the name of the input file (base name with file extension removed).
 *
 * // TODO: remove when axom::Path supports this
 *
 * @param[in] file_path path to a file
 * @return name of input file
 */
std::string getInputFileName(const std::string& file_path);

/**
 * @brief Defines the schema for a vector in R^{1,2,3} space
 * @param[inout] container The base container on which to define the schema
 */
void defineVectorInputFileSchema(axom::inlet::Container& container);

/**
 * @brief The information required from the input file for an mfem::(Vector)(Function)Coefficient
 */
struct CoefficientInputOptions {
  /**
   * @brief The type for coefficient functions that are vector-valued
   *
   */
  using VecFunc = std::function<void(const mfem::Vector&, double, mfem::Vector&)>;

  /**
   * @brief The type for coefficient functions that are scalar-valued
   *
   */
  using ScalarFunc = std::function<double(const mfem::Vector&, double)>;
  /**
   * @brief The scalar std::function corresponding to a function coefficient
   */
  ScalarFunc scalar_function;

  /**
   * @brief The vector std::function corresponding to a function coefficient
   */
  VecFunc vector_function;

  /**
   * @brief The scalar constant associated with the coefficient
   */
  std::optional<double> scalar_constant;

  /**
   * @brief The vector constant associated with the coefficient
   */
  std::optional<mfem::Vector> vector_constant;

  /**
   * @brief Scalar piecewise constant definition map
   */
  std::unordered_map<int, double> scalar_pw_const;

  /**
   * @brief Vector piecewise constant definition map
   */
  std::unordered_map<int, mfem::Vector> vector_pw_const;

  /**
   * @brief The component to which a scalar coefficient should be applied
   */
  std::optional<int> component;
  /**
   * @brief Returns whether the contained function corresponds to a vector coefficient
   */
  bool isVector() const;
  /**
   * @brief Constructs a vector coefficient with the requested dimension
   */
  std::unique_ptr<mfem::VectorCoefficient> constructVector(const int dim = 3) const;
  /**
   * @brief Constructs a scalar coefficient
   */
  std::unique_ptr<mfem::Coefficient> constructScalar() const;
  /**
   * @brief Defines the input file schema on the provided inlet container
   */
  static void defineInputFileSchema(axom::inlet::Container& container);
};

/**
 * @brief The information required from the input file for a boundary condition
 */
struct BoundaryConditionInputOptions {
  /**
   * @brief The mesh attributes on which to apply the boundary condition
   */
  std::set<int> attrs{};
  /**
   * @brief The information from the input file on the BC coefficient
   */
  CoefficientInputOptions coef_opts;
  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] container Inlet's Container to which fields should be added
   **/
  static void defineInputFileSchema(axom::inlet::Container& container);
};

}  // namespace smith::input

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by Inlet
 */
template <>
struct FromInlet<mfem::Vector> {
  /// @brief Returns created object from Inlet container
  mfem::Vector operator()(const axom::inlet::Container& base);
};

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by Inlet
 */
template <>
struct FromInlet<smith::input::CoefficientInputOptions> {
  /// @brief Returns created object from Inlet container
  smith::input::CoefficientInputOptions operator()(const axom::inlet::Container& base);
};

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by Inlet
 */
template <>
struct FromInlet<smith::input::BoundaryConditionInputOptions> {
  /// @brief Returns created object from Inlet container
  smith::input::BoundaryConditionInputOptions operator()(const axom::inlet::Container& base);
};
