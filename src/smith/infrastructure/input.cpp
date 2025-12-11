// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/infrastructure/input.hpp"

#include <cstdlib>
#include <climits>
#include <algorithm>
#include <cstddef>
#include <utility>

#include "axom/core.hpp"

#include "smith/infrastructure/logger.hpp"
#include "smith/numerics/solver_config.hpp"

namespace smith::input {

axom::inlet::Inlet initialize(axom::sidre::DataStore& datastore, const std::string& input_file_path,
                              const Language language, const std::string& sidre_path)
{
  // Initialize Inlet
  std::unique_ptr<axom::inlet::Reader> reader;
  if (language == Language::Lua) {
    reader = std::make_unique<axom::inlet::LuaReader>();
  } else if (language == Language::JSON) {
    reader = std::make_unique<axom::inlet::JSONReader>();
  } else if (language == Language::YAML) {
    reader = std::make_unique<axom::inlet::YAMLReader>();
  }

  if (axom::utilities::filesystem::pathExists(input_file_path)) {
    reader->parseFile(input_file_path);
  }

  // Store inlet data under its own group
  if (datastore.getRoot()->hasGroup(sidre_path)) {
    // If this is a restart, wipe out the previous input file
    datastore.getRoot()->destroyGroup(sidre_path);
  }
  axom::sidre::Group* inlet_root = datastore.getRoot()->createGroup(sidre_path);
  return axom::inlet::Inlet(std::move(reader), inlet_root);
}

std::string findMeshFilePath(const std::string& mesh_path, const std::string& input_file_path)
{
  using namespace axom::utilities;

  // Check if given path exists
  if (filesystem::pathExists(mesh_path)) {
    return mesh_path;
  }

  // Check relative to input file
  std::string input_file_dir = fullDirectoryFromPath(input_file_path);
  std::string possible_path = filesystem::joinPath(input_file_dir, mesh_path);
  if (filesystem::pathExists(possible_path)) {
    return possible_path;
  }

  // Failed to find mesh file
  std::string msg = axom::fmt::format("Input file: Given mesh file does not exist: '{0}'", mesh_path);
  SLIC_ERROR_ROOT(msg);
  return "";
}

std::string fullDirectoryFromPath(const std::string& path)
{
  char actualpath[PATH_MAX + 1];
  char* ptr = realpath(path.c_str(), actualpath);
  if (ptr == nullptr) {
    SLIC_ERROR_ROOT("Failed to find absolute path from input file.");
  }
  std::string dir;
  axom::utilities::filesystem::getDirName(dir, std::string(actualpath));
  return dir;
}

std::string getInputFileName(const std::string& file_path)
{
  axom::Path path(file_path);
  std::string basename = path.baseName();
  std::string name;

  std::size_t index = basename.find_last_of(".");
  if (index != std::string::npos) {
    name = basename.substr(0, index);
  } else {
    name = basename;
  }

  return name;
}

void defineVectorInputFileSchema(axom::inlet::Container& container)
{
  // TODO: I had to remove the required tag on x as we now have an optional vector input in the coefficients. IT would
  // be nice to support "If this exists, this subcomponent is required."
  container.addDouble("x", "x-component of vector");
  container.addDouble("y", "y-component of vector");
  container.addDouble("z", "z-component of vector");
}

void BoundaryConditionInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  container.addIntArray("attrs", "Boundary attributes to which the BC should be applied");
  CoefficientInputOptions::defineInputFileSchema(container);
}

bool CoefficientInputOptions::isVector() const
{
  return vector_function || vector_constant || (!vector_pw_const.empty());
}

std::unique_ptr<mfem::VectorCoefficient> CoefficientInputOptions::constructVector(const int dim) const
{
  SLIC_ERROR_ROOT_IF(!isVector(), "Cannot construct a vector coefficient from scalar input");

  if (vector_function) {
    return std::make_unique<mfem::VectorFunctionCoefficient>(dim, vector_function);
  } else if (vector_constant) {
    return std::make_unique<mfem::VectorConstantCoefficient>(*vector_constant);
  } else if (!vector_pw_const.empty()) {
    // Find the maximum mesh attribute
    auto max_attr_elem = std::max_element(vector_pw_const.begin(), vector_pw_const.end(),
                                          [](auto a, auto b) { return a.first < b.first; });

    // Create the vector array coefficient. We will use this as an array of piecewise constant scalars
    auto vec_pw_coeff = std::make_unique<mfem::VectorArrayCoefficient>(max_attr_elem->second.Size());

    // Loop over each spatial dimension
    for (int i = 0; i < max_attr_elem->second.Size(); ++i) {
      // Create an mfem vector for the attributes
      // Note that this vector expects zero indexing
      mfem::Vector pw_constants(max_attr_elem->first);
      pw_constants = 0.0;

      for (auto& entry : vector_pw_const) {
        pw_constants(entry.first - 1) = entry.second[i];
      }

      // Set the spatial dimension coefficient to a newly constructed scalar piecewise coefficient
      vec_pw_coeff->Set(i, new mfem::PWConstCoefficient(pw_constants));
    }
    return vec_pw_coeff;

  } else {
    SLIC_ERROR(
        "Trying to build a vector coefficient without specifying a vector_function, vector_constant, or "
        "vector_piecewise_constant.");
    return nullptr;
  }
}

std::unique_ptr<mfem::Coefficient> CoefficientInputOptions::constructScalar() const
{
  SLIC_ERROR_ROOT_IF(isVector(), "Cannot construct a scalar coefficient from vector input");

  if (scalar_function) {
    return std::make_unique<mfem::FunctionCoefficient>(scalar_function);
  } else if (scalar_constant) {
    return std::make_unique<mfem::ConstantCoefficient>(*scalar_constant);
  } else if (!scalar_pw_const.empty()) {
    // First, find the element with the maximum attribute key
    auto max_attr_elem = std::max_element(scalar_pw_const.begin(), scalar_pw_const.end(),
                                          [](auto a, auto b) { return a.first < b.first; });

    // Check for a valid iterator to avoid compiler warnings
    if (max_attr_elem != scalar_pw_const.end()) {
      // Create an mfem vector for the attributes
      // Note that this vector expects zero indexing
      mfem::Vector pw_constants(max_attr_elem->first);
      pw_constants = 0.0;

      for (auto& entry : scalar_pw_const) {
        pw_constants(entry.first - 1) = entry.second;
      }

      // Create the MFEM coefficient
      return std::make_unique<mfem::PWConstCoefficient>(pw_constants);
    }
  }

  SLIC_ERROR_ROOT(
      "Trying to build a scalar coefficient without specifying a scalar_function, constant, or piecewise_constant.");
  return nullptr;
}

void CoefficientInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Vectors are implemented as lua usertypes and can be converted to/from mfem::Vector
  container.addFunction("vector_function", axom::inlet::FunctionTag::Vector,
                        {axom::inlet::FunctionTag::Vector, axom::inlet::FunctionTag::Double},
                        "The function to use for an mfem::VectorFunctionCoefficient");
  container.addFunction("scalar_function", axom::inlet::FunctionTag::Double,
                        {axom::inlet::FunctionTag::Vector, axom::inlet::FunctionTag::Double},
                        "The function to use for an mfem::FunctionCoefficient");
  container.addInt("component", "The vector component to which the scalar coefficient should be applied");

  container.addDouble("constant", "The constant scalar value to use as the coefficient");

  auto& vector_container = container.addStruct("vector_constant", "The constant vector to use as the coefficient");
  smith::input::defineVectorInputFileSchema(vector_container);

  container.addDoubleArray("piecewise_constant",
                           "Map of mesh attributes to constant values to use as a piecewise coefficient");

  auto& pw_vector_container = container.addStructArray(
      "vector_piecewise_constant", "Map of mesh attributes to constant vectors to use as a piecewise coefficient");
  smith::input::defineVectorInputFileSchema(pw_vector_container);
}

}  // namespace smith::input

mfem::Vector FromInlet<mfem::Vector>::operator()(const axom::inlet::Container& base)
{
  mfem::Vector result(3);  // Allocate up front since it's small
  result[0] = base["x"];
  if (base.contains("y")) {
    result[1] = base["y"];
    if (base.contains("z")) {
      result[2] = base["z"];
    } else {
      result.SetSize(2);  // Shrink to a 2D vector, leaving the data intact
    }
  } else {
    result.SetSize(1);  // Shrink to a 1D vector, leaving the data intact
  }
  return result;
}

smith::input::BoundaryConditionInputOptions FromInlet<smith::input::BoundaryConditionInputOptions>::operator()(
    const axom::inlet::Container& base)
{
  smith::input::BoundaryConditionInputOptions result{.attrs = {},
                                                     .coef_opts = base.get<smith::input::CoefficientInputOptions>()};
  // Build a set with just the values of the map
  auto bdr_attr_map = base["attrs"].get<std::unordered_map<int, int>>();
  for (const auto& [_, val] : bdr_attr_map) {
    result.attrs.insert(val);
  }
  return result;
}

smith::input::CoefficientInputOptions FromInlet<smith::input::CoefficientInputOptions>::operator()(
    const axom::inlet::Container& base)
{
  smith::input::CoefficientInputOptions result;

  // Create a counter for definition of the coefficient
  int coefficient_definitions = 0;

  // Check if functions have been assigned and store them appropriately
  if (base.contains("vector_function")) {
    auto func = base["vector_function"]
                    .get<std::function<axom::inlet::FunctionType::Vector(axom::inlet::FunctionType::Vector, double)>>();
    result.vector_function = [func(std::move(func))](const mfem::Vector& input, double t, mfem::Vector& output) {
      auto ret = func(axom::inlet::FunctionType::Vector{input.GetData(), input.Size()}, t);
      // Copy from the primal vector into the MFEM vector
      std::copy(ret.vec.data(), ret.vec.data() + input.Size(), output.GetData());
    };
    coefficient_definitions++;
  }

  if (base.contains("scalar_function")) {
    auto func = base["scalar_function"].get<std::function<double(axom::inlet::FunctionType::Vector, double)>>();
    result.scalar_function = [func(std::move(func))](const mfem::Vector& input, double t) {
      return func(axom::inlet::FunctionType::Vector{input.GetData(), input.Size()}, t);
    };
    coefficient_definitions++;
  }

  if (base.contains("constant")) {
    result.scalar_constant = base["constant"];
    coefficient_definitions++;
  }

  if (base.contains("vector_constant")) {
    result.vector_constant = base["vector_constant"].get<mfem::Vector>();
    coefficient_definitions++;
  }

  if (base.contains("piecewise_constant")) {
    result.scalar_pw_const = base["piecewise_constant"].get<std::unordered_map<int, double>>();
    coefficient_definitions++;
  }

  if (base.contains("vector_piecewise_constant")) {
    result.vector_pw_const = base["vector_piecewise_constant"].get<std::unordered_map<int, mfem::Vector>>();
    coefficient_definitions++;
  }

  // If scalar valued, check of a component
  if (result.scalar_constant || result.scalar_function || !result.scalar_pw_const.empty()) {
    // If component input exists, set it in the option struct
    if (base.contains("component")) {
      result.component = base["component"];
    }
  }

  SLIC_ERROR_ROOT_IF(coefficient_definitions > 1,
                     "Coefficient has multiple definitions. Please use only one of (constant, vector_constant, "
                     "piecewise_constant, vector_piecewise_constant, scalar_function, vector_function");
  SLIC_ERROR_ROOT_IF(coefficient_definitions == 0, "Coefficient definition does not contain known type.");

  return result;
}
