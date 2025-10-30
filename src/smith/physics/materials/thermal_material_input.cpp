// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/materials/thermal_material_input.hpp"
#include <string>

#include "smith/numerics/functional/tensor.hpp"
#include "smith/physics/materials/thermal_material.hpp"

namespace smith {

void ThermalMaterialInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Define schema with each thermal material parameter
  container.addString("model", "The model of material").required(true);
  container.addDouble("density", "Initial mass density");
  container.addDouble("cp", "The specific heat capacity");

  // For LinearIsotropicConductor
  container.addDouble("kappa", "The conductivity parameter");

  // For LinearConductor
  container.addInt("dim", "Dimension of conductivity tensor parameter");
  auto& kappa_tensor_container = container.addStruct("kappa_tensor", "The conductivity tensor parameter");
  kappa_tensor_container.addDoubleArray("row1", "First row of the conductivity tensor parameter");
  kappa_tensor_container.addDoubleArray("row2", "Second row of the conductivity tensor parameter");
  kappa_tensor_container.addDoubleArray("row3", "Third row of the conductivity tensor parameter");

  // Verify
  container.registerVerifier([](const axom::inlet::Container& c) -> bool {
    axom::inlet::InletType double_type = axom::inlet::InletType::Double;
    axom::inlet::InletType int_type = axom::inlet::InletType::Integer;
    axom::inlet::InletType obj_type = axom::inlet::InletType::Object;
    axom::inlet::InletType coll_type = axom::inlet::InletType::Collection;
    std::string model = c["model"];
    bool density_present = c.contains("density") && (c["density"].type() == double_type);
    bool cp_present = c.contains("cp") && (c["cp"].type() == double_type);
    bool kappa_present = c.contains("kappa") && (c["kappa"].type() == double_type);
    bool dim_present = c.contains("dim") && (c["dim"].type() == int_type);
    bool kappa_tensor_present = c.contains("kappa_tensor") && (c["kappa_tensor"].type() == obj_type);

    if (model == "LinearIsotropicConductor") {
      return density_present && cp_present && kappa_present && !dim_present && !kappa_tensor_present;
    } else if (model == "LinearConductor") {
      if (density_present && cp_present && !kappa_present && dim_present && kappa_tensor_present) {
        // Verify rows of kappa tensor struct is an array of doubles and is of proper size
        int dim = c["dim"];
        bool row1_present = c.contains("kappa_tensor/row1") && (c["kappa_tensor/row1"].type() == coll_type);
        bool row2_present = c.contains("kappa_tensor/row2") && (c["kappa_tensor/row2"].type() == coll_type);
        bool row3_present = c.contains("kappa_tensor/row3") && (c["kappa_tensor/row3"].type() == coll_type);
        auto row1_size = c["kappa_tensor/row1"].get<std::vector<double>>().size();
        auto row2_size = c["kappa_tensor/row2"].get<std::vector<double>>().size();
        auto row3_size = c["kappa_tensor/row3"].get<std::vector<double>>().size();

        if (dim == 2) {
          return row1_present && (row1_size == 2) && row2_present && (row2_size == 2) && !row3_present;
        } else if (dim == 3) {
          return row1_present && (row1_size == 3) && row2_present && (row2_size == 3) && row3_present &&
                 (row3_size == 3);
        }
      }
    }

    return false;
  });
}

}  // namespace smith

std::vector<std::vector<double>> FromInlet<std::vector<std::vector<double>>>::operator()(
    const axom::inlet::Container& base)
{
  std::vector<std::vector<double>> result;

  result.push_back(base["row1"].get<std::vector<double>>());
  result.push_back(base["row2"].get<std::vector<double>>());
  result.push_back(base["row3"].get<std::vector<double>>());

  return result;
}

smith::var_thermal_material_t FromInlet<smith::var_thermal_material_t>::operator()(const axom::inlet::Container& base)
{
  smith::var_thermal_material_t result;
  std::string model = base["model"];

  if (model == "LinearIsotropicConductor") {
    result = smith::heat_transfer::LinearIsotropicConductor(base["density"], base["cp"], base["kappa"]);
  } else if (model == "LinearConductor") {
    // Store tensor in a vector temporarily, then
    // set the tensor values and material result based on the dimension
    int dim = base["dim"];
    if (dim == 2) {
      std::vector<std::vector<double>> v = {base["kappa_tensor"]["row1"], base["kappa_tensor"]["row2"]};
      smith::tensor<double, 2, 2> cond = {{{v[0][0], v[0][1]}, {v[1][0], v[1][1]}}};
      result = smith::heat_transfer::LinearConductor<2>(base["density"], base["cp"], cond);
    } else if (dim == 3) {
      std::vector<std::vector<double>> v = {base["kappa_tensor"]["row1"], base["kappa_tensor"]["row2"],
                                            base["kappa_tensor"]["row3"]};
      smith::tensor<double, 3, 3> cond = {
          {{v[0][0], v[0][1], v[0][2]}, {v[1][0], v[1][1], v[1][2]}, {v[2][0], v[2][1], v[2][2]}}};
      result = smith::heat_transfer::LinearConductor<3>(base["density"], base["cp"], cond);
    }
  }

  return result;
}
