// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/materials/solid_material_input.hpp"

namespace serac {

void HardeningInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Shared between both hardening laws
  container.addString("law", "Name of the hardening law (e.g. PowerLawHardening)").required(true);
  container.addDouble("sigma_y", "Yield strength");

  // PowerLawHardening
  container.addDouble("n", "Hardening index in reciprocal form");
  container.addDouble("eps0", "Reference value of accumulated plastic strain");

  // VoceHardening
  container.addDouble("sigma_sat", "Saturation value of flow strength");
  container.addDouble("strain_constant", "Constant dictating how fast the exponential decays");

  // Verify
  container.registerVerifier([](const axom::inlet::Container& c) -> bool {
    axom::inlet::InletType double_type = axom::inlet::InletType::Double;
    bool sigma_y_present = c.contains("sigma_y") && (c["sigma_y"].type() == double_type);
    bool n_present = c.contains("n") && (c["n"].type() == double_type);
    bool eps0_present = c.contains("eps0") && (c["eps0"].type() == double_type);
    bool sigma_sat_present = c.contains("sigma_sat") && (c["sigma_sat"].type() == double_type);
    bool strain_constant_present = c.contains("strain_constant") && (c["strain_constant"].type() == double_type);

    std::string law = c["law"];
    if (law == "PowerLawHardening") {
      return sigma_y_present && n_present && eps0_present && !sigma_sat_present && !sigma_sat_present;
    } else if (law == "VoceHardening") {
      return sigma_y_present && !n_present && !eps0_present && sigma_sat_present && strain_constant_present;
    }

    return false;
  });
}

}  // namespace serac

serac::var_hardening_t FromInlet<serac::var_hardening_t>::operator()(const axom::inlet::Container& base)
{
  serac::var_hardening_t result;
  std::string law = base["law"];
  if (law == "PowerLawHardening") {
    result =
        serac::solid_mechanics::PowerLawHardening{.sigma_y = base["sigma_y"], .n = base["n"], .eps0 = base["eps0"]};
  } else if (law == "VoceHardening") {
    result = serac::solid_mechanics::VoceHardening{
        .sigma_y = base["sigma_y"], .sigma_sat = base["sigma_sat"], .strain_constant = base["strain_constant"]};
  }
  return result;
}
