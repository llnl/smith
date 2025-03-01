// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/materials/solid_material_input.hpp"

namespace serac {

void SolidMaterialInputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // Define schema with each solid material parameter
  container.addString("model", "The model of material (e.g. NeoHookean)").required(true);
  container.addDouble("density", "Initial mass density");

  // Solid mechanics (neo-hookean, linear isotropic)
  container.addDouble("mu", "The shear modulus");
  container.addDouble("K", "The bulk modulus");

  // Solid mechanics (j2, j2nonlinear)
  container.addDouble("E", "Young's modulus");
  container.addDouble("nu", "Poisson's ratio");
  container.addDouble("Hk", "Kinematic hardening constant");
  auto& hardening_container = container.addStruct("hardening", "Hardening law");
  HardeningInputOptions::defineInputFileSchema(hardening_container);

  // Verify
  container.registerVerifier([](const axom::inlet::Container& c) -> bool {
    axom::inlet::InletType double_type = axom::inlet::InletType::Double;
    axom::inlet::InletType obj_type = axom::inlet::InletType::Object;
    bool density_present = c.contains("density") && (c["density"].type() == double_type);
    bool mu_present = c.contains("mu") && (c["mu"].type() == double_type);
    bool K_present = c.contains("K") && (c["K"].type() == double_type);
    bool E_present = c.contains("E") && (c["E"].type() == double_type);
    bool nu_present = c.contains("nu") && (c["nu"].type() == double_type);
    bool Hk_present = c.contains("Hk") && (c["Hk"].type() == double_type);
    bool hardening_present = c.contains("hardening") && (c["hardening"].type() == obj_type);

    std::string model = c["model"];
    if (model == "NeoHookean" || model == "LinearIsotropic") {
      return density_present && mu_present && K_present && !E_present && !nu_present && !hardening_present;
    } else if (model == "J2SmallStrain") {
      return density_present && !mu_present && !K_present && E_present && nu_present && Hk_present && hardening_present;
    }

    return false;
  });
}

}  // namespace serac

serac::var_solid_material_t FromInlet<serac::var_solid_material_t>::operator()(const axom::inlet::Container& base)
{
  serac::var_solid_material_t result;
  std::string model = base["model"];

  if (model == "NeoHookean") {
    result = serac::solid_mechanics::NeoHookean{.density = base["density"], .K = base["K"], .G = base["mu"]};
  } else if (model == "LinearIsotropic") {
    result = serac::solid_mechanics::LinearIsotropic{.density = base["density"], .K = base["K"], .G = base["mu"]};
  } else if (model == "J2SmallStrain") {
    serac::var_hardening_t hardening = base["hardening"].get<serac::var_hardening_t>();

    if (std::holds_alternative<serac::solid_mechanics::LinearHardening>(hardening)) {
      result = serac::solid_mechanics::J2SmallStrain<serac::solid_mechanics::LinearHardening>{
          .E = base["E"],
          .nu = base["nu"],
          .hardening = std::get<serac::solid_mechanics::LinearHardening>(hardening),
          .Hk = base["Hk"],
          .density = base["density"]};
    } else if (std::holds_alternative<serac::solid_mechanics::PowerLawHardening>(hardening)) {
      result = serac::solid_mechanics::J2SmallStrain<serac::solid_mechanics::PowerLawHardening>{
          .E = base["E"],
          .nu = base["nu"],
          .hardening = std::get<serac::solid_mechanics::PowerLawHardening>(hardening),
          .Hk = base["Hk"],
          .density = base["density"]};
    } else if (std::holds_alternative<serac::solid_mechanics::VoceHardening>(hardening)) {
      result = serac::solid_mechanics::J2SmallStrain<serac::solid_mechanics::VoceHardening>{
          .E = base["E"],
          .nu = base["nu"],
          .hardening = std::get<serac::solid_mechanics::VoceHardening>(hardening),
          .Hk = base["Hk"],
          .density = base["density"]};
    }
  }

  return result;
}
