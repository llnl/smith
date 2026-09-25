// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <set>
#include <vector>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/pchip.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/smith_config.hpp"

namespace smith {

namespace {

template <typename ModulusType, typename GradientType, int dim>
auto linearIsotropicStress(ModulusType youngs_modulus, double poisson_ratio,
                           const tensor<GradientType, dim, dim>& displacement_gradient)
{
  const auto shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
  const auto bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio));
  const auto lambda = bulk_modulus - 2.0 * shear_modulus / 3.0;
  const auto strain = 0.5 * (displacement_gradient + transpose(displacement_gradient));
  return lambda * tr(strain) * Identity<dim>() + 2.0 * shear_modulus * strain;
}

struct TemperatureDependentLinearIsotropic {
  using State = Empty;

  double density;
  double poisson_ratio;
  PchipView youngs_modulus;

  template <typename GradientType, typename TemperatureType, int dim>
  auto operator()(State&, const tensor<GradientType, dim, dim>& displacement_gradient,
                  TemperatureType temperature) const
  {
    const auto modulus = youngs_modulus(get<0>(temperature));
    return linearIsotropicStress(modulus, poisson_ratio, displacement_gradient);
  }
};

}  // namespace

TEST(TabulatedSolidProperties, MaterialAndTractionCallbacks)
{
  constexpr int order = 1;
  constexpr int dim = 2;
  constexpr double poisson_ratio = 0.25;
  const std::vector temperatures{0.0, 1.0, 2.0, 4.0};
  const std::vector moduli{10.0, 15.0, 18.0, 20.0};
  const PchipData youngs_modulus_data(temperatures, moduli);
  const auto youngs_modulus = youngs_modulus_data.view();

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "test_tabulated_solid_properties");

  auto mesh = std::make_shared<Mesh>(buildMeshFromFile(SMITH_REPO_DIR "/data/meshes/patch2D_quads.mesh"), "mesh");
  mesh->addDomainOfBoundaryElements("essential_boundary", by_attr<dim>(std::set<int>{1, 4}));

  NonlinearSolverOptions nonlinear_options{.relative_tol = 1.0e-13, .absolute_tol = 1.0e-13};
  SolidMechanics<order, dim, Parameters<H1<order>>> solid(nonlinear_options, solid_mechanics::default_linear_options,
                                                          solid_mechanics::default_quasistatic_options, "solid", mesh,
                                                          {"temperature"});

  FiniteElementState temperature(mesh->mfemParMesh(), H1<order>{}, "temperature");
  temperature = 1.5;
  solid.setParameter(0, temperature);

  TemperatureDependentLinearIsotropic material{
      .density = 1.0, .poisson_ratio = poisson_ratio, .youngs_modulus = youngs_modulus};
  solid.setMaterial(DependsOn<0>{}, material, mesh->entireBody());

  constexpr tensor<double, dim, dim> displacement_gradient{{{0.02, 0.01}, {-0.01, 0.03}}};
  constexpr tensor<double, dim> translation{{0.1, -0.2}};
  solid.setDisplacementBCs(
      [=](tensor<double, dim> position, double) { return displacement_gradient * position + translation; },
      mesh->domain("essential_boundary"));

  solid.setTraction(
      DependsOn<0>{},
      [=](auto, auto normal, double, auto temperature_value) {
        const auto modulus = youngs_modulus(get<0>(temperature_value));
        const auto stress = linearIsotropicStress(modulus, poisson_ratio, displacement_gradient);
        return stress * normal;
      },
      mesh->entireBoundary());

  solid.completeSetup();
  solid.advanceTimestep(1.0);

  auto exact_displacement = [=](const mfem::Vector& position, mfem::Vector& displacement) {
    const tensor<double, dim> position_tensor{{position[0], position[1]}};
    const auto displacement_tensor = displacement_gradient * position_tensor + translation;
    displacement[0] = displacement_tensor[0];
    displacement[1] = displacement_tensor[1];
  };
  mfem::VectorFunctionCoefficient exact_solution(dim, exact_displacement);
  EXPECT_LT(computeL2Error(solid.displacement(), exact_solution), 1.0e-11);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager application_manager(argc, argv);
  return RUN_ALL_TESTS();
}
