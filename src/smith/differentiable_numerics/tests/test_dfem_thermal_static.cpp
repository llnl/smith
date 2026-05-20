// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <gtest/gtest.h>
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/dfem_weak_form.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "mfem/fem/dfem/fieldoperator.hpp"

using namespace smith;

namespace dfem_thermal_static_test {

struct ThermalMaterialQf {
  static constexpr int dim = 2;
  SMITH_HOST_DEVICE inline void operator()(
      const mfem::future::tensor<double, 2>& grad_T,
      const double& k,
      const double& w,
      mfem::future::tensor<double, 2>& out_grad_v) const
  {
    auto flux = k * grad_T;
    out_grad_v = flux * w;
  }
};

struct ThermalSourceQf {
  static constexpr int dim = 2;
  SMITH_HOST_DEVICE inline void operator()(
      const mfem::future::tensor<double, 2>& X,
      const double& w,
      double& out_v) const
  {
    double x = X[0];
    double y = X[1];
    double source = 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
    out_v = -source * w;
  }
};

} // namespace dfem_thermal_static_test

using namespace dfem_thermal_static_test;

struct DfemThermalStaticFixture : public testing::Test {
  axom::sidre::DataStore datastore;
  std::shared_ptr<Mesh> mesh;

  void setupMesh(int n_elements, std::string tag)
  {
    StateManager::reset();
    StateManager::initialize(datastore, "dfem_thermal_static_" + tag);
    auto mfem_mesh = mfem::Mesh::MakeCartesian2D(n_elements, n_elements, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    mesh = std::make_shared<Mesh>(std::move(mfem_mesh), "thermal_mesh_" + tag);
  }

  void SetUp() override
  {
    setupMesh(32, "default");
  }

  void TearDown() override { StateManager::reset(); }
};

TEST_F(DfemThermalStaticFixture, ManufacturedSolutionAndVJP)
{
  // 1. Setup spaces and states
  mfem::H1_FECollection fec(1, mesh->mfemParMesh().Dimension());
  mfem::ParFiniteElementSpace fes(&mesh->mfemParMesh(), &fec);

  auto temp = std::make_unique<FiniteElementState>(fes, "temperature");
  temp->UseDevice(true);
  *temp = 0.0;

  auto k_param = std::make_unique<FiniteElementState>(fes, "conductivity");
  k_param->UseDevice(true);
  *k_param = 1.0;

  auto coords_space = dynamic_cast<const mfem::ParFiniteElementSpace*>(mesh->mfemParMesh().GetNodes()->FESpace());
  SLIC_ERROR_IF(coords_space == nullptr, "coords_space is nullptr! GetNodes()->FESpace() is not a ParFiniteElementSpace.");
  auto coords = std::make_unique<FiniteElementState>(*coords_space, "coords");
  coords->UseDevice(true);
  coords->setFromGridFunction(*dynamic_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes()));

  // 2. Setup DfemWeakForm
  std::vector<const mfem::ParFiniteElementSpace*> input_spaces = {&temp->space(), &k_param->space(), coords_space};
  DfemWeakForm weak_form("thermal", mesh, temp->space(), input_spaces);

  mfem::Array<int> all_attrs;
  all_attrs.Append(1);

  const mfem::IntegrationRule* ir = &mfem::IntRules.Get(mesh->mfemParMesh().GetElementGeometry(0), 2);

  // Material: integral(k * grad(T) . grad(v))
  weak_form.addBodyIntegral(all_attrs, ThermalMaterialQf{},
                            mfem::future::tuple{mfem::future::Gradient<0>{}, mfem::future::Value<1>{}, mfem::future::Weight{}},
                            mfem::future::tuple{mfem::future::Gradient<3>{}},
                            *ir,
                            std::integer_sequence<size_t, 0, 1>{}); // Differentiate w.r.t temp and k_param

  // Source: integral(-source * v)
  weak_form.addBodyIntegral(all_attrs, ThermalSourceQf{},
                            mfem::future::tuple{mfem::future::Value<2>{}, mfem::future::Weight{}},
                            mfem::future::tuple{mfem::future::Value<3>{}},
                            *ir,
                            std::integer_sequence<size_t>{});

  // 3. Solve (Linear Problem)
  std::vector<ConstFieldPtr> fields = {temp.get(), k_param.get(), coords.get()};
  
  auto res = weak_form.residual(TimeInfo(0, 1), nullptr, fields);
  auto jac = weak_form.jacobian(TimeInfo(0, 1), nullptr, fields, {1.0, 0.0, 0.0});

  FiniteElementState temp_tangent(fes, "temp_tangent");
  temp_tangent.Randomize(3);
  FiniteElementDual jac_times_dir(fes, "jac_times_dir");
  jac->Mult(temp_tangent, jac_times_dir);

  constexpr double eps = 1.0e-7;
  FiniteElementState temp_plus(*temp);
  FiniteElementState temp_minus(*temp);
  temp_plus.Add(eps, temp_tangent);
  temp_minus.Add(-eps, temp_tangent);

  auto res_plus = weak_form.residual(TimeInfo(0, 1), nullptr, {&temp_plus, k_param.get(), coords.get()});
  auto res_minus = weak_form.residual(TimeInfo(0, 1), nullptr, {&temp_minus, k_param.get(), coords.get()});
  FiniteElementDual fd_jac_action(fes, "fd_jac_action");
  fd_jac_action = res_plus;
  fd_jac_action -= res_minus;
  fd_jac_action *= 0.5 / eps;

  jac_times_dir -= fd_jac_action;
  EXPECT_NEAR(jac_times_dir.Norml2(), 0.0, 1.0e-6);

}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  ApplicationManager app(argc, argv);
  return RUN_ALL_TESTS();
}
