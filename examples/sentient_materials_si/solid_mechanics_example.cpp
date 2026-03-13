
// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mfem/fem/dfem/doperator.hpp>
#include <mfem/linalg/tensor.hpp>
#include <string>
#include <vector>

#include "quasistatic_solid.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/solid_dynamics_system.hpp"
// clang-format off
//clang-format on
namespace example_m {

static constexpr int dim = 3;
static constexpr int vdim = dim;
static constexpr int displacement_order = 1;

using smith::cross;
using smith::dev;
using smith::dot;
using smith::get;
using smith::inner;
using smith::norm;
using smith::tr;
using smith::transpose;

template <typename T, int d>
auto greenStrain(const smith::tensor<T, d, d>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

template<typename T, int d>
void setIdentity(smith::tensor<T, d, d> & F){
  for (size_t i = 0; i < d; i++){
    for (size_t j = 0; j < d; j++){
      F(i, j) = static_cast<T>(i == j);
    }
  }
}

struct GreenSaintVenantElasticMaterial {
  double density;
  double E0;
  double nu;
//   double C_v;
  double alpha_T;
  double Temperature_ref;
//   double kappa;
  double Temperature;

  using State = smith::Empty;


  template <typename T1, typename T2, int d>
  auto operator()(double, State&, const smith::tensor<T1, d, d>& grad_u, const smith::tensor<T2, d, d>& /*grad_v*/) const
  {

      // Concatenating results


    auto E = E0;
    const auto K = E / (3.0 * (1.0 - 2.0 * nu));
    const auto G = 0.5 * E / (1.0 + nu);
    const auto Eg = greenStrain<T1, d>(grad_u);
    const auto trEg = tr(Eg);

    static constexpr auto I = smith::Identity<d>();
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - d * alpha_T * (Temperature - Temperature_ref)) * I;
;
    auto F = grad_u + I;
    const auto Piola = dot(F, S);

    // auto greenStrainRate =
    //     0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    return smith::tuple{Piola};
  }

  static constexpr int numParameters() { return 1; }
};

smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack,
                                          .preconditioner = smith::Preconditioner::HypreJacobi,
                                          // For this coupled, non-symmetric block system, very tight Krylov tolerances
                                          // can be prohibitively expensive; rely on the nonlinear iterations instead.
                                          .relative_tol = 1e-4,
                                          .absolute_tol = 0.0,
                                          .max_iterations = 300,
                                          .print_level = 0};

smith::NonlinearSolverOptions nonlinear_opts{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                             .relative_tol = 1.0e-8,
                                             .absolute_tol = 1.0e-8,
                                             .max_iterations = 200,
                                             .max_line_search_iterations = 30,
                                             .print_level = 1};

int runCoupledWithState(const std::shared_ptr<smith::Mesh>& mesh, double dt, double T)
{
  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double Temperature = 0.0;

  using MaterialModel = GreenSaintVenantElasticMaterial;
  // Disable thermal eigenstrain in this solid-only example to avoid large initial stresses
  // that can prevent Newton from converging from the zero initial guess.
  MaterialModel material{rho, E0, nu, 0.0, 0.0, Temperature};

  auto solver = smith::buildDifferentiableNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh);
  auto system = smith::buildQuasiStaticSolidMechanicsSystem<dim, displacement_order>(mesh, solver);


  system.setMaterial(material, mesh->entireBodyName());

  system.disp_bc->setVectorBCs<dim>(mesh->domain("left"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    // Keep the loading modest so the first Newton solves are well-conditioned.
    bc[0] = 1.0 * t;
    return bc;
  });
  system.disp_bc->setFixedVectorBCs<dim, vdim>(mesh->domain("right"));


  // Initialize displacement fields (avoid solver starting from uninitialized/NaN values).
  auto& disp_pred = const_cast<smith::FiniteElementState&>(*system.field_store->getField("displacement_predicted").get());
  disp_pred.setFromFieldFunction([](smith::tensor<double, dim>) { return smith::tensor<double, dim>{}; });
  const_cast<smith::FiniteElementState&>(*system.field_store->getField("displacement").get()) = disp_pred;


  std::string pv_dir = "paraview_quasistaticsolid";
  auto pv_writer = smith::createParaviewWriter(*mesh, system.getStateFields(), pv_dir);

  double time = 0.0;
  int cycle = 0;

  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();
  auto params = system.getParameterFields();

  pv_writer.write(cycle, time, states);

  size_t step = 0;

  while (time < T) {
    smith::TimeInfo t_info(time, dt, step);
    auto [new_states, reactions] = system.advancer->advanceState(t_info, shape_disp, states, params);
    states = std::move(new_states);

    // std::cout << "step " << step << " max reaction (disp)=" << reactions[0].get()->Normlinf()
    //           << " (temp)=" << reactions[1].get()->Normlinf() << " (state)=" << reactions[2].get()->Normlinf() << "\n";

    time += dt;
    cycle++;
    pv_writer.write(cycle, time, states);

    step++;
  }
  std::cout << "Wrote ParaView output to '" << pv_dir << "'\n";
  return 0;
}

int test_example(){

  using namespace mfem::future;
  using mfem::future::tensor;
  using mfem::future::make_tensor;

  std::shared_ptr<DifferentiableOperator> a;
  return 0;
}

}  // namespace example_etm

int main(int argc, char** argv)
{
  smith::ApplicationManager applicationManager(argc, argv);

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: extended_thermomechanics [--nx=<int>] [--ny=<int>] [--nz=<int>] [--dt=<real>] [--T=<real>]\n";
      std::cout << "Defaults: nx=12 ny=2 nz=2 dt=0.01 T=0.1\n";
      return 0;
    }
  }

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid");

  double length = 1.0;
  double width = 0.04;
  int num_elements_x = 60;
  int num_elements_y = 10;
  int num_elements_z = 10;
  double dt = 0.01;
  double T = 1.0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto parse_int = [&](const char* prefix, int& value) {
      const std::string p(prefix);
      if (arg.rfind(p, 0) == 0) value = std::stoi(arg.substr(p.size()));
    };
    auto parse_double = [&](const char* prefix, double& value) {
      const std::string p(prefix);
      if (arg.rfind(p, 0) == 0) value = std::stod(arg.substr(p.size()));
    };
    parse_int("--nx=", num_elements_x);
    parse_int("--ny=", num_elements_y);
    parse_int("--nz=", num_elements_z);
    parse_double("--dt=", dt);
    parse_double("--T=", T);
  }

  auto mfem_shape = mfem::Element::QUADRILATERAL;
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(num_elements_x, num_elements_y, num_elements_z, mfem_shape, length, width, width),
      "mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<example_m::dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<example_m::dim>(5));

  return example_m::runCoupledWithState(mesh, dt, T);
}
