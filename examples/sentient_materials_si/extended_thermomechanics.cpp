// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include "mfem.hpp"
#include <string>
#include <vector>

#include "extended_thermomechanics.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
// clang-format off
//clang-format on
namespace example_etm {

static constexpr int dim = 3;
static constexpr int vdim = dim;
static constexpr int StateMatrixDim =
        (dim == 2) ? 3 :
        (dim == 3) ? 6 :
        -1;
static constexpr int Statedim = StateMatrixDim + 1;
static constexpr int displacement_order = 1;
static constexpr int temperature_order = 1;
using StateSpace = smith::L2<0>;
using ExtendedStateSpace = smith::L2<0, Statedim>;

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

struct GreenSaintVenantThermoelasticWithExtendedStateMaterial {
  double density;
  double E0;
  double nu;
  double C_v;
  double alpha_T;
  double theta_ref;
  double kappa;

  using State = smith::Empty;

  template <int d>
  struct SymmetricStatePacking {
    static_assert(d >= 1, "Invalid matrix dimension.");
    static constexpr int sym_size = d * (d + 1) / 2;

    template <typename T, int sd>
    static smith::tensor<T, sd> pack(const T& scalar, const smith::tensor<T, d, d>& symm)
    {
      static_assert(sd == 1 + sym_size, "Packed state size mismatch.");
      smith::tensor<T, sd> out{};
      out[0] = scalar;
      int k = 1;
      for (int i = 0; i < d; ++i) {
        for (int j = i; j < d; ++j) {
          out[k++] = symm(i, j);
        }
      }
      return out;
    }

    template <typename T, int sd>
    static auto unpack(const smith::tensor<T, sd>& in)
    {
      static_assert(sd == 1 + sym_size, "Packed state size mismatch.");
      T scalar = in[0];
      smith::tensor<T, d, d> symm{};
      int k = 1;
      for (int i = 0; i < d; ++i) {
        for (int j = i; j < d; ++j) {
          symm(i, j) = in[k];
          symm(j, i) = in[k];
          ++k;
        }
      }
      return smith::tuple{scalar, symm};
    }
  };

  template <typename T1, typename T2, typename T3, typename T4, typename T5, int d, int sd>
  auto operator()(double, State&, const smith::tensor<T1, d, d>& grad_u, const smith::tensor<T2, d, d>& grad_v, T3 theta,
                  const smith::tensor<T4, d>& grad_theta, const smith::tensor<T5, sd>& alpha_old) const
  {
    // Calculate Alpha new using the old variables to be used

    auto [w_old, F_old] = SymmetricStatePacking<d>::template unpack<T5, sd>(alpha_old);

      // Extracting 0 index scalar value and calculating rate of change
auto w_new = w_old;
auto F_new = F_old;

      // Concatenating results


    auto E = E0;
    const auto K = E / (3.0 * (1.0 - 2.0 * nu));
    const auto G = 0.5 * E / (1.0 + nu);
    const auto Eg = greenStrain<T1, d>(grad_u);
    const auto trEg = tr(Eg);

    static constexpr auto I = smith::Identity<d>();
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - d * alpha_T * (theta - theta_ref)) * I;
    auto F = grad_u + I;
    const auto Piola = dot(F, S);

    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto s0 = -d * K * alpha_T * (theta + 273.1) * tr(greenStrainRate);
    const auto q0 = -kappa * grad_theta;

    auto alpha_new = SymmetricStatePacking<d>::template pack<T5, sd>(w_new, F_new);
    return smith::tuple{Piola, C_v, s0, q0, alpha_new};
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

int runCoupledWithState(const std::shared_ptr<smith::Mesh>& mesh, double dt, double T, double alpha_T)
{
  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  double initial_temperature = 0.0;

  using MaterialModel = GreenSaintVenantThermoelasticWithExtendedStateMaterial;
  MaterialModel material{rho, E0, nu, specific_heat, alpha_T, 1.0, kappa};

  auto solver = smith::buildNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh);

  auto system = smith::buildExtendedThermoMechanicsSystem<dim, displacement_order, temperature_order, ExtendedStateSpace>(
      mesh, solver, "");

  system.setMaterial(material, mesh->entireBodyName());

  system.disp_bc->setVectorBCs<dim>(mesh->domain("left"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    // Keep the loading modest so the first Newton solves are well-conditioned.
    bc[0] = 0.01 * t;
    return bc;
  });
  system.disp_bc->template setFixedVectorBCs<dim, vdim>(mesh->domain("right"));

  system.temperature_bc->setFixedScalarBCs<dim>(mesh->domain("left"));
  // system.temperature_bc->setFixedScalarBCs<dim>(mesh->domain("right"));

  system.addThermalHeatSource(mesh->entireBodyName(),
                              [](double, auto, auto, auto, auto, auto, auto, auto, auto...) { return 0.0; });

  // Initialize displacement fields (avoid solver starting from uninitialized/NaN values).
  auto& disp_pred = const_cast<smith::FiniteElementState&>(*system.field_store->getField("displacement_predicted").get());
  disp_pred.setFromFieldFunction([](smith::tensor<double, dim>) { return smith::tensor<double, dim>{}; });
  const_cast<smith::FiniteElementState&>(*system.field_store->getField("displacement").get()) = disp_pred;

  // Initialize temperature fields
  auto& temp_pred =
      const_cast<smith::FiniteElementState&>(*system.field_store->getField("temperature_predicted").get());
  temp_pred.setFromFieldFunction([=](smith::tensor<double, dim>) { return initial_temperature; });
  const_cast<smith::FiniteElementState&>(*system.field_store->getField("temperature").get()) = temp_pred;

  // Initialize extended state fields (w, Feinv in Mandel-like packing)
  auto& state_pred = const_cast<smith::FiniteElementState&>(*system.field_store->getField("state_predicted").get());
  state_pred.setFromFieldFunction([](smith::tensor<double, dim>) {
    smith::tensor<double, Statedim> alpha{};
    auto [w_0, F_0] = MaterialModel::SymmetricStatePacking<dim>::unpack(alpha);
    w_0 = 0.0;
    setIdentity(F_0);
    alpha = MaterialModel::SymmetricStatePacking<dim>::pack<double, Statedim>(w_0, F_0);
    return alpha;
  });
  const_cast<smith::FiniteElementState&>(*system.field_store->getField("state").get()) = state_pred;

  std::string pv_dir = "paraview_extended_thermomechanics";
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
      std::cout
          << "Usage: extended_thermomechanics [--nx=<int>] [--ny=<int>] [--nz=<int>] [--dt=<real>] [--T=<real>] "
             "[--alpha=<real>]\n";
      std::cout << "Defaults: nx=12 ny=2 nz=2 dt=0.01 T=0.1 alpha=0.0\n";
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
  double alpha_T = 0.0;

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
    parse_double("--alpha=", alpha_T);
  }

  auto mfem_shape = mfem::Element::QUADRILATERAL;
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(num_elements_x, num_elements_y, num_elements_z, mfem_shape, length, width, width),
      "mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<example_etm::dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<example_etm::dim>(5));

  return example_etm::runCoupledWithState(mesh, dt, T, alpha_T);
}
