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

#include "extended_thermomechanics.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "smith/differentiable_numerics/differentiable_solver.hpp"
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

  template <typename T, int sd>
  static T StateScalar(const smith::tensor<T, sd>& v)
  {
    return v[0];
  }

  template<typename T, int d, int sd>
  static smith::tensor<T, d, d> StateMatrix(const smith::tensor<T, sd>& v)
  {
    smith::tensor<T, d, d> vmat;
    switch (d) {
      case 2: {
        vmat(0, 0) = v[1];
        vmat(0, 1) = v[2];
        vmat(1, 1) = v[3];

        // symmetries
        vmat(1, 0) = v[2];

        break;
      }
      case 3:{
        vmat(0, 0) = v[1];
        vmat(0, 1) = v[2];
        vmat(0, 2) = v[3];
        vmat(1, 1) = v[4];
        vmat(1, 2) = v[5];
        vmat(2, 2) = v[6];

        // symmetries
        vmat(1, 0) = v[2];
        vmat(2, 0) = v[3];
        vmat(2, 1) = v[5];
        break;
      } default: {

      }
    };

    return vmat;
  }

  template<typename T, int d, int sd>
  static smith::tensor<T, sd> StateMuxer(const T& v0, const smith::tensor<T, d, d>& vmat)
  {
    smith::tensor<T, sd> vmux;
    vmux[0] = v0;
    switch (d) {
      case 2: {
        vmux[1] = vmat(0, 0);
        vmux[2] = vmat(0, 1);
        vmux[3] = vmat(1, 1);
        // no need to transfer symmetries
        break;
      } case 3: {
        vmux[1] = vmat(0, 0);
        vmux[2] = vmat(0, 1);
        vmux[3] = vmat(0, 2);
        vmux[4] = vmat(1, 1);
        vmux[5] = vmat(1, 2);
        vmux[6] = vmat(2, 2);
        break;
      } default: {

      }
    }
    return vmux;

  }
  template<typename T, int d, int sd> 
  static auto StateDemuxer(const smith::tensor<T, sd> & v) {
    auto w = StateScalar<T, sd>(v);
    auto F = StateMatrix<T, d, sd>(v);
    return smith::tuple{w, F};
  }

  template <typename T1, typename T2, typename T3, typename T4, typename T5, int d, int sd>
  auto operator()(double, State&, const smith::tensor<T1, d, d>& grad_u, const smith::tensor<T2, d, d>& grad_v, T3 theta,
                  const smith::tensor<T4, d>& grad_theta, const smith::tensor<T5, sd>& alpha_old) const
  {
    // Calculate Alpha new using the old variables to be used

    auto [w_old, F_old] = StateDemuxer<T5, d, sd>(alpha_old);

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
    const auto s0 = -d * K * alpha_T * (theta + 273.1) * tr(greenStrainRate) + 0.0 * E;
    const auto q0 = -kappa * grad_theta;

      auto alpha_new = StateMuxer<T5, d, sd>(w_new, F_new);
    return smith::tuple{Piola, C_v, s0, q0, alpha_new};
  }

  static constexpr int numParameters() { return 1; }
};

smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack,
                                          .relative_tol = 1e-8,
                                          .absolute_tol = 1e-8,
                                          .max_iterations = 200,
                                          .print_level = 0};

smith::NonlinearSolverOptions nonlinear_opts{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                             .relative_tol = 1.0e-10,
                                             .absolute_tol = 1.0e-10,
                                             .max_iterations = 200,
                                             .max_line_search_iterations = 30,
                                             .print_level = 1};

int runCoupledWithState(const std::shared_ptr<smith::Mesh>& mesh)
{
  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.0;
  double kappa = 0.1;
  double initial_temperature = 0.0;

  using MaterialModel = GreenSaintVenantThermoelasticWithExtendedStateMaterial;
  MaterialModel material{rho, E0, nu, specific_heat, 0.0, 1.0, kappa};

  auto solver = smith::buildDifferentiableNonlinearBlockSolver(nonlinear_opts, linear_options, *mesh);

  auto system = smith::buildExtendedThermoMechanicsSystem<dim, displacement_order, temperature_order, ExtendedStateSpace>(
      mesh, solver, "");

  system.setMaterial(material, mesh->entireBodyName());

  system.disp_bc->setVectorBCs<dim>(mesh->domain("left"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    bc[0] = 1.0 * t;
    return bc;
  });
  system.disp_bc->setFixedVectorBCs<dim, vdim>(mesh->domain("right"));

  system.temperature_bc->setFixedScalarBCs<dim>(mesh->domain("left"));
  // system.temperature_bc->setFixedScalarBCs<dim>(mesh->domain("right"));

  system.addThermalHeatSource(mesh->entireBodyName(),
                              [](double, auto, auto, auto, auto, auto, auto, auto, auto...) { return 25.0; });

  // Initialize temperature fields
  auto& temp_pred =
      const_cast<smith::FiniteElementState&>(*system.field_store->getField("temperature_predicted").get());
  temp_pred.setFromFieldFunction([=](smith::tensor<double, dim>) { return initial_temperature; });
  const_cast<smith::FiniteElementState&>(*system.field_store->getField("temperature").get()) = temp_pred;

  // Initialize extended state fields (w, Feinv in Mandel-like packing)
  auto& state_pred = const_cast<smith::FiniteElementState&>(*system.field_store->getField("state_predicted").get());
  state_pred.setFromFieldFunction([](smith::tensor<double, dim>) {
    smith::tensor<double, Statedim> alpha{};
    auto [w_0, F_0] = MaterialModel::StateDemuxer<double, dim>(alpha);
    w_0 = 0.0;
    setIdentity(F_0);
    alpha = MaterialModel::StateMuxer<double, dim, Statedim>(w_0, F_0);
    return alpha;
  });
  const_cast<smith::FiniteElementState&>(*system.field_store->getField("state").get()) = state_pred;

  std::string pv_dir = "paraview_extended_thermomechanics";
  auto pv_writer = smith::createParaviewWriter(*mesh, system.getStateFields(), pv_dir);

  double dt = 0.01;
  double time = 0.0;
  double T = 1.0;
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

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid");

  double length = 1.0;
  double width = 0.04;
  int num_elements_x = 120;
  int num_elements_y = 20;
  int num_elements_z = 20;
  auto mfem_shape = mfem::Element::QUADRILATERAL;
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(num_elements_x, num_elements_y, num_elements_z, mfem_shape, length, width, width),
      "mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<example_etm::dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<example_etm::dim>(5));

  return example_etm::runCoupledWithState(mesh);
}
