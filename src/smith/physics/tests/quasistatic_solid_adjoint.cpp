// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cassert>
#include <string>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/accelerator.hpp"
#include "smith/numerics/functional/differentiate_wrt.hpp"
#include "smith/numerics/functional/finite_element.hpp"  // for H1
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/geometry.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/functional/tuple.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/base_physics.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

struct ParameterizedLinearIsotropicSolid {
  using State = ::smith::Empty;  ///< this material has no internal variables

  template <int dim, typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto operator()(State&, const ::smith::tensor<T1, dim, dim>& u_grad, const T2& E_tuple,
                                    const T3& v_tuple) const
  {
    auto E = ::smith::get<0>(E_tuple);                            // Young's modulus VALUE
    auto v = ::smith::get<0>(v_tuple);                            // Poisson's ratio VALUE
    auto lambda = E * v / ((1.0 + v) * (1.0 - 2.0 * v));          // Lamé's first parameter
    auto mu = E / (2.0 * (1.0 + v));                              // Lamé's second parameter
    const auto I = ::smith::Identity<dim>();                      // identity matrix
    auto strain = ::smith::sym(u_grad);                           // small strain tensor
    return lambda * ::smith::tr(strain) * I + 2.0 * mu * strain;  // Cauchy stress
  }
  static constexpr double density{1.0};  ///< mass density, for dynamics problems
};

struct ParameterizedNeoHookeanSolid {
  using State = ::smith::Empty;  // this material has no internal variables

  template <int dim, typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto operator()(State&, const ::smith::tensor<T1, dim, dim>& du_dX, const T2& E_tuple,
                                    const T3& v_tuple) const
  {
    using std::log1p;
    constexpr auto I = smith::Identity<dim>();
    auto E = smith::get<0>(E_tuple);
    auto v = smith::get<0>(v_tuple);
    auto G = E / (2.0 * (1.0 + v));
    auto lambda = (E * v) / ((1.0 + v) * (1.0 - 2.0 * v));
    auto B_minus_I = du_dX * smith::transpose(du_dX) + smith::transpose(du_dX) + du_dX;
    auto J_minus_1 = smith::detApIm1(du_dX);
    auto J = J_minus_1 + 1;
    return (lambda * log1p(J_minus_1) * I + G * B_minus_I) / J;
  }
  static constexpr double density{1.0};  ///< mass density, for dynamics problems
};

namespace smith {

constexpr int DIM = 3;
constexpr int ORDER = 1;

const std::string mesh_tag = "mesh";
const std::string physics_prefix = "solid";

using paramFES = smith::L2<0>;
using uFES = smith::H1<ORDER, DIM>;
using qoiType = smith::Functional<double(paramFES, paramFES, uFES)>;

double forwardPass(smith::BasePhysics* solid, qoiType* qoi, mfem::ParMesh* /*meshPtr*/, int nTimeSteps, double timeStep,
                   std::string /*saveName*/)
{
  solid->resetStates();

  double qoiValue = 0.0;
  const smith::FiniteElementState& E = solid->parameter("E");
  const smith::FiniteElementState& v = solid->parameter("v");
  const smith::FiniteElementState& u = solid->state("displacement");

  double prev = (*qoi)(solid->time(), E, v, u);
  for (int i = 0; i < nTimeSteps; i++) {
    // solve
    solid->advanceTimestep(timeStep);

    // accumulate
    double curr = (*qoi)(solid->time(), E, v, u);
    qoiValue += timeStep * 0.5 * (prev + curr);  // trapezoid
    prev = curr;
  }
  return qoiValue;
}

void adjointPass(smith::BasePhysics* solid, qoiType* qoi, int nTimeSteps, double timeStep,
                 mfem::ParFiniteElementSpace& param_space, double& Ederiv, double& vderiv)
{
  smith::FiniteElementDual Egrad(param_space);
  smith::FiniteElementDual vgrad(param_space);
  const smith::FiniteElementState& E = solid->parameter("E");
  const smith::FiniteElementState& v = solid->parameter("v");
  for (int i = nTimeSteps; i > 0; i--) {
    const smith::FiniteElementState& u = solid->loadCheckpointedState("displacement", i);
    double scalar = (i == nTimeSteps) ? 0.5 * timeStep : timeStep;

    auto dQoI_dE = ::smith::get<1>((*qoi)(::smith::DifferentiateWRT<0>{}, solid->time(), E, v, u));
    std::unique_ptr<::mfem::HypreParVector> assembled_Egrad = dQoI_dE.assemble();
    *assembled_Egrad *= scalar;
    Egrad += *assembled_Egrad;

    auto dQoI_dv = ::smith::get<1>((*qoi)(::smith::DifferentiateWRT<1>{}, solid->time(), E, v, u));
    std::unique_ptr<::mfem::HypreParVector> assembled_vgrad = dQoI_dv.assemble();
    *assembled_vgrad *= scalar;
    vgrad += *assembled_vgrad;

    auto dQoI_du = ::smith::get<1>((*qoi)(::smith::DifferentiateWRT<2>{}, solid->time(), E, v, u));
    std::unique_ptr<::mfem::HypreParVector> assembled_ugrad = dQoI_du.assemble();

    smith::FiniteElementDual adjointLoad(u.space());
    adjointLoad = *assembled_ugrad;
    adjointLoad *= scalar;
    solid->setAdjointLoad({{"displacement", adjointLoad}});

    solid->reverseAdjointTimestep();

    smith::FiniteElementDual const& Edual = solid->computeTimestepSensitivity(0);
    Egrad += Edual;
    smith::FiniteElementDual const& vdual = solid->computeTimestepSensitivity(1);
    vgrad += vdual;
  }
  Ederiv = Egrad(0);
  vderiv = vgrad(0);
}

TEST(quasistatic, finiteDifference)
{
  // set up mesh
  ::axom::sidre::DataStore datastore;
  ::smith::StateManager::initialize(datastore, "sidreDataStore");

  auto mesh =
      std::make_shared<smith::Mesh>(mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::HEXAHEDRON), mesh_tag, 0, 0);
  auto meshPtr = &mesh->mfemParMesh();
  assert(meshPtr->SpaceDimension() == DIM);

  std::string xmax_face_domain_name = "xmax_face";
  std::string ymax_face_domain_name = "ymax_face";
  std::string zmin_face_domain_name = "zmin_face";
  std::string zmax_face_domain_name = "zmax_face";
  mesh->addDomainOfBoundaryElements(xmax_face_domain_name, by_attr<DIM>(3));
  mesh->addDomainOfBoundaryElements(ymax_face_domain_name, by_attr<DIM>(4));
  mesh->addDomainOfBoundaryElements(zmin_face_domain_name, by_attr<DIM>(1));
  mesh->addDomainOfBoundaryElements(zmax_face_domain_name, by_attr<DIM>(6));

  // set up solver
  using solidType = smith::SolidMechanics<ORDER, DIM, ::smith::Parameters<paramFES, paramFES>>;
  auto nonlinear_options = smith::NonlinearSolverOptions{.nonlin_solver = ::smith::NonlinearSolver::Newton,
                                                         .relative_tol = 1e-6,
                                                         .absolute_tol = 1e-8,
                                                         .max_iterations = 10,
                                                         .print_level = 1};
  auto smithSolid = ::std::make_unique<solidType>(nonlinear_options, smith::solid_mechanics::direct_linear_options,
                                                  ::smith::solid_mechanics::default_quasistatic_options, physics_prefix,
                                                  mesh, std::vector<std::string>{"E", "v"});

  using materialType = ParameterizedNeoHookeanSolid;
  materialType material;

  smithSolid->setMaterial(::smith::DependsOn<0, 1>{}, material, mesh->entireBody());

  smithSolid->setFixedBCs(mesh->domain(xmax_face_domain_name), Component::X);
  smithSolid->setFixedBCs(mesh->domain(ymax_face_domain_name), Component::Y);
  smithSolid->setFixedBCs(mesh->domain(zmin_face_domain_name), Component::Z);

  smithSolid->setDisplacementBCs([](vec3, double time) { return vec3{{0.0, 0.0, time}}; },
                                 mesh->domain(zmax_face_domain_name), Component::Z);

  double E0 = 1.0;
  double v0 = 0.3;
  ::smith::FiniteElementState Estate(smithSolid->parameter(smithSolid->parameterNames()[0]));
  ::smith::FiniteElementState vstate(smithSolid->parameter(smithSolid->parameterNames()[1]));
  Estate = E0;
  vstate = v0;
  smithSolid->setParameter(0, Estate);
  smithSolid->setParameter(1, vstate);

  smithSolid->completeSetup();

  // set up QoI
  auto [param_space, _] = ::smith::generateParFiniteElementSpace<paramFES>(meshPtr);
  const ::mfem::ParFiniteElementSpace* u_space = &smithSolid->state("displacement").space();

  std::array<const ::mfem::ParFiniteElementSpace*, 3> qoiFES = {param_space.get(), param_space.get(), u_space};
  auto qoi = std::make_unique<qoiType>(qoiFES);
  qoi->AddDomainIntegral(
      smith::Dimension<DIM>{}, smith::DependsOn<0, 1, 2>{},
      [&](auto time, auto, auto E, auto v, auto u) {
        auto du_dx = ::smith::get<1>(u);
        auto state = ::smith::Empty{};
        auto stress = material(state, du_dx, E, v);
        return stress[2][2] * time;
      },
      mesh->entireBody());

  int nTimeSteps = 3;
  double timeStep = 0.8;
  forwardPass(smithSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "f0");

  // ADJOINT GRADIENT
  double Ederiv, vderiv;
  adjointPass(smithSolid.get(), qoi.get(), nTimeSteps, timeStep, *param_space, Ederiv, vderiv);

  smithSolid->resetAdjointStates();

  double Ederiv2, vderiv2;
  adjointPass(smithSolid.get(), qoi.get(), nTimeSteps, timeStep, *param_space, Ederiv2, vderiv2);
  EXPECT_EQ(Ederiv, Ederiv2);
  EXPECT_EQ(vderiv, vderiv2);

  // FINITE DIFFERENCE GRADIENT
  double h = 1e-7;

  Estate = E0 + h;
  smithSolid->setParameter(0, Estate);
  double fpE = forwardPass(smithSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "fpE");

  Estate = E0 - h;
  smithSolid->setParameter(0, Estate);
  double fmE = forwardPass(smithSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "fmE");

  Estate = E0;
  smithSolid->setParameter(0, Estate);

  vstate = v0 + h;
  smithSolid->setParameter(1, vstate);
  double fpv = forwardPass(smithSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "fpv");

  vstate = v0 - h;
  smithSolid->setParameter(1, vstate);
  double fmv = forwardPass(smithSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "fmv");

  ASSERT_NEAR(Ederiv, (fpE - fmE) / (2. * h), 1e-7);
  ASSERT_NEAR(vderiv, (fpv - fmv) / (2. * h), 1e-7);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
