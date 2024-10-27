// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <functional>
#include <set>
#include <string>

#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/materials/solid_material.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/terminator.hpp"

struct ParameterizedLinearIsotropicSolid {
  using State = ::serac::Empty;  ///< this material has no internal variables

  template <int dim, typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE auto operator()(State&, const ::serac::tensor<T1, dim, dim>& u_grad, const T2& E_tuple,
                                    const T3& v_tuple) const
  {
    auto       E      = ::serac::get<0>(E_tuple);                 // Young's modulus VALUE
    auto       v      = ::serac::get<0>(v_tuple);                 // Poisson's ratio VALUE
    auto       lambda = E * v / ((1.0 + v) * (1.0 - 2.0 * v));    // Lamé's first parameter
    auto       mu     = E / (2.0 * (1.0 + v));                    // Lamé's second parameter
    const auto I      = ::serac::Identity<dim>();                 // identity matrix
    auto       strain = ::serac::sym(u_grad);                     // small strain tensor
    return lambda * ::serac::tr(strain) * I + 2.0 * mu * strain;  // Cauchy stress
  }
  static constexpr double density{1.0};  ///< mass density, for dynamics problems
};

struct ParameterizedNeoHookeanSolid {
  using State = ::serac::Empty;  // this material has no internal variables

  template <int dim, typename T1, typename T2, typename T3>
  SERAC_HOST_DEVICE auto operator()(State&, const ::serac::tensor<T1, dim, dim>& du_dX, const T2& E_tuple,
                                    const T3& v_tuple) const
  {
    using std::log1p;
    constexpr auto I         = serac::Identity<dim>();
    auto           E         = serac::get<0>(E_tuple);
    auto           v         = serac::get<0>(v_tuple);
    auto           G         = E / (2.0 * (1.0 + v));
    auto           lambda    = (E * v) / ((1.0 + v) * (1.0 - 2.0 * v));
    auto           B_minus_I = du_dX * serac::transpose(du_dX) + serac::transpose(du_dX) + du_dX;
    auto           J_minus_1 = serac::detApIm1(du_dX);
    auto           J         = J_minus_1 + 1;
    return (lambda * log1p(J_minus_1) * I + G * B_minus_I) / J;
  }
  static constexpr double density{1.0};  ///< mass density, for dynamics problems
};

namespace serac {

constexpr int DIM   = 3;
constexpr int ORDER = 1;

const std::string mesh_tag       = "mesh";
const std::string physics_prefix = "solid";

using paramFES = serac::L2<0>;
using uFES     = serac::H1<ORDER, DIM>;
using qoiType  = serac::Functional<double(paramFES, paramFES, uFES)>;

double forwardPass(serac::BasePhysics* solid, qoiType* qoi, mfem::ParMesh* /*meshPtr*/, int nTimeSteps, double timeStep,
                   std::string /*saveName*/)
{
  solid->resetStates();

  double                           qoiValue = 0.0;
  const serac::FiniteElementState& E        = solid->parameter("E");
  const serac::FiniteElementState& v        = solid->parameter("v");
  const serac::FiniteElementState& u        = solid->state("displacement");

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

void adjointPass(serac::BasePhysics* solid, qoiType* qoi, int nTimeSteps, double timeStep,
                 mfem::ParFiniteElementSpace& param_space, double& Ederiv, double& vderiv)
{
  serac::FiniteElementDual         Egrad(param_space);
  serac::FiniteElementDual         vgrad(param_space);
  const serac::FiniteElementState& E = solid->parameter("E");
  const serac::FiniteElementState& v = solid->parameter("v");
  for (int i = nTimeSteps; i > 0; i--) {
    const serac::FiniteElementState& u      = solid->loadCheckpointedState("displacement", i);
    double                           scalar = (i == nTimeSteps) ? 0.5 * timeStep : timeStep;

    auto dQoI_dE = ::serac::get<1>((*qoi)(::serac::DifferentiateWRT<0>{}, solid->time(), E, v, u));
    std::unique_ptr<::mfem::HypreParVector> assembled_Egrad = dQoI_dE.assemble();
    *assembled_Egrad *= scalar;
    Egrad += *assembled_Egrad;

    auto dQoI_dv = ::serac::get<1>((*qoi)(::serac::DifferentiateWRT<1>{}, solid->time(), E, v, u));
    std::unique_ptr<::mfem::HypreParVector> assembled_vgrad = dQoI_dv.assemble();
    *assembled_vgrad *= scalar;
    vgrad += *assembled_vgrad;

    auto dQoI_du = ::serac::get<1>((*qoi)(::serac::DifferentiateWRT<2>{}, solid->time(), E, v, u));
    std::unique_ptr<::mfem::HypreParVector> assembled_ugrad = dQoI_du.assemble();

    serac::FiniteElementDual adjointLoad(u.space());
    adjointLoad = *assembled_ugrad;
    adjointLoad *= scalar;
    solid->setAdjointLoad({{"displacement", adjointLoad}});

    solid->reverseAdjointTimestep();

    serac::FiniteElementDual const& Edual = solid->computeTimestepSensitivity(0);
    Egrad += Edual;
    serac::FiniteElementDual const& vdual = solid->computeTimestepSensitivity(1);
    vgrad += vdual;
  }
  Ederiv = Egrad(0);
  vderiv = vgrad(0);
}

TEST(quasistatic, finiteDifference)
{
  // set up mesh
  ::axom::sidre::DataStore datastore;
  ::serac::StateManager::initialize(datastore, "sidreDataStore");

  mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::HEXAHEDRON);
  assert(mesh.SpaceDimension() == DIM);
  auto             pmesh   = ::std::make_unique<::mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  ::mfem::ParMesh* meshPtr = &::serac::StateManager::setMesh(::std::move(pmesh), mesh_tag);

  // set up solver
  using solidType        = serac::SolidMechanics<ORDER, DIM, ::serac::Parameters<paramFES, paramFES>>;
  auto nonlinear_options = serac::NonlinearSolverOptions{.nonlin_solver  = ::serac::NonlinearSolver::Newton,
                                                         .relative_tol   = 1e-6,
                                                         .absolute_tol   = 1e-8,
                                                         .max_iterations = 10,
                                                         .print_level    = 1};
  auto seracSolid = ::std::make_unique<solidType>(nonlinear_options, serac::solid_mechanics::direct_linear_options,
                                                  ::serac::solid_mechanics::default_quasistatic_options,
                                                  ::serac::GeometricNonlinearities::On, physics_prefix, mesh_tag,
                                                  std::vector<std::string>{"E", "v"});

  using materialType = ParameterizedNeoHookeanSolid;
  materialType material;
  seracSolid->setMaterial(::serac::DependsOn<0, 1>{}, material);

  seracSolid->setDisplacementBCs(
      {3}, [](const mfem::Vector&) { return 0.0; }, 0);
  seracSolid->setDisplacementBCs(
      {4}, [](const mfem::Vector&) { return 0.0; }, 1);
  seracSolid->setDisplacementBCs(
      {1}, [](const mfem::Vector&) { return 0.0; }, 2);

  // serac::Domain loadRegion = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));
  // seracSolid->setTraction([](auto, auto n, auto) {return 1.0*n;}, loadRegion);
  seracSolid->setDisplacementBCs({6}, [](const mfem::Vector&, double time, mfem::Vector& u) { return u[2] = time; });

  double                      E0 = 1.0;
  double                      v0 = 0.3;
  ::serac::FiniteElementState Estate(seracSolid->parameter(seracSolid->parameterNames()[0]));
  ::serac::FiniteElementState vstate(seracSolid->parameter(seracSolid->parameterNames()[1]));
  Estate = E0;
  vstate = v0;
  seracSolid->setParameter(0, Estate);
  seracSolid->setParameter(1, vstate);

  seracSolid->completeSetup();

  // set up QoI
  auto [param_space, _]                        = ::serac::generateParFiniteElementSpace<paramFES>(meshPtr);
  const ::mfem::ParFiniteElementSpace* u_space = &seracSolid->state("displacement").space();

  std::array<const ::mfem::ParFiniteElementSpace*, 3> qoiFES = {param_space.get(), param_space.get(), u_space};
  auto                                                qoi    = std::make_unique<qoiType>(qoiFES);
  qoi->AddDomainIntegral(
      serac::Dimension<DIM>{}, serac::DependsOn<0, 1, 2>{},
      [&](auto time, auto, auto E, auto v, auto u) {
        auto du_dx  = ::serac::get<1>(u);
        auto state  = ::serac::Empty{};
        auto stress = material(state, du_dx, E, v);
        return stress[2][2] * time;
      },
      *meshPtr);

  int    nTimeSteps = 3;
  double timeStep   = 0.8;
  forwardPass(seracSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "f0");

  // ADJOINT GRADIENT
  double Ederiv, vderiv;
  adjointPass(seracSolid.get(), qoi.get(), nTimeSteps, timeStep, *param_space, Ederiv, vderiv);

  seracSolid->resetAdjointStates();

  double Ederiv2, vderiv2;
  adjointPass(seracSolid.get(), qoi.get(), nTimeSteps, timeStep, *param_space, Ederiv2, vderiv2);
  EXPECT_EQ(Ederiv, Ederiv2);
  EXPECT_EQ(vderiv, vderiv2);

  // FINITE DIFFERENCE GRADIENT
  double h = 1e-7;

  Estate = E0 + h;
  seracSolid->setParameter(0, Estate);
  double fpE = forwardPass(seracSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "fpE");

  Estate = E0 - h;
  seracSolid->setParameter(0, Estate);
  double fmE = forwardPass(seracSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "fmE");

  Estate = E0;
  seracSolid->setParameter(0, Estate);

  vstate = v0 + h;
  seracSolid->setParameter(1, vstate);
  double fpv = forwardPass(seracSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "fpv");

  vstate = v0 - h;
  seracSolid->setParameter(1, vstate);
  double fmv = forwardPass(seracSolid.get(), qoi.get(), meshPtr, nTimeSteps, timeStep, "fmv");

  ASSERT_NEAR(Ederiv, (fpE - fmE) / (2. * h), 1e-7);
  ASSERT_NEAR(vderiv, (fpv - fmv) / (2. * h), 1e-7);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  serac::exitGracefully(result);

  return result;
}
