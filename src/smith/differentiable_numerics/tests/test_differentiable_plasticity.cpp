// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include "gtest/gtest.h"

#include "smith/smith_config.hpp"
#include "smith/smith.hpp"

namespace smith {

LinearSolverOptions primal_lin_opts{.linear_solver = LinearSolver::CG,
                                    .preconditioner = Preconditioner::HypreAMG,
                                    .relative_tol = 1e-15,
                                    .absolute_tol = 1e-15,
                                    .max_iterations = 200,
                                    .print_level = 0};

NonlinearSolverOptions primal_nonlin_opts{.nonlin_solver = NonlinearSolver::Newton,
                                          .relative_tol = 1e-9,
                                          .absolute_tol = 1e-10,
                                          .max_iterations = 25,
                                          .print_level = 2};

LinearSolverOptions state_lin_opts{.linear_solver = LinearSolver::CG,
                                   .preconditioner = Preconditioner::HypreJacobi,
                                   .relative_tol = 1e-10,
                                   .absolute_tol = 1e-10,
                                   .max_iterations = 200,
                                   .print_level = 0};

NonlinearSolverOptions state_nonlin_opts{.nonlin_solver = NonlinearSolver::Newton,
                                         .relative_tol = 1e-7,
                                         .absolute_tol = 1e-8,
                                         .max_iterations = 25,
                                         .print_level = 2};

/// @brief Differentiable J2 material with nonlinear isotropic hardening and linear kinematic hardening
template <int dim, typename HardeningType>
class DifferentiableJ2SmallStrain {
 public:
  DifferentiableJ2SmallStrain(HardeningType hardeningModel, double youngsModulus, double poissonsRatio,
                              double hardeningModulus, double density)
      : hardening(hardeningModel), E(youngsModulus), nu(poissonsRatio), Hk(hardeningModulus), rho(density)
  {
  }

  /** @brief calculate the first Piola stress, given the displacement gradient and previous staggered solve material
   * state */
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto firstPiolaStress(double /* dt */, const tensor<T1, dim, dim>& du_dX,
                                          const tensor<T2, dim, dim>& Fp, const T3& /* epsilon_p */) const
  {
    constexpr auto I = Identity<dim>();
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);

    auto el_strain = sym(du_dX) - Fp;
    auto p = K * tr(el_strain);
    auto s = 2.0 * G * dev(el_strain);

    return s + p * I;
  }

  /** @brief calculate the plastic deformation gradient */
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto plasticDeformGrad(double dt, const tensor<T1, dim, dim>& Fp_old, const T2& epsilon_dot,
                                           const tensor<T3, dim, dim>& du_dX) const
  {
    using std::sqrt;
    const double G = 0.5 * E / (1.0 + nu);

    auto el_strain = sym(du_dX) - Fp_old;
    auto s = 2.0 * G * dev(el_strain);
    auto sigma_b = 2.0 / 3.0 * Hk * Fp_old;
    auto eta = s - sigma_b;
    auto q = sqrt(1.5) * norm(eta);

    auto Np = 1.5 * eta / q;

    return Fp_old + epsilon_dot * dt * Np;
  }

  /** @brief calculate the plastic strain */
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto plasticStrain(double dt, const T1& epsilon_p, const T2& epsilon_dot,
                                       const tensor<T3, dim, dim>& Fp_old, const tensor<T4, dim, dim>& du_dX) const
  {
    using std::sqrt;
    const double G = 0.5 * E / (1.0 + nu);

    auto el_strain = sym(du_dX) - Fp_old;
    auto s = 2.0 * G * dev(el_strain);
    auto sigma_b = 2.0 / 3.0 * Hk * Fp_old;
    auto eta = s - sigma_b;
    auto q = sqrt(1.5) * norm(eta);

    auto epsilon_dot_predict = q / dt - (3.0 * G + Hk) * epsilon_dot - this->hardening(epsilon_p, epsilon_dot) / dt;
    auto non_negativity = epsilon_dot_predict >= 0.0;
    return epsilon_dot_predict * non_negativity;
  }

 private:
  HardeningType hardening;  ///< Flow stress hardening model
  double E;                 ///< Young's modulus
  double nu;                ///< Poisson's ratio
  double Hk;                ///< Kinematic hardening modulus
  double rho;               ///< Mass density
};

TEST(DifferentiablePlasticity, J2SmallStrainLinearHardening)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 0;
  int parallel_refinement = 0;

  static constexpr int dim = 3;
  static constexpr int order = 2;

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "plasticity_small_strain");

  std::string filename = SMITH_REPO_DIR "/data/meshes/beam-hex.mesh";
  const std::string meshtag = "mesh";
  auto mesh = std::make_shared<Mesh>(buildMeshFromFile(filename), meshtag, serial_refinement, parallel_refinement);

  auto staggered_coupled_solver = std::make_shared<CoupledSystemSolver>(50);
  auto primal_block_solver = buildNonlinearBlockSolver(primal_nonlin_opts, primal_lin_opts, *mesh);
  auto strain_block_solver = buildNonlinearBlockSolver(state_nonlin_opts, state_lin_opts, *mesh);
  auto defgrad_block_solver = buildNonlinearBlockSolver(state_nonlin_opts, state_lin_opts, *mesh);

  staggered_coupled_solver->addSubsystemSolver({0}, primal_block_solver);
  staggered_coupled_solver->addSubsystemSolver({2}, strain_block_solver);
  staggered_coupled_solver->addSubsystemSolver({1}, defgrad_block_solver);

  auto plastic_mechanics_system = buildPlasticMechanicsSystem<dim, order>(mesh, staggered_coupled_solver);

  using Hardening = solid_mechanics::LinearHardening;
  Hardening hardening{.sigma_y = 40.0, .Hi = 50.0, .eta = 0.0};
  DifferentiableJ2SmallStrain<dim, Hardening> mat(hardening, 1e+4, 0.25, 5.0, 1.0);

  plastic_mechanics_system.setMaterial(mesh->entireBodyName(), mat);
  plastic_mechanics_system.setPlasticity(mesh->entireBodyName(), mat);

  // prescribe zero displacement at the supported end of the beam,
  mesh->addDomainOfBoundaryElements("support", by_attr<dim>(1));
  plastic_mechanics_system.disp_bc->setFixedVectorBCs<dim>(mesh->domain("support"));

  // apply a displacement along z to the the tip of the beam
  mesh->addDomainOfBoundaryElements("tip", by_attr<dim>(2));
  auto translated_in_z = [](double t, tensor<double, dim>) {
    tensor<double, dim> u{};
    u[2] = t * (t - 1);
    return u;
  };
  plastic_mechanics_system.disp_bc->setVectorBCs<dim>(mesh->domain("tip"), {2}, translated_in_z);

  double dt = 0.1;
  double time = 0.0;
  auto shape_disp = plastic_mechanics_system.field_store->getShapeDisp();
  auto states = plastic_mechanics_system.getStateFields();
  auto params = plastic_mechanics_system.getParameterFields();

  auto pv_writer = createParaviewWriter(*mesh, states, "J2_small_strain");
  pv_writer.write(0, 0.0, states);

  std::vector<ReactionState> reactions;
  for (size_t step = 0; step < 10; ++step) {
    std::tie(states, reactions) =
        plastic_mechanics_system.advancer->advanceState(TimeInfo(time, dt, step), shape_disp, states, params);
    time += dt;
    pv_writer.write(step + 1, time, states);
  }

  auto displacement = plastic_mechanics_system.field_store->getField(plastic_mechanics_system.prefix("displacement"));
  double final_disp_l2 = norm(*displacement.get(), 2.0);
  EXPECT_GT(final_disp_l2, 0.0);
}

/// @brief Parameterized differentiable J2 material with nonlinear isotropic hardening and linear kinematic hardening
template <int dim, typename HardeningType>
class ParameterizedDifferentiableJ2SmallStrain {
 public:
  ParameterizedDifferentiableJ2SmallStrain(HardeningType hardeningModel, double baseYoungsModulus, double poissonsRatio,
                                           double baseHardeningModulus, double density)
      : hardening(hardeningModel), E0(baseYoungsModulus), nu(poissonsRatio), Hk0(baseHardeningModulus), rho(density)
  {
  }

  /** @brief calculate the first Piola stress, given the displacement gradient and previous staggered solve material
   * state */
  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  SMITH_HOST_DEVICE auto firstPiolaStress(double /* dt */, const tensor<T1, dim, dim>& du_dX,
                                          const tensor<T2, dim, dim>& Fp, const T3& /* epsilon_p */, const T4& deltaE,
                                          const T5& /* deltaHk */) const
  {
    constexpr auto I = Identity<dim>();
    auto K = (E0 + get<VALUE>(deltaE)) / (3.0 * (1.0 - 2.0 * nu));
    auto G = 0.5 * (E0 + get<VALUE>(deltaE)) / (1.0 + nu);

    auto el_strain = sym(du_dX) - Fp;
    auto p = K * tr(el_strain);
    auto s = 2.0 * G * dev(el_strain);

    return s + p * I;
  }

  /** @brief calculate the plastic deformation gradient */
  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  SMITH_HOST_DEVICE auto plasticDeformGrad(double dt, const tensor<T1, dim, dim>& Fp_old, const T2& epsilon_dot,
                                           const tensor<T3, dim, dim>& du_dX, const T4& deltaE, const T5& deltaHk) const
  {
    using std::sqrt;
    auto G = 0.5 * (E0 + get<VALUE>(deltaE)) / (1.0 + nu);
    auto Hk = Hk0 + get<VALUE>(deltaHk);

    auto el_strain = sym(du_dX) - Fp_old;
    auto s = 2.0 * G * dev(el_strain);
    auto sigma_b = 2.0 / 3.0 * Hk * Fp_old;
    auto eta = s - sigma_b;
    auto q = sqrt(1.5) * norm(eta);

    auto Np = 1.5 * eta / q;

    return Fp_old + epsilon_dot * dt * Np;
  }

  /** @brief calculate the plastic strain */
  template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
  SMITH_HOST_DEVICE auto plasticStrain(double dt, const T1& epsilon_p, const T2& epsilon_dot,
                                       const tensor<T3, dim, dim>& Fp_old, const tensor<T4, dim, dim>& du_dX,
                                       const T5& deltaE, const T6& deltaHk) const
  {
    using std::sqrt;
    auto G = 0.5 * (E0 + get<VALUE>(deltaE)) / (1.0 + nu);
    auto Hk = Hk0 + get<VALUE>(deltaHk);

    auto el_strain = sym(du_dX) - Fp_old;
    auto s = 2.0 * G * dev(el_strain);
    auto sigma_b = 2.0 / 3.0 * Hk * Fp_old;
    auto eta = s - sigma_b;
    auto q = sqrt(1.5) * norm(eta);

    auto epsilon_dot_predict = q / dt - (3.0 * G + Hk) * epsilon_dot - this->hardening(epsilon_p, epsilon_dot) / dt;
    auto non_negativity = epsilon_dot_predict >= 0.0;
    return epsilon_dot_predict * non_negativity;
  }

 private:
  HardeningType hardening;  ///< Flow stress hardening model
  double E0;                ///< Base Young's modulus
  double nu;                ///< Poisson's ratio
  double Hk0;               ///< Base kinematic hardening modulus
  double rho;               ///< Mass density
};

TEST(DifferentiablePlasticity, PlasticLoadingFinitDiff)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string physics_name = "differentiable_plasticity";

  state_nonlin_opts.print_level = 0;
  primal_nonlin_opts.print_level = 0;

  int serial_refinement = 0;
  int parallel_refinement = 0;

  static constexpr int dim = 3;
  static constexpr int order = 2;

  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "plasticity_small_strain");

  std::string filename = SMITH_REPO_DIR "/data/meshes/beam-hex.mesh";
  const std::string meshtag = "mesh";
  auto mesh = std::make_shared<Mesh>(buildMeshFromFile(filename), meshtag, serial_refinement, parallel_refinement);

  auto staggered_coupled_solver = std::make_shared<CoupledSystemSolver>(100);
  auto primal_block_solver = buildNonlinearBlockSolver(primal_nonlin_opts, primal_lin_opts, *mesh);
  auto strain_block_solver = buildNonlinearBlockSolver(state_nonlin_opts, state_lin_opts, *mesh);
  auto defgrad_block_solver = buildNonlinearBlockSolver(state_nonlin_opts, state_lin_opts, *mesh);

  staggered_coupled_solver->addSubsystemSolver({0}, primal_block_solver);
  staggered_coupled_solver->addSubsystemSolver({2}, strain_block_solver);
  staggered_coupled_solver->addSubsystemSolver({1}, defgrad_block_solver);

  auto plastic_mechanics_system = buildPlasticMechanicsSystem<dim, order>(
      mesh, staggered_coupled_solver, physics_name, FieldType<L2<0>>("deltaE"), FieldType<L2<0>>("deltaHk"));

  using Hardening = solid_mechanics::LinearHardening;
  Hardening hardening{.sigma_y = 25.0, .Hi = 50.0, .eta = 0.0};
  ParameterizedDifferentiableJ2SmallStrain<dim, Hardening> mat(hardening, 1.5e+4, 0.25, 500.0, 1.0);

  plastic_mechanics_system.setMaterial(mesh->entireBodyName(), mat);
  plastic_mechanics_system.setPlasticity(mesh->entireBodyName(), mat);

  // prescribe zero displacement at the supported end of the beam,
  mesh->addDomainOfBoundaryElements("support", by_attr<dim>(1));
  plastic_mechanics_system.disp_bc->setFixedVectorBCs<dim>(mesh->domain("support"));

  // apply a displacement along z to the the tip of the beam
  mesh->addDomainOfBoundaryElements("tip", by_attr<dim>(2));
  auto translated_in_z = [](double t, tensor<double, dim>) {
    tensor<double, dim> u{};
    u[2] = t;
    return u;
  };
  plastic_mechanics_system.disp_bc->setVectorBCs<dim>(mesh->domain("tip"), {2}, translated_in_z);

  std::shared_ptr<DifferentiablePhysics> differentiable_plasticity =
      plastic_mechanics_system.createDifferentiablePhysics(physics_name);

  auto pv_writer = createParaviewWriter(*mesh, differentiable_plasticity->getFieldStatesAndParamStates(), physics_name,
                                        smith::ParaviewWriter::Options{.write_duals = false});
  pv_writer.write(0, differentiable_plasticity->time(), differentiable_plasticity->getFieldStatesAndParamStates());

  double dt = 0.1;
  for (size_t m = 0; m < 2; ++m) {
    differentiable_plasticity->advanceTimestep(dt);
    pv_writer.write(m + 1, differentiable_plasticity->time(),
                    differentiable_plasticity->getFieldStatesAndParamStates());
  }

  auto shape_disp = plastic_mechanics_system.field_store->getShapeDisp();
  auto params = plastic_mechanics_system.getParameterFields();

  auto displacement = plastic_mechanics_system.field_store->getField(plastic_mechanics_system.prefix("displacement"));
  auto disp_squared = 0.5 * innerProduct(displacement, displacement);
  gretl::set_as_objective(disp_squared);

  EXPECT_GT(checkGradWrt(disp_squared, shape_disp, 2e-4, 4, true), 0.95);
  EXPECT_GT(checkGradWrt(disp_squared, params[0], 100.0, 4, true), 0.95);
  EXPECT_GT(checkGradWrt(disp_squared, params[1], 100.0, 4, true), 0.95);

  std::vector<FieldState> sensitivity_fields{shape_disp, params[0], params[1]};
  auto sens_writer = createParaviewWriter(*mesh, sensitivity_fields, physics_name + "_final_sensitivities");

  auto& graph = disp_squared.data_store();
  graph.reset();
  gretl::set_as_objective(disp_squared);
  graph.back_prop();

  sens_writer.write(0, differentiable_plasticity->time(), sensitivity_fields);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
