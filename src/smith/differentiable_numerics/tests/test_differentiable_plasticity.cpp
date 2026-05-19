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

/// @brief Differentiable J2 material with nonlinear isotropic hardening and linear kinematic hardening
template <int dim, typename HardeningType>
class DifferentiableJ2SmallStrain {
 public:
  DifferentiableJ2SmallStrain(HardeningType hardeningModel, double youngsModulus, double poissonsRatio,
                              double hardeningModulus, double density)
      : hardening(hardeningModel), E(youngsModulus), nu(poissonsRatio), Hk(hardeningModulus), rho(density)
  {}

  /** @brief calculate the first Piola stress, given the displacement gradient and previous staggered solve material state */
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto firstPiolaStress(const tensor<T1, dim, dim>& du_dX, const tensor<T2, dim, dim>& Fp,
                                          const T3& /* epsilon_p */) const
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
  // Hanyu : combine plasticDeformGrad and plasticStrain
  template <typename T1, typename T2, typename T3>
  SMITH_HOST_DEVICE auto plasticDeformGrad(double dt, const tensor<T1, dim, dim>& Fp_old,
                                           const T2& epsilon_dot, const tensor<T3, dim, dim>& du_dX) const
  {
    using std::sqrt;
    constexpr auto I = Identity<dim>();
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);

    auto el_strain = sym(du_dX) - Fp_old;
    auto p = K * tr(el_strain);
    auto s = 2.0 * G * dev(el_strain);
    auto sigma_b = 2.0 / 3.0 * Hk * Fp_old;
    auto eta = s - sigma_b;
    auto q = sqrt(1.5) * norm(eta);

    auto Np = 1.5 * eta / q;

    return Fp_old + epsilon_dot * dt * Np;
  }

  /** @brief calculate the plastic strain */
  template <typename T1, typename T2, typename T3, typename T4>
  SMITH_HOST_DEVICE auto plasticStrain(double dt, const T1& epsilon_p, const T2& epsilon_dot, const tensor<T3, dim, dim>& Fp_old,
                                       const tensor<T4, dim, dim>& du_dX) const
  {
    using std::sqrt;
    constexpr auto I = Identity<dim>();
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);

    auto el_strain = sym(du_dX) - Fp_old;
    auto p = K * tr(el_strain);
    auto s = 2.0 * G * dev(el_strain);
    auto sigma_b = 2.0 / 3.0 * Hk * Fp_old;
    auto eta = s - sigma_b;
    auto q = sqrt(1.5) * norm(eta);

    return q / dt - (3.0 * G + Hk) * epsilon_dot - this->hardening(epsilon_p, epsilon_dot) / dt;
  }

 private:
  HardeningType hardening;  ///< Flow stress hardening model
  double E;                 ///< Young's modulus
  double nu;                ///< Poisson's ratio
  double Hk;                ///< Kinematic hardening modulus
  double rho;               ///< Mass density
};

smith::LinearSolverOptions primal_lin_opts{.linear_solver = smith::LinearSolver::CG,
                                           .preconditioner = smith::Preconditioner::HypreAMG,
                                           .relative_tol = 1e-6,
                                           .absolute_tol = 1e-10,
                                           .max_iterations = 200,
                                           .print_level = 0};

smith::NonlinearSolverOptions primal_nonlin_opts{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                 .relative_tol = 1.0e-6,
                                                 .absolute_tol = 1.0e-8,
                                                 .max_iterations = 25,
                                                 .print_level = 1};

smith::LinearSolverOptions state_lin_opts{.linear_solver = smith::LinearSolver::GMRES,
                                          .preconditioner = smith::Preconditioner::HypreAMG,
                                          .relative_tol = 1e-6,
                                          .absolute_tol = 1e-10,
                                          .max_iterations = 200,
                                          .print_level = 0};

smith::NonlinearSolverOptions state_nonlin_opts{.nonlin_solver = NonlinearSolver::NewtonLineSearch,
                                                .relative_tol = 1.0e-6,
                                                .absolute_tol = 1.0e-8,
                                                .max_iterations = 25,
                                                .print_level = 1};

TEST(DifferentiablePlasticity, J2SmallStrainLinearHardening)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 1;
  int parallel_refinement = 1;

  static constexpr int dim = 3;
  static constexpr int order = 1;

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "plasticity_small_strain");

  std::string filename = SMITH_REPO_DIR "/data/meshes/patch3D_tets.mesh";
  const std::string meshtag = "mesh";
  auto mesh = std::make_shared<smith::Mesh>(smith::buildMeshFromFile(filename), meshtag, serial_refinement,
                                            parallel_refinement);

  auto staggered_coupled_solver = std::make_shared<CoupledSystemSolver>(10);
  auto primal_block_solver = buildNonlinearBlockSolver(primal_nonlin_opts, primal_lin_opts, *mesh);
  auto state_block_solver = buildNonlinearBlockSolver(state_nonlin_opts, state_lin_opts, *mesh);
  staggered_coupled_solver->addSubsystemSolver({0}, primal_block_solver);
  staggered_coupled_solver->addSubsystemSolver({1, 2}, state_block_solver);

  auto plastic_mechanics_system = buildPlasticMechanicsSystem<dim, order>(mesh, staggered_coupled_solver);

  using Hardening = solid_mechanics::LinearHardening;
  Hardening hardening{.sigma_y = 50.0, .Hi = 50.0, .eta = 0.0};
  DifferentiableJ2SmallStrain<dim, Hardening> mat(hardening, 1e+4, 0.25, 5.0, 1.0);

  plastic_mechanics_system.setMaterial(mesh->entireBodyName(), mat);
}

} // namespace smith
