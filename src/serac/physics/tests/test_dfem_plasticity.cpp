// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/infrastructure/application_manager.hpp"
#include "serac/physics/dfem_weak_form.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/state_manager.hpp"

#include "serac/physics/tests/physics_test_utils.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

namespace serac {

struct SmoothJ2 {
  static constexpr int dim = 3;         ///< spatial dimension
  static constexpr int n_internal_states = 10;
  static constexpr double tol = 1e-10;  ///< relative tolerance on residual mag to judge convergence of return map

  double E;                 ///< Young's modulus
  double nu;                ///< Poisson's ratio
  double sigma_y;           ///< Yield strength
  double Hi;                ///< Isotropic hardening modulus
  double rho;               ///< Mass density

  /// @brief variables required to characterize the hysteresis response
  struct InternalState {
    mfem::future::tensor<double, dim, dim> plastic_strain;  ///< plastic strain
    double accumulated_plastic_strain;        ///< uniaxial equivalent plastic strain
  };

  MFEM_HOST_DEVICE inline
  InternalState unpack_internal_state(const mfem::future::tensor<double, n_internal_states>& packed_state) const
  {
      // we could use type punning here to avoid copies
      auto plastic_strain = mfem::future::make_tensor<dim, dim>(
         [&packed_state](int i, int j) { return packed_state[dim*i + j]; });
      double accumulated_plastic_strain = packed_state[n_internal_states - 1];
      return {plastic_strain, accumulated_plastic_strain};
  }

  MFEM_HOST_DEVICE inline
  mfem::future::tensor<double, n_internal_states> pack_internal_state(
    const mfem::future::tensor<double, dim, dim>& plastic_strain, 
    double accumulated_plastic_strain) const
  {
      mfem::future::tensor<double, n_internal_states> packed_state{};
      for (int i = 0, ij = 0; i < dim; i++) {
         for (int j = 0; j < dim; j++, ij++) {
            packed_state[ij] = plastic_strain[i][j];
         }
      }
      packed_state[n_internal_states - 1] = accumulated_plastic_strain;
      return packed_state;
  }

  MFEM_HOST_DEVICE inline
  mfem::future::tuple<tensor<double, dim, dim>, mfem::future::tensor<double, n_internal_states>>
  update(double /* dt */, const mfem::future::tensor<double, dim, dim>& dudX, 
    const mfem::future::tensor<double, n_internal_states>& internal_state) const
  {
      auto I = mfem::future::IdentityMatrix<dim>();
      const double K = E / (3.0 * (1.0 - 2.0 * nu));
      const double G = 0.5 * E / (1.0 + nu);

      auto [plastic_strain, accumulated_plastic_strain] = unpack_internal_state(internal_state);

      // (i) elastic predictor
      auto el_strain = mfem::future::sym(dudX) - plastic_strain;
      auto p = K * tr(el_strain);
      auto s = 2.0 * G * mfem::future::dev(el_strain);
      auto q = sqrt(1.5) * mfem::future::norm(s);
      double delta_eqps = 0.0;

      auto flow_strength = [this](double eqps) { return this->sigma_y + this->Hi*eqps; };

      // (ii) admissibility
      if (q - (sigma_y + Hi*accumulated_plastic_strain) > tol*sigma_y) {
         // (iii) return mapping
         double delta_eqps = (q - sigma_y - Hi*accumulated_plastic_strain)/(3*G + Hi);
         auto Np = 1.5 * s / q;
         s -= 2.0 * G * delta_eqps * Np;
         plastic_strain += delta_eqps * Np;
         accumulated_plastic_strain += delta_eqps;
      }
      auto stress = s + p * I;
      auto internal_state_new = pack_internal_state(plastic_strain, accumulated_plastic_strain);
      return {stress, internal_state_new};
  }

  SERAC_HOST_DEVICE auto pkStress(double, const mfem::future::tensor<double, dim, dim>& du_dX,
                                  const mfem::future::tensor<double, dim, dim>&, 
                                  const tensor<double, n_internal_states>& internal_state) const
  {
    const double dt = 1.0;
    auto [stress, internal_state_new] = update(dt, du_dX, internal_state);
    return stress;
  }

  SERAC_HOST_DEVICE auto internalStateNew(double, const mfem::future::tensor<double, dim, dim>& du_dX,
    const mfem::future::tensor<double, dim, dim>&, 
    const tensor<double, n_internal_states>& internal_state) const
  {
    const double dt = 1.0;
    auto [stress, internal_state_new] = update(dt, du_dX, internal_state);
    return internal_state_new;
  }

  SERAC_HOST_DEVICE double density() const { return rho; }
};

TEST(Dfem, Plasticity)
{
    constexpr int dim = 3;
    const std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
    axom::sidre::DataStore datastore;   
    serac::StateManager::initialize(datastore, "dfem_plasticity");
    mesh = std::make_shared<serac::Mesh>(buildMeshFromFile(filename),
                                         "this_mesh_name", 0, 0);

    LinearSolverOptions linear_options{.linear_solver = LinearSolver::CG, .print_level = 0};

    NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                           .relative_tol = 1.0e-13,
                                           .absolute_tol = 1.0e-13,
                                           .max_iterations = 20,
                                           .print_level = 1};

    enum STATE
    {
        DISPLACEMENT,
        VELOCITY,
        ACCELERATION,
        COORDINATES
    };

    enum PARAMS
    {
        // none for now
    };

    using VectorSpace = serac::H1<disp_order, dim>;

    serac::FiniteElementState disp = serac::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
    serac::FiniteElementState velo = serac::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
    serac::FiniteElementState accel = serac::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());

    using Material = SmoothJ2;

    int ir_order = 2;
    const mfem::IntegrationRule& displacement_ir = mfem::IntRules.Get(disp.space().GetFE(0)->GetGeomType(), ir_order);
    bool use_tensor_product = false;
    mfem::future::UniformParameterSpace internal_state_space(mesh->mfemParMesh(), displacement_ir, Material.n_internal_states, use_tensor_product);
    mfem::future::ParameterFunction internal_state(internal_state_space);

    
}

} // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}