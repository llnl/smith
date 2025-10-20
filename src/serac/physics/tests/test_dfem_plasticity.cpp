// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mfem/fem/pgridfunc.hpp>
#include <string>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/infrastructure/application_manager.hpp"
#include "serac/mesh_utils/mesh_utils_base.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/physics/dfem_solid_weak_form.hpp"
#include "serac/physics/field_types.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/finite_element_dual.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/state_manager.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

namespace serac {

struct SmoothJ2 {
  static constexpr int dim = 3;  ///< spatial dimension
  static constexpr int N_INTERNAL_STATES = 10;
  static constexpr double tol = 1e-10;  ///< relative tolerance on residual mag to judge convergence of return map

  double E;        ///< Young's modulus
  double nu;       ///< Poisson's ratio
  double sigma_y;  ///< Yield strength
  double Hi;       ///< Isotropic hardening modulus
  double rho;      ///< Mass density

  /// @brief variables required to characterize the hysteresis response
  struct InternalState {
    mfem::future::tensor<double, dim, dim> plastic_strain;  ///< plastic strain
    double accumulated_plastic_strain;                      ///< uniaxial equivalent plastic strain
  };

  MFEM_HOST_DEVICE inline InternalState unpack_internal_state(
      const mfem::future::tensor<double, N_INTERNAL_STATES>& packed_state) const
  {
    // we could use type punning here to avoid copies
    auto plastic_strain =
        mfem::future::make_tensor<dim, dim>([&packed_state](int i, int j) { return packed_state[dim * i + j]; });
    double accumulated_plastic_strain = packed_state[N_INTERNAL_STATES - 1];
    return {plastic_strain, accumulated_plastic_strain};
  }

  MFEM_HOST_DEVICE inline mfem::future::tensor<double, N_INTERNAL_STATES> pack_internal_state(
      const mfem::future::tensor<double, dim, dim>& plastic_strain, double accumulated_plastic_strain) const
  {
    mfem::future::tensor<double, N_INTERNAL_STATES> packed_state{};
    for (int i = 0, ij = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++, ij++) {
        packed_state[ij] = plastic_strain[i][j];
      }
    }
    packed_state[N_INTERNAL_STATES - 1] = accumulated_plastic_strain;
    return packed_state;
  }

  MFEM_HOST_DEVICE inline mfem::future::tuple<mfem::future::tensor<double, dim, dim>,
                                              mfem::future::tensor<double, N_INTERNAL_STATES>>
  update(double /* dt */, const mfem::future::tensor<double, dim, dim>& dudX,
         const mfem::future::tensor<double, N_INTERNAL_STATES>& internal_state) const
  {
    auto I = mfem::future::IdentityMatrix<dim>();
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);

    auto [plastic_strain, accumulated_plastic_strain] = unpack_internal_state(internal_state);

    // (i) elastic predictor
    auto el_strain = mfem::future::sym(dudX) - plastic_strain;
    auto p = K * tr(el_strain);
    auto s = 2.0 * G * mfem::future::dev(el_strain);
    auto q = std::sqrt(1.5) * mfem::future::norm(s);

    // auto flow_strength = [this](double eqps) { return this->sigma_y + this->Hi * eqps; };

    // (ii) admissibility
    if (q - (sigma_y + Hi * accumulated_plastic_strain) > tol * sigma_y) {
      // (iii) return mapping
      double delta_eqps = (q - sigma_y - Hi * accumulated_plastic_strain) / (3 * G + Hi);
      auto Np = 1.5 * s / q;
      s -= 2.0 * G * delta_eqps * Np;
      plastic_strain += delta_eqps * Np;
      accumulated_plastic_strain += delta_eqps;
    }
    auto stress = s + p * I;
    auto internal_state_new = pack_internal_state(plastic_strain, accumulated_plastic_strain);
    return mfem::future::make_tuple(stress, internal_state_new);
  }

  SERAC_HOST_DEVICE auto pkStress(double, const mfem::future::tensor<double, dim, dim>& du_dX,
                                  const mfem::future::tensor<double, dim, dim>&,
                                  const mfem::future::tensor<double, N_INTERNAL_STATES>& internal_state) const
  {
    const double dt = 1.0;
    auto [stress, internal_state_new] = update(dt, du_dX, internal_state);
    return stress;
  }

  SERAC_HOST_DEVICE auto internalStateNew(double, const mfem::future::tensor<double, dim, dim>& du_dX,
                                          const mfem::future::tensor<double, dim, dim>&,
                                          const mfem::future::tensor<double, N_INTERNAL_STATES>& internal_state) const
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
  constexpr int disp_order = 1;
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "dfem_plasticity");
  auto mfem_mesh = buildMeshFromFile(filename);
  auto mesh = std::make_shared<serac::Mesh>(std::move(mfem_mesh), "amesh", 0, 0);
  mfem::out << "Constructed mesh" << std::endl;

  // TODO: add these when we have a solver
  // LinearSolverOptions linear_options{.linear_solver = LinearSolver::CG, .print_level = 0};

  // NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
  //                                          .relative_tol = 1.0e-13,
  //                                          .absolute_tol = 1.0e-13,
  //                                          .max_iterations = 20,
  //                                          .print_level = 1};

  // NOTE: we can grab STATE from SolidT::STATE
  // enum STATE
  // {
  //   DISPLACEMENT,
  //   VELOCITY,
  //   ACCELERATION,
  //   COORDINATES
  // };

  enum PARAMS
  {
    J2_INTERNAL_STATE
  };

  // create material
  using Material = SmoothJ2;
  const double E = 1.0e3;
  const double nu = 0.25;
  const double sigma_y = 0.53333*1000;
  auto mat = Material{.E = E, .nu = nu, .sigma_y = sigma_y, .Hi = 40.0, .rho = 1.0};

  // create fields
  using KinematicSpace = H1<disp_order, dim>;
  FiniteElementState disp = StateManager::newState(KinematicSpace{}, "displacement", mesh->tag());
  FiniteElementState velo = StateManager::newState(KinematicSpace{}, "velocity", mesh->tag());
  FiniteElementState accel = StateManager::newState(KinematicSpace{}, "acceleration", mesh->tag());
  FiniteElementState coords = StateManager::newState(KinematicSpace{}, "coordinates", mesh->tag());

  int ir_order = 2;
  const mfem::IntegrationRule& displacement_ir = mfem::IntRules.Get(disp.space().GetFE(0)->GetGeomType(), ir_order);
  bool use_tensor_product = false;
  mfem::future::UniformParameterSpace internal_state_space(mesh->mfemParMesh(), displacement_ir, mat.N_INTERNAL_STATES,
                                                           use_tensor_product);
  mfem::future::ParameterFunction internal_state(internal_state_space);

  // initialize fields

  /* Get boundary dofs for applying forcing, don't need this until we have a solver
  mfem::Array<int> bdr_attr_is_ess(mesh->mfemParMesh().bdr_attributes.Max());
  mfem::Array<int> displacement_ess_tdof;
  mfem::Array<int> bc_tdof;
  bdr_attr_is_ess = 0; // reset
  bdr_attr_is_ess[0] = 1; // flag boundary 0 (1 in mesh)
  disp.space().GetEssentialTrueDofs(bdr_attr_is_ess, bc_tdof, 0); // get x-dir dofs
  for (auto td : bc_tdof) { displacement_ess_tdof.Append(td); };

  bdr_attr_is_ess = 0; // reset
  bdr_attr_is_ess[1] = 1; // flag boundary 1 (2 in mesh)
  disp.space().GetEssentialTrueDofs(bdr_attr_is_ess, bc_tdof, 0); // get x-dir dofs
  for (auto td : bc_tdof) { displacement_ess_tdof.Append(td); };
  */

  // set displacement to uniaxial stress solution
  auto applied_displacement = [nu](tensor<double, dim> X) {
    double strain = 0.01;
    return tensor<double, dim>{strain*X[0], -nu*strain*X[1], -nu*strain*X[2]}; };
  disp.setFromFieldFunction(applied_displacement);
  velo = 0.0;
  accel = 0.0;
  coords.setFromGridFunction(static_cast<mfem::ParGridFunction&>(*mesh->mfemParMesh().GetNodes()));
  internal_state = 0.0;

  // set up physics
  constexpr bool is_quasi_static = true;
  constexpr bool use_lumped_mass = false;
  using SolidT = DfemSolidWeakForm<is_quasi_static, use_lumped_mass>;
  auto physics = SolidT("plasticity", mesh, disp.space(), {&internal_state_space}, {});
  mfem::Array<int> entire_domain{1, 2};
  physics.setMaterial<Material, InternalVariableParameter<J2_INTERNAL_STATE, Material::N_INTERNAL_STATES>>(
      entire_domain, mat, displacement_ir);

  double t = 0.0;
  double dt = 1.0;
  FiniteElementDual reaction = StateManager::newDual(KinematicSpace{}, "reactions", mesh->tag());
  reaction = physics.residual(t, dt, &disp, {&disp, &velo, &accel, &coords}, {&internal_state});
  mfem::out << "reaction = \n";
  reaction.Print();

  // make a gridFunction of the reactions for plotting
  mfem::ParGridFunction reaction_gf(&reaction.space());
  reaction.linearForm().ParallelAssemble(reaction_gf.GetTrueVector());
  reaction_gf.SetFromTrueVector();

  mfem::ParaViewDataCollection dc("dfem_plasticity_pv", &(mesh->mfemParMesh()));
  dc.SetHighOrderOutput(true);
  dc.SetLevelsOfDetail(1);
  dc.RegisterField("displacement", &disp.gridFunction());
  dc.RegisterField("reaction", &reaction_gf);
  //dc.RegisterQField("internal_state", &output_internal_state);
  dc.SetCycle(0);
  dc.Save();

  // physics.residual()
  // implement physics.vjp()

  // set loads

  // advance
  //  solve for u
  //    residual and jvp
  //  use u to update internal
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}