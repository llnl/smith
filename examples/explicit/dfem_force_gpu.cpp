// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "mfem.hpp"
#include "smith/infrastructure/accelerator.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "smith/physics/functional_weak_form.hpp"
#include "smith/physics/dfem_solid_weak_form.hpp"
#include "smith/physics/dfem_mass_weak_form.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

const std::string MESHTAG = "mesh";

namespace mfem {
namespace future {

/**
 * @brief Compute Green's strain from the displacement gradient
 */
template <typename T, int dim>
SMITH_HOST_DEVICE auto greenStrain(const tensor<T, dim, dim>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

}  // namespace future
}  // namespace mfem

namespace smith {

// NOTE (EBC): NeoHookean is not working with dfem on device with HIP, since some needed LLVM intrinsics are not
// implemented in Enzyme with the call to log1p()/log().
struct StVenantKirchhoffWithFieldDensityDfem {
  static constexpr int dim = 2;

  /**
   * @brief stress calculation for a St. Venant Kirchhoff material model
   *
   * @tparam T Type of the displacement gradient components (number-like)
   *
   * @param[in] grad_u Displacement gradient
   *
   * @return The first Piola stress
   */
  template <typename T, int dim, typename Density>
  SMITH_HOST_DEVICE auto pkStress(double, const mfem::future::tensor<T, dim, dim>& du_dX,
                                  const mfem::future::tensor<T, dim, dim>&, const Density&) const
  {
    auto I = mfem::future::IdentityMatrix<dim>();
    auto F = du_dX + I;
    const auto E = mfem::future::greenStrain(du_dX);

    // stress
    const auto S = K * mfem::future::tr(E) * I + 2.0 * G * mfem::future::dev(E);
    return mfem::future::tuple{mfem::future::dot(F, S)};
  }

  /// @brief interpolates density field
  template <typename Density>
  SMITH_HOST_DEVICE auto density(const Density& density) const
  {
    return density;
  }

  double K;  ///< Bulk modulus
  double G;  ///< Shear modulus
};

}  // namespace smith

int main(int argc, char* argv[])
{
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using SolidMaterialDfem = smith::StVenantKirchhoffWithFieldDensityDfem;

  enum STATE
  {
    DISPLACEMENT,
    VELOCITY,
    ACCELERATION,
    COORDINATES
  };

  enum PARAMS
  {
    DENSITY
  };

  // Command line-modifiable variables
  int n_els = 8;
  bool use_gpu = false;
  bool write_output = false;

  // Handle command line arguments
  axom::CLI::App app{"Explicit dynamics"};
  // Mesh options
  app.add_option("--nels", n_els, "Number of elements in the x and y directions")->check(axom::CLI::PositiveNumber);
  // GPU options
  app.add_flag("--gpu,!--no-gpu", use_gpu, "Execute on GPU (where available)");
  // Output options
  app.add_flag("--output,!--no-output", write_output, "Save output to disk (e.g. for debugging)");

  // Need to allow extra arguments for PETSc support
  app.set_help_flag("--help");
  app.allow_extras()->parse(argc, argv);

  auto exec_space = use_gpu ? smith::ExecutionSpace::GPU : smith::ExecutionSpace::CPU;

  smith::ApplicationManager applicationManager(argc, argv, MPI_COMM_WORLD, true, exec_space);

  MPI_Barrier(MPI_COMM_WORLD);

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid_dynamics");

  // create mesh
  constexpr double length = 1.0;
  constexpr double width = 1.0;
  int nel_x = n_els;
  int nel_y = n_els;
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian2D(nel_x, nel_y, element_shape, true, length, width), MESHTAG, 0, 0);

  // create residual evaluator
  using VectorSpace = smith::H1<disp_order, dim>;
  using DensitySpace = smith::L2<disp_order - 1>;
  smith::FiniteElementState disp = smith::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
  smith::FiniteElementState velo = smith::StateManager::newState(VectorSpace{}, "velocity", mesh->tag());
  smith::FiniteElementState accel = smith::StateManager::newState(VectorSpace{}, "acceleration", mesh->tag());
  smith::FiniteElementState coords = smith::StateManager::newState(VectorSpace{}, "coordinates", mesh->tag());
  smith::FiniteElementState density = smith::StateManager::newState(DensitySpace{}, "density", mesh->tag());

  std::vector<smith::FiniteElementState> states{disp, velo, accel, coords};
  std::vector<smith::FiniteElementState> params{density};

  std::string physics_name = "solid";
  double E = 1.0e3;
  double nu = 0.3;

  using SolidT = smith::DfemSolidWeakForm;
  auto solid_dfem_weak_form =
      std::make_shared<SolidT>(physics_name, mesh, states[DISPLACEMENT].space(), getSpaces(params));

  SolidMaterialDfem dfem_mat;
  dfem_mat.K = E / (3.0 * (1.0 - 2.0 * nu));  // bulk modulus
  dfem_mat.G = E / (2.0 * (1.0 + nu));        // shear modulus
  int ir_order = 3;
  const mfem::IntegrationRule& displacement_ir = mfem::IntRules.Get(disp.space().GetFE(0)->GetGeomType(), ir_order);
  mfem::Array<int> solid_attrib({1});
  solid_dfem_weak_form->setMaterial<SolidMaterialDfem, smith::ScalarParameter<0>>(solid_attrib, dfem_mat,
                                                                                  displacement_ir);

  mfem::future::tensor<mfem::real_t, dim> g({0.0, -9.81});  // gravity vector
  mfem::future::tuple<mfem::future::Value<SolidT::DISPLACEMENT>, mfem::future::Value<SolidT::VELOCITY>,
                      mfem::future::Value<SolidT::ACCELERATION>, mfem::future::Gradient<SolidT::COORDINATES>,
                      mfem::future::Weight, mfem::future::Value<SolidT::NUM_STATE_VARS>>
      g_inputs{};
  mfem::future::tuple<mfem::future::Value<SolidT::NUM_STATE_VARS + 1>> g_outputs{};
  solid_dfem_weak_form->addBodyIntegral(
      solid_attrib,
      [=] SMITH_HOST_DEVICE(const mfem::future::tensor<mfem::real_t, dim>&,
                            const mfem::future::tensor<mfem::real_t, dim>&,
                            const mfem::future::tensor<mfem::real_t, dim>&,
                            const mfem::future::tensor<mfem::real_t, dim, dim>& dX_dxi, mfem::real_t weight, double) {
        auto J = mfem::future::det(dX_dxi) * weight;
        return mfem::future::tuple{g * J};
      },
      g_inputs, g_outputs, displacement_ir, std::index_sequence<>{});

  states[DISPLACEMENT] = 0.0;
  states[VELOCITY].setFromFieldFunction([](smith::tensor<double, dim>) {
    smith::tensor<double, dim> u({0.0, -1.0});
    return u;
  });
  states[ACCELERATION] = 0.0;
  states[COORDINATES].setFromGridFunction(static_cast<mfem::ParGridFunction&>(*mesh->mfemParMesh().GetNodes()));
  params[DENSITY] = 1.0;

  double time = 0.0;
  constexpr double dt = 0.0001;
  constexpr size_t num_steps = 5000;

  const auto& u = states[DISPLACEMENT];
  const auto& v = states[VELOCITY];
  const auto& a = states[ACCELERATION];

  auto v_pred = v;
  v_pred.Add(0.5 * dt, a);
  auto u_pred = u;
  u_pred.Add(dt, v_pred);

  std::vector<smith::ConstFieldPtr> pred_states = {&u_pred, &v_pred, &a, &states[COORDINATES], &params[DENSITY]};

  axom::utilities::Timer timer(true);
  for (size_t step = 0; step < num_steps; ++step) {
    auto no_mass_resid = solid_dfem_weak_form->residual(smith::TimeInfo(time, dt), &u_pred, pred_states);
    time += dt;
  }
  timer.stop();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cout << "Total time: " << timer.elapsedTimeInMilliSec() << " milliseconds" << std::endl;
  }

  return 0;
}
