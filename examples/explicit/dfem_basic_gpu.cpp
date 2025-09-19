// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "mfem.hpp"
#include "serac/infrastructure/accelerator.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/state_manager.hpp"

#include "serac/physics/functional_weak_form.hpp"
#include "serac/physics/dfem_weak_form.hpp"

auto element_shape = mfem::Element::QUADRILATERAL;

const std::string MESHTAG = "mesh";

namespace mfem {
namespace future {

/**
 * @brief Compute Green's strain from the displacement gradient
 */
template <typename T, int dim>
SERAC_HOST_DEVICE auto greenStrain(const tensor<T, dim, dim>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

}  // namespace future
}  // namespace mfem

namespace serac {

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
  template <typename T, int dim>
  SERAC_HOST_DEVICE auto pkStress(double, const mfem::future::tensor<T, dim, dim>& du_dX) const
  {
    auto I = mfem::future::IdentityMatrix<dim>();
    auto F = du_dX + I;
    const auto E = mfem::future::greenStrain(du_dX);

    // stress
    const auto S = K * mfem::future::tr(E) * I + 2.0 * G * mfem::future::dev(E);
    return mfem::future::tuple{mfem::future::dot(F, S)};
  }

  /// @brief interpolates density field
  SERAC_HOST_DEVICE auto density() const
  {
    return rho;
  }

  double K;  ///< Bulk modulus
  double G;  ///< Shear modulus
  double rho;
};

template <typename Material>
struct StressDivQFunction {
  SERAC_HOST_DEVICE inline auto operator()(
      // mfem::real_t dt, // TODO: figure out how to pass this in
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& du_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dX_dxi, mfem::real_t weight) const
  {
    auto dxi_dX = mfem::future::inv(dX_dxi);
    auto du_dX = mfem::future::dot(du_dxi, dxi_dX);
    double dt = 1.0;  // TODO: figure out how to pass this in to the qfunction
    auto P = mfem::future::get<0>(material.pkStress(dt, du_dX));
    auto JxW = mfem::future::det(dX_dxi) * weight * mfem::future::transpose(dxi_dX);
    return mfem::future::tuple{-P * JxW};
  }

  Material material;  ///< the material model to use for computing the stress
};

}  // namespace serac

int main(int argc, char* argv[])
{
  static constexpr int dim = 2;
  static constexpr int disp_order = 1;

  using SolidMaterialDfem = serac::StVenantKirchhoffWithFieldDensityDfem;

  enum STATE
  {
    DISPLACEMENT,
    COORDINATES
  };

  enum PARAMS
  {
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

  auto exec_space = use_gpu ? serac::ExecutionSpace::GPU : serac::ExecutionSpace::CPU;

  serac::ApplicationManager applicationManager(argc, argv, MPI_COMM_WORLD, exec_space);

  MPI_Barrier(MPI_COMM_WORLD);

  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_dynamics");

  // create mesh
  constexpr double length = 1.0;
  constexpr double width = 1.0;
  int nel_x = n_els;
  int nel_y = n_els;
  auto mesh = std::make_shared<serac::Mesh>(
      mfem::Mesh::MakeCartesian2D(nel_x, nel_y, element_shape, true, length, width), MESHTAG, 0, 0);

  // create residual evaluator
  using VectorSpace = serac::H1<disp_order, dim>;
  serac::FiniteElementState disp = serac::StateManager::newState(VectorSpace{}, "displacement", mesh->tag());
  serac::FiniteElementState coords = serac::StateManager::newState(VectorSpace{}, "coordinates", mesh->tag());

  std::vector<serac::FiniteElementState> states{disp, coords};
  std::vector<serac::FiniteElementState> params{};

  std::string physics_name = "solid";
  double E = 1.0e3;
  double nu = 0.3;

  using SolidT = serac::DfemWeakForm;
  auto dfem_weak_form =
      std::make_shared<SolidT>(physics_name, mesh, states[DISPLACEMENT].space(), getSpaces(states));

  SolidMaterialDfem dfem_mat;
  dfem_mat.K = E / (3.0 * (1.0 - 2.0 * nu));  // bulk modulus
  dfem_mat.G = E / (2.0 * (1.0 + nu));        // shear modulus
  int ir_order = 3;
  const mfem::IntegrationRule& displacement_ir = mfem::IntRules.Get(disp.space().GetFE(0)->GetGeomType(), ir_order);
  mfem::Array<int> solid_attrib({1});
  auto stress_div_integral = serac::StressDivQFunction<SolidMaterialDfem>{.material = dfem_mat};
  mfem::future::tuple<mfem::future::Gradient<DISPLACEMENT>, mfem::future::Gradient<COORDINATES>, mfem::future::Weight>
      stress_div_integral_inputs{};
  mfem::future::tuple<mfem::future::Gradient<2>> stress_div_integral_outputs{};
  dfem_weak_form->addBodyIntegral(solid_attrib, stress_div_integral, stress_div_integral_inputs,
                                  stress_div_integral_outputs, displacement_ir, std::index_sequence<>{});

  mfem::future::tensor<mfem::real_t, dim> g({0.0, -9.81});  // gravity vector
  mfem::future::tuple<mfem::future::Value<DISPLACEMENT>, mfem::future::Gradient<COORDINATES>,
                      mfem::future::Weight>
      g_inputs{};
  mfem::future::tuple<mfem::future::Value<2>> g_outputs{};
  dfem_weak_form->addBodyIntegral(
      solid_attrib,
      [=] SERAC_HOST_DEVICE(const mfem::future::tensor<mfem::real_t, dim>&,
                            const mfem::future::tensor<mfem::real_t, dim, dim>& dX_dxi, mfem::real_t weight) {
        auto J = mfem::future::det(dX_dxi) * weight;
        return mfem::future::tuple{g * J};
      },
      g_inputs, g_outputs, displacement_ir, std::index_sequence<>{});

  states[DISPLACEMENT] = 0.0;
  states[COORDINATES].setFromGridFunction(static_cast<mfem::ParGridFunction&>(*mesh->mfemParMesh().GetNodes()));

  double time = 0.0;
  constexpr double dt = 0.0001;
  constexpr size_t num_steps = 5000;

  const auto& u = states[DISPLACEMENT];
  auto u_pred = u;
  std::vector<serac::ConstFieldPtr> pred_states = {&u_pred, &states[COORDINATES]};

  // mfem::real_t max_resid = 0.0;
  axom::utilities::Timer timer(true);
  for (size_t step = 0; step < num_steps; ++step) {
    // auto u_ptr = states[DISPLACEMENT].Write();
    // auto x_ptr = states[COORDINATES].Read();
    // mfem::forall_switch(u.UseDevice(), u.Size(), [=] MFEM_HOST_DEVICE(int i) { u_ptr[i] = time * x_ptr[i]; });
    // auto u_pred = u;
    // std::vector<serac::ConstFieldPtr> pred_states = {&u_pred, &states[COORDINATES]};
    auto no_mass_resid = dfem_weak_form->residual(time, dt, &u_pred, pred_states);
    // max_resid += no_mass_resid.Max();
    time += dt;
  }
  timer.stop();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cout << "Total time: " << timer.elapsedTimeInMilliSec() << " milliseconds" << std::endl;
    // std::cout << "Max residual: " << max_resid << std::endl;
  }

  return 0;
}
