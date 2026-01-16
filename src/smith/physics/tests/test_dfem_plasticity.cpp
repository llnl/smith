// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <memory>
#include <ostream>
#include <string>
#include <fstream> // for debugging, remove in final

#include <axom/slic/interface/slic_macros.hpp>
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/physics/dfem_solid_weak_form.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/state_manager.hpp"


namespace smith {

struct NewtonSettings {
  int max_iters;
  double residual_abs_tol;
  double residual_rel_tol;
};

template <auto f, typename T>
__attribute__((noinline))
void newton_scalar_impl(const double* x0_ptr, const T* p_ptr, double* x_ptr) {
  const double& x0 = *x0_ptr;
  const T& p = *p_ptr;

  auto fprime = [&p](double x) {
    double dx = 1.0;
    return __enzyme_fwddiff<double>((void*)+f, enzyme_dup, x, dx, enzyme_const, p);
  };

  // TODO: make this in input argument. Have to deal with making this a non-differentiated argument.
  NewtonSettings settings{.max_iters = 50, .residual_abs_tol = 1e-10, .residual_rel_tol = 0.0};

  double x = x0;
  double r0 = f(x, p);
  int iters = 0;
  for (int i = 0; i < settings.max_iters; i++, iters++) {
        double r = f(x, p);
        if (std::abs(r) < settings.residual_abs_tol || std::abs(r/r0) < settings.residual_rel_tol) {
            break;
        }
        double J = fprime(x);
        x -= r/J;
    }
    *x_ptr = x;
    SLIC_WARNING_IF(iters == settings.max_iters, axom::fmt::format("Scalar nonlinear solver failed to converge in alloted iterations ({}), abs(r) = {}", settings.max_iters, std::abs(f(x, p))));
}


template <auto f, typename T>
void newton_scalar_impl_fwddiff(const double* x0, const double* /* dx0 */,
                                const T* p, const T* dp,
                                double* x, double* dx)
{
  newton_scalar_impl<f>(x0, p, x);
  double dfdx = __enzyme_fwddiff<double>((void*)+f, enzyme_dup, *x, 1.0, enzyme_const, *p);
  double dfdp = __enzyme_fwddiff<double>((void*)+f, enzyme_const, *x, enzyme_dup, *p, *dp);
  *dx = -dfdp/dfdx;
}


template<auto f, typename T>
double newton_scalar(double x0, T p) {
  double x;
  newton_scalar_impl<f>(&x0, &p, &x);
  return x;
}


double sqrt_residual(double x, double a) {
    return x*x - a;
};


__attribute__((used))
void *  __enzyme_register_derivative_newton_scalar_on_sqrt[2] = {
  (void*) newton_scalar_impl<sqrt_residual, double>, 
  (void*) newton_scalar_impl_fwddiff<sqrt_residual, double> 
};


TEST(Enzyme, Newton) {
  double z = 2.0;
  //NewtonSettings settings{.max_iters = 50, .residual_abs_tol = 0.0, .residual_rel_tol = 1e-10};
  double x0 = 0.5*z;
  double x = newton_scalar<sqrt_residual>(x0, z);
  EXPECT_NEAR(x, std::sqrt(z), 1e-9);

  /* QUESTION FOR BILL:
  I would like to use the commented code below (where the settings struct is marked enzyme_const).
  However, this causes enzyme to fail to register the custom derivative rule for newton_scalar.
  The problem is the integer field in the `settings` struct. If I make it a double, the line below works.
  Are custom derivatives forced to use the default activity rules?
  */
  //double dx_dz = __enzyme_fwddiff<double>((void*) newton_scalar<sqrt_residual>, enzyme_const, x0, enzyme_const, settings, enzyme_dup, z, dz);
  
  //NewtonSettings dummy_settings_dual{};
  double z_dot = 1.0;
  double dx_dz = __enzyme_fwddiff<double>((void*) newton_scalar<sqrt_residual, double>, enzyme_const, x0, enzyme_dup, z, z_dot);
  double dx_dz_gold = 0.5/x;
  EXPECT_NEAR(dx_dz, dx_dz_gold, 1e-9);
}


double nthroot_residual(double x, mfem::future::tuple<double, double> p) {
  auto power = mfem::future::get<0>(p);
  auto a = mfem::future::get<1>(p);
  return std::pow(x, power) - a;
}

__attribute__((used))
void *  __enzyme_register_derivative_newton_scalar_on_nthrt[2] = {
  (void*) newton_scalar_impl<nthroot_residual, mfem::future::tuple<double, double>>, 
  (void*) newton_scalar_impl_fwddiff<nthroot_residual, mfem::future::tuple<double, double>> 
};

TEST(Enzyme, NewtonWithTupleParams) {
  double index = 3.0;
  double radicand = 8.0;
  mfem::future::tuple params{index, radicand};
  double x0 = radicand / 2.0;
  double y = newton_scalar<nthroot_residual>(x0, params);
  EXPECT_NEAR(y, 2.0, 1e-9);

  mfem::future::tuple params_dot{0.0, 1.0};
  double dy_da = __enzyme_fwddiff<double>((void*) newton_scalar<nthroot_residual, mfem::future::tuple<double, double>>, enzyme_const, x0, enzyme_dup, params, params_dot);
  EXPECT_NEAR(dy_da, 1.0/3.0*std::pow(radicand, -2.0/3.0), 1e-9);
}


double nthroot_residual_with_vector(double x, std::vector<double> p) {
  auto power = p[0];
  auto a = p[1];
  return std::pow(x, power) - a;
}

__attribute__((used))
void *  __enzyme_register_derivative_nthroot_residual_with_vector[2] = {
  (void*) newton_scalar_impl<nthroot_residual_with_vector, std::vector<double>>, 
  (void*) newton_scalar_impl_fwddiff<nthroot_residual_with_vector, std::vector<double>> 
};

TEST(Enzyme, NewtonWithVectorParams) {
  double index = 3.0;
  double radicand = 8.0;
  std::vector<double> params{index, radicand};
  double x0 = radicand / 2.0;
  double y = newton_scalar<nthroot_residual_with_vector>(x0, params);
  EXPECT_NEAR(y, 2.0, 1e-9);

  std::vector<double> params_dot{0.0, 1.0};
  double dy_da = __enzyme_fwddiff<double>((void*) newton_scalar<nthroot_residual_with_vector, std::vector<double>>, enzyme_const, x0, enzyme_dup, params, params_dot);
  EXPECT_NEAR(dy_da, 1.0/3.0*std::pow(radicand, -2.0/3.0), 1e-9);
}

double square_and_shift(double x, double p) {
  return x*x - p;
}

mfem::future::tensor<double, 3, 3> sym_and_shift(
    const mfem::future::tensor<double, 3, 3>& X, const mfem::future::tensor<double, 3, 3>& p)
{
  return mfem::future::sym(X) - p;
}


mfem::future::tensor<double, 3, 3> sym_only(const mfem::future::tensor<double, 3, 3>& dudX) {
  return mfem::future::sym(dudX);
}


TEST(Enzyme, ConstTensorDifferentiatesCorrectly) {
  double x = 2.0;
  double x_dot = 1.0;
  double q = 3.0;
  double dydx = __enzyme_fwddiff<double>((void*)square_and_shift, enzyme_dup, x, x_dot, enzyme_const, q);
  EXPECT_DOUBLE_EQ(dydx, 4.0);

  // Checks that fucntions are differentiated correctly when tensor arguments are marked enzyme_const
  mfem::future::tensor<double, 3, 3> H{{{1.0, 0, 0}, {0, -0.5, 0}, {0, 0, -0.25}}};
  mfem::future::tensor<double, 3,3 > p{{{1.0, 0, 0}, {0, 2.0, 0}, {0, 0, 3.0}}};
  mfem::future::tensor<double, 3, 3> H_dot{{{0, 1, 0}, {0, 0, 0}, {0, 0, 0}}};
  // Adding `enzyme_runtime_activity` to the argument list makes the test pass, but I don't know why yet.
  auto z_dot = __enzyme_fwddiff<mfem::future::tensor<double, 3, 3>>((void*)sym_and_shift, enzyme_dup, enzyme_dup, H, H_dot, enzyme_const, p);

  mfem::out << "sym and shift output " << z_dot << std::endl;
  
  mfem::future::tensor<double, 3, 3> expected_z_dot{{{0, 0.5, 0}, {0.5, 0, 0}, {0, 0, 0}}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_DOUBLE_EQ(z_dot[i][j], expected_z_dot.values[i][j]);
    }
  }
  
  // This is ok, so the problem is definitely with enzyme_const. 
  // this confirms the sym() function is differentiated corectly
  auto ok = __enzyme_fwddiff<mfem::future::tensor<double, 3, 3>>((void*)sym_only, enzyme_dup, H, H_dot);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_DOUBLE_EQ(ok[i][j], expected_z_dot.values[i][j]);
    }
  }
}


struct UniaxialSolution {
  double E, nu, sigma_y, Hi;

  double axial_strain(double t) const { return std::sin(M_PI_2*t); };

  double axial_stress(double t) const {
    double e = axial_strain(t);
    return E/(E + Hi)*(Hi*e + sigma_y);
  }

  tensor<double, 3> displacement(tensor<double, 3> X, double t) const
  {
    double ex = axial_strain(t);
    if (ex < sigma_y/E) return {X[0]*ex, -nu*ex*X[1], -nu*ex*X[2]};

    auto ep = plastic_strain(t);
    double eex = ex - ep[0];
    double eey = -nu*eex;
    tensor<double, 3> strain{ex, ep[1] + eey, ep[2] + eey};
    return {strain[0]*X[0], strain[1]*X[1], strain[2]*X[2]};
  }

  tensor<double, 3> plastic_strain(double t) const
  {
    double ex = axial_strain(t);
    double epx = (E*ex - sigma_y)/(E + Hi);
    return {epx, -0.5*epx, -0.5*epx};
  }
};

struct J2Linear {
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

  SMITH_HOST_DEVICE auto pkStress(double, const mfem::future::tensor<double, dim, dim>& du_dX,
                                  const mfem::future::tensor<double, dim, dim>&,
                                  const mfem::future::tensor<double, N_INTERNAL_STATES>& internal_state) const
  {
    const double dt = 1.0;
    auto [stress, internal_state_new] = update(dt, du_dX, internal_state);
    return stress;
  }

  SMITH_HOST_DEVICE auto internalStateNew(double, const mfem::future::tensor<double, dim, dim>& du_dX,
                                          const mfem::future::tensor<double, dim, dim>&,
                                          const mfem::future::tensor<double, N_INTERNAL_STATES>& internal_state) const
  {
    const double dt = 1.0;
    auto [stress, internal_state_new] = update(dt, du_dX, internal_state);
    return internal_state_new;
  }

  template <typename... Parameters>
  SMITH_HOST_DEVICE double density(Parameters...) const { return rho; }
};


double static_flow_strength(double eqps, double sigma_y, double Hi) {
  return sigma_y + Hi*eqps;
}

double flow_strength(double eqps, double dt, double eqps_old, double sigma_y, double Hi, double m, double eqps_dot_0) {
  double S = static_flow_strength(eqps, sigma_y, Hi);
  double C = std::pow(2.0, 1.0/m) - 1.0;
  double eqps_dot = (eqps - eqps_old)/dt;
  return S*(std::pow(1.0 + C*eqps_dot/eqps_dot_0, m) - 1.0);
}

double potential(double eqps, double dt, double eqps_old, double sigma_y, double Hi, double m, 
                 double eqps_dot_0) {
    const double alpha = 1.0;
    double eqps_a = (1.0 - alpha)*eqps_old + alpha*eqps;
    double eqps_dot = (eqps - eqps_old)/dt;
    double S = static_flow_strength(eqps_a, sigma_y, Hi);
    double C = std::pow(2.0, 1.0/m) - 1.0;
    double pfactor = std::pow(1.0 + C*eqps_dot/eqps_dot_0, m + 1.0) - 1.0;
    return dt*S*(eqps_dot_0/(C*(m + 1.0))*pfactor - eqps_dot);
}


using J2ResidualParams = mfem::future::tuple<double, double, double, double, double, double, double, double>;

double j2_rate_dependent_residual(double eqps, J2ResidualParams p) {
  auto [sigma_eff_trial, dt, mu, eqps_old, sigma_y, Hi, m, eqps_dot_0] = p;
  //double strength = __enzyme_fwddiff<double>((void*) potential, enzyme_dup, eqps, 1.0, enzyme_const, dt, enzyme_const, eqps_old, enzyme_const, sigma_y, enzyme_const, Hi, enzyme_const, m, enzyme_const, eqps_dot_0);
  double strength = flow_strength(eqps, dt, eqps_old, sigma_y, Hi, m, eqps_dot_0);
  return sigma_eff_trial - 3*mu*(eqps - eqps_old) - strength;
}

__attribute__((used))
void *  __enzyme_register_derivative_newton_scalar_on_j2[2] = {
  (void*) newton_scalar_impl<j2_rate_dependent_residual, J2ResidualParams>,
  (void*) newton_scalar_impl_fwddiff<j2_rate_dependent_residual, J2ResidualParams>
};

struct J2Smooth {
  static constexpr int dim = 3;  ///< spatial dimension
  static constexpr int N_INTERNAL_STATES = 10;
  static constexpr double tol = 1e-10;  ///< relative tolerance on residual mag to judge convergence of return map

  double E;        ///< Young's modulus
  double nu;       ///< Poisson's ratio
  double sigma_y;  ///< Yield strength
  double Hi;       ///< Isotropic hardening modulus
  double m;        ///< Rate sensitivity
  double edot_0;   ///< Reference plastic strain rate
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
  update(double dt, const mfem::future::tensor<double, dim, dim>& dudX,
         const mfem::future::tensor<double, N_INTERNAL_STATES>& internal_state) const
  {
    auto I = mfem::future::IdentityMatrix<dim>();
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);

    auto [plastic_strain, accumulated_plastic_strain_old] = unpack_internal_state(internal_state);
    
    auto el_strain = mfem::future::sym(dudX) - plastic_strain;
    auto p = K * tr(el_strain);
    auto s = 2.0 * G * mfem::future::dev(el_strain);
    auto q = std::sqrt(1.5) * mfem::future::norm(s);
    double denom = q > 0.0? q : 1.0;
    auto Np = 1.5 * s / denom;

    double accumulated_plastic_strain = newton_scalar<j2_rate_dependent_residual, J2ResidualParams>(accumulated_plastic_strain_old, mfem::future::make_tuple(q, dt, G, accumulated_plastic_strain_old, sigma_y, Hi, m, edot_0));
    double delta_eqps = accumulated_plastic_strain - accumulated_plastic_strain_old;
    s -= 2.0 * G * delta_eqps * Np;
    plastic_strain += delta_eqps * Np;

    auto stress = s + p * I;
    auto internal_state_new = pack_internal_state(plastic_strain, accumulated_plastic_strain);
    return mfem::future::make_tuple(stress, internal_state_new);
  }

  SMITH_HOST_DEVICE auto pkStress(double, const mfem::future::tensor<double, dim, dim>& du_dX,
                                  const mfem::future::tensor<double, dim, dim>&,
                                  const mfem::future::tensor<double, N_INTERNAL_STATES>& internal_state) const
  {
    const double dt = 1.0;
    auto [stress, internal_state_new] = update(dt, du_dX, internal_state);
    return stress;
  }

  SMITH_HOST_DEVICE auto internalStateNew(double, const mfem::future::tensor<double, dim, dim>& du_dX,
                                          const mfem::future::tensor<double, dim, dim>&,
                                          const mfem::future::tensor<double, N_INTERNAL_STATES>& internal_state) const
  {
    const double dt = 1.0;
    auto [stress, internal_state_new] = update(dt, du_dX, internal_state);
    return internal_state_new;
  }

  template <typename... Parameters>
  SMITH_HOST_DEVICE double density(Parameters...) const { return rho; }
};


/* q-function for internal state update. Located here in the test for development.
   This should be moved inside a weak form class. */
template <typename Material, typename... Parameters>
struct InternalStateQFunction {
  SMITH_HOST_DEVICE inline auto operator()(
      // mfem::real_t dt, // TODO: figure out how to pass this in
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& du_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dv_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dX_dxi,
      Parameters::QFunctionInput... params) const
  {
    auto dxi_dX = mfem::future::inv(dX_dxi);
    auto du_dX = mfem::future::dot(du_dxi, dxi_dX);
    auto dv_dX = mfem::future::dot(dv_dxi, dxi_dX);
    double dt = 1.0;  // TODO: figure out how to pass this in to the qfunction
    return mfem::future::make_tuple(material.internalStateNew(dt, du_dX, dv_dX, params...));
  }

  Material material;  ///< the material model to use for computing the stress
};

template <typename Material>
struct InternalStateVirtualWorkQFunction {
  SMITH_HOST_DEVICE inline auto operator()(
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& du_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dv_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dX_dxi,
      const mfem::future::tensor<mfem::real_t, 10>& Q,
      const mfem::future::tensor<mfem::real_t, 10>& Q_bar) const
  {
    auto dxi_dX = mfem::future::inv(dX_dxi);
    auto du_dX = mfem::future::dot(du_dxi, dxi_dX);
    auto dv_dX = mfem::future::dot(dv_dxi, dxi_dX);
    double dt = 1.0;  // TODO: figure out how to pass this in to the qfunction
    auto Q_new = material.internalStateNew(dt, du_dX, dv_dX, Q);
    return mfem::future::make_tuple(mfem::future::inner(Q_new, Q_bar));
  }

  Material material;  ///< the material model to use for computing the stress
};


class BeamMeshFixture : public testing::Test {
  public: 
    static constexpr int dim = 3;
    static constexpr double LENGTH = 8.0;
    static constexpr double DEPTH = 1.0;

  protected:
    BeamMeshFixture()
    {
      StateManager::initialize(datastore, "beam_problem");

      std::string filename = SMITH_REPO_DIR "/data/meshes/beam-hex.mesh";
      mfem::ParMesh& setMesh(std::unique_ptr<mfem::ParMesh> pmesh, const std::string& mesh_tag);
      auto mfem_mesh = mfem::Mesh::MakeCartesian3D(8, 1, 1, mfem::Element::HEXAHEDRON, LENGTH, DEPTH, DEPTH);
      mesh = std::make_shared<smith::Mesh>(std::move(mfem_mesh), "amesh", 0, 0);
      entire_domain = mesh->mfemParMesh().attributes;
      boundaries["left_end"] = 4;
      boundaries["right_end"] = 2;
    }

    axom::sidre::DataStore datastore;
    std::shared_ptr<smith::Mesh> mesh;
    mfem::Array<int> entire_domain;
    std::map<std::string, int> boundaries;
};


class DfemSolidTest : public BeamMeshFixture {
  public: 
    static constexpr int disp_order = 1;
    static constexpr int ir_order = 2;
    
    using KinematicSpace = H1<disp_order, dim>;
  
    using Material = J2Linear;
    static constexpr double E = 1.0e3;
    static constexpr double nu = 0.25;
    static constexpr double sigma_y = 9.0;
    static constexpr double Hi = 40.0;

    enum PARAMS
    {
      J2_INTERNAL_STATE
    };

    static constexpr bool use_tensor_product = false;

    enum IsvStates {COORDINATES, DISPLACEMENT, VELOCITY, INTERNAL_VARIABLES, DUAL_INTERNAL_VARIABLES};

  protected:
    DfemSolidTest() :
      mat{.E = E, .nu = nu, .sigma_y = sigma_y, .Hi = Hi, .rho = 1.0},
      disp(StateManager::newState(KinematicSpace{}, "displacement", mesh->tag())),
      velo(StateManager::newState(KinematicSpace{}, "velocity", mesh->tag())),
      accel(StateManager::newState(KinematicSpace{}, "acceleration", mesh->tag())),
      coords(StateManager::newState(KinematicSpace{}, "coordinates", mesh->tag())),
      ir(mfem::IntRules.Get(disp.space().GetFE(0)->GetGeomType(), ir_order)),
      internal_state_space(mesh->mfemParMesh(), ir, mat.N_INTERNAL_STATES, use_tensor_product),
      internal_state(internal_state_space),
      physics(DfemSolidWeakForm("plasticity", mesh, disp.space(), {&internal_state_space}, {})),
      update_internal_state(
        {mfem::future::FieldDescriptor(DfemSolidWeakForm<>::STATE::NUM_STATES, &internal_state_space)},
        {mfem::future::FieldDescriptor(DfemSolidWeakForm<>::STATE::DISPLACEMENT, &disp.space()),
        mfem::future::FieldDescriptor(DfemSolidWeakForm<>::STATE::VELOCITY, &velo.space()),
        mfem::future::FieldDescriptor(DfemSolidWeakForm<>::STATE::COORDINATES, &coords.space())},
        mesh->mfemParMesh())
    {
      coords.setFromGridFunction(static_cast<mfem::ParGridFunction&>(*mesh->mfemParMesh().GetNodes()));

      physics.setMaterial<Material, InternalVariableParameter<J2_INTERNAL_STATE, Material::N_INTERNAL_STATES>>(
        entire_domain, mat, ir);

      mfem::future::tuple<mfem::future::Gradient<DfemSolidWeakForm<>::STATE::DISPLACEMENT>,
                          mfem::future::Gradient<DfemSolidWeakForm<>::STATE::VELOCITY>,
                          mfem::future::Gradient<DfemSolidWeakForm<>::STATE::COORDINATES>,
                          mfem::future::Identity<DfemSolidWeakForm<>::STATE::NUM_STATES>> internal_state_qf_inputs{};

      mfem::future::tuple<mfem::future::Identity<DfemSolidWeakForm<>::STATE::NUM_STATES>> internal_state_qf_outputs{};

      InternalStateQFunction<Material, InternalVariableParameter<4, Material::N_INTERNAL_STATES>> update_internal_state_qf{
        .material = mat};
      // update_internal_state.DisableTensorProductStructure(); // Disabling breaks it
      update_internal_state.AddDomainIntegrator(
        update_internal_state_qf, internal_state_qf_inputs, internal_state_qf_outputs, ir,
        entire_domain, 
        std::index_sequence<DfemSolidWeakForm<>::STATE::DISPLACEMENT, DfemSolidWeakForm<>::STATE::NUM_STATES>{});
    }

    Material mat;
    FiniteElementState disp;
    FiniteElementState velo;
    FiniteElementState accel;
    FiniteElementState coords;
    const mfem::IntegrationRule ir;
    mfem::future::UniformParameterSpace internal_state_space;
    mfem::future::ParameterFunction internal_state;
    DfemSolidWeakForm<false, false> physics; //< operator to compute force residual
    mfem::future::DifferentiableOperator update_internal_state; //< operator to compute new internal state vars
};


TEST_F(DfemSolidTest, PlasticityPatchTest)
{
  // set displacement to uniaxial stress solution
  UniaxialSolution exact_solution{.E= E, .nu = nu, .sigma_y = sigma_y, .Hi = Hi};
  disp.setFromFieldFunction([exact_solution](tensor<double, dim> X) { return exact_solution.displacement(X, 1.0); });
  velo = 0.0;
  accel = 0.0;
  internal_state = 0.0;

  // Use dFEM-basedweak form to compute reactions
  double t = 1.0;
  double dt = 1.0;
  FiniteElementDual reaction = StateManager::newDual(KinematicSpace{}, "reactions", mesh->tag());
  reaction = physics.residual(t, dt, &disp, {&disp, &velo, &accel, &coords}, {&internal_state});
  
  // Get boundary dofs for computing resultants
  mfem::Array<int> bdr_attr_is_ess(mesh->mfemParMesh().bdr_attributes.Max());
  bdr_attr_is_ess = 0; // reset
  
  bdr_attr_is_ess[boundaries["left_end"]] = 1;
  mfem::Array<int> left_x_tdof;
  disp.space().GetEssentialTrueDofs(bdr_attr_is_ess, left_x_tdof, 0); // get x-dir dofs

  bdr_attr_is_ess = 0; // reset
  bdr_attr_is_ess[boundaries["right_end"]] = 1; // flag boundary 2 (1 in mesh)
  mfem::Array<int> right_x_tdof;
  disp.space().GetEssentialTrueDofs(bdr_attr_is_ess, right_x_tdof, 0); // get x-dir dofs

  double Fx = 0;
  for (auto td : left_x_tdof) {
    Fx += reaction[td];
  }
  double exact_resultant = exact_solution.axial_stress(t)*DEPTH*DEPTH;
  ASSERT_GT(std::abs(exact_resultant), 0.0); // ensure test is not trivial
  EXPECT_NEAR(Fx, exact_resultant, 1e-10);

  Fx = 0;
  for (auto td : right_x_tdof) {
    Fx += reaction[td];
  }
  EXPECT_NEAR(Fx, -exact_resultant, 1e-10);

  // make a gridFunction of the reactions for plotting
  mfem::ParGridFunction reaction_gf(&reaction.space());
  reaction.linearForm().ParallelAssemble(reaction_gf.GetTrueVector());
  reaction_gf.SetFromTrueVector();

  mfem::future::ParameterFunction internal_state_new(internal_state_space);
  update_internal_state.SetParameters({&disp, &velo, &coords});
  update_internal_state.Mult(internal_state, internal_state_new);
  
  mfem::out << "internal state data:\n";
  mfem::out << "vdim = " << internal_state_space.GetVDim() << std::endl;
  mfem::out << "dtq.nqpt = " << internal_state_space.GetDofToQuad().nqpt << std::endl;
  mfem::out << "dtq.ndof = " << internal_state_space.GetDofToQuad().ndof << std::endl;
  mfem::out << "tsize = " << internal_state_space.GetTrueVSize() << std::endl;
  mfem::out << "vsize = " << internal_state_space.GetVSize() << std::endl;
  mfem::out << "ne = " << internal_state_space.GetNE() << std::endl;
  mfem::out << "dim = " << internal_state_space.Dimension() << std::endl;
  auto pstrain = exact_solution.plastic_strain(t);
  for (int e = 0, i = 0; e < internal_state_space.GetNE(); e++) {
    for (int qp = 0; qp < ir.GetNPoints(); qp++) {
      // plastic strain tensor
      EXPECT_NEAR(internal_state_new[i + 0], pstrain[0], 1e-10);
      EXPECT_NEAR(internal_state_new[i + 4], pstrain[1], 1e-10);
      EXPECT_NEAR(internal_state_new[i + 8], pstrain[2], 1e-10);

      // EQPS
      EXPECT_NEAR(internal_state_new[i + 9], pstrain[0], 1e-10);
      i += mat.N_INTERNAL_STATES;
    }
  }

  // uncomment to view output
  //
  // mfem::ParaViewDataCollection dc("dfem_plasticity_pv", &(mesh->mfemParMesh()));
  // dc.SetHighOrderOutput(true);
  // dc.SetLevelsOfDetail(1);
  // dc.RegisterField("displacement", &disp.gridFunction());
  // dc.RegisterField("reaction", &reaction_gf);
  // // dc.RegisterQField("internal_state", &output_internal_state);
  // dc.SetCycle(0);
  // dc.Save();
}


TEST_F(DfemSolidTest, DifferentiateInternalStateUpdate)
{
  // set displacement to uniaxial stress solution
  UniaxialSolution exact_solution{.E= E, .nu = nu, .sigma_y = sigma_y, .Hi = Hi};
  disp.setFromFieldFunction([exact_solution](tensor<double, dim> X) { return exact_solution.displacement(X, 1.0); });
  velo = 0.0;
  accel = 0.0;
  internal_state = 0.0;

  double t = 1.0;
  double dt = 1.0;

  mfem::future::ParameterFunction internal_state_new(internal_state_space);
  update_internal_state.SetParameters({&disp, &velo, &coords});
  update_internal_state.Mult(internal_state, internal_state_new);

  //
  // VJP of internal state update
  //
  std::vector<mfem::Vector*> primals_l{&internal_state};
  std::vector<FiniteElementState*> fields{&disp, &velo, &coords};
  std::vector<mfem::Vector*> params_l;
  params_l.reserve(fields.size());
  for (size_t i = 0; i < fields.size(); ++i) {
    params_l.push_back(&fields[i]->gridFunction());
  }
  auto derivative_taker = update_internal_state.GetDerivative(0, primals_l, params_l);
  FiniteElementState du = StateManager::newState(KinematicSpace{}, "tangent_disp", mesh->tag());
  du.Randomize(0);
  du *= 0.001;
  mfem::future::ParameterFunction dQ(internal_state_space);
  derivative_taker->Mult(du, dQ);

  // check with directional finite difference
  double fd_eps = 1e-5;
  FiniteElementState disp_p = StateManager::newState(KinematicSpace{}, "displacement_perturbed", mesh->tag());
  disp_p = disp;
  disp_p.Add(fd_eps, du);
  update_internal_state.SetParameters({&disp_p, &velo, &coords});
  mfem::future::ParameterFunction Qnew_p(internal_state_space);
  update_internal_state.Mult(internal_state, Qnew_p);
  mfem::future::ParameterFunction dQ_h(Qnew_p);
  dQ_h.Add(-1.0, internal_state_new);
  dQ_h *= 1.0 / fd_eps;

  mfem::future::ParameterFunction error(internal_state_space);
  error.Set(1.0, dQ_h);
  error.Add(-1.0, dQ);
  mfem::out << "error norm = " << error.Norml2() << std::endl;
  EXPECT_LT(error.Norml2(), 1e-4*internal_state_new.Norml2());

  //
  // VJP of internal state update
  //
  

  mfem::future::ParameterFunction Q_bar(internal_state_space);
  const int rand_seed = 1;
  Q_bar.Randomize(rand_seed);

  mfem::future::DifferentiableOperator update_internal_state_virtual_work(
      {mfem::future::FieldDescriptor(IsvStates::INTERNAL_VARIABLES, &internal_state_space)},
      {mfem::future::FieldDescriptor(IsvStates::DISPLACEMENT, &disp.space()),
       mfem::future::FieldDescriptor(IsvStates::VELOCITY, &velo.space()),
       mfem::future::FieldDescriptor(IsvStates::COORDINATES, &coords.space()),
       mfem::future::FieldDescriptor(IsvStates::DUAL_INTERNAL_VARIABLES, &internal_state_space)},
      mesh->mfemParMesh());
  update_internal_state_virtual_work.DisableTensorProductStructure();

  mfem::future::tuple<mfem::future::Gradient<IsvStates::DISPLACEMENT>,
                      mfem::future::Gradient<IsvStates::VELOCITY>,
                      mfem::future::Gradient<IsvStates::COORDINATES>,
                      mfem::future::Identity<IsvStates::INTERNAL_VARIABLES>,
                      mfem::future::Identity<IsvStates::DUAL_INTERNAL_VARIABLES>>
      update_internal_state_virtual_work_qf_inputs{};
  mfem::future::tuple<mfem::future::Sum<IsvStates::INTERNAL_VARIABLES>>
      update_internal_state_virtual_work_outputs{};
  InternalStateVirtualWorkQFunction<Material>
      update_internal_state_virtual_work_qf{.material = mat};
  update_internal_state_virtual_work.AddDomainIntegrator(
      update_internal_state_virtual_work_qf, update_internal_state_virtual_work_qf_inputs,
      update_internal_state_virtual_work_outputs, ir, entire_domain, std::index_sequence<IsvStates::INTERNAL_VARIABLES, IsvStates::DISPLACEMENT>{});

  update_internal_state_virtual_work.SetParameters({&disp, &velo, &coords, &Q_bar});

  params_l.push_back(&Q_bar);

  // For reverse mode, we should be able to ask for ALL upstream derivatives at once,
  // but dFEM makes you choose one upstream.
  // (This is likely because it is computing the full Jacobian wrt that variable at quad points)
  auto Q_vjp_u = update_internal_state_virtual_work.GetDerivative(IsvStates::DISPLACEMENT, primals_l, params_l);

  FiniteElementDual u_bar = StateManager::newDual(KinematicSpace{}, "u_bar", mesh->tag());

  // This is the code we want to write, but MultTranspose is not implemented yet.
#if 0
  mfem::Vector seed(1);
  seed = 1.0;
  Q_vjp_u->MultTranspose(seed, u_bar);
#endif
  
  // Instead, we have to do this until MultTranspose is implmented
  mfem::Vector direction(disp.Size());
  direction = 0.0;
  mfem::Vector u_bar_i(1);
  for (int i = 0; i < disp.Size(); i++) {
    direction[i] = 1.0;
    Q_vjp_u->Mult(direction, u_bar_i);
    u_bar[i] = u_bar_i[0];
    direction[i] = 0.0;
  }

  // TODO: make above work in parallel
  mfem::out << "u_bar = " << std::endl;
  u_bar.Print();
}

using Tensor3D = mfem::future::tensor<mfem::real_t, 3, 3>;
using InternalVars = mfem::future::tensor<mfem::real_t, 10>;

double transverse_stress(double e22, mfem::future::tuple<J2Smooth*, double, Tensor3D, Tensor3D, InternalVars> p) {
  auto material = *mfem::future::get<0>(p);
  auto dt = mfem::future::get<1>(p);
  auto du_dX = mfem::future::get<2>(p);
  auto dv_dX = mfem::future::get<3>(p);
  auto Q = mfem::future::get<4>(p);
  du_dX[1][1] = du_dX[2][2] = e22;
  auto sigma = material.pkStress(dt, du_dX, dv_dX, Q);
  return sigma[1][1];
}


TEST(EnzymeMaterial, Stress) {
  J2Smooth material{.E = 70.0e3, .nu=0.34, .sigma_y = 240.0, .Hi = 1750.0, .m = 0.25, .edot_0 = 250e-6, .rho = 1.0};
  Tensor3D disp_grad{};
  Tensor3D vel_grad{};
  InternalVars q{}, q_old{};
  std::ofstream stress_file("stress_strain.csv");
  stress_file << 0.0 << " " << 0.0 << " " << 0.0 << "\n";
  const double strain_inc = 250e-6;
  double dt = 1.0;
  for (int i = 0; i < 200; i++) {
    q_old = q;
    disp_grad[0][0] += strain_inc;
    double e22 = newton_scalar<transverse_stress>(disp_grad[1][1], mfem::future::make_tuple(&material, dt, disp_grad, vel_grad, q_old));
    disp_grad[1][1] = disp_grad[2][2] = e22;
    auto [sigma, q_new] = material.update(dt, disp_grad, q_old);
    q = q_new;
    stress_file << disp_grad[0][0] << " " << sigma[0][0] << " " << sigma[1][1] << "\n";
  }
  stress_file.close();
}


Tensor3D compute_stress(J2Smooth* mat, Tensor3D du_dX, InternalVars Q) {
  return mat->pkStress(1.0, du_dX, Tensor3D{}, Q);
}

mfem::future::tensor<double, 3, 3, 3, 3> compute_tangent(J2Smooth* mat, Tensor3D du_dX, InternalVars Q) {
  mfem::future::tensor<double, 3, 3, 3, 3> C;
  for (int k = 0; k < 3; k++) {
    for (int l = 0; l < 3; l++) {
      Tensor3D direction{};
      direction[k][l] = 1.0;
      auto ds = __enzyme_fwddiff<Tensor3D>((void*)compute_stress, enzyme_const, mat, enzyme_dup, du_dX, direction, enzyme_dup, Q, InternalVars{});
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          C[i][j][k][l] = ds[i][j];
        }
      }
    }
  }
  return C;
}


TEST(J2Smooth, FDTangent) {
  // compare Elastic-plastic tangents computed with Enzyme with finite differences
  Tensor3D H{{{0.05, 0, 0}, {0, -0.0242778, 0}, {0, 0, -0.0242778}}};
  InternalVars q_old{{0.0452424, 0, 0, 0, -0.0226212, 0, 0, 0, -0.0226212, 0.0452424}};
  J2Smooth material{.E = 70.0e3, .nu=0.34, .sigma_y = 240.0, .Hi = 1750.0, .m = 0.25, .edot_0 = 250e-6, .rho = 1.0};
  double dt = 1.0;
  auto stress = material.pkStress(dt, H, Tensor3D{}, q_old);

  auto A = compute_tangent(&material, H, q_old);
  // mfem::out << "Enzyme tangent = \n" << A << std::endl;

  mfem::future::tensor<double, 3, 3, 3, 3> A_h;
  double epsilon = -1e-5;
  for (int k = 0; k < 3; k++) {
    for (int l = 0; l < 3; l++) {
      Tensor3D dH{};
      dH[k][l] = 1.0;
      auto stress_p = material.pkStress(dt, H + epsilon*dH, Tensor3D{}, q_old);
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          A_h[i][j][k][l] = (stress_p[i][j] - stress[i][j])/epsilon;
          EXPECT_NEAR(A_h[i][j][k][l], A[i][j][k][l], 1e-4*material.E);
        }
      }
    } 
  }
  // mfem::out << "finite difference tangent = \n" << A_h << std::endl;

  // double lm = (A[8][4]*A[0][8] - A[0][4]*A[8][8])/(A[4][4]*A[8][8] - A[8][4]*A[4][8]);
  // double uniaxial_mod = A[0][0] + 2*lm*A[4][0];
  // mfem::out << "Uniaxial stress tangent modulus = " << uniaxial_mod << std::endl;
}


}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}

// Questions for dFEM team:
// For reverse mode, it doesn't make sense to ask for a single upstream variable.
// We don't want to re-compute all the quadrature point jacobians to get upstream derivatives for another variable.
