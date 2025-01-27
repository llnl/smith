// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file liquid_crystal_elastomer.hpp
 *
 * @brief Brighenti's constitutive model for liquid crystal elastomers
 * Cosma, M. P., & Brighenti, R. (2022). Controlled morphing of architected liquid crystal elastomer elements:
 *  modeling and simulations. Mechanics Research Communications, 121, 103858.
 *
 * see https://doi.org/10.1016/j.ijsolstr.2021.02.023
 */

#pragma once

#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/dual.hpp"
#include "serac/physics/materials/solid_material.hpp"

namespace serac {

/**
 * @brief Brighenti's liquid crystal elastomer model
 */
struct LiquidCrystElastomerBrighenti {
  /// this model is only intended to be used in 3D
  static constexpr int dim = 3;

  /// internal variables for the liquid crystal elastomer model
  struct State {
    tensor<double, dim, dim> deformation_gradient;  ///< F from the last timestep
    tensor<double, dim, dim> distribution_tensor;   ///< mu from the last timestep
    double temperature;                             ///< temperature at the last timestep
  };

  /**
   * @brief Constructor
   *
   * @param rho mass density of the material (in reference configuration)
   * @param shear_modulus Shear modulus of the material
   * @param bulk_modulus Bulk modulus of the material
   * @param order_constant temperature-valued constant in exponential factor for order parameter
   * @param order_parameter Initial value of the order parameter
   * @param transition_temperature Characteristic temperature of the order-disorder transition
   * @param N_b_squared Number of Kunh segments/chain, times square of Kuhn segment length
   */
  LiquidCrystElastomerBrighenti(double rho, double shear_modulus, double bulk_modulus, double order_constant,
                                double order_parameter, double transition_temperature, double N_b_squared)
      : density(rho),
        shear_modulus_(shear_modulus),
        bulk_modulus_(bulk_modulus),
        order_constant_(order_constant),
        initial_order_parameter_(order_parameter),
        transition_temperature_(transition_temperature),
        N_b_squared_(N_b_squared)
  {
    SLIC_ERROR_ROOT_IF(density <= 0.0, "Density must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(shear_modulus_ <= 0.0, "Shear modulus must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(bulk_modulus_ <= 0.0, "Bulk modulus must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(order_constant_ <= 0.0, "Order constant must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(transition_temperature_ <= 0.0,
                       "The transition temperature must be positive in the LCE material model.");
  }

  /**
   * @brief Material response
   *
   * @tparam DispGradType number-like type for the displacement gradient tensor
   * @tparam OrderParamType number-like type for the order parameter
   * @tparam GammaAngleType number-like type for the orientation angle gamma
   *
   * @param[in,out] state A state variable object for this material. The value is updated in place.
   * @param[in] displacement_grad displacement gradient with respect to the reference configuration
   * @param[in] temperature_tuple the temperature
   * @param[in] gamma_tuple the first polar angle used to define the liquid crystal orientation vector
   * @return The calculated material response (Cauchy stress) for the material
   */
  template <typename DispGradType, typename OrderParamType, typename GammaAngleType>
  SERAC_HOST_DEVICE auto operator()(State& state, const tensor<DispGradType, dim, dim>& displacement_grad,
                                    OrderParamType temperature_tuple, GammaAngleType gamma_tuple) const
  {
    using std::cos, std::sin;

    // get the values from the packed value/gradient tuples
    auto temperature = get<0>(temperature_tuple);
    auto gamma = get<0>(gamma_tuple);

    auto I = Identity<dim>();
    double q0 = initial_order_parameter_;
    tensor normal{{cos(gamma), sin(gamma), 0.0 * gamma}};

    if (norm(state.deformation_gradient) == 0) {
      state.distribution_tensor = get_value((N_b_squared_ / 3.0) * ((1 - q0) * I) + (3 * q0 * outer(normal, normal)));
      state.deformation_gradient = get_value(displacement_grad) + I;
      state.temperature = get_value(temperature);
    }

    // kinematics
    auto F = displacement_grad + I;
    auto F_old = state.deformation_gradient;
    auto F_hat = dot(F, inv(F_old));
    auto J = det(F);

    // Distribution tensor function of nematic order tensor
    auto mu = calculateDistributionTensor(state, F_hat, temperature, normal);
    auto mu0 = (N_b_squared_ / 3.0) * ((1 - q0) * I) + (3 * q0 * outer(normal, normal));

    // stress output
    // Note to Jorge-Luis: the paper is omits a prefactor of 1/J in the
    // Cauchy stress equation because they assume J = 1 strictly
    // (the incompressible limit). It needs to be retained for this
    // compressible model.
    const double lambda = bulk_modulus_ - (2.0 / 3.0) * shear_modulus_;
    using std::log;
    auto stress = ((3 * shear_modulus_ / N_b_squared_) * (mu - mu0) + lambda * log(J) * I) / J;

    // update state variables
    state.deformation_gradient = get_value(F);
    state.temperature = get_value(temperature);
    state.distribution_tensor = get_value(mu);

    return stress;
  }

  /// -------------------------------------------------------

  /**
   * @brief Compute the distribution tensor using Brighenti's model
   *
   * @tparam S Type of the deformation gradient
   * @tparam T Type of angle of the liquid crystals
   * @tparam U Type of the unit length normal of liquid crystal alignment (first-order tensor)
   *
   * @param state State variables for this material
   * @param F_hat Dot product of the current and previous deformation gradients
   * @param theta In-plane angle of the liquid crystals in the elastomer matrix
   * @param normal Unit length normal of angle alignment
   *
   * @return distribution tensor
   */
  template <typename S, typename T, typename U>
  auto calculateDistributionTensor(const State& state, const tensor<S, dim, dim>& F_hat, const T theta,
                                   const tensor<U, dim>& normal) const
  {
    // Nematic order scalar
    using std::exp;
    auto theta_old = state.temperature;
    auto q_old = initial_order_parameter_ / (1 + exp((theta_old - transition_temperature_) / order_constant_));
    auto q = initial_order_parameter_ / (1 + exp((theta - transition_temperature_) / order_constant_));

    // Nematic order tensor
    constexpr auto I = Identity<dim>();
    auto n_dyad = outer(normal, normal);
    // BT: These are different than what Jorge-Luis had. I found the papers
    // to be confusing on this point. I'm extrapolating from Eq (7)
    // in https://doi.org/10.1016/j.mechrescom.2022.103858
    // Well-defined validation problems would help to confirm.
    auto Q_old = 0.5 * ((1.0 - q_old) * I + 3.0 * q_old * n_dyad);
    auto Q = 0.5 * ((1.0 - q) * I + 3.0 * q * n_dyad);

    // Polar decomposition of incremental deformation gradient
    auto U_hat = matrix_sqrt(transpose(F_hat) * F_hat);
    auto R_hat = F_hat * inv(U_hat);

    // Distribution tensor (using 'Strang Splitting' approach)
    double alpha = 2.0 * N_b_squared_ / 3.0;
    auto mu_old = state.distribution_tensor;
    auto mu_hat = mu_old + alpha * (Q - Q_old);
    auto mu_a = dot(F_hat, dot(mu_hat, transpose(F_hat)));
    auto mu_b = alpha * (Q - dot(R_hat, dot(Q, transpose(R_hat))));

    return mu_a + mu_b;
  }

  /// -------------------------------------------------------

  // Sam: please forgive some of the tautological
  // explanations below, I'm not knowledgeable enough
  // about this model to write meaningful descriptions,
  // so these placeholders really only exist to satisfy
  // our doxygen requirements
  //
  // suggestions are welcome

  double density;                   ///<  mass density
  double shear_modulus_;            ///< shear modulus in stress-free configuration
  double bulk_modulus_;             ///< bulk modulus in stress-free configuration
  double order_constant_;           ///< Order constant
  double initial_order_parameter_;  ///< initial value of order parameter
  double transition_temperature_;   ///< Transition temperature

  // BT: I think this can be removed - it looks like it cancels out every place it appears.
  double N_b_squared_;  ///< Kuhn segment parameters.
};

/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------

/**
 * @brief Bertoldi's liquid crystal elastomer model
 * Paper: Li, S., Librandi, G., Yao, Y., Richard, A. J., Schneider‐Yamamura, A., Aizenberg, J., & Bertoldi, K. (2021).
 * Controlling Liquid Crystal Orientations for Programmable Anisotropic Transformations in Cellular Microstructures.
 * Advanced Materials, 33(42), 2105024.
 */
struct LiquidCrystalElastomerBertoldi {
  using State = Empty;  ///< this material has no internal variables

  /// this model is only intended to be used in 3D
  static constexpr int dim = 3;

  /**
   * @brief Constructor
   *
   * @param rho mass density of the material (in reference configuration)
   * @param young_modulus Bulk modulus of the material
   * @param poisson_ratio Poisson ratio of the material
   * @param initial_order_parameter Initial value of the order parameter
   * @param beta_parameter Parameter for degree of coupling between elastic and nematic energies
   */
  LiquidCrystalElastomerBertoldi(double rho, double young_modulus, double poisson_ratio, double initial_order_parameter,
                                 double beta_parameter)
      : density(rho),
        young_modulus_(young_modulus),
        poisson_ratio_(poisson_ratio),
        initial_order_parameter_(initial_order_parameter),
        beta_parameter_(beta_parameter)
  {
    SLIC_ERROR_ROOT_IF(density <= 0.0, "Density must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(young_modulus_ <= 0.0, "Bulk modulus must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(poisson_ratio_ <= 0.0, "Poisson ratio must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(initial_order_parameter_ <= 0.0,
                       "Initial order parameter must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(beta_parameter_ <= 0.0, "The beta parameter must be positive in the LCE material model.");
  }

  /// -------------------------------------------------------

  /**
   * @brief Material response
   *
   * @tparam DispGradType number-like type for the displacement gradient tensor
   * @tparam OrderParamType number-like type for the order parameter
   * @tparam GammaAngleType number-like type for the orientation angle gamma
   * @tparam EtaAngleType number-like type for the orientation angle eta
   *
   * @param[in] displacement_grad displacement gradient with respect to the reference configuration
   * @param[in] inst_order_param_tuple the current order parameter
   * @param[in] gamma_tuple the first polar angle used to define the liquid crystal orientation vector
   * @param[in] eta_tuple the second polar angle used to define the liquid crystal orientation vector
   * @return The calculated material response (Cauchy stress) for the material
   */
  template <typename DispGradType, typename OrderParamType, typename GammaAngleType, typename EtaAngleType>
  SERAC_HOST_DEVICE auto operator()(State& /*state*/, const tensor<DispGradType, dim, dim>& displacement_grad,
                                    OrderParamType inst_order_param_tuple, GammaAngleType gamma_tuple,
                                    EtaAngleType eta_tuple) const
  {
    using std::cos, std::sin;

    // Compute the normal
    auto gamma = get<0>(gamma_tuple);
    auto eta = get<0>(eta_tuple);
    tensor normal{{cos(gamma) * cos(eta), sin(gamma) * cos(eta), 0.0 * gamma + sin(eta)}};

    // Get order parameters
    auto St = get<0>(inst_order_param_tuple);
    double S0 = initial_order_parameter_;

    const double lambda = poisson_ratio_ * young_modulus_ / (1.0 + poisson_ratio_) / (1.0 - 2.0 * poisson_ratio_);
    const double mu = young_modulus_ / 2.0 / (1.0 + poisson_ratio_);

    // kinematics
    auto I = Identity<dim>();
    auto F = displacement_grad + I;
    auto C = dot(transpose(F), F);
    auto E = 0.5 * (C - I);
    auto J = det(F);

    // Compute the second Piola-Kirchhoff stress, i.e., \partial strain_energy / \partial E
    auto S_stress_1 = lambda * tr(E) * I;
    auto S_stress_2 = 2 * mu * E;
    auto S_stress_3 = -0.5 * beta_parameter_ * (St - S0) * (3 * outer(normal, normal) - I);
// std::cout<< ".... young_modulus_ = "<< young_modulus_ << std::endl;
// std::cout<< ".... poisson_ratio_ = "<< poisson_ratio_ << std::endl;
// std::cout<< ".... beta_parameter_ = "<< beta_parameter_ << std::endl;
// std::cout<< ".... normal = "<< normal[0] << std::endl;
// std::cout<< ".... St = "<< St << " .... S0 = "<< S0 << std::endl;
// std::cout<< ".... S_stress_1 = " << std::endl;
// std::cout<<S_stress_1;
// std::cout<< ".... S_stress_2 = " << std::endl;
// std::cout<<S_stress_2;
// std::cout<< ".... S_stress_3 = " << std::endl;
// std::cout<<S_stress_3;
// std::cout<< std::endl;
// exit(0);
    // transform from second Piola-Lichhoff to Cauchy stress
    auto stress = 1.0 / J * F * (S_stress_1 + S_stress_2 + S_stress_3) * transpose(F);

    return stress;
  }

  /// -------------------------------------------------------

  /**
   * @brief Compute the strain energy
   *
   * @tparam DispGradType Type of the displacement gradient
   * @tparam orderParamType Type of order parameter (level of alignment)
   * @tparam GammaAngleType Type of the in-plane angle
   * @tparam EtaAngleType Type of the out-of-plane angle
   *
   * @param[in] displacement_grad Displacement gradient
   * @param[in] inst_order_param_tuple Instantaneous order parameter
   * @param[in] gamma_tuple In-plane angle of alignment of liquid crystal in elastomer matrix
   * @param[in] eta_tuple Out-of-plane angle of alignment of liquid crystal in elastomer matrix
   *
   * @return strain energy
   */
  template <typename DispGradType, typename orderParamType, typename GammaAngleType, typename EtaAngleType>
  auto calculateStrainEnergy(const State& /*state*/, const tensor<DispGradType, dim, dim>& displacement_grad,
                             orderParamType inst_order_param_tuple, GammaAngleType gamma_tuple,
                             EtaAngleType eta_tuple) const
  {
    using std::cos, std::sin;

    // Compute the normal
    auto gamma = get<0>(gamma_tuple);
    auto eta = get<0>(eta_tuple);
    tensor normal{{cos(gamma) * cos(eta), sin(gamma) * cos(eta), 0.0 * gamma + sin(eta)}};

    // Get order parameters
    auto St = get<0>(inst_order_param_tuple);
    double S0 = initial_order_parameter_;

    const double lambda = poisson_ratio_ * young_modulus_ / (1.0 + poisson_ratio_) / (1.0 - 2.0 * poisson_ratio_);
    const double mu = young_modulus_ / 2.0 / (1.0 + poisson_ratio_);

    // kinematics
    auto I = Identity<dim>();
    auto F = displacement_grad + I;
    auto C = dot(transpose(F), F);
    auto E = 0.5 * (C - I);

    // Compute the second Piola-Kirchhoff stress, i.e., \partial strain_energy / \partial E
    auto strain_energy_1 = 0.5 * lambda * tr(E) * tr(E);
    auto strain_energy_2 = mu * inner(E, E);
    auto strain_energy_3 = -0.5 * beta_parameter_ * (St - S0) * inner(3 * outer(normal, normal) - I, E);

    auto strain_energy = strain_energy_1 + strain_energy_2 + strain_energy_3;

    return strain_energy;
  }

  double density;                   ///<  mass density
  double young_modulus_;            ///< Young's modulus in stress-free configuration
  double poisson_ratio_;            ///< poisson's ratio
  double initial_order_parameter_;  ///< initial value of order parameter
  double beta_parameter_;           ///< Degree of coupling between elastic and nematic energies
};

/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------
/// -----------------------------------------------------------------------------

/**
 * @brief Zhang's liquid crystal elastomer model
 * Paper: Li, W., & Zhang, X. S. (2023). Arbitrary curvature programming of thermo-active liquid 
 * crystal elastomer via topology optimization. Computer Methods in Applied Mechanics and 
 * Engineering, 417, 116393..
 */
struct LiquidCrystalElastomerZhang {
  using State = Empty;  ///< this material has no internal variables

  /// this model is only intended to be used in 3D
  static constexpr int dim = 3;

  /**
   * @brief Constructor
   *
   * @param rho mass density of the material (in reference configuration)
   * @param shear_mod Shear modulus of the material
   * @param initial_order_param Parameter for degree of coupling between elastic and nematic energies
   * @param omega_param Parameter for degree of swelling
   * @param bulk_mod Bulk modulus of the material
   */
  LiquidCrystalElastomerZhang(double rho, double shear_mod, double initial_order_param, 
                              double omega_param, double bulk_mod)
      : density(rho),
        shear_mod_(shear_mod),
        initial_order_param_(initial_order_param),
        omega_param_(omega_param),
        bulk_mod_(bulk_mod)
  {
    SLIC_ERROR_ROOT_IF(density <= 0.0, "Density must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(shear_mod_ <= 0.0, "Shear modulus must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(initial_order_param_ <= 0.0, "Initial order parameter must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(omega_param_ <= 0.0, "Omega parameter must be positive in the LCE material model.");
    SLIC_ERROR_ROOT_IF(bulk_mod_ <= 0.0, "Bulk modulus must be positive in the LCE material model.");
  }

  /// -------------------------------------------------------

  /**
   * @brief Material response
   *
   * @tparam DispGradType number-like type for the displacement gradient tensor
   * @tparam OrderParamType number-like type for the order parameter
   * @tparam GammaAngleType number-like type for the orientation angle gamma
   * @tparam EtaAngleType number-like type for the orientation angle eta
   *
   * @param[in] displacement_grad displacement gradient with respect to the reference configuration
   * @param[in] inst_order_param_tuple the current order parameter
   * @param[in] gamma_tuple the first polar angle used to define the liquid crystal orientation vector
   * @param[in] eta_tuple the second polar angle used to define the liquid crystal orientation vector
   * @return The calculated material response (Cauchy stress) for the material
   */
  template <typename DispGradType, typename OrderParamType, typename GammaAngleType, typename EtaAngleType>
  SERAC_HOST_DEVICE auto operator()(State& /*state*/, const tensor<DispGradType, dim, dim>& displacement_grad,
                                    OrderParamType inst_order_param_tuple, GammaAngleType gamma_tuple,
                                    EtaAngleType eta_tuple) const
  {
    using std::cos, std::sin, std::pow, std::log;

    // Compute the normal
    auto   gamma = get<0>(gamma_tuple);
    auto   eta   = get<0>(eta_tuple);
    tensor normal{{cos(gamma) * cos(eta), sin(gamma) * cos(eta), 0.0 * gamma + sin(eta)}};

    // Get order parameters
    auto   St = get<0>(inst_order_param_tuple);
    const double S0 = initial_order_param_;
    const double mu   = shear_mod_;
    const double bulk = bulk_mod_;
    const double omega = omega_param_;

    // Kinematics
    auto I    = Identity<dim>();
    auto F    = displacement_grad + I;
    auto Finv = inv(F);
    auto J    = det(F);
    auto Fbar = pow(J, -1.0/3.0) * F;
    auto FbarInv = inv(Fbar);
    auto Cbar = dot(transpose(Fbar), Fbar);

  // Compute partial derivatives of strain energy wrt Cbar
    auto dWLCEdCbar_1 = mu/2.0*(1.0-S0)/(1.0-St) * I;
    auto dWLCEdCbar_2 = mu/2.0*((3.0*S0)/(1.0+2.0*St) - omega) * outer(normal, normal);
    auto dWLCEdCbar_3 = -mu/2.0*((3.0*St)*(1.0-S0)/(1.0-St)/(1.0+2.0*St) - omega) / dot(normal, dot(Cbar,normal)) * 
                    ( outer(normal, dot(Cbar,normal)) + 
                      outer(dot(transpose(Cbar),normal), normal) - 
                      inner(normal, dot(dot(Cbar,Cbar), normal)) * outer(normal, normal)/dot(normal, dot(Cbar,normal)) 
                    );

    auto dWLCEdCbar = dWLCEdCbar_1 + dWLCEdCbar_2 + dWLCEdCbar_3;

    // Compute partial derivative os Cbar wrt strain
    auto dCbardF = make_tensor<3, 3, 3, 3>([&](auto i, auto j, auto m, auto n) {
        return pow(J, -1.0/3.0) * ( Fbar(m,j)*I(i,n) + Fbar(m,i)*I(j,n) - 2.0/3.0*Cbar(i,j)*FbarInv(n,m));
      });

    // Compute strain energy derivatives wrt strain (PK stress)
    auto dWLCEdF = make_tensor<3, 3>([&](auto m, auto n) {
        auto temp = 0.0 * dWLCEdCbar(0,0) * dCbardF(0,0,0,0);
        for ( int i=0; i<3; i++)
          for ( int j=0; j<3; j++) {
            temp = temp + dWLCEdCbar(i,j) * dCbardF(i,j,m,n); 
          }
        return temp;
      });

    // Compute incompressibility component
    auto dWIncdF = bulk*log(J)*transpose(Finv) + 0.0*dWLCEdF ;

    // Compute total dW/dF
    auto dWdF  = dWLCEdF + dWIncdF;

    // Transform from first Piola-Kirchhoff (dWdF) to Cauchy stress
    return 1.0 / J * dWdF* transpose(F);
  }

  /// -------------------------------------------------------

  /**
   * @brief Compute the strain energy
   *
   * @tparam DispGradType Type of the displacement gradient
   * @tparam orderParamType Type of order parameter (level of alignment)
   * @tparam GammaAngleType Type of the in-plane angle
   * @tparam EtaAngleType Type of the out-of-plane angle
   *
   * @param[in] displacement_grad Displacement gradient
   * @param[in] inst_order_param_tuple Instantaneous order parameter
   * @param[in] gamma_tuple In-plane angle of alignment of liquid crystal in elastomer matrix
   * @param[in] eta_tuple Out-of-plane angle of alignment of liquid crystal in elastomer matrix
   *
   * @return strain energy
   */
  template <typename DispGradType, typename orderParamType, typename GammaAngleType, typename EtaAngleType>
  auto calculateStrainEnergy(const State& /*state*/, const tensor<DispGradType, dim, dim>& displacement_grad,
                             orderParamType inst_order_param_tuple, GammaAngleType gamma_tuple,
                             EtaAngleType eta_tuple) const
  {
    using std::cos, std::sin, std::pow, std::log;

    // Compute the normal
    auto   gamma = get<0>(gamma_tuple);
    auto   eta   = get<0>(eta_tuple);
    tensor normal{{cos(gamma) * cos(eta), sin(gamma) * cos(eta), 0.0 * gamma + sin(eta)}};

    // Get order parameters
    auto   St = get<0>(inst_order_param_tuple);
    const double S0 = initial_order_param_;
    const double mu   = shear_mod_;
    const double bulk = bulk_mod_;
    const double omega = omega_param_;

    // kinematics
    auto I    = Identity<dim>();
    auto F    = displacement_grad + I;
    auto J    = det(F);
    auto Fbar = pow(J, -1.0/3.0) * F;
    auto Cbar = dot(transpose(Fbar), Fbar);

    // Compute the strain_energy
    auto W_1 = mu/2.0 * ( (1.0-S0)/(1.0-St) * tr(Cbar) + 0.0*inner(normal, dot(Cbar,normal)) );
    auto W_2 = mu/2.0 * ( (3.0*S0)/(1.0+2.0*St) - omega ) * inner(normal, dot(Cbar,normal));
    auto W_3 = -mu/2.0 * ((3.0*St)*(1.0-S0)/(1.0-St)/(1.0+2.0*St) - omega) * 
                    ( inner(normal, dot(dot(Cbar,Cbar), normal)) / inner(normal, dot(Cbar,normal)) );
    auto W_Inc = bulk/2.0*log(J)*log(J) + 0.0*W_2 ;

    auto W = W_1 + W_2 + W_3 + W_Inc;

    return W;
  }

  double density;               ///<  mass density
  double shear_mod_;            ///< Shear modulus in stress-free configuration
  double initial_order_param_;  ///< initial value of order parameter
  double omega_param_;          ///< Degree of coupling between elastic and nematic energies
  double bulk_mod_;             ///< Bulk modulus in stress-free configuration
};

}  // namespace serac
