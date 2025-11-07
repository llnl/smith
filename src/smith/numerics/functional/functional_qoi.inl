// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file functional_qoi.inl
 *
 * @brief a specialization of smith::Functional for quantities of interest
 */

namespace smith {

/**
 * @brief this class behaves like a Prolongation operator, except is specialized for
 * the case of a quantity of interest. The action of its MultTranspose() operator (the
 * only thing it is used for) sums the values from different processors.
 */
struct QoIProlongation {
  QoIProlongation() {}

  /// @brief create a QoIProlongation for a Quantity of Interest
  QoIProlongation(MPI_Comm c) : comm(c) {}

  /// @brief unimplemented: do not use
  void Mult(const mfem::Vector&, mfem::Vector&) const { SLIC_ERROR_ROOT("QoIProlongation::Mult() is not defined"); }

  /// @brief set the value of output to the distributed sum over input values from different processors
  void MultTranspose(const mfem::Vector& input, mfem::Vector& output) const
  {
    // const_cast to work around clang@14.0.6 compiler error:
    //   "argument type 'const double *' doesn't match specified 'MPI' type tag that requires 'double *'"
    MPI_Allreduce(const_cast<double*>(&input[0]), &output[0], 1, MPI_DOUBLE, MPI_SUM, comm);
  }

  MPI_Comm comm;  ///< MPI communicator used to carry out the distributed reduction
};

/**
 * @brief this class behaves like a Restriction operator, except is specialized for
 * the case of a quantity of interest. The action of its ScatterAdd() operator (the
 * only thing it is used for) sums the values on this local processor.
 */
struct QoIElementRestriction {
  /**
   * @brief element-to-global ScatterAdd operation used in FEM assembly, for quantities of interest
   *
   * @param input the values from each element
   * @param output the total of those elemental values
   */
  void ScatterAdd(const mfem::Vector& input, mfem::Vector& output) const { output[0] += input.Sum(); }
};

/**
 * @brief a partial template specialization of Functional with test == double, implying "quantity of interest"
 */
template <typename... trials, ExecutionSpace exec>
class Functional<double(trials...), exec> {
  using test = QOI;
  static constexpr tuple<trials...> trial_spaces{};
  static constexpr uint32_t num_trial_spaces = sizeof...(trials);
  static constexpr auto Q = std::max({test::order, trials::order...}) + 1;

  class Gradient;

  // clang-format off
  template <uint32_t i>
  struct operator_paren_return {
    using type = typename std::conditional<
        i == NO_DIFFERENTIATION,          // if `i` is greater than or equal to zero,
        double,                           // wise, we just return the value
        smith::tuple<double&, Gradient&>  // otherwise, we return the value and the derivative w.r.t arg `i`
        >::type;
  };
  // clang-format on

 public:
  /**
   * @brief Constructs using a @p mfem::ParFiniteElementSpace object corresponding to the trial space
   * @param[in] trial_fes The trial space
   */
  Functional(std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_fes)
      : test_fec_(0, trial_fes[0]->GetMesh()->Dimension()),
        test_space_(dynamic_cast<mfem::ParMesh*>(trial_fes[0]->GetMesh()), &test_fec_, 1, smith::ordering),
        trial_space_(trial_fes),
        mem_type(mfem::Device::GetMemoryType())
  {
    SMITH_MARK_FUNCTION;

    std::array<Family, num_trial_spaces> trial_families = {trials::family...};
    std::array<int, num_trial_spaces> trial_orders = {trials::order...};
    std::array<int, num_trial_spaces> trial_components = {trials::components...};

    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      trial_function_spaces_[i] = {trial_families[i], trial_orders[i], trial_components[i]};

      P_trial_[i] = trial_space_[i]->GetProlongationMatrix();

      // For L2 spaces, configure a ParGridFunction with external data to exchange face neighbor data.
      // This is for DG method where interior faces on the processor boundary need to access data on
      // the neighrbor element
      if (trial_function_spaces_[i].family == Family::L2) {
        // Set the vector size to store tdof data
        input_ldof_values_[i].SetSize(P_trial_[i]->Height(), mem_type);

        // Initialize face neighbor data vector
        updateFaceNbrData(trial_space_[i], trial_pargrid_functions_[i], input_ldof_values_[i]);

        // Set the local input vector size to store local + ghost data
        input_L_[i].SetSize(input_ldof_values_[i].Size() + trial_pargrid_functions_[i].FaceNbrData().Size(), mem_type);
      } else {
        input_L_[i].SetSize(P_trial_[i]->Height(), mem_type);
      }

      // create the necessary number of empty mfem::Vectors, to be resized later
      input_E_.push_back({});
      input_E_buffer_.push_back({});
    }

    G_test_ = QoIElementRestriction();
    P_test_ = QoIProlongation(trial_fes[0]->GetParMesh()->GetComm());

    output_L_.SetSize(1, mem_type);

    output_T_.SetSize(1, mem_type);

    // gradient objects depend on some member variables in
    // Functional, so we initialize the gradient objects last
    // to ensure that those member variables are initialized first
    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      grad_.emplace_back(*this, i);
    }
  }

  /**
   * @brief Adds a domain integral term to the Functional object
   * @tparam dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   * @param[in] qdata The data structure containing per-quadrature-point data
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */

  template <int dim, int... args, typename lambda, typename qpt_data_type = Nothing>
  void AddDomainIntegral(Dimension<dim>, DependsOn<args...>, const lambda& integrand, Domain& domain,
                         std::shared_ptr<QuadratureData<qpt_data_type>> qdata = NoQData)
  {
    if (domain.mesh_.GetNE() == 0) return;

    SLIC_ERROR_ROOT_IF(dim != domain.mesh_.Dimension(), "invalid mesh dimension for domain integral");

    check_for_unsupported_elements(domain.mesh_);
    check_for_missing_nodal_gridfunc(domain.mesh_);

    std::vector<uint32_t> arg_vec = {args...};
    for (uint32_t i : arg_vec) {
      domain.insert_restriction(trial_space_[i], trial_function_spaces_[i]);
    }

    using signature = test(decltype(smith::type<args>(trial_spaces))...);
    integrals_.push_back(
        MakeDomainIntegral<signature, Q, dim>(domain, integrand, qdata, std::vector<uint32_t>{args...}));
  }

  /**
   * @tparam dim The dimension of the boundary element (1 for line, 2 for quad, etc)
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain which elements make up the domain of integration
   *
   * @brief Adds a boundary integral term to the Functional object
   *
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int dim, int... args, typename lambda>
  void AddBoundaryIntegral(Dimension<dim>, DependsOn<args...>, const lambda& integrand, Domain& domain)
  {
    auto num_bdr_elements = domain.mesh_.GetNBE();
    if (num_bdr_elements == 0) return;

    SLIC_ERROR_ROOT_IF(dim != domain.dim_, "invalid domain of integration for boundary integral");

    check_for_missing_nodal_gridfunc(domain.mesh_);

    std::vector<uint32_t> arg_vec = {args...};
    for (uint32_t i : arg_vec) {
      domain.insert_restriction(trial_space_[i], trial_function_spaces_[i]);
    }

    using signature = test(decltype(smith::type<args>(trial_spaces))...);
    integrals_.push_back(MakeBoundaryIntegral<signature, Q, dim>(domain, integrand, arg_vec));
  }

  /**
   * @tparam dim The dimension of the boundary element (1 for line, 2 for quad, etc)
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain which elements make up the domain of integration
   *
   * @brief Adds a interior face integral term to the Functional object
   *
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int dim, int... args, typename Integrand>
  void AddInteriorFaceIntegral(Dimension<dim>, DependsOn<args...>, const Integrand& integrand, Domain& domain)
  {
    check_for_missing_nodal_gridfunc(domain.mesh_);

    std::vector<uint32_t> arg_vec = {args...};
    for (uint32_t i : arg_vec) {
      domain.insert_restriction(trial_space_[i], trial_function_spaces_[i]);
    }

    using signature = test(decltype(smith::type<args>(trial_spaces))...);
    integrals_.push_back(MakeInteriorFaceIntegral<signature, Q, dim>(domain, integrand, arg_vec));
  }

  /**
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] which_args a tag type used to indicate which trial spaces are required by this calculation
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   * @param[in] data The data structure containing per-quadrature-point data
   *
   * @brief Adds an area integral, i.e., over 2D elements in R^2
   */
  template <int... args, typename lambda, typename qpt_data_type = Nothing>
  void AddAreaIntegral(DependsOn<args...> which_args, const lambda& integrand, Domain& domain,
                       std::shared_ptr<QuadratureData<qpt_data_type>>& data = NoQData)
  {
    AddDomainIntegral(Dimension<2>{}, which_args, integrand, domain, data);
  }

  /**
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] which_args a tag type used to indicate which trial spaces are required by this calculation
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   * @param[in] data The data structure containing per-quadrature-point data
   *
   * @brief Adds a volume integral, i.e., over 3D elements in R^3
   */
  template <int... args, typename lambda, typename qpt_data_type = Nothing>
  void AddVolumeIntegral(DependsOn<args...> which_args, const lambda& integrand, Domain& domain,
                         std::shared_ptr<QuadratureData<qpt_data_type>>& data = NoQData)
  {
    AddDomainIntegral(Dimension<3>{}, which_args, integrand, domain, data);
  }

  /// @brief alias for Functional::AddBoundaryIntegral(Dimension<2>{}, integrand, domain);
  template <int... args, typename lambda>
  void AddSurfaceIntegral(DependsOn<args...> which_args, const lambda& integrand, Domain& domain)
  {
    AddBoundaryIntegral(Dimension<2>{}, which_args, integrand, domain);
  }

  /**
   * @brief this function computes the directional derivative of the quantity of interest functional
   *
   * @param input_T a T-vector to apply the action of gradient to
   * @param which describes which trial space input_T corresponds to
   *
   * note: it accepts exactly `num_trial_spaces` arguments of type mfem::Vector. Additionally, one of those
   * arguments may be a dual_vector, to indicate that Functional::operator() should not only evaluate the
   * element calculations, but also differentiate them w.r.t. the specified dual_vector argument
   */
  double ActionOfGradient(const mfem::Vector& input_T, uint32_t which) const
  {
    // Please refer to 'ActionOfGradient' in 'functional.hpp' for detailed comments of this implementation
    if (trial_function_spaces_[which].family == Family::L2) {
      P_trial_[which]->Mult(input_T, input_ldof_values_[which]);
      input_L_[which].SetVector(input_ldof_values_[which], 0);
      updateFaceNbrData(trial_space_[which], trial_pargrid_functions_[which], input_ldof_values_[which]);

      if (trial_space_[which]->GetVDim() == 1) {
        input_L_[which].SetVector(trial_pargrid_functions_[which].FaceNbrData(), input_ldof_values_[which].Size());
      } else {
        if (trial_space_[which]->GetOrdering() == mfem::Ordering::byVDIM) {
          int local_size = input_ldof_values_[which].Size();
          appendFaceNbrData(trial_space_[which], trial_pargrid_functions_[which], local_size, input_L_[which]);
        } else {
          SLIC_ERROR_ROOT("Unsupported: L2 vector field ordered by nodes");
        }
      }
    } else {
      P_trial_[which]->Mult(input_T, input_L_[which]);
    }

    output_L_ = 0.0;

    for (auto& integral : integrals_) {
      if (integral.DependsOn(which)) {
        Domain& dom = integral.domain_;

        const smith::BlockElementRestriction& G_trial = dom.get_restriction(trial_function_spaces_[which]);
        input_E_buffer_[which].SetSize(int(G_trial.ESize()));
        input_E_[which].Update(input_E_buffer_[which], G_trial.bOffsets());
        G_trial.Gather(input_L_[which], input_E_[which]);

        output_E_buffer_.SetSize(dom.total_elements());
        output_E_.Update(output_E_buffer_, dom.bOffsets());

        integral.GradientMult(input_E_[which], output_E_, which);

        // make sure shared interior faces are integrated only once
        if (dom.type_ == Domain::Type::InteriorFaces) {
          for (int shared_id : dom.shared_interior_face_ids_) {
            output_E_[shared_id] *= 0.5;
          }
        }

        // scatter-add to compute QoI value for the local processor
        G_test_.ScatterAdd(output_E_, output_L_);
      }
    }

    // compute global QoI value by summing values from different processors
    P_test_.MultTranspose(output_L_, output_T_);

    return output_T_[0];
  }

  /**
   * @brief this function lets the user evaluate the smith::Functional with the given trial space values
   *
   * @param t the time
   * @param args the input T-vectors
   *
   * note: it accepts exactly `num_trial_spaces` arguments of type mfem::Vector. Additionally, one of those
   * arguments may be a dual_vector, to indicate that Functional::operator() should not only evaluate the
   * element calculations, but also differentiate them w.r.t. the specified dual_vector argument
   */
  template <uint32_t wrt, typename... T>
  typename operator_paren_return<wrt>::type operator()(DifferentiateWRT<wrt>, double t, const T&... args)
  {
    const mfem::Vector* input_T[] = {&static_cast<const mfem::Vector&>(args)...};

    // Please refer to 'operator()' in 'functional.hpp' for detailed comments of this implementation
    // get the values for each local processor
    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      if (trial_function_spaces_[i].family == Family::L2) {
        P_trial_[i]->Mult(*input_T[i], input_ldof_values_[i]);
        input_L_[i].SetVector(input_ldof_values_[i], 0);
        updateFaceNbrData(trial_space_[i], trial_pargrid_functions_[i], input_ldof_values_[i]);

        if (trial_space_[i]->GetVDim() == 1) {
          input_L_[i].SetVector(trial_pargrid_functions_[i].FaceNbrData(), input_ldof_values_[i].Size());
        } else {
          if (trial_space_[i]->GetOrdering() == mfem::Ordering::byVDIM) {
            int local_size = input_ldof_values_[i].Size();
            appendFaceNbrData(trial_space_[i], trial_pargrid_functions_[i], local_size, input_L_[i]);
          } else {
            SLIC_ERROR_ROOT("Unsupported: L2 vector field ordered by nodes");
          }
        }
      } else {
        P_trial_[i]->Mult(*input_T[i], input_L_[i]);
      }
    }

    output_L_ = 0.0;

    for (auto& integral : integrals_) {
      Domain& dom = integral.domain_;

      for (auto i : integral.active_trial_spaces_) {
        const smith::BlockElementRestriction& G_trial = dom.get_restriction(trial_function_spaces_[i]);
        input_E_buffer_[i].SetSize(int(G_trial.ESize()));
        input_E_[i].Update(input_E_buffer_[i], G_trial.bOffsets());
        G_trial.Gather(input_L_[i], input_E_[i]);
      }

      output_E_buffer_.SetSize(dom.total_elements());
      output_E_.Update(output_E_buffer_, dom.bOffsets());

      const bool update_qdata = false;
      integral.Mult(t, input_E_, output_E_, wrt, update_qdata);

      // make sure shared interior faces are integrated only once
      if (dom.type_ == Domain::Type::InteriorFaces) {
        for (int shared_id : dom.shared_interior_face_ids_) {
          output_E_[shared_id] *= 0.5;
        }
      }

      // scatter-add to compute QoI value for the local processor
      G_test_.ScatterAdd(output_E_, output_L_);
    }

    // compute global QoI value by summing values from different processors
    P_test_.MultTranspose(output_L_, output_T_);

    if constexpr (wrt != NO_DIFFERENTIATION) {
      // if the user has indicated they'd like to evaluate and differentiate w.r.t.
      // a specific argument, then we return both the value and gradient w.r.t. that argument
      //
      // mfem::Vector arg0 = ...;
      // mfem::Vector arg1 = ...;
      // e.g. auto [value, gradient_wrt_arg1] = my_functional(arg0, differentiate_wrt(arg1));
      return {output_T_[0], grad_[wrt]};
    }

    if constexpr (wrt == NO_DIFFERENTIATION) {
      // if the user passes only `mfem::Vector`s then we assume they only want the output value
      //
      // mfem::Vector arg0 = ...;
      // mfem::Vector arg1 = ...;
      // e.g. mfem::Vector value = my_functional(arg0, arg1);
      return output_T_[0];
    }
  }

  /// @overload
  template <typename... T>
  auto operator()(double t, const T&... args)
  {
    // below we add 0 so the number of differentiated arguments defaults to 0 if trial spaces are not provided
    constexpr int num_differentiated_arguments = (std::is_same_v<T, differentiate_wrt_this> + ... + 0);
    static_assert(num_differentiated_arguments <= 1,
                  "Error: Functional::operator() can only differentiate w.r.t. 1 argument a time");
    static_assert(sizeof...(T) == num_trial_spaces,
                  "Error: Functional::operator() must take exactly as many arguments as trial spaces");

    [[maybe_unused]] constexpr uint32_t i = index_of_differentiation<T...>();

    return (*this)(DifferentiateWRT<i>{}, t, args...);
  }

 private:
  /**
   * @brief Indicates whether to obtain values or gradients from a calculation
   */
  enum class Operation
  {
    Mult,
    GradientMult
  };

  /**
   * @brief mfem::Operator that produces the gradient of a @p Functional from a @p Mult
   */
  class Gradient {
   public:
    /**
     * @brief Constructs a Gradient wrapper that references a parent @p Functional
     * @param[in] f The @p Functional to use for gradient calculations
     */
    Gradient(Functional<double(trials...)>& f, uint32_t which = 0)
        : form_(f), which_argument(which), gradient_L_(f.trial_space_[which]->GetVSize())
    {
    }

    double operator()(const mfem::Vector& x) const { return form_.ActionOfGradient(x, which_argument); }

    uint64_t max_buffer_size()
    {
      uint64_t max_entries = 0;
      for (auto& integral : form_.integrals_) {
        if (integral.DependsOn(which_argument)) {
          Domain& dom = integral.domain_;
          const auto& G_trial = dom.get_restriction(form_.trial_function_spaces_[which_argument]);
          for (const auto& [geom, test_restriction] : G_trial.restrictions) {
            const auto& trial_restriction = G_trial.restrictions.at(geom);
            uint64_t entries_per_element = trial_restriction.nodes_per_elem * trial_restriction.components;
            max_entries = std::max(max_entries, trial_restriction.num_elements * entries_per_element);
          }
        }
      }
      return max_entries;
    }

    std::unique_ptr<mfem::HypreParVector> assemble()
    {
      // The mfem method ParFiniteElementSpace.NewTrueDofVector should really be marked const
      std::unique_ptr<mfem::HypreParVector> gradient_T(
          const_cast<mfem::ParFiniteElementSpace*>(form_.trial_space_[which_argument])->NewTrueDofVector());

      gradient_L_ = 0.0;

      std::vector<double> K_elem_buffer(max_buffer_size());

      std::map<mfem::Geometry::Type, ExecArray<double, 3, exec>> element_gradients[Domain::num_types];

      int lcols = form_.trial_space_[which_argument]->GetVSize();

      ////////////////////////////////////////////////////////////////////////////////

      for (auto& integral : form_.integrals_) {
        Domain& dom = integral.domain_;

        // if this integral's derivative isn't identically zero
        if (integral.functional_to_integral_index_.count(which_argument) > 0) {
          uint32_t id = integral.functional_to_integral_index_.at(which_argument);
          const auto& G_trial = dom.get_restriction(form_.trial_function_spaces_[which_argument]);
          for (const auto& [geom, calculate_element_gradients] : integral.element_gradient_[id]) {
            const auto& trial_restriction = G_trial.restrictions.at(geom);

            // prepare a buffer to hold the element matrices
            CPUArrayView<double, 3> K_e(K_elem_buffer.data(), trial_restriction.num_elements, 1,
                                        trial_restriction.nodes_per_elem * trial_restriction.components);
            detail::zero_out(K_e);

            calculate_element_gradients(K_e);

            const std::vector<int>& element_ids = integral.domain_.get(geom);

            uint32_t cols_per_elem = uint32_t(trial_restriction.nodes_per_elem * trial_restriction.components);
            std::vector<DoF> trial_vdofs(cols_per_elem);

            for (uint32_t e = 0; e < element_ids.size(); e++) {
              trial_restriction.GetElementVDofs(int(e), trial_vdofs);

              for (uint32_t i = 0; i < cols_per_elem; i++) {
                int col = int(trial_vdofs[i].index());

                // only add local DG dof gradients (other FE spaces satisfy col < lcols inherently)
                if (col < lcols) {
                  gradient_L_[col] += K_e(e, 0, i);
                }
              }
            }
          }
        }
      }

      ////////////////////////////////////////////////////////////////////////////////

      form_.P_trial_[which_argument]->MultTranspose(gradient_L_, *gradient_T);

      return gradient_T;
    }

    friend auto assemble(Gradient& g) { return g.assemble(); }

   private:
    /**
     * @brief The "parent" @p Functional to calculate gradients with
     */
    Functional<double(trials...), exec>& form_;

    uint32_t which_argument;

    mfem::Vector gradient_L_;
  };

  /// @brief Manages DOFs for the test space
  const mfem::L2_FECollection test_fec_;
  const mfem::ParFiniteElementSpace test_space_;

  /// @brief Manages DOFs for the trial space
  std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_space_;

  std::array<FunctionSpace, num_trial_spaces> trial_function_spaces_;

  /// @brief Cache for access of neighbor element info for processor boundary faces
  mutable std::array<mfem::ParGridFunction, num_trial_spaces> trial_pargrid_functions_;

  /// @brief Cache to store values of L2 ldofs on the local processor
  mutable std::array<mfem::Vector, num_trial_spaces> input_ldof_values_;

  /**
   * @brief Operator that converts true (global) DOF values to local (current rank) DOF values
   * for the test space
   */
  const mfem::Operator* P_trial_[num_trial_spaces];

  /// @brief The input set of local DOF values (i.e., on the current rank)
  mutable mfem::Vector input_L_[num_trial_spaces];

  mutable std::vector<mfem::Vector> input_E_buffer_;
  mutable std::vector<mfem::BlockVector> input_E_;

  mutable std::vector<Integral> integrals_;

  mutable mfem::Vector output_E_buffer_;
  mutable mfem::BlockVector output_E_;

  QoIElementRestriction G_test_;

  /// @brief The output set of local DOF values (i.e., on the current rank)
  mutable mfem::Vector output_L_;

  QoIProlongation P_test_;

  /// @brief The set of true DOF values, a reference to this member is returned by @p operator()
  mutable mfem::Vector output_T_;

  /// @brief The objects representing the gradients w.r.t. each input argument of the Functional
  mutable std::vector<Gradient> grad_;

  const mfem::MemoryType mem_type;
};

}  // namespace smith
