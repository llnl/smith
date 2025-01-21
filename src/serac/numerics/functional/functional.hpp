// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file functional.hpp
 *
 * @brief Implementation of the quadrature-function-based functional enabling rapid development of FEM formulations
 */

#pragma once

#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/integral.hpp"
#include "serac/numerics/functional/differentiate_wrt.hpp"

#include "serac/numerics/functional/element_restriction.hpp"

#include "serac/numerics/functional/domain.hpp"

#include <array>
#include <vector>

namespace serac {

template <int... i>
struct DependsOn {
};

/**
 * @brief given a list of types, this function returns the index that corresponds to the type `dual_vector`.
 *
 * @tparam T a list of types, containing at most 1 `differentiate_wrt_this`
 *
 * e.g.
 * @code{.cpp}
 * static_assert(index_of_dual_vector < foo, bar, differentiate_wrt_this, baz, qux >() == 2);
 * @endcode
 */
template <typename... T>
constexpr uint32_t index_of_differentiation()
{
  constexpr uint32_t n = sizeof...(T);
  bool matching[] = {std::is_same_v<T, differentiate_wrt_this>...};
  for (uint32_t i = 0; i < n; i++) {
    if (matching[i]) {
      return i;
    }
  }
  return NO_DIFFERENTIATION;
}

/**
 * @brief Compile-time alias for index of differentiation
 */
template <int ind>
struct Index {
  /**
   * @brief Returns the index
   */
  constexpr operator int() { return ind; }
};

/// function for verifying that the mesh has been fully initialized
inline void check_for_missing_nodal_gridfunc(const mfem::Mesh& mesh)
{
  if (mesh.GetNodes() == nullptr) {
    SLIC_ERROR_ROOT(
        R"errmsg(
      the provided mesh does not have a nodal gridfunction.
      If you created an mfem::Mesh manually, make sure that the
      following member functions are invoked before use

      > mfem::Mesh::EnsureNodes();
      > mfem::ParMesh::ExchangeFaceNbrData();

      or else the mfem::Mesh won't be fully initialized
      )errmsg";);
  }
}

/// function for verifying that there are no unsupported element types in the mesh
inline void check_for_unsupported_elements(const mfem::Mesh& mesh)
{
  int num_elements = mesh.GetNE();
  for (int e = 0; e < num_elements; e++) {
    auto type = mesh.GetElementType(e);
    if (type == mfem::Element::POINT || type == mfem::Element::WEDGE || type == mfem::Element::PYRAMID) {
      SLIC_ERROR_ROOT("Mesh contains unsupported element type");
    }
  }
}

/**
 * @brief function for verifying that DG spaces aren't used on interior face integrals over meshes that contain "shared"
 * faces
 *
 * sam: I would like to support these "shared" faces, but apparently mfem handles them in a fundamentally different way
 * than the "finite element operator decomposition" pattern used by everything else (see: "ExchangeFaceNbrData")
 */
inline void check_interior_face_compatibility(const mfem::Mesh& mesh, const FunctionSpace space)
{
  if (space.family == Family::L2) {
    const mfem::ParMesh* pmesh = dynamic_cast<const mfem::ParMesh*>(&mesh);
    if (pmesh) {
      SLIC_ERROR_IF(
          pmesh->GetNSharedFaces() > 0,
          "interior face integrals involving DG function spaces don't currently support meshes with shared faces");
    }
  }
}

/**
 * @brief create an mfem::ParFiniteElementSpace from one of serac's
 * tag types: H1, Hcurl, L2
 *
 * @tparam function_space a tag type containing the kind of function space and polynomial order
 * @param mesh the mesh on which the space is defined
 * @return a pair containing the new finite element space and associated finite element collection
 */
template <typename function_space>
inline std::pair<std::unique_ptr<mfem::ParFiniteElementSpace>, std::unique_ptr<mfem::FiniteElementCollection>>
generateParFiniteElementSpace(mfem::ParMesh* mesh)
{
  const int dim = mesh->Dimension();
  std::unique_ptr<mfem::FiniteElementCollection> fec;

  switch (function_space::family) {
    case Family::H1:
      fec = std::make_unique<mfem::H1_FECollection>(function_space::order, dim);
      break;
    case Family::HCURL:
      fec = std::make_unique<mfem::ND_FECollection>(function_space::order, dim);
      break;
    case Family::HDIV:
      fec = std::make_unique<mfem::RT_FECollection>(function_space::order, dim);
      break;
    case Family::L2:
      // We use GaussLobatto basis functions as this is what is used for the serac::Functional FE kernels
      fec = std::make_unique<mfem::L2_FECollection>(function_space::order, dim, mfem::BasisType::GaussLobatto);
      break;
    default:
      return std::pair<std::unique_ptr<mfem::ParFiniteElementSpace>, std::unique_ptr<mfem::FiniteElementCollection>>(
          nullptr, nullptr);
      break;
  }

  auto fes =
      std::make_unique<mfem::ParFiniteElementSpace>(mesh, fec.get(), function_space::components, serac::ordering);

  return std::pair(std::move(fes), std::move(fec));
}

/// @cond
template <typename T, ExecutionSpace exec = serac::default_execution_space>
class Functional;
/// @endcond

/**
 * @brief Intended to be like @p std::function for finite element kernels
 *
 * That is: you tell it the inputs (trial spaces) for a kernel, and the outputs (test space) like @p std::function.
 *
 * For example, this code represents a function that takes an integer argument and returns a double:
 * @code{.cpp}
 * std::function< double(int) > my_func;
 * @endcode
 * And this represents a function that takes values from an Hcurl field and returns a
 * residual vector associated with an H1 field:
 * @code{.cpp}
 * Functional< H1(Hcurl) > my_residual;
 * @endcode
 *
 * @tparam test The space of test functions to use
 * @tparam trial The space of trial functions to use
 * @tparam exec whether to carry out calculations on CPU or GPU
 *
 * To use this class, you use the methods @p Functional::Add****Integral(integrand,domain_of_integration)
 * where @p integrand is a q-function lambda or functor and @p domain_of_integration is an @p mfem::mesh
 *
 * @see https://libceed.readthedocs.io/en/latest/libCEEDapi/#theoretical-framework for additional
 * information on the idea behind a quadrature function and its inputs/outputs
 *
 * @code{.cpp}
 * // for domains made up of quadrilaterals embedded in R^2
 * my_residual.AddAreaIntegral(integrand, domain_of_integration);
 * // alternatively...
 * my_residual.AddDomainIntegral(Dimension<2>{}, integrand, domain_of_integration);
 *
 * // for domains made up of quadrilaterals embedded in R^3
 * my_residual.AddSurfaceIntegral(integrand, domain_of_integration);
 *
 * // for domains made up of hexahedra embedded in R^3
 * my_residual.AddVolumeIntegral(integrand, domain_of_integration);
 * // alternatively...
 * my_residual.AddDomainIntegral(Dimension<3>{}, integrand, domain_of_integration);
 * @endcode
 */
template <typename test, typename... trials, ExecutionSpace exec>
class Functional<test(trials...), exec> {
  static constexpr tuple<trials...> trial_spaces{};
  static constexpr uint32_t num_trial_spaces = sizeof...(trials);
  static constexpr auto Q = std::max({test::order, trials::order...}) + 1;

  static constexpr mfem::Geometry::Type elem_geom[4] = {mfem::Geometry::INVALID, mfem::Geometry::SEGMENT,
                                                        mfem::Geometry::SQUARE, mfem::Geometry::CUBE};
  static constexpr mfem::Geometry::Type simplex_geom[4] = {mfem::Geometry::INVALID, mfem::Geometry::SEGMENT,
                                                           mfem::Geometry::TRIANGLE, mfem::Geometry::TETRAHEDRON};

  class Gradient;

  // clang-format off
  template <uint32_t i>
  struct operator_paren_return {
    using type = typename std::conditional<
        i == NO_DIFFERENTIATION,               // if `i` indicates that we want to skip differentiation
        mfem::Vector&,                         // we just return the value
        serac::tuple<mfem::Vector&, Gradient&> // otherwise we return the value and the derivative w.r.t arg `i`
        >::type;
  };
  // clang-format on

 public:
  /**
   * @brief Constructs using @p mfem::ParFiniteElementSpace objects corresponding to the test/trial spaces
   * @param[in] test_fes The (non-qoi) test space
   * @param[in] trial_fes The trial space
   */
  Functional(const mfem::ParFiniteElementSpace* test_fes,
             std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_fes)
      : update_qdata_(false), test_space_(test_fes), trial_space_(trial_fes), mem_type(mfem::Device::GetMemoryType())
  {
    SERAC_MARK_FUNCTION;

    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      P_trial_[i] = trial_space_[i]->GetProlongationMatrix();

      input_L_[i].SetSize(P_trial_[i]->Height(), mfem::Device::GetMemoryType());

      // create the necessary number of empty mfem::Vectors, to be resized later
      input_E_.push_back({});
      input_E_buffer_.push_back({});
    }

    test_function_space_ = {test::family, test::order, test::components};

    std::array<Family, num_trial_spaces> trial_families = {trials::family...};
    std::array<int, num_trial_spaces> trial_orders = {trials::order...};
    std::array<int, num_trial_spaces> trial_components = {trials::components...};
    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      trial_function_spaces_[i] = {trial_families[i], trial_orders[i], trial_components[i]};
    }

    // for (auto type : {Domain::Type::Elements, Domain::Type::BoundaryElements, Domain::Type::InteriorFaces}) {
    //   output_E_[type].Update(G_test_[type].bOffsets(), mem_type);
    // }

    P_test_ = test_space_->GetProlongationMatrix();

    output_L_.SetSize(P_test_->Height(), mem_type);

    output_T_.SetSize(test_fes->GetTrueVSize(), mem_type);

    // gradient objects depend on some member variables in
    // Functional, so we initialize the gradient objects last
    // to ensure that those member variables are initialized first
    for (uint32_t i = 0; i < num_trial_spaces; i++) {
      grad_.emplace_back(*this, i);
    }
  }

  /**
   * @brief Adds a domain integral term to the weak formulation of the PDE
   * @tparam dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   * @param[inout] qdata The data for each quadrature point
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
    domain.insert_restriction(test_space_, test_function_space_);

    using signature = test(decltype(serac::type<args>(trial_spaces))...);
    integrals_.push_back(
        MakeDomainIntegral<signature, Q, dim>(domain, integrand, qdata, std::vector<uint32_t>{args...}));
  }

  /**
   * @brief Adds a boundary integral term to the weak formulation of the PDE
   * @tparam dim The dimension of the boundary element (1 for line, 2 for quad, etc)
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
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
    domain.insert_restriction(test_space_, test_function_space_);

    using signature = test(decltype(serac::type<args>(trial_spaces))...);
    integrals_.push_back(MakeBoundaryIntegral<signature, Q, dim>(domain, integrand, std::vector<uint32_t>{args...}));
  }

  /**
   * @brief TODO
   */
  template <int dim, int... args, typename Integrand>
  void AddInteriorFaceIntegral(Dimension<dim>, DependsOn<args...>, const Integrand& integrand, Domain& domain)
  {
    check_for_missing_nodal_gridfunc(domain.mesh_);

    std::vector<uint32_t> arg_vec = {args...};
    for (uint32_t i : arg_vec) {
      domain.insert_restriction(trial_space_[i], trial_function_spaces_[i]);
      check_interior_face_compatibility(domain.mesh_, trial_function_spaces_[i]);
    }
    domain.insert_restriction(test_space_, test_function_space_);
    check_interior_face_compatibility(domain.mesh_, test_function_space_);

    using signature = test(decltype(serac::type<args>(trial_spaces))...);
    integrals_.push_back(
        MakeInteriorFaceIntegral<signature, Q, dim>(domain, integrand, std::vector<uint32_t>{args...}));
  }

  /**
   * @brief Adds an area integral, i.e., over 2D elements in R^2 space
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] which_args a tag type used to indicate which trial spaces are required by this calculation
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   * @param[inout] data The data for each quadrature point
   */
  template <int... args, typename lambda, typename qpt_data_type = Nothing>
  void AddAreaIntegral(DependsOn<args...> which_args, const lambda& integrand, Domain& domain,
                       std::shared_ptr<QuadratureData<qpt_data_type>> data = NoQData)
  {
    AddDomainIntegral(Dimension<2>{}, which_args, integrand, domain, data);
  }

  /**
   * @brief Adds a volume integral, i.e., over 3D elements in R^3 space
   * @tparam lambda the type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] which_args a tag type used to indicate which trial spaces are required by this calculation
   * @param[in] integrand The quadrature function
   * @param[in] domain The mesh to evaluate the integral on
   * @param[inout] data The data for each quadrature point
   */
  template <int... args, typename lambda, typename qpt_data_type = Nothing>
  void AddVolumeIntegral(DependsOn<args...> which_args, const lambda& integrand, Domain& domain,
                         std::shared_ptr<QuadratureData<qpt_data_type>> data = NoQData)
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
   * @brief this function computes the directional derivative of `serac::Functional::operator()`
   *
   * @param input_T the T-vector to apply the action of gradient to
   * @param output_T the T-vector where the resulting values are stored
   * @param which describes which trial space input_T corresponds to
   *
   * @note: it accepts exactly `num_trial_spaces` arguments of type mfem::Vector. Additionally, one of those
   * arguments may be a dual_vector, to indicate that Functional::operator() should not only evaluate the
   * element calculations, but also differentiate them w.r.t. the specified dual_vector argument
   */
  void ActionOfGradient(const mfem::Vector& input_T, mfem::Vector& output_T, uint32_t which) const
  {
    P_trial_[which]->Mult(input_T, input_L_[which]);

    output_L_ = 0.0;

    for (auto& integral : integrals_) {
      if (integral.DependsOn(which)) {
        Domain& dom = integral.domain_;

        const serac::BlockElementRestriction& G_trial = dom.get_restriction(trial_function_spaces_[which]);
        input_E_buffer_[which].SetSize(int(G_trial.ESize()));
        input_E_[which].Update(input_E_buffer_[which], G_trial.bOffsets());
        G_trial.Gather(input_L_[which], input_E_[which]);

        const serac::BlockElementRestriction& G_test = dom.get_restriction(test_function_space_);
        output_E_buffer_.SetSize(int(G_test.ESize()));
        output_E_.Update(output_E_buffer_, G_test.bOffsets());

        integral.GradientMult(input_E_[which], output_E_, which);

        // scatter-add to compute residuals on the local processor
        G_test.ScatterAdd(output_E_, output_L_);
      }
    }

    // scatter-add to compute global residuals
    P_test_->MultTranspose(output_L_, output_T);
  }

  /**
   * @brief this function lets the user evaluate the serac::Functional with the given trial space values
   *
   * note: it accepts exactly `num_trial_spaces` arguments of type mfem::Vector. Additionally, one of those
   * arguments may be a dual_vector, to indicate that Functional::operator() should not only evaluate the
   * element calculations, but also differentiate them w.r.t. the specified dual_vector argument
   *
   * @tparam T the types of the arguments passed in
   * @param t the time
   * @param args the trial space dofs used to carry out the calculation,
   *  at most one of which may be of the type `differentiate_wrt_this(mfem::Vector)`
   */
  template <uint32_t wrt, typename... T>
  typename operator_paren_return<wrt>::type operator()(DifferentiateWRT<wrt>, double t, const T&... args)
  {
    const mfem::Vector* input_T[] = {&static_cast<const mfem::Vector&>(args)...};

    // get the values for each local processor
    for (uint32_t i = 0; i < num_trial_spaces; i++) {
#if 0
      if (trial_function_spaces_[i].family == Family::L2) {

        // make a PGF on the fly
        // TODO: don't allocate/deallocate this data every invocation
        mfem::ParGridFunction X;
        X.MakeRef(trial_space_[i], input_L_[i].GetData());

        // call exchange face nbr
        X.ExchangeFaceNbrData();

        // copy input_L[i] and facenbrdata [i] into a common array like:
        //          first part        second part
        //    [   ---   L    ---   |  ---  FND ---  ]
        mfem::Vector first_part;
        first_part.MakeRef(input_L_[i], 0);
        P_trial_[i]->Mult(*input_T[i], first_part);

        mfem::Vector second_part;
        second_part.MakeRef(input_L_[i], first_part.Size());
        second_part = X.FaceNbrData();

      } else {

        P_trial_[i]->Mult(*input_T[i], input_L_[i]);

      }
#else
      P_trial_[i]->Mult(*input_T[i], input_L_[i]);
#endif
    }

    output_L_ = 0.0;

    for (auto& integral : integrals_) {
      Domain& dom = integral.domain_;

      const serac::BlockElementRestriction& G_test = dom.get_restriction(test_function_space_);

      for (auto i : integral.active_trial_spaces_) {
        const serac::BlockElementRestriction& G_trial = dom.get_restriction(trial_function_spaces_[i]);
        input_E_buffer_[i].SetSize(int(G_trial.ESize()));
        input_E_[i].Update(input_E_buffer_[i], G_trial.bOffsets());
        G_trial.Gather(input_L_[i], input_E_[i]);
      }

      output_E_buffer_.SetSize(int(G_test.ESize()));
      output_E_.Update(output_E_buffer_, G_test.bOffsets());
      integral.Mult(t, input_E_, output_E_, wrt, update_qdata_);

      // scatter-add to compute residuals on the local processor
      G_test.ScatterAdd(output_E_, output_L_);
    }

    // scatter-add to compute global residuals
    P_test_->MultTranspose(output_L_, output_T_);

    if constexpr (wrt != NO_DIFFERENTIATION) {
      // if the user has indicated they'd like to evaluate and differentiate w.r.t.
      // a specific argument, then we return both the value and gradient w.r.t. that argument
      //
      // mfem::Vector arg0 = ...;
      // mfem::Vector arg1 = ...;
      // e.g. auto [value, gradient_wrt_arg1] = my_functional(arg0, differentiate_wrt(arg1));
      return {output_T_, grad_[wrt]};
    }

    if constexpr (wrt == NO_DIFFERENTIATION) {
      // if the user passes only `mfem::Vector`s then we assume they only want the output value
      //
      // mfem::Vector arg0 = ...;
      // mfem::Vector arg1 = ...;
      // e.g. mfem::Vector value = my_functional(arg0, arg1);
      return output_T_;
    }
  }

  /// @overload
  template <typename... T>
  auto operator()(double t, const T&... args)
  {
    constexpr int num_differentiated_arguments = (std::is_same_v<T, differentiate_wrt_this> + ...);
    static_assert(num_differentiated_arguments <= 1,
                  "Error: Functional::operator() can only differentiate w.r.t. 1 argument a time");
    static_assert(sizeof...(T) == num_trial_spaces,
                  "Error: Functional::operator() must take exactly as many arguments as trial spaces");

    [[maybe_unused]] constexpr uint32_t i = index_of_differentiation<T...>();

    return (*this)(DifferentiateWRT<i>{}, t, args...);
  }

  /**
   * @brief A flag to update the quadrature data for this operator following the computation
   *
   * Typically this is set to false during nonlinear solution iterations and is set to true for the
   * final pass once equilibrium is found.
   *
   * @param update_flag A flag to update the related quadrature data
   */
  void updateQdata(bool update_flag) { update_qdata_ = update_flag; }

 private:
  /// @brief flag for denoting when a residual evaluation should update the material state buffers
  bool update_qdata_;

  /**
   * @brief mfem::Operator representing the gradient matrix that
   * can compute the action of the gradient (with operator()),
   * or assemble the sparse matrix representation through implicit conversion to mfem::HypreParMatrix *
   */
  class Gradient : public mfem::Operator {
   public:
    /**
     * @brief Constructs a Gradient wrapper that references a parent @p Functional
     * @param[in] f The @p Functional to use for gradient calculations
     */
    Gradient(Functional<test(trials...), exec>& f, uint32_t which = 0)
        : mfem::Operator(f.test_space_->GetTrueVSize(), f.trial_space_[which]->GetTrueVSize()),
          form_(f),
          which_argument(which),
          test_space_(f.test_space_),
          trial_space_(f.trial_space_[which]),
          df_(f.test_space_->GetTrueVSize())
    {
      SERAC_MARK_FUNCTION;
    }

    /**
     * @brief implement that action of the gradient: df := df_dx * dx
     * @param[in] dx a small perturbation in the trial space
     * @param[in] df the resulting small perturbation in the residuals
     */
    virtual void Mult(const mfem::Vector& dx, mfem::Vector& df) const override
    {
      form_.ActionOfGradient(dx, df, which_argument);
    }

    /// @brief syntactic sugar:  df_dx.Mult(dx, df)  <=>  mfem::Vector df = df_dx(dx);
    mfem::Vector& operator()(const mfem::Vector& dx)
    {
      form_.ActionOfGradient(dx, df_, which_argument);
      return df_;
    }

    void initialize_sparsity_pattern()
    {
      using row_col = std::tuple<int, int>;

      std::set<row_col> nonzero_entries;

      for (auto& integral : form_.integrals_) {
        if (integral.DependsOn(which_argument)) {
          Domain& dom = integral.domain_;
          const auto& G_test = dom.get_restriction(form_.test_function_space_);
          const auto& G_trial = dom.get_restriction(form_.trial_function_spaces_[which_argument]);
          for (const auto& [geom, test_restriction] : G_test.restrictions) {
            const auto& trial_restriction = G_trial.restrictions.at(geom);

            // the degrees of freedom associated with the rows/columns of the e^th element stiffness matrix
            std::vector<int> test_vdofs(test_restriction.nodes_per_elem * test_restriction.components);
            std::vector<int> trial_vdofs(trial_restriction.nodes_per_elem * trial_restriction.components);

            auto num_elements = static_cast<uint32_t>(test_restriction.num_elements);
            for (uint32_t e = 0; e < num_elements; e++) {
              for (uint32_t i = 0; i < test_restriction.nodes_per_elem; i++) {
                auto test_dof = test_restriction.dof_info(e, i);
                for (uint32_t j = 0; j < test_restriction.components; j++) {
                  test_vdofs[i * test_restriction.components + j] = int(test_restriction.GetVDof(test_dof, j).index());
                }
              }

              for (uint32_t i = 0; i < trial_restriction.nodes_per_elem; i++) {
                auto trial_dof = trial_restriction.dof_info(e, i);
                for (uint32_t j = 0; j < trial_restriction.components; j++) {
                  trial_vdofs[i * trial_restriction.components + j] =
                      int(trial_restriction.GetVDof(trial_dof, j).index());
                }
              }

              for (int row : test_vdofs) {
                for (int col : trial_vdofs) {
                  nonzero_entries.insert({row, col});
                }
              }
            }
          }
        }
      }

      uint64_t nnz = nonzero_entries.size();
      int nrows = form_.output_L_.Size();

      row_ptr.resize(uint32_t(nrows + 1));
      col_ind.resize(nnz);

      int nz = 0;
      int last_row = -1;
      for (auto [row, col] : nonzero_entries) {
        col_ind[uint32_t(nz)] = col;
        for (int i = last_row + 1; i <= row; i++) {
          row_ptr[uint32_t(i)] = nz;
        }
        last_row = row;
        nz++;
      }
      for (int i = last_row + 1; i <= nrows; i++) {
        row_ptr[uint32_t(i)] = nz;
      }
    };

    uint64_t max_buffer_size()
    {
      uint64_t max_entries = 0;
      for (auto& integral : form_.integrals_) {
        if (integral.DependsOn(which_argument)) {
          Domain& dom = integral.domain_;
          const auto& G_test = dom.get_restriction(form_.test_function_space_);
          const auto& G_trial = dom.get_restriction(form_.trial_function_spaces_[which_argument]);
          for (const auto& [geom, test_restriction] : G_test.restrictions) {
            const auto& trial_restriction = G_trial.restrictions.at(geom);
            uint64_t nrows_per_element = test_restriction.nodes_per_elem * test_restriction.components;
            uint64_t ncols_per_element = trial_restriction.nodes_per_elem * trial_restriction.components;
            uint64_t entries_per_element = nrows_per_element * ncols_per_element;
            uint64_t entries_needed = test_restriction.num_elements * entries_per_element;
            max_entries = std::max(entries_needed, max_entries);
          }
        }
      }
      return max_entries;
    }

    std::unique_ptr<mfem::HypreParMatrix> assemble()
    {
      if (row_ptr.empty()) {
        initialize_sparsity_pattern();
      }

      // since we own the storage for row_ptr, col_ind, values,
      // we ask mfem to not deallocate those pointers in the SparseMatrix dtor
      constexpr bool sparse_matrix_frees_graph_ptrs = false;
      constexpr bool sparse_matrix_frees_values_ptr = false;
      constexpr bool col_ind_is_sorted = true;

      // note: we make a copy of col_ind since mfem::HypreParMatrix
      //       changes it in the constructor
      std::vector<int> col_ind_copy = col_ind;

      int nnz = row_ptr.back();
      std::vector<double> values(uint32_t(nnz), 0.0);
      auto A_local = mfem::SparseMatrix(row_ptr.data(), col_ind_copy.data(), values.data(), form_.output_L_.Size(),
                                        form_.input_L_[which_argument].Size(), sparse_matrix_frees_graph_ptrs,
                                        sparse_matrix_frees_values_ptr, col_ind_is_sorted);

      std::vector<double> K_elem_buffer(max_buffer_size());

      for (auto& integral : form_.integrals_) {
        // if this integral's derivative isn't identically zero
        if (integral.functional_to_integral_index_.count(which_argument) > 0) {
          Domain& dom = integral.domain_;

          uint32_t id = integral.functional_to_integral_index_.at(which_argument);
          const auto& G_test = dom.get_restriction(form_.test_function_space_);
          const auto& G_trial = dom.get_restriction(form_.trial_function_spaces_[which_argument]);
          for (const auto& [geom, calculate_element_matrices_func] : integral.element_gradient_[id]) {
            const auto& test_restriction = G_test.restrictions.at(geom);
            const auto& trial_restriction = G_trial.restrictions.at(geom);

            // prepare a buffer to hold the element matrices
            CPUArrayView<double, 3> K_e(K_elem_buffer.data(), test_restriction.num_elements,
                                        trial_restriction.nodes_per_elem * trial_restriction.components,
                                        test_restriction.nodes_per_elem * test_restriction.components);
            detail::zero_out(K_e);

            // perform the actual calculations
            calculate_element_matrices_func(K_e);

            const std::vector<int>& element_ids = integral.domain_.get(geom);

            uint32_t rows_per_elem = uint32_t(test_restriction.nodes_per_elem * test_restriction.components);
            uint32_t cols_per_elem = uint32_t(trial_restriction.nodes_per_elem * trial_restriction.components);

            std::vector<DoF> test_vdofs(rows_per_elem);
            std::vector<DoF> trial_vdofs(cols_per_elem);

            for (uint32_t e = 0; e < element_ids.size(); e++) {
              test_restriction.GetElementVDofs(int(e), test_vdofs);
              trial_restriction.GetElementVDofs(int(e), trial_vdofs);

              for (uint32_t i = 0; i < cols_per_elem; i++) {
                int col = int(trial_vdofs[i].index());

                for (uint32_t j = 0; j < rows_per_elem; j++) {
                  int row = int(test_vdofs[j].index());
                  A_local.SearchRow(row, col) += K_e(e, i, j);
                }
              }
            }
          }
        }
      }

      auto* R = form_.test_space_->Dof_TrueDof_Matrix();

      auto* A_hypre =
          new mfem::HypreParMatrix(test_space_->GetComm(), test_space_->GlobalVSize(), trial_space_->GlobalVSize(),
                                   test_space_->GetDofOffsets(), trial_space_->GetDofOffsets(), &A_local);

      auto* P = trial_space_->Dof_TrueDof_Matrix();

      std::unique_ptr<mfem::HypreParMatrix> A(mfem::RAP(R, A_hypre, P));

      delete A_hypre;

      return A;
    };

    friend auto assemble(Gradient& g) { return g.assemble(); }

   private:
    /// @brief The "parent" @p Functional to calculate gradients with
    Functional<test(trials...), exec>& form_;

    std::vector<int> row_ptr;
    std::vector<int> col_ind;

    /**
     * @brief this member variable tells us which argument the associated Functional this gradient
     *  corresponds to:
     *
     *  e.g.
     *    Functional< test(trial0, trial1, trial2) > f(...);
     *    grad<0>(f) == df_dtrial0
     *    grad<1>(f) == df_dtrial1
     *    grad<2>(f) == df_dtrial2
     */
    uint32_t which_argument;

    /// @brief shallow copy of the test space from the associated Functional
    const mfem::ParFiniteElementSpace* test_space_;

    /// @brief shallow copy of the trial space from the associated Functional
    const mfem::ParFiniteElementSpace* trial_space_;

    /// @brief storage for computing the action-of-gradient output
    mfem::Vector df_;
  };

  /// @brief Manages DOFs for the test space
  const mfem::ParFiniteElementSpace* test_space_;

  /// @brief Manages DOFs for the trial space
  std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_space_;

  std::array<FunctionSpace, num_trial_spaces> trial_function_spaces_;
  FunctionSpace test_function_space_;

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

  /// @brief The output set of local DOF values (i.e., on the current rank)
  mutable mfem::Vector output_L_;

  const mfem::Operator* P_test_;

  /// @brief The set of true DOF values, a reference to this member is returned by @p operator()
  mutable mfem::Vector output_T_;

  /// @brief The objects representing the gradients w.r.t. each input argument of the Functional
  mutable std::vector<Gradient> grad_;

  const mfem::MemoryType mem_type;
};

}  // namespace serac

#include "functional_qoi.inl"
