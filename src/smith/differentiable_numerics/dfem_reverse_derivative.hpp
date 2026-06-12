// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// ===========================================================================
// MakeReverseGradientOperator — turns a STANDARD dFEM residual qfunction
// (the kind one would feed to DifferentiableOperator::AddDomainIntegrator)
// into a reverse-mode VJP operator. The user writes the physics once; the
// factory injects a seed field V, contracts the physics outputs against the
// matching V fop (Gradient<V> for Gradient<U>, Value<V> for Value<U>, ...),
// builds an internal scalar QoI density, and reverse-differentiates it via
// Enzyme to produce ∂/∂(each chosen input field) in a single Mult.
//
// User code:
//
//   struct ResidualQF {            // standard dFEM residual qfunction
//      MFEM_HOST_DEVICE inline
//      void operator()(const tensor<real_t,DIM,DIM>& dudxi,
//                      const real_t& E_val,
//                      const tensor<real_t,DIM,DIM>& J,
//                      const real_t& w,
//                      tensor<real_t,DIM,DIM>& dvdxi) const { ... }
//   };
//
//   auto rev = MakeReverseGradientOperator<V, U, EFLD>(
//        ResidualQF{},
//        tuple{Gradient<U>{}, Value<EFLD>{}, Gradient<COORDS>{}, Weight{}},
//        tuple{Gradient<U>{}},                          // physics outputs
//        {{U,&vfes},{V,&vfes},{EFLD,&efes},{COORDS,mfes}},
//        *ir, all_domain_attr, pmesh);
//
//   MultiVector X{u_t, v_t, E_t, nodes_t};
//   MultiVector Y{grad_u, grad_E};
//   rev->Mult(X, Y);     // Y holds VJP w.r.t. U and EFLD (seed = v_t)
//
// The first template parameter `VFieldId` is the field id of the seed V.
// The remaining template parameters `DiffFieldIds...` are the field ids of
// the physics inputs to differentiate. Each output FOp of the physics qf is
// rebound to field id `VFieldId` (Gradient<U> → Gradient<V>, etc.) and added
// as a non-differentiated input to the synthesized scalar density qf.
//
// All Enzyme calls are synthesized internally; output seed is 1.
// ===========================================================================
#pragma once

#include "mfem/fem/dfem/doperator.hpp"
#include "mfem/fem/dfem/backends/local_qf/prelude.hpp"
#include "mfem/fem/dfem/util.hpp"
#include "mfem/fem/dfem/fieldoperator.hpp"
#include "mfem/linalg/tensor.hpp"
#include "mfem/general/enzyme.hpp"

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace mfem::future::reverse
{

// --- map the head N types of a dfem tuple into a std::tuple -----------------
namespace detail
{
template <typename DFEMTuple, typename Seq> struct head_std_tuple_impl;

template <typename... Ts, std::size_t... Is>
struct head_std_tuple_impl<mfem::future::tuple<Ts...>, std::index_sequence<Is...>>
{
   using type = std::tuple<
      typename mfem::future::tuple_element<Is,
                                           mfem::future::tuple<Ts...>>::type...>;
};

template <typename DFEMTuple, std::size_t N>
using head_std_tuple = typename head_std_tuple_impl<
                       DFEMTuple, std::make_index_sequence<N>>::type;

// Slice [Start, Start+Count) of a dfem tuple into a std::tuple.
template <std::size_t Start, std::size_t Count, typename DFEMTuple,
          typename Seq = std::make_index_sequence<Count>>
struct slice_std_tuple_impl;

template <std::size_t Start, std::size_t Count, typename... Ts,
          std::size_t... Is>
struct slice_std_tuple_impl<Start, Count, mfem::future::tuple<Ts...>,
       std::index_sequence<Is...>>
{
   using type = std::tuple<
      typename mfem::future::tuple_element<Start + Is,
                                           mfem::future::tuple<Ts...>>::type...>;
};

template <typename DFEMTuple, std::size_t Start, std::size_t Count>
using slice_std_tuple = typename slice_std_tuple_impl<Start, Count,
      DFEMTuple>::type;

// Decay each element of a std::tuple. (decay_tuple in util.hpp is for
// mfem::future::tuple only.)
template <typename T> struct decay_std_tuple;
template <typename... Ts>
struct decay_std_tuple<std::tuple<Ts...>>
{
   using type = std::tuple<std::remove_cv_t<std::remove_reference_t<Ts>>...>;
};
template <typename T>
using decay_std_tuple_t = typename decay_std_tuple<T>::type;

// Generic scalar inner product between a physics output and the matching v
// dual. Compile-time overload set: scalar*scalar, dot(rank-1,rank-1),
// ddot(rank-2,rank-2). Used to build the synthesized scalar QoI density.
template <typename T>
MFEM_HOST_DEVICE inline
std::enable_if_t<std::is_arithmetic<T>::value, T>
weak_inner(const T &a, const T &b) { return a * b; }

template <typename S, typename T, int m>
MFEM_HOST_DEVICE inline
auto weak_inner(const tensor<S, m> &a, const tensor<T, m> &b)
-> decltype(dot(a, b))
{ return dot(a, b); }

template <typename S, typename T, int m, int n>
MFEM_HOST_DEVICE inline
auto weak_inner(const tensor<S, m, n> &a, const tensor<T, m, n> &b)
-> decltype(ddot(a, b))
{ return ddot(a, b); }

// Rebind a FieldOperator's field id (Gradient<U> with NewId=V → Gradient<V>).
// Only Value, Gradient, Sum, Identity carry a field id; Weight does not and
// is rejected by static_assert (it has no v-dual interpretation).
template <typename Op, int NewId> struct rebind_field_id;

template <int OldId, int NewId>
struct rebind_field_id<Value<OldId>, NewId>     { using type = Value<NewId>; };
template <int OldId, int NewId>
struct rebind_field_id<Gradient<OldId>, NewId>  { using type = Gradient<NewId>; };
template <int OldId, int NewId>
struct rebind_field_id<Sum<OldId>, NewId>       { using type = Sum<NewId>; };
template <int OldId, int NewId>
struct rebind_field_id<Identity<OldId>, NewId>  { using type = Identity<NewId>; };

template <typename Op, int NewId>
using rebind_field_id_t = typename rebind_field_id<Op, NewId>::type;

// Find the position of a field-id `Id` in a pack of FOp types.
template <int Id, typename... FOps>
constexpr std::size_t fop_index_for_id()
{
   constexpr int ids[] = { FOps::GetFieldId() ... };
   for (std::size_t i = 0; i < sizeof...(FOps); ++i)
   {
      if (ids[i] == Id) { return i; }
   }
   return static_cast<std::size_t>(-1);
}

template <typename... Seqs> struct concat_index_sequences;

template <>
struct concat_index_sequences<>
{
   using type = std::index_sequence<>;
};

template <std::size_t... Is>
struct concat_index_sequences<std::index_sequence<Is...>>
{
   using type = std::index_sequence<Is...>;
};

template <std::size_t... Is, std::size_t... Js, typename... Rest>
struct concat_index_sequences<std::index_sequence<Is...>,
                              std::index_sequence<Js...>,
                              Rest...>
   : concat_index_sequences<std::index_sequence<Is..., Js...>, Rest...>
{};

template <typename... Seqs>
using concat_index_sequences_t = typename concat_index_sequences<Seqs...>::type;

template <int Id, std::size_t Pos, typename... FOps>
struct fop_positions_for_id_impl;

template <int Id, std::size_t Pos>
struct fop_positions_for_id_impl<Id, Pos>
{
   using type = std::index_sequence<>;
   static constexpr bool found = false;
};

template <int Id, std::size_t Pos, typename FOp, typename... Rest>
struct fop_positions_for_id_impl<Id, Pos, FOp, Rest...>
{
private:
   using tail = fop_positions_for_id_impl<Id, Pos + 1, Rest...>;

public:
   using type = std::conditional_t<
      FOp::GetFieldId() == Id,
      concat_index_sequences_t<std::index_sequence<Pos>, typename tail::type>,
      typename tail::type>;
   static constexpr bool found = (FOp::GetFieldId() == Id) || tail::found;
};

template <int Id, typename... FOps>
using fop_positions_for_id =
   typename fop_positions_for_id_impl<Id, 0, FOps...>::type;

template <int Id, typename... FOps>
inline constexpr bool fop_has_id =
   fop_positions_for_id_impl<Id, 0, FOps...>::found;

} // namespace detail

// --- ReverseAdapter qfunction -----------------------------------------------
//
// Calls __enzyme_autodiff with seed=1 and uniform enzyme_dup for every input,
// then extracts the shadows for the user-selected DiffIs... positions.

template <typename DensityQF, typename DiffSeq, typename InsStdTuple>
struct ReverseAdapter;

template <typename DensityQF, std::size_t... DiffIs, typename... Ins>
struct ReverseAdapter<DensityQF, std::index_sequence<DiffIs...>,
       std::tuple<Ins...>>
{
   DensityQF inner;

   static constexpr std::size_t NIn = sizeof...(Ins);

   MFEM_HOST_DEVICE static void density_wrapper(
      const DensityQF *qf,
      std::decay_t<Ins> *... ins,
      real_t *q_out)
   {
      (*qf)(*ins..., *q_out);
   }

   MFEM_HOST_DEVICE inline
   void operator()(
      const std::decay_t<Ins> &... in_vals,
      std::decay_t<typename std::tuple_element<DiffIs,
                                               std::tuple<Ins...>>::type> &... user_shadows) const
   {
      std::tuple<std::decay_t<Ins>...> primals{in_vals...};
      std::tuple<std::decay_t<Ins>...> shadows{};

      real_t q    = real_t(0);
      real_t seed = real_t(1);

      do_autodiff(primals, shadows, q, seed,
                  std::make_index_sequence<NIn> {});

      ((user_shadows = std::get<DiffIs>(shadows)), ...);
   }

private:
   template <typename PrimalsT, typename ShadowsT, std::size_t... AllIs>
   MFEM_HOST_DEVICE inline
   void do_autodiff(
      PrimalsT &primals,
      ShadowsT &shadows,
      real_t &q,
      real_t &seed,
      std::index_sequence<AllIs...>) const
   {
#ifdef MFEM_USE_ENZYME
      auto all = std::tuple_cat(
                    std::make_tuple(
                       reinterpret_cast<void *>(&density_wrapper),
                       enzyme_const, const_cast<DensityQF *>(&inner)),
                    std::make_tuple(enzyme_dup,
                                    &std::get<AllIs>(primals),
                                    &std::get<AllIs>(shadows))...,
                    std::make_tuple(enzyme_dupnoneed, &q, &seed));
      std::apply([](auto &&... args)
      {
         __enzyme_autodiff<void>(args...);
      }, all);
#else
      MFEM_ABORT("MakeReverseGradientOperator requires MFEM_USE_ENZYME");
#endif
   }
};

// --- WeakFormDensityQF ------------------------------------------------------
//
// Wraps a user-supplied physics residual qfunction. Synthesizes a scalar QoI
// density by calling the physics qf to compute its outputs, then summing the
// inner products with the v duals (passed in as additional inputs).
//
// Parameter pack layout:
//   PhysIns... : decayed types of the physics qf's input parameters
//   PhysOuts...: decayed types of the physics qf's output parameters
// The synthesized operator()'s signature is
//   (PhysIns..., PhysOuts..., real_t& q_out)
// where the second pack is the v duals (one per physics output, same type).

template <typename PhysicsQF, typename PhysInsTuple, typename PhysOutsTuple>
struct WeakFormDensityQF;

template <typename PhysicsQF, typename... PhysIns, typename... PhysOuts>
struct WeakFormDensityQF<PhysicsQF, std::tuple<PhysIns...>,
       std::tuple<PhysOuts...>>
{
   PhysicsQF physics;

   MFEM_HOST_DEVICE inline
   void operator()(
      const PhysIns &... in_vals,
      const PhysOuts &... v_duals,
      real_t &q_out) const
   {
      // Default-initialize outputs, call physics, contract with v duals.
      std::tuple<PhysOuts...> outs{};
      call_physics(in_vals..., outs,
                   std::make_index_sequence<sizeof...(PhysOuts)> {});
      q_out = real_t(0);
      accumulate(outs, std::forward_as_tuple(v_duals...), q_out,
                 std::make_index_sequence<sizeof...(PhysOuts)> {});
   }

private:
   template <std::size_t... Os>
   MFEM_HOST_DEVICE inline
   void call_physics(const PhysIns &... in_vals,
                     std::tuple<PhysOuts...> &outs,
                     std::index_sequence<Os...>) const
   {
      physics(in_vals..., std::get<Os>(outs)...);
   }

   template <typename OutsT, typename VsT, std::size_t... Os>
   MFEM_HOST_DEVICE inline
   void accumulate(const OutsT &outs, const VsT &vs, real_t &q,
                   std::index_sequence<Os...>) const
   {
      ((q += detail::weak_inner(std::get<Os>(outs), std::get<Os>(vs))), ...);
   }
};

// --- Internal helper: build adapter + integrator (scalar-density form) ------
//
// Takes a *scalar-density* qfunction (one whose final argument is real_t&
// q_out) and an augmented input FOp tuple. This is the layer that talks to
// dFEM. The public factory below produces the density qf and augmented fops
// via WeakFormDensityQF, then calls in here.

template <typename DensityQF,
          typename... InputFOps,
          std::size_t... DiffPositions>
void add_with_positions(
   mfem::future::DifferentiableOperator& dop,
   DensityQF density_qf,
   mfem::future::tuple<InputFOps...> input_fops,
   const IntegrationRule &ir,
   const Array<int> &domain_attrs,
   std::index_sequence<DiffPositions...>)
{
   using qf_sig = typename get_function_signature<DensityQF>::type;
   using qf_params = typename qf_sig::parameter_ts;
   constexpr std::size_t NIn = sizeof...(InputFOps);

   using ins_std_tuple = detail::head_std_tuple<qf_params, NIn>;
   using AdapterT = ReverseAdapter<DensityQF,
                                   std::index_sequence<DiffPositions...>,
                                   ins_std_tuple>;

   auto output_fops = mfem::future::tuple{
      mfem::future::get<DiffPositions>(input_fops)...
   };

   AdapterT adapter{ std::move(density_qf) };
   dop.template AddDomainIntegrator<LocalQFBackend>(
      adapter,
      input_fops,
      output_fops,
      ir, domain_attrs, std::integer_sequence<size_t> {});
}

// --- Boundary variant of add_with_positions ---------------------------------

template <typename DensityQF,
          typename... InputFOps,
          std::size_t... DiffPositions>
void add_boundary_with_positions(
   mfem::future::DifferentiableOperator& dop,
   DensityQF density_qf,
   mfem::future::tuple<InputFOps...> input_fops,
   const IntegrationRule &ir,
   const Array<int> &boundary_attrs,
   std::index_sequence<DiffPositions...>)
{
   using qf_sig = typename get_function_signature<DensityQF>::type;
   using qf_params = typename qf_sig::parameter_ts;
   constexpr std::size_t NIn = sizeof...(InputFOps);

   using ins_std_tuple = detail::head_std_tuple<qf_params, NIn>;
   using AdapterT = ReverseAdapter<DensityQF,
                                   std::index_sequence<DiffPositions...>,
                                   ins_std_tuple>;

   auto output_fops = mfem::future::tuple{
      mfem::future::get<DiffPositions>(input_fops)...
   };

   AdapterT adapter{ std::move(density_qf) };
   dop.template AddBoundaryIntegrator<LocalQFBackend>(
      adapter,
      input_fops,
      output_fops,
      ir, boundary_attrs, std::integer_sequence<size_t> {});
}

// --- Public factory ---------------------------------------------------------

template <int VFieldId, std::size_t... DiffFieldIds,
          typename PhysicsQF,
          typename... PhysInputFOps,
          typename... PhysOutputFOps>
void AddReverseGradientIntegrator(
   mfem::future::DifferentiableOperator& dop,
   PhysicsQF physics_qf,
   mfem::future::tuple<PhysInputFOps...> physics_input_fops,
   [[maybe_unused]] mfem::future::tuple<PhysOutputFOps...> physics_output_fops,
   const IntegrationRule &ir,
   const Array<int> &domain_attrs,
   std::integer_sequence<std::size_t, DiffFieldIds...>)
{
   // Physics qf signature: parameter_ts = (PhysIns..., PhysOuts...) with
   // outputs taken by mutable reference. Decay to value types for both packs.
   using phys_sig = typename get_function_signature<PhysicsQF>::type;
   using phys_params = typename phys_sig::parameter_ts;
   constexpr std::size_t NPhysIn  = sizeof...(PhysInputFOps);
   constexpr std::size_t NPhysOut = sizeof...(PhysOutputFOps);

   // Split phys_params into (head NPhysIn) and (next NPhysOut), both decayed.
   using PhysInsRaw  = detail::head_std_tuple<phys_params, NPhysIn>;
   using PhysOutsRaw = detail::slice_std_tuple<phys_params, NPhysIn, NPhysOut>;
   using PhysInsTuple  = detail::decay_std_tuple_t<PhysInsRaw>;
   using PhysOutsTuple = detail::decay_std_tuple_t<PhysOutsRaw>;

   using DensityQF = WeakFormDensityQF<PhysicsQF, PhysInsTuple, PhysOutsTuple>;

   // Augmented FOp tuple = physics_input_fops ++ rebind(physics_output_fops -> V).
   // All these FOps are stateless aggregates with constexpr default ctors,
   // so we can just default-construct the merged tuple type. (mfem::future
   // ::tuple has no tuple_cat.)
   (void)physics_input_fops;
   mfem::future::tuple<PhysInputFOps...,
       detail::rebind_field_id_t<PhysOutputFOps, VFieldId>...> augmented_input_fops{};

   // Compile-time positions of DiffFieldIds in the augmented input FOp tuple.
   static_assert(((DiffFieldIds != static_cast<std::size_t>(VFieldId)) && ...),
                 "VFieldId must not be one of the differentiated field ids");
   static_assert(((detail::fop_has_id<DiffFieldIds,
                   PhysInputFOps..., detail::rebind_field_id_t<PhysOutputFOps, VFieldId>...>) && ...),
                 "a requested diff field id does not appear in physics_input_fops");

   DensityQF density_qf{ std::move(physics_qf) };
   using DiffPositions = detail::concat_index_sequences_t<
                         detail::fop_positions_for_id<DiffFieldIds,
                         PhysInputFOps...,
                         detail::rebind_field_id_t<PhysOutputFOps, VFieldId>...>...>;
   add_with_positions(
             dop,
             std::move(density_qf),
             augmented_input_fops,
             ir, domain_attrs,
             DiffPositions {});
}

// --- Boundary variant of AddReverseGradientIntegrator -----------------------

template <int VFieldId, std::size_t... DiffFieldIds,
          typename PhysicsQF,
          typename... PhysInputFOps,
          typename... PhysOutputFOps>
void AddReverseBoundaryGradientIntegrator(
   mfem::future::DifferentiableOperator& dop,
   PhysicsQF physics_qf,
   mfem::future::tuple<PhysInputFOps...> physics_input_fops,
   [[maybe_unused]] mfem::future::tuple<PhysOutputFOps...> physics_output_fops,
   const IntegrationRule &ir,
   const Array<int> &boundary_attrs,
   std::integer_sequence<std::size_t, DiffFieldIds...>)
{
   using phys_sig = typename get_function_signature<PhysicsQF>::type;
   using phys_params = typename phys_sig::parameter_ts;
   constexpr std::size_t NPhysIn  = sizeof...(PhysInputFOps);
   constexpr std::size_t NPhysOut = sizeof...(PhysOutputFOps);

   using PhysInsRaw  = detail::head_std_tuple<phys_params, NPhysIn>;
   using PhysOutsRaw = detail::slice_std_tuple<phys_params, NPhysIn, NPhysOut>;
   using PhysInsTuple  = detail::decay_std_tuple_t<PhysInsRaw>;
   using PhysOutsTuple = detail::decay_std_tuple_t<PhysOutsRaw>;

   using DensityQF = WeakFormDensityQF<PhysicsQF, PhysInsTuple, PhysOutsTuple>;

   (void)physics_input_fops;
   mfem::future::tuple<PhysInputFOps...,
       detail::rebind_field_id_t<PhysOutputFOps, VFieldId>...> augmented_input_fops{};

   static_assert(((DiffFieldIds != static_cast<std::size_t>(VFieldId)) && ...),
                 "VFieldId must not be one of the differentiated field ids");
   static_assert(((detail::fop_has_id<DiffFieldIds,
                   PhysInputFOps..., detail::rebind_field_id_t<PhysOutputFOps, VFieldId>...>) && ...),
                 "a requested diff field id does not appear in physics_input_fops");

   DensityQF density_qf{ std::move(physics_qf) };
   using DiffPositions = detail::concat_index_sequences_t<
                         detail::fop_positions_for_id<DiffFieldIds,
                         PhysInputFOps...,
                         detail::rebind_field_id_t<PhysOutputFOps, VFieldId>...>...>;
   add_boundary_with_positions(
             dop,
             std::move(density_qf),
             augmented_input_fops,
             ir, boundary_attrs,
             DiffPositions {});
}

template <int VFieldId, int... DiffFieldIds,
          typename PhysicsQF,
          typename... PhysInputFOps,
          typename... PhysOutputFOps>
std::unique_ptr<DifferentiableOperator> MakeReverseGradientOperator(
   PhysicsQF physics_qf,
   mfem::future::tuple<PhysInputFOps...> physics_input_fops,
   mfem::future::tuple<PhysOutputFOps...> physics_output_fops,
   const std::vector<FieldDescriptor> &inputs,
   const IntegrationRule &ir,
   const Array<int> &domain_attrs,
   ParMesh &pmesh)
{
   constexpr int diff_ids[] = { DiffFieldIds... };
   std::vector<FieldDescriptor> outputs;
   outputs.reserve(sizeof...(DiffFieldIds));
   for (int id : diff_ids)
   {
      bool found = false;
      for (const auto &fd : inputs)
      {
         if (static_cast<int>(fd.id) == id)
         {
            outputs.push_back(fd);
            found = true;
            break;
         }
      }
      MFEM_VERIFY(found, "diff field id has no matching FieldDescriptor "
                  "in inputs vector");
   }

   auto dop = std::make_unique<DifferentiableOperator>(inputs, outputs, pmesh);
   AddReverseGradientIntegrator<VFieldId>(*dop, std::move(physics_qf),
                                          physics_input_fops, physics_output_fops,
                                          ir, domain_attrs,
                                          std::integer_sequence<std::size_t, static_cast<std::size_t>(DiffFieldIds)...>{});
   return dop;
}

// --- ScalarSumDensityQF ------------------------------------------------------
//
// Wraps a physics residual qf whose output FOps are all Sum<...> (so the v
// seed is a global scalar 1 — pure scalar QoI). Synthesizes a scalar density
// qf with signature (physics_inputs..., real_t& q_out) that calls physics and
// accumulates each output directly into q_out (no v duals needed).

template <typename PhysicsQF, typename PhysInsTuple, typename PhysOutsTuple>
struct ScalarSumDensityQF;

template <typename PhysicsQF, typename... PhysIns, typename... PhysOuts>
struct ScalarSumDensityQF<PhysicsQF, std::tuple<PhysIns...>,
       std::tuple<PhysOuts...>>
{
   PhysicsQF physics;

   MFEM_HOST_DEVICE inline
   void operator()(
      const PhysIns &... in_vals,
      real_t &q_out) const
   {
      std::tuple<PhysOuts...> outs{};
      call_physics(in_vals..., outs,
                   std::make_index_sequence<sizeof...(PhysOuts)> {});
      q_out = real_t(0);
      accumulate(outs, q_out,
                 std::make_index_sequence<sizeof...(PhysOuts)> {});
   }

private:
   template <std::size_t... Os>
   MFEM_HOST_DEVICE inline
   void call_physics(const PhysIns &... in_vals,
                     std::tuple<PhysOuts...> &outs,
                     std::index_sequence<Os...>) const
   {
      physics(in_vals..., std::get<Os>(outs)...);
   }

   template <typename OutsT, std::size_t... Os>
   MFEM_HOST_DEVICE inline
   void accumulate(const OutsT &outs, real_t &q,
                   std::index_sequence<Os...>) const
   {
      ((q += std::get<Os>(outs)), ...);
   }
};

// --- Scalar-QoI factory: all physics outputs are Sum<> (v ≡ 1) ---------------
//
// MakeReverseScalarSumOperator<DiffFieldIds...>(
//     physics_qf,
//     physics_input_fops,     // dfem tuple of FOps for the physics qf inputs
//     physics_output_fops,    // dfem tuple of FOps — all MUST be Sum<>
//     inputs,                 // FieldDescriptors for all primal inputs (no V)
//     ir, attrs, pmesh)
//
// Returned operator's Mult takes the primal inputs and produces one output
// per DiffFieldIds... — the partials of the globally-summed scalar QoI w.r.t.
// each chosen input field.

template <int... DiffFieldIds,
          typename PhysicsQF,
          typename... PhysInputFOps,
          typename... PhysOutputFOps>
std::unique_ptr<DifferentiableOperator> MakeReverseScalarSumOperator(
   PhysicsQF physics_qf,
   mfem::future::tuple<PhysInputFOps...> physics_input_fops,
   [[maybe_unused]] mfem::future::tuple<PhysOutputFOps...> physics_output_fops,
   const std::vector<FieldDescriptor> &inputs,
   const IntegrationRule &ir,
   const Array<int> &domain_attrs,
   ParMesh &pmesh)
{
   static_assert((is_sum_fop<PhysOutputFOps>::value && ...),
                 "MakeReverseScalarSumOperator requires all physics output "
                 "FOps to be Sum<...> (use MakeReverseGradientOperator if you "
                 "need v-dual contractions)");

   using phys_sig = typename get_function_signature<PhysicsQF>::type;
   using phys_params = typename phys_sig::parameter_ts;
   constexpr std::size_t NPhysIn  = sizeof...(PhysInputFOps);
   constexpr std::size_t NPhysOut = sizeof...(PhysOutputFOps);

   using PhysInsRaw  = detail::head_std_tuple<phys_params, NPhysIn>;
   using PhysOutsRaw = detail::slice_std_tuple<phys_params, NPhysIn, NPhysOut>;
   using PhysInsTuple  = detail::decay_std_tuple_t<PhysInsRaw>;
   using PhysOutsTuple = detail::decay_std_tuple_t<PhysOutsRaw>;

   using DensityQF = ScalarSumDensityQF<PhysicsQF, PhysInsTuple, PhysOutsTuple>;

   static_assert(((detail::fop_has_id<DiffFieldIds,
                   PhysInputFOps...>) && ...),
                 "a requested diff field id does not appear in physics_input_fops");

   DensityQF density_qf{ std::move(physics_qf) };
   using DiffPositions = detail::concat_index_sequences_t<
                         detail::fop_positions_for_id<DiffFieldIds,
                         PhysInputFOps...>...>;
   return make_with_positions(
             std::move(density_qf),
             physics_input_fops,
             inputs, ir, domain_attrs, pmesh,
             DiffPositions {});
}

} // namespace mfem::future::reverse
