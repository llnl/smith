// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file variant.hpp
 *
 * @brief This file contains the declaration of a two-element variant type
 *
 * This is necessary to work around an issue reported to LC in April 2021 regarding
 * the use of the GCC 8 libstdc++ variant header with NVCC 11.  As of July 2021 this
 * has not been fixed.  Additionally, the non-recursive implementation here should reduce
 * compile times, though this effect may be limited for a variant with only two alternatives.
 */

#pragma once

#include <memory>
#include <type_traits>

namespace smith {

namespace detail {

/**
 * @brief Storage abstraction to provide trivial destructor when both variant types are trivially destructible
 * @tparam T0 The type of the first variant element
 * @tparam T1 The type of the second variant element
 * @tparam SFINAE Used to switch between the trivially destructible and not-trivially destructible implementations
 */
template <typename T0, typename T1, typename SFINAE = void>
struct variant_storage {
  /**
   * @brief The index of the active member
   */
  int index_ = 0;
  union {
    T0 t0_;  ///< type 0
    T1 t1_;  ///< type 1
  };

  /**
   * @brief Copy constructor for nontrivial types
   * Placement-new is required to correctly initialize the union
   * @param[in] other The variant_storage to copy from
   */
  constexpr variant_storage(const variant_storage& other) : index_(other.index_)
  {
    switch (index_) {
      case 0: {
        new (&t0_) T0(other.t0_);
        break;
      }
      case 1: {
        new (&t1_) T1(other.t1_);
        break;
      }
    }
  };

  /**
   * @brief Move constructor for nontrivial types
   * Placement-new is required to correctly initialize the union
   * @param[in] other The variant_storage to move from
   */
  constexpr variant_storage(variant_storage&& other) : index_(other.index_)
  {
    switch (index_) {
      case 0: {
        new (&t0_) T0(std::move(other.t0_));
        break;
      }
      case 1: {
        new (&t1_) T1(std::move(other.t1_));
        break;
      }
    }
  };

  /**
   * @brief Resets the union by destroying the active member
   * @note The index is technically invalid here after this function is called
   */
  constexpr void clear()
  {
    switch (index_) {
      case 0: {
        t0_.~T0();
        break;
      }
      case 1: {
        t1_.~T1();
        break;
      }
    }
  }

  /**
   * @brief Default constructor
   * Default initializes the first member of the variant
   */
  constexpr variant_storage() : index_{0}, t0_{} {}

  /**
   * @brief Destroys the variant by calling the destructor of the active member
   */
  ~variant_storage() { clear(); }
};

/**
 * @overload
 */
template <typename T0, typename T1>
struct variant_storage<T0, T1,
                       std::enable_if_t<std::is_trivially_destructible_v<T0> && std::is_trivially_destructible_v<T1>>> {
  /**
   * @brief The index of the active member
   */
  int index_ = 0;
  union {
    /**
     * @brief The storage for the first data type
     */
    T0 t0_;
    /**
     * @brief The storage for the second data type
     */
    T1 t1_;
  };

  /**
   * @brief Default constructor
   * Default initializes the first member of the variant
   */
  constexpr variant_storage() : index_{0}, t0_{} {}

  /**
   * @brief No-op clear as both member types are trivially destructible
   */
  constexpr void clear() {}
};

/**
 * @brief Determines if T can be assigned to a variant<T0, T1>
 * @tparam T The type on the right-hand side of the assignment
 * @tparam T0 The first member type of the variant
 * @tparam T1 The second member type of the variant
 */
template <typename T, typename T0, typename T1>
struct is_variant_assignable {
  /**
   * @brief If T can be assigned to the variant type
   */
  constexpr static bool value = std::is_same_v<std::decay_t<T>, T0> || std::is_assignable_v<T0, T> ||
                                std::is_same_v<std::decay_t<T>, T1> || std::is_assignable_v<T1, T>;
};

}  // namespace detail

/**
 * @brief Obtains the type at index @p I of a variant<T0, T1>
 * @tparam I The index to find the corresponding type for
 * @tparam T0 The first member type of the variant
 * @tparam T1 The second member type of the variant
 */
template <int I, typename T0, typename T1>
struct variant_alternative;

/**
 * @brief Obtains the type at index 0 of a variant<T0, T1>
 * @tparam T0 The first member type of the variant
 * @tparam T1 The second member type of the variant
 */
template <typename T0, typename T1>
struct variant_alternative<0, T0, T1> {
  /**
   * @brief The type of the first member
   */
  using type = T0;
};

/**
 * @brief Obtains the type at index 1 of a variant<T0, T1>
 * @tparam T0 The first member type of the variant
 * @tparam T1 The second member type of the variant
 */
template <typename T0, typename T1>
struct variant_alternative<1, T0, T1> {
  /**
   * @brief The type of the second member
   */
  using type = T1;
};

/**
 * @brief A simple variant type that supports only two elements
 *
 * Avoids the recursive template instantiation associated with std::variant
 * and provides a (roughly) identical interface that is fully constexpr (when both types are
 * trivially destructible)
 * @tparam T0 The first member type of the variant
 * @tparam T1 The second member type of the variant
 */
template <typename T0, typename T1>
struct variant {
  /**
   * @brief Storage abstraction used to provide constexpr functionality when applicable
   */
  detail::variant_storage<T0, T1> storage_;

  /**
   * @brief Default constructor - will default-initialize first member of the variant
   */
  constexpr variant() = default;

  /**
   * @brief Default constructor
   * @note These are needed explicitly so the variant(T&&) doesn't match first
   */
  constexpr variant(const variant&) = default;

  /**
   * @brief Default constructor
   */
  constexpr variant(variant&&) = default;

  /**
   * @brief "Parameterized" constructor with which a value can be assigned
   * @tparam T The type of the parameter to assign to one of the variant elements
   * @param[in] t_ The parameter to assign to the variant's contents
   * @pre The parameter type @p T must be equal to or assignable to either @p T0 or @p T1
   * @note If the conversion is ambiguous, i.e., if @a t is equal or convertible to *both* @p T0 and @p T1,
   * the first element of the variant - of type @p T0 - will be assigned to
   */
  template <typename T, typename SFINAE = std::enable_if_t<detail::is_variant_assignable<T, T0, T1>::value>>
  constexpr variant(T&& t_)
  {
    if constexpr (std::is_same_v<std::decay_t<T>, T0> || std::is_assignable_v<T0, T>) {
      storage_.index_ = 0;
      new (&storage_.t0_) T0(std::forward<T>(t_));
    } else if constexpr (std::is_same_v<std::decay_t<T>, T1> || std::is_assignable_v<T1, T>) {
      storage_.index_ = 1;
      new (&storage_.t1_) T1(std::forward<T>(t_));
    } else {
      static_assert(sizeof(T) < 0, "Type not supported");
    }
  }

  /**
   * @brief Default assignment operator
   * @note These are needed explicitly so the operator=(T&&) doesn't match first
   *
   * @return The modified variant
   */
  constexpr variant& operator=(const variant&) = default;

  /**
   * @brief Default assignment operator
   *
   * @return The modified variant
   */
  constexpr variant& operator=(variant&&) = default;

  /**
   * "Parameterized" assignment with which a value can be assigned
   * @tparam T The type of the parameter to assign to one of the variant elements
   * @param[in] t The parameter to assign to the variant's contents
   * @see variant::variant(T&& t) for notes and preconditions
   */
  template <typename T, typename SFINAE = std::enable_if_t<detail::is_variant_assignable<T, T0, T1>::value>>
  constexpr variant& operator=(T&& t)
  {
    if constexpr (std::is_same_v<std::decay_t<T>, T0> || std::is_assignable_v<T0, T>) {
      if (storage_.index_ != 0) {
        storage_.clear();
      }
      storage_.t0_ = std::forward<T>(t);
      storage_.index_ = 0;
    } else if constexpr (std::is_same_v<std::decay_t<T>, T1> || std::is_assignable_v<T1, T>) {
      if (storage_.index_ != 1) {
        storage_.clear();
      }
      storage_.t1_ = std::forward<T>(t);
      storage_.index_ = 1;
    } else {
      static_assert(sizeof(T) < 0, "Type not supported");
    }
    return *this;
  }

  /**
   * @brief Returns the index of the active variant member
   */
  constexpr int index() const { return storage_.index_; }

  /**
   * @brief Returns the variant member at the provided index
   * @tparam I The index of the element to retrieve
   * @see std::variant::get
   */
  template <int I>
  friend constexpr typename variant_alternative<I, T0, T1>::type& get(variant& v)
  {
    if constexpr (I == 0) {
      return v.storage_.t0_;
    }
    if constexpr (I == 1) {
      return v.storage_.t1_;
    }
  }

  /// @overload
  template <int I>
  friend constexpr const typename variant_alternative<I, T0, T1>::type& get(const variant& v)
  {
    if constexpr (I == 0) {
      return v.storage_.t0_;
    }
    if constexpr (I == 1) {
      return v.storage_.t1_;
    }
  }
};

/**
 * @brief Returns the variant member of specified type
 * @tparam T The type of the element to retrieve
 * @tparam T0 The first member type of the variant
 * @tparam T1 The second member type of the variant
 * @param[in] v The variant to return the element of
 * @see std::variant::get
 * @pre T must be either @p T0 or @p T1
 * @note If T == T0 == T1, the element at index 0 will be returned
 */
template <typename T, typename T0, typename T1>
constexpr T& get(variant<T0, T1>& v)
{
  if constexpr (std::is_same_v<T, T0>) {
    return get<0>(v);
  } else if constexpr (std::is_same_v<T, T1>) {
    return get<1>(v);
  }
}

/// @overload
template <typename T, typename T0, typename T1>
constexpr const T& get(const variant<T0, T1>& v)
{
  if constexpr (std::is_same_v<T, T0>) {
    return get<0>(v);
  } else if constexpr (std::is_same_v<T, T1>) {
    return get<1>(v);
  }
}

/**
 * @brief Applies a functor to the active variant element
 * @param[in] visitor The functor to apply
 * @param[in] v The variant to apply the functor to
 * @see std::visit
 */
template <typename Visitor, typename Variant>
constexpr decltype(auto) visit(Visitor visitor, Variant&& v)
{
  if (v.index() == 0) {
    return visitor(get<0>(v));
  } else {
    return visitor(get<1>(v));
  }
}

/**
 * @brief Checks whether a variant's active member is of a certain type
 * @tparam T The type to check for
 * @param[in] v The variant to check
 * @see std::holds_alternative
 */
template <typename T, typename T0, typename T1>
bool holds_alternative(const variant<T0, T1>& v)
{
  if constexpr (std::is_same_v<T, T0>) {
    return v.index() == 0;
  } else if constexpr (std::is_same_v<T, T1>) {
    return v.index() == 1;
  }
  return false;
}

/**
 * @brief Returns the member of requested type if it's active, otherwise @p nullptr
 * @tparam T The type to check for
 * @param[in] v The variant to check/retrieve from
 * @see std::get_if
 */
template <typename T, typename T0, typename T1>
T* get_if(variant<T0, T1>* v)
{
  if constexpr (std::is_same_v<T, T0>) {
    return (v->index() == 0) ? &get<0>(*v) : nullptr;
  } else if constexpr (std::is_same_v<T, T1>) {
    return (v->index() == 1) ? &get<1>(*v) : nullptr;
  }
  return nullptr;
}

/// @overload
template <typename T, typename T0, typename T1>
const T* get_if(const variant<T0, T1>* v)
{
  if constexpr (std::is_same_v<T, T0>) {
    return (v->index() == 0) ? &get<0>(*v) : nullptr;
  } else if constexpr (std::is_same_v<T, T1>) {
    return (v->index() == 1) ? &get<1>(*v) : nullptr;
  }
  return nullptr;
}

namespace detail {

/**
 * @brief A helper type for uniform semantics over owning/non-owning pointers
 *
 * This logic is needed to integrate with the mesh and field reconstruction logic
 * provided by Sidre's MFEMSidreDataCollection.  When a Smith restart occurs, the
 * saved data is used to construct fully functional mfem::(Par)Mesh and
 * mfem::(Par)GridFunction objects.  The FiniteElementCollection and (Par)FiniteElementSpace
 * objects are intermediates in the construction of these objects and are therefore owned
 * by the MFEMSidreDataCollection in the case of a restart/reconstruction.  In a normal run,
 * Smith constructs the mesh and fields, so these FEColl and FESpace objects are owned
 * by Smith.  In both cases, the MFEMSidreDataCollection maintains ownership of the mesh
 * and field objects themselves.
 */
template <typename T>
using MaybeOwningPointer = variant<T*, std::unique_ptr<T>>;

/**
 * @brief Retrieves a reference to the underlying object in a MaybeOwningPointer
 * @param[in] obj The object to dereference
 */
template <typename T>
static T& retrieve(MaybeOwningPointer<T>& obj)
{
  return visit([](auto&& ptr) -> T& { return *ptr; }, obj);
}
/// @overload
template <typename T>
static const T& retrieve(const MaybeOwningPointer<T>& obj)
{
  return visit([](auto&& ptr) -> const T& { return *ptr; }, obj);
}

}  // namespace detail
}  // namespace smith
