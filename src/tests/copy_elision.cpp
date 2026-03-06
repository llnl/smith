// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include "mfem.hpp"
#include <cstddef>

class ArrayCtr {
 private:
  mfem::Array<double> array_;
  static std::size_t copy_;
  static std::size_t default_;

 public:
  ArrayCtr() { default_++; }
  ArrayCtr(const ArrayCtr& other) : array_(other.array_) { copy_++; }

  static auto numDefaultCalls() { return default_; }
  static auto numCopyCalls() { return copy_; }

  void Append(double elem) { array_.Append(elem); }
};

std::size_t ArrayCtr::copy_ = 0;
std::size_t ArrayCtr::default_ = 0;

ArrayCtr doubleArrayMaker()
{
  constexpr int size = 10;
  ArrayCtr result;
  // Just a random loop - if nothing was done to the array before returning,
  // the compiler might be able to do an unnamed returned value optimimization
  for (auto i = 0; i < size; i++) {
    if (i % 2) {
      result.Append(i + 7 % 3);
    }
  }
  return result;
}

TEST(NRVO, NRVOMfemArray)
{
  ArrayCtr arr = doubleArrayMaker();
  ASSERT_EQ(ArrayCtr::numDefaultCalls(), 1);
  ASSERT_EQ(ArrayCtr::numCopyCalls(), 0);
}
