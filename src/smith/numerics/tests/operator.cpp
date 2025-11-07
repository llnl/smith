// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "axom/slic/core/SimpleLogger.hpp"
#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/numerics/stdfunction_operator.hpp"
#include "smith/infrastructure/application_manager.hpp"

namespace smith {

TEST(Operators, RectangularOperator)
{
  // profile mesh refinement
  MPI_Barrier(MPI_COMM_WORLD);

  // Construct a sample rectangular matrix
  mfem::DenseMatrix mat(2, 3);
  mat(0, 0) = 0.0;
  mat(0, 1) = 1.0;
  mat(0, 2) = 2.0;
  mat(1, 0) = 3.0;
  mat(1, 1) = 4.0;
  mat(1, 2) = 5.0;

  // Construct an input vector to test the rectangular stdfunction operator
  mfem::Vector in(3);
  in(0) = 6.0;
  in(1) = 7.0;
  in(2) = 8.0;

  // Construct container vectors for the output
  mfem::Vector out_1(2);
  out_1 = 0.0;

  mfem::Vector out_2(2);
  out_2 = 0.0;

  // Test the constructors for the rectangular stdfunction operators
  mfem_ext::StdFunctionOperator a(2, 3);

  mfem_ext::StdFunctionOperator b(2, 3, [&mat](const mfem::Vector& input, mfem::Vector& out) { mat.Mult(input, out); });

  mfem_ext::StdFunctionOperator c(
      2, 3, [&mat](const mfem::Vector& input, mfem::Vector& out) { mat.Mult(input, out); },
      [&mat](const mfem::Vector&) -> mfem::Operator& { return mat; });

  // Compute the action of the equivalent rectangular operators
  mat.Mult(in, out_1);
  c.Mult(in, out_2);

  // Compare the results
  for (int i = 0; i < out_1.Size(); ++i) {
    EXPECT_DOUBLE_EQ(out_1(i), out_2(i));
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
