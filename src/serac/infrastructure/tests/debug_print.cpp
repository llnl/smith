// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>

#include "serac/infrastructure/debug_print.hpp"
#include "serac/serac_config.hpp"

class SlicErrorException : public std::exception {};

namespace serac {

TEST(DebugPrint, typeToString)
{
  int i = 0;
  std::string str = "test";
  double d = 3.14;

  EXPECT_EQ(typeToString(i), "int");
  EXPECT_EQ(typeToString(str), "std::string");
  EXPECT_EQ(typeToString(d), "double");

  const int ci = 0;
  const std::string cstr = "test";
  const double cd = 3.14;

  EXPECT_EQ(typeToString(ci), "const int");
  EXPECT_EQ(typeToString(cstr), "const std::string");
  EXPECT_EQ(typeToString(cd), "const double");
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
