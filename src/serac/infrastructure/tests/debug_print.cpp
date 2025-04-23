// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>

#include "serac/infrastructure/debug_print.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/serac_config.hpp"

namespace serac {

TEST(DebugPrint, typeToString)
{
  int i = 0;
  std::string str = "test";
  std::string str_type = "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >";
  double d = 3.14;

  EXPECT_EQ(typeToString(i), "int");
  EXPECT_EQ(typeToString(str), str_type);
  EXPECT_EQ(typeToString(d), "double");

  const int ci = 0;
  const std::string cstr = "test";
  const double cd = 3.14;

  EXPECT_EQ(typeToString(ci), "const int");
  EXPECT_EQ(typeToString(cstr), "const " + str_type);
  EXPECT_EQ(typeToString(cd), "const double");
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
