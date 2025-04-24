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

namespace detail {
struct Material {
  struct State {
    double x;
  };
  double y;
};

}  // namespace detail

TEST(DebugPrint, typeString)
{
  int i = 0;
  std::string str = "test";
  std::string longStrType = "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >";
  double d = 3.14;
  detail::Material m;
  detail::Material::State ms;

  EXPECT_EQ(typeString(i), "int");
  std::string testStr = typeString(str);
  EXPECT_TRUE(testStr == "std::string" || testStr == longStrType)
      << "Expected type string to be either 'std::string' or '" << longStrType << "', but got: " << testStr;
  EXPECT_EQ(typeString(d), "double");
  EXPECT_EQ(typeString(m), "serac::detail::Material");
  EXPECT_EQ(typeString(ms), "serac::detail::Material::State");

  const int ci = 0;
  const std::string cstr = "test";
  const double cd = 3.14;
  const detail::Material::State cms{5.0};
  const detail::Material cm{6.0};

  EXPECT_EQ(typeString(ci), "const int");
  testStr = typeString(cstr);
  EXPECT_TRUE(testStr == "const std::string" || testStr == "const " + longStrType)
      << "Expected type string to be either 'const std::string' or 'const " << longStrType << "', but got: " << testStr;
  EXPECT_EQ(typeString(cd), "const double");
  EXPECT_EQ(typeString(cm), "const serac::detail::Material");
  EXPECT_EQ(typeString(cms), "const serac::detail::Material::State");
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
