// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <string>

#include "gtest/gtest.h"

#include "smith/infrastructure/debug_print.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/smith_config.hpp"

namespace smith {

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
  std::string gnuStdStr = "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >";
  std::string llvmStdStr = "std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>";
  double d = 3.14;
  detail::Material m;
  detail::Material::State ms;

  EXPECT_EQ(typeString(i), "int");
  std::string testStr = typeString(str);
  EXPECT_TRUE(testStr == "std::string" || testStr == gnuStdStr || testStr == llvmStdStr)
      << "Expected type string to be either 'std::string' or '" << gnuStdStr << "' or '" << llvmStdStr
      << "', but got: " << testStr;
  EXPECT_EQ(typeString(d), "double");
  EXPECT_EQ(typeString(m), "smith::detail::Material");
  EXPECT_EQ(typeString(ms), "smith::detail::Material::State");

  const int ci = 0;
  const std::string cstr = "test";
  const double cd = 3.14;
  const detail::Material::State cms{5.0};
  const detail::Material cm{6.0};

  EXPECT_EQ(typeString(ci), "const int");
  testStr = typeString(cstr);
  EXPECT_TRUE(testStr == "const std::string" || testStr == "const " + gnuStdStr || testStr == "const " + llvmStdStr)
      << "Expected type string to be either 'const std::string' or 'const " << gnuStdStr << "', or 'const "
      << llvmStdStr << "', but got: " << testStr;
  EXPECT_EQ(typeString(cd), "const double");
  EXPECT_EQ(typeString(cm), "const smith::detail::Material");
  EXPECT_EQ(typeString(cms), "const smith::detail::Material::State");
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
