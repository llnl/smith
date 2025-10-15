// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "serac/serac_config.hpp"
#include "serac/infrastructure/application_manager.hpp"

namespace {
  // Stash copies that tests can read (after gtest strips its flags)
  int    g_argc = 0;
  char** g_argv = nullptr;
}

namespace serac {

TEST(ApplicationManager, Lifetime)
{
  serac::ApplicationManager applicationManager(g_argc, g_argv);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);  // removes --gtest_* flags
  g_argc = argc;                           // store leftovers for tests
  g_argv = argv;
  return RUN_ALL_TESTS();
}
