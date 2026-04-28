// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"

namespace {
// Stash copies that tests can read (after gtest strips its flags)
int g_argc = 0;
char** g_argv = nullptr;
}  // namespace

namespace smith {

TEST(ApplicationManager, Lifetime) { smith::ApplicationManager applicationManager(g_argc, g_argv); }

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);  // removes --gtest_* flags
  g_argc = argc;                           // store leftovers for tests
  g_argv = argv;
  return RUN_ALL_TESTS();
}
