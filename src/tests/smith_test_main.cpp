// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file
 * @brief Shared test driver used by aggregated TOSS 4 Cray test executables.
 */

#include "gtest/gtest.h"

#include "smith/infrastructure/application_manager.hpp"

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager application_manager(argc, argv);
  return RUN_ALL_TESTS();
}
