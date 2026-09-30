// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <vector>

#include "gtest/gtest.h"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/pchip.hpp"

namespace smith {

TEST(Pchip, MonotoneAndClamped)
{
  const std::vector coordinates{0.0, 1.0, 2.0, 4.0};
  const std::vector values{10.0, 15.0, 18.0, 20.0};
  const PchipData data(coordinates, values);
  const auto interpolant = data.view();

  EXPECT_DOUBLE_EQ(interpolant(-1.0), values.front());
  EXPECT_DOUBLE_EQ(interpolant(5.0), values.back());

  for (std::size_t i = 0; i + 1 < coordinates.size(); ++i) {
    EXPECT_DOUBLE_EQ(interpolant(coordinates[i]), values[i]);
    const double midpoint_value = interpolant(0.5 * (coordinates[i] + coordinates[i + 1]));
    EXPECT_GE(midpoint_value, values[i]);
    EXPECT_LE(midpoint_value, values[i + 1]);
  }
  EXPECT_DOUBLE_EQ(interpolant(coordinates.back()), values.back());
}

TEST(Pchip, SupportsRuntimeSizedData)
{
  constexpr std::size_t point_count = 64;
  std::vector<double> coordinates(point_count);
  std::vector<double> values(point_count);
  for (std::size_t i = 0; i < point_count; ++i) {
    coordinates[i] = static_cast<double>(i);
    values[i] = 2.0 * coordinates[i] + 1.0;
  }

  const PchipData data(coordinates, values);
  const auto interpolant = data.view();
  EXPECT_NEAR(interpolant(31.5), 64.0, 1.0e-14);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager application_manager(argc, argv);
  return RUN_ALL_TESTS();
}
