// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "smith/differentiable_numerics/make_time_info_material.hpp"
#include "smith/physics/common.hpp"

namespace smith {

struct TestMaterialState {};

struct StaticMaterial {
  using State = TestMaterialState;

  double density = 2.0;

  double operator()(State&, double grad_u, double param) const { return density + grad_u + param; }
};

TEST(TimeInfoMaterial, ForwardsStaticMaterial)
{
  auto material = makeTimeInfoMaterial(StaticMaterial{});
  StaticMaterial::State state;

  EXPECT_EQ(material.density, 2.0);
  EXPECT_EQ(material(TimeInfo(10.0, 20.0), state, 1.0, 100.0, 3.0), 6.0);
  EXPECT_EQ(material.material.density, 2.0);
}

}  // namespace smith
