#include <gtest/gtest.h>

#include "common.hpp"

TEST(UnitTest, StridesColumnMajor2D) {
  stack::array< uint32_t, 2 > shape = {4, 6};
  stack::array< uint32_t, 2 > strides = nd::compute_strides(shape, nd::ordering::col_major);
  EXPECT_EQ(strides[0], 1);
  EXPECT_EQ(strides[1], 4);

  strides = nd::compute_strides(shape, nd::ordering::col_major, 4);
  EXPECT_EQ(strides[0], 4);
  EXPECT_EQ(strides[1], 16);
}

TEST(UnitTest, StridesColumnMajor3D) {
  stack::array< uint32_t, 3 > shape = {4, 6, 7};
  stack::array< uint32_t, 3 > strides = nd::compute_strides(shape, nd::ordering::col_major);
  EXPECT_EQ(strides[0], 1);
  EXPECT_EQ(strides[1], 4);
  EXPECT_EQ(strides[2], 24);

  strides = nd::compute_strides(shape, nd::ordering::col_major, 4);
  EXPECT_EQ(strides[0], 4);
  EXPECT_EQ(strides[1], 16);
  EXPECT_EQ(strides[2], 96);
}

TEST(UnitTest, StridesRowMajor2D) {
  stack::array< uint32_t, 2 > shape = {4, 6};
  stack::array< uint32_t, 2 > strides = nd::compute_strides(shape, nd::ordering::row_major);
  EXPECT_EQ(strides[0], 6);
  EXPECT_EQ(strides[1], 1);

  strides = nd::compute_strides(shape, nd::ordering::row_major, 4);
  EXPECT_EQ(strides[0], 24);
  EXPECT_EQ(strides[1], 4);
}

TEST(UnitTest, StridesRowMajor3D) {
  stack::array< uint32_t, 3 > shape = {4, 6, 7};
  stack::array< uint32_t, 3 > strides = nd::compute_strides(shape, nd::ordering::row_major);
  EXPECT_EQ(strides[0], 42);
  EXPECT_EQ(strides[1], 7);
  EXPECT_EQ(strides[2], 1);

  strides = nd::compute_strides(shape, nd::ordering::row_major, 4);
  EXPECT_EQ(strides[0], 168);
  EXPECT_EQ(strides[1], 28);
  EXPECT_EQ(strides[2], 4);
}