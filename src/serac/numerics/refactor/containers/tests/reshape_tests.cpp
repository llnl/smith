#include <gtest/gtest.h>

#include "common.hpp"

TEST(UnitTest, Flattening2D) {
  nd::array< double, 2 > arr2D = make_patterned_ndarray(stack::array{7u, 8u});
  nd::view< double > view1D = flatten(arr2D);
  EXPECT_EQ(view1D(0), 0);
  EXPECT_EQ(view1D(10), 12);
  EXPECT_EQ(view1D(24), 30);
}

TEST(UnitTest, Flattening3D) {
  nd::array< double, 3 > arr3D = make_patterned_ndarray(stack::array{6u, 7u, 8u});
  nd::view< double > view1D = flatten(arr3D);
  EXPECT_EQ(view1D(0), 0);
  EXPECT_EQ(view1D(10), 12);
  EXPECT_EQ(view1D(24), 30);
  EXPECT_EQ(view1D(57), 101);
}

TEST(UnitTest, Reshape) {

  nd::array< double, 2 > arr2D = make_patterned_ndarray(stack::array{8u, 3u});

  // select column 1 from 2D array
  // {
  //   { 0, ( 1),  2},
  //   {10, (11), 12},
  //   {20, (21), 22},
  //   {30, (31), 32},
  //   {40, (41), 42},
  //   {50, (51), 52},
  //   {60, (61), 62},
  //   {70, (71), 72}
  // }
  nd::view< double > view1D = arr2D(nd::range{0,8}, 1);
  EXPECT_EQ(view1D(0),  1);
  EXPECT_EQ(view1D(1), 11);
  EXPECT_EQ(view1D(2), 21);

  nd::view< double, 3 > view3D = nd::reshape<3>(view1D, {2,2,2});
  EXPECT_EQ(view3D(0, 0, 0),  1);
  EXPECT_EQ(view3D(1, 1, 0), 61);
  EXPECT_EQ(view3D(1, 0, 1), 51);
  EXPECT_EQ(view3D(0, 1, 0), 21);

  view3D = nd::reshape<3>(view1D, {2,2,2}, nd::ordering::col_major);
  EXPECT_EQ(view3D(0, 0, 0),  1);
  EXPECT_EQ(view3D(0, 1, 1), 61);
  EXPECT_EQ(view3D(1, 0, 1), 51);
  EXPECT_EQ(view3D(0, 1, 0), 21);
}