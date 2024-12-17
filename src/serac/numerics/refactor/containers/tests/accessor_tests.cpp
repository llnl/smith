#include <gtest/gtest.h>

#include "common.hpp"

TEST(UnitTest, BasicAccess) {

  nd::array< double, 2 > arr2D = make_patterned_ndarray(stack::array{7u, 8u});
  EXPECT_EQ(arr2D(0, 0), 00);
  EXPECT_EQ(arr2D(1, 0), 10);
  EXPECT_EQ(arr2D(2, 1), 21);
  EXPECT_EQ(arr2D(4, 2), 42);
  EXPECT_EQ(arr2D(6, 7), 67);

  nd::view< double, 2 > view2D = arr2D;
  EXPECT_EQ(view2D(0, 0), 00);
  EXPECT_EQ(view2D(1, 0), 10);
  EXPECT_EQ(view2D(2, 1), 21);
  EXPECT_EQ(view2D(4, 2), 42);
  EXPECT_EQ(view2D(6, 7), 67);

  nd::array< double, 3 > arr3D = make_patterned_ndarray(stack::array{6u, 7u, 8u});
  EXPECT_EQ(arr3D(0, 0, 0), 000);
  EXPECT_EQ(arr3D(1, 3, 0), 130);
  EXPECT_EQ(arr3D(2, 0, 2), 202);
  EXPECT_EQ(arr3D(3, 1, 7), 317);
  EXPECT_EQ(arr3D(5, 6, 7), 567);

  //#ifdef NDARRAY_ENABLE_BOUNDS_CHECKING
  //  EXPECT_EQ(arr3D(10, 10, 10), 1000);
  //#endif

  nd::view< double, 3 > view3D = arr3D;
  EXPECT_EQ(view3D(0, 0, 0), 000);
  EXPECT_EQ(view3D(1, 3, 0), 130);
  EXPECT_EQ(view3D(2, 0, 2), 202);
  EXPECT_EQ(view3D(3, 1, 7), 317);
  EXPECT_EQ(view3D(5, 6, 7), 567);

}


TEST(UnitTest, ArraySlicing) {

  nd::array< double, 3 > arr3D = make_patterned_ndarray(stack::array{6u, 7u, 8u});

  double value = arr3D(1, 2, 3);
  arr3D(1,2,3) = value;

  nd::view< double, 2 > slice1 = arr3D(nd::range{2,5}, 4);
  EXPECT_EQ(slice1.stride[0], arr3D.stride[0]);
  EXPECT_EQ(slice1.stride[1], arr3D.stride[2]);
  EXPECT_EQ(slice1.shape[0], 3);
  EXPECT_EQ(slice1.shape[1], 8);
  EXPECT_EQ(slice1(0,0), arr3D(2,4,0));
  EXPECT_EQ(slice1(1,1), arr3D(3,4,1));
  EXPECT_EQ(slice1(2,2), arr3D(4,4,2));

  nd::view< double, 1 > slice2 = arr3D(1, 2, nd::range{2,5});
  EXPECT_EQ(slice2.stride[0], arr3D.stride[2]);
  EXPECT_EQ(slice2.shape[0], 3);
  EXPECT_EQ(slice2(0), arr3D(1,2,2));
  EXPECT_EQ(slice2(1), arr3D(1,2,3));
  EXPECT_EQ(slice2(2), arr3D(1,2,4));

  nd::view< double, 2 > slice3 = arr3D(nd::range{2,5}, nd::range{1,2}, 4);
  EXPECT_EQ(slice3.stride[0], arr3D.stride[0]);
  EXPECT_EQ(slice3.stride[1], arr3D.stride[1]);
  EXPECT_EQ(slice3.shape[0], 3);
  EXPECT_EQ(slice3.shape[1], 1);
  EXPECT_EQ(slice3(0,0), arr3D(2,1,4));
  EXPECT_EQ(slice3(1,0), arr3D(3,1,4));
  EXPECT_EQ(slice3(2,0), arr3D(4,1,4));

  nd::view< double, 3 > slice4 = arr3D(nd::range{1, 2}, nd::range{2, 5}, nd::range{1,2});
  EXPECT_EQ(slice4.stride[0], arr3D.stride[0]);
  EXPECT_EQ(slice4.stride[1], arr3D.stride[1]);
  EXPECT_EQ(slice4.stride[2], arr3D.stride[2]);
  EXPECT_EQ(slice4.shape[0], 1);
  EXPECT_EQ(slice4.shape[1], 3);
  EXPECT_EQ(slice4.shape[2], 1);
  EXPECT_EQ(slice4(0,0,0), arr3D(1, 2, 1));
  EXPECT_EQ(slice4(0,1,0), arr3D(1, 3, 1));
  EXPECT_EQ(slice4(0,2,0), arr3D(1, 4, 1));

}

TEST(UnitTest, ConstArraySlicing) {

  const nd::array< double, 3 > arr3D = make_patterned_ndarray(stack::array{6u, 7u, 8u});

  nd::view< const double, 2 > slice1 = arr3D(nd::range{2,5}, 4);
  EXPECT_EQ(slice1.stride[0], arr3D.stride[0]);
  EXPECT_EQ(slice1.stride[1], arr3D.stride[2]);
  EXPECT_EQ(slice1.shape[0], 3);
  EXPECT_EQ(slice1.shape[1], 8);
  EXPECT_EQ(slice1(0,0), arr3D(2,4,0));
  EXPECT_EQ(slice1(1,1), arr3D(3,4,1));
  EXPECT_EQ(slice1(2,2), arr3D(4,4,2));

  nd::view< const double, 1 > slice2 = arr3D(1, 2, nd::range{2,5});
  EXPECT_EQ(slice2.stride[0], arr3D.stride[2]);
  EXPECT_EQ(slice2.shape[0], 3);
  EXPECT_EQ(slice2(0), arr3D(1,2,2));
  EXPECT_EQ(slice2(1), arr3D(1,2,3));
  EXPECT_EQ(slice2(2), arr3D(1,2,4));

  nd::view< const double, 2 > slice3 = arr3D(nd::range{2,5}, nd::range{1,2}, 4);
  EXPECT_EQ(slice3.stride[0], arr3D.stride[0]);
  EXPECT_EQ(slice3.stride[1], arr3D.stride[1]);
  EXPECT_EQ(slice3.shape[0], 3);
  EXPECT_EQ(slice3.shape[1], 1);
  EXPECT_EQ(slice3(0,0), arr3D(2,1,4));
  EXPECT_EQ(slice3(1,0), arr3D(3,1,4));
  EXPECT_EQ(slice3(2,0), arr3D(4,1,4));

  nd::view< const double, 3 > slice4 = arr3D(nd::range{1, 2}, nd::range{2, 5}, nd::range{1,2});
  EXPECT_EQ(slice4.stride[0], arr3D.stride[0]);
  EXPECT_EQ(slice4.stride[1], arr3D.stride[1]);
  EXPECT_EQ(slice4.stride[2], arr3D.stride[2]);
  EXPECT_EQ(slice4.shape[0], 1);
  EXPECT_EQ(slice4.shape[1], 3);
  EXPECT_EQ(slice4.shape[2], 1);
  EXPECT_EQ(slice4(0,0,0), arr3D(1, 2, 1));
  EXPECT_EQ(slice4(0,1,0), arr3D(1, 3, 1));
  EXPECT_EQ(slice4(0,2,0), arr3D(1, 4, 1));

}

TEST(UnitTest, ViewConstSlicing) {

  const nd::array< double, 3 > arr3D = make_patterned_ndarray(stack::array{6u, 7u, 8u});
  nd::view< const double, 3 > view3D = arr3D;

  nd::view< const double, 2 > slice1 = view3D(nd::range{2,5}, 4);
  EXPECT_EQ(slice1.stride[0], arr3D.stride[0]);
  EXPECT_EQ(slice1.stride[1], arr3D.stride[2]);
  EXPECT_EQ(slice1.shape[0], 3);
  EXPECT_EQ(slice1.shape[1], 8);
  EXPECT_EQ(slice1(0,0), arr3D(2,4,0));
  EXPECT_EQ(slice1(1,1), arr3D(3,4,1));
  EXPECT_EQ(slice1(2,2), arr3D(4,4,2));

  nd::view< const double, 1 > slice2 = view3D(1, 2, nd::range{2,5});
  EXPECT_EQ(slice2.stride[0], arr3D.stride[2]);
  EXPECT_EQ(slice2.shape[0], 3);
  EXPECT_EQ(slice2(0), arr3D(1,2,2));
  EXPECT_EQ(slice2(1), arr3D(1,2,3));
  EXPECT_EQ(slice2(2), arr3D(1,2,4));

  nd::view< const double, 2 > slice3 = view3D(nd::range{2,5}, nd::range{1,2}, 4);
  EXPECT_EQ(slice3.stride[0], arr3D.stride[0]);
  EXPECT_EQ(slice3.stride[1], arr3D.stride[1]);
  EXPECT_EQ(slice3.shape[0], 3);
  EXPECT_EQ(slice3.shape[1], 1);
  EXPECT_EQ(slice3(0,0), arr3D(2,1,4));
  EXPECT_EQ(slice3(1,0), arr3D(3,1,4));
  EXPECT_EQ(slice3(2,0), arr3D(4,1,4));

  nd::view< const double, 3 > slice4 = view3D(nd::range{1, 2}, nd::range{2, 5}, nd::range{1,2});
  EXPECT_EQ(slice4.stride[0], arr3D.stride[0]);
  EXPECT_EQ(slice4.stride[1], arr3D.stride[1]);
  EXPECT_EQ(slice4.stride[2], arr3D.stride[2]);
  EXPECT_EQ(slice4.shape[0], 1);
  EXPECT_EQ(slice4.shape[1], 3);
  EXPECT_EQ(slice4.shape[2], 1);
  EXPECT_EQ(slice4(0,0,0), arr3D(1, 2, 1));
  EXPECT_EQ(slice4(0,1,0), arr3D(1, 3, 1));
  EXPECT_EQ(slice4(0,2,0), arr3D(1, 4, 1));

}

TEST(UnitTest, ConstViewSlicing) {

  nd::array< double, 3 > arr3D = make_patterned_ndarray(stack::array{6u, 7u, 8u});
  const nd::view< double, 3 > view3D = arr3D;

  nd::view< double, 2 > slice1 = view3D(nd::range{2,5}, 4);
  EXPECT_EQ(slice1.stride[0], arr3D.stride[0]);
  EXPECT_EQ(slice1.stride[1], arr3D.stride[2]);
  EXPECT_EQ(slice1.shape[0], 3);
  EXPECT_EQ(slice1.shape[1], 8);
  EXPECT_EQ(slice1(0,0), arr3D(2,4,0));
  EXPECT_EQ(slice1(1,1), arr3D(3,4,1));
  EXPECT_EQ(slice1(2,2), arr3D(4,4,2));

  nd::view< double, 1 > slice2 = view3D(1, 2, nd::range{2,5});
  EXPECT_EQ(slice2.stride[0], arr3D.stride[2]);
  EXPECT_EQ(slice2.shape[0], 3);
  EXPECT_EQ(slice2(0), arr3D(1,2,2));
  EXPECT_EQ(slice2(1), arr3D(1,2,3));
  EXPECT_EQ(slice2(2), arr3D(1,2,4));

  nd::view< double, 2 > slice3 = view3D(nd::range{2,5}, nd::range{1,2}, 4);
  EXPECT_EQ(slice3.stride[0], arr3D.stride[0]);
  EXPECT_EQ(slice3.stride[1], arr3D.stride[1]);
  EXPECT_EQ(slice3.shape[0], 3);
  EXPECT_EQ(slice3.shape[1], 1);
  EXPECT_EQ(slice3(0,0), arr3D(2,1,4));
  EXPECT_EQ(slice3(1,0), arr3D(3,1,4));
  EXPECT_EQ(slice3(2,0), arr3D(4,1,4));

  nd::view< double, 3 > slice4 = view3D(nd::range{1, 2}, nd::range{2, 5}, nd::range{1,2});
  EXPECT_EQ(slice4.stride[0], arr3D.stride[0]);
  EXPECT_EQ(slice4.stride[1], arr3D.stride[1]);
  EXPECT_EQ(slice4.stride[2], arr3D.stride[2]);
  EXPECT_EQ(slice4.shape[0], 1);
  EXPECT_EQ(slice4.shape[1], 3);
  EXPECT_EQ(slice4.shape[2], 1);
  EXPECT_EQ(slice4(0,0,0), arr3D(1, 2, 1));
  EXPECT_EQ(slice4(0,1,0), arr3D(1, 3, 1));
  EXPECT_EQ(slice4(0,2,0), arr3D(1, 4, 1));

}