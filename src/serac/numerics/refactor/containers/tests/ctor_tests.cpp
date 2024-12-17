#include <gtest/gtest.h>

#include "common.hpp"

TEST(UnitTest, DefaultCtor2D) {
  nd::array< double, 2 > arr2D;
}

TEST(UnitTest, DefaultCtor3D) {
  nd::array< double, 3 > arr3D;
}

TEST(UnitTest, CopyCtor2D) {
  nd::array< double, 2 > arr2D({3, 4});
  nd::array< double, 2 > copy2D = arr2D;
}

TEST(UnitTest, CopyCtor3D) {
  nd::array< double, 3 > arr3D({2, 3, 4});
  nd::array< double, 3 > copy3D = arr3D;
}

TEST(UnitTest, MoveCtor2D) {
  nd::array< double, 2 > arr2D({3, 4});
  nd::array< double, 2 > copy2D = std::move(arr2D);
  EXPECT_EQ(arr2D.data(), nullptr);
}

TEST(UnitTest, MoveCtor3D) {
  nd::array< double, 3 > arr3D({2, 3, 4});
  nd::array< double, 3 > copy3D = std::move(arr3D);
  EXPECT_EQ(arr3D.data(), nullptr);
}

TEST(UnitTest, CopyAssignmentCtor2D) {
  nd::array< double, 2 > arr2D({3, 4});
  nd::array< double, 2 > copy2D;
  copy2D = arr2D;
}

TEST(UnitTest, CopyAssignmentCtor3D) {
  nd::array< double, 3 > arr3D({2, 3, 4});
  nd::array< double, 3 > copy3D;
  copy3D = arr3D;
}

TEST(UnitTest, MoveAssignmentCtor2D) {
  nd::array< double, 2 > arr2D({3, 4});
  nd::array< double, 2 > copy2D;
  copy2D = std::move(arr2D);
  EXPECT_EQ(arr2D.data(), nullptr);
}

TEST(UnitTest, MoveAssignmentCtor3D) {
  nd::array< double, 3 > arr3D({2, 3, 4});
  nd::array< double, 3 > copy3D;
  copy3D = std::move(arr3D);
  EXPECT_EQ(arr3D.data(), nullptr);
}
