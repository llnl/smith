#include <gtest/gtest.h>

#include "common.hpp"

TEST(UnitTest, BasicAccessOnCPU) {

  nd::array< double, 2 > arr2D = make_patterned_ndarray_on_GPU(stack::array{7u, 8u});
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

  nd::array< double, 3 > arr3D = make_patterned_ndarray_on_GPU(stack::array{6u, 7u, 8u});
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

__global__ void accessor_kernel(int * num_errors, nd::view<double, 2> v) {
  *num_errors += (v(0, 0) != 00);
  *num_errors += (v(1, 0) != 10);
  *num_errors += (v(2, 1) != 21);
  *num_errors += (v(4, 2) != 42);
  *num_errors += (v(6, 7) != 67);
}

__global__ void accessor_kernel(int * num_errors, nd::view<double, 3> v) {
  *num_errors += (v(0, 0, 0) != 000);
  *num_errors += (v(1, 3, 0) != 130);
  *num_errors += (v(2, 0, 2) != 202);
  *num_errors += (v(3, 1, 7) != 317);
  *num_errors += (v(5, 6, 7) != 567);
}

TEST(UnitTest, BasicAccessOnGPU) {
  int * errors;
  cudaMallocManaged(&errors, sizeof(int));

////////////////////////////////////////////////////////////////////////////////

  nd::array< double, 2 > arr2D = make_patterned_ndarray_on_GPU(stack::array{7u, 8u});

  *errors = 0;
  accessor_kernel<<<1,1>>>(errors, arr2D);
  cudaDeviceSynchronize();
  EXPECT_EQ(*errors, 0);

////////////////////////////////////////////////////////////////////////////////

  nd::array< double, 3 > arr3D = make_patterned_ndarray_on_GPU(stack::array{6u, 7u, 8u});

  *errors = 0;
  accessor_kernel<<<1,1>>>(errors, arr3D);
  cudaDeviceSynchronize();
  EXPECT_EQ(*errors, 0);

////////////////////////////////////////////////////////////////////////////////

  cudaFree(errors);
}

template < typename S, typename T >
__global__ void slicing_kernel(int * num_errors, nd::view<T, 3> arr3D) {

  nd::view< S, 2 > slice1 = arr3D(nd::range{2,5}, 4);
  *num_errors += (slice1.stride[0] != arr3D.stride[0]);
  *num_errors += (slice1.stride[1] != arr3D.stride[2]);
  *num_errors += (slice1.shape[0] != 3);
  *num_errors += (slice1.shape[1] != 8);
  *num_errors += (slice1(0,0) != arr3D(2,4,0));
  *num_errors += (slice1(1,1) != arr3D(3,4,1));
  *num_errors += (slice1(2,2) != arr3D(4,4,2));

  nd::view< S, 1 > slice2 = arr3D(1, 2, nd::range{2,5});
  *num_errors += (slice2.stride[0] != arr3D.stride[2]);
  *num_errors += (slice2.shape[0] != 3);
  *num_errors += (slice2(0) != arr3D(1,2,2));
  *num_errors += (slice2(1) != arr3D(1,2,3));
  *num_errors += (slice2(2) != arr3D(1,2,4));

  nd::view< S, 2 > slice3 = arr3D(nd::range{2,5}, nd::range{1,2}, 4);
  *num_errors += (slice3.stride[0] != arr3D.stride[0]);
  *num_errors += (slice3.stride[1] != arr3D.stride[1]);
  *num_errors += (slice3.shape[0] != 3);
  *num_errors += (slice3.shape[1] != 1);
  *num_errors += (slice3(0,0) != arr3D(2,1,4));
  *num_errors += (slice3(1,0) != arr3D(3,1,4));
  *num_errors += (slice3(2,0) != arr3D(4,1,4));

  nd::view< S, 3 > slice4 = arr3D(nd::range{1, 2}, nd::range{2, 5}, nd::range{1,2});
  *num_errors += (slice4.stride[0] != arr3D.stride[0]);
  *num_errors += (slice4.stride[1] != arr3D.stride[1]);
  *num_errors += (slice4.stride[2] != arr3D.stride[2]);
  *num_errors += (slice4.shape[0] != 1);
  *num_errors += (slice4.shape[1] != 3);
  *num_errors += (slice4.shape[2] != 1);
  *num_errors += (slice4(0,0,0) != arr3D(1, 2, 1));
  *num_errors += (slice4(0,1,0) != arr3D(1, 3, 1));
  *num_errors += (slice4(0,2,0) != arr3D(1, 4, 1));
}

TEST(UnitTest, ArraySlicingOnGPU) {
  int * errors;
  cudaMallocManaged(&errors, sizeof(int));

////////////////////////////////////////////////////////////////////////////////

  nd::array< double, 3 > arr3D = make_patterned_ndarray_on_GPU(stack::array{6u, 7u, 8u});

  *errors = 0;
  slicing_kernel<double, double><<<1,1>>>(errors, arr3D);
  cudaDeviceSynchronize();
  EXPECT_EQ(*errors, 0);

////////////////////////////////////////////////////////////////////////////////

  cudaFree(errors);
}

TEST(UnitTest, ConstArraySlicingOnGPU) {
  int * errors;
  cudaMallocManaged(&errors, sizeof(int));

////////////////////////////////////////////////////////////////////////////////

  nd::array< double, 3 > arr3D = make_patterned_ndarray_on_GPU(stack::array{6u, 7u, 8u});

  *errors = 0;
  slicing_kernel<const double, double><<<1,1>>>(errors, arr3D);
  cudaDeviceSynchronize();
  EXPECT_EQ(*errors, 0);

////////////////////////////////////////////////////////////////////////////////

  cudaFree(errors);
}
