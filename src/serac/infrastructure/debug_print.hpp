// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "mfem.hpp"
#include "axom/core.hpp"

#include <string>
#include <fstream>
#include <iomanip>
#include <vector>

#include "serac/infrastructure/memory.hpp"
#include "serac/numerics/functional/element_restriction.hpp"

/**
 * @brief write an array of values out to file, in a space-separated format
 * @tparam T the type of each value in the array
 * @param v the values to write to file
 * @param filename the name of the output file
 */
template <typename T>
void write_to_file(std::vector<T> v, std::string filename)
{
  std::ofstream outfile(filename);
  for (int i = 0; i < v.size(); i++) {
    outfile << v[i] << std::endl;
  }
  outfile.close();
}

/**
 * @brief write an array of doubles out to file, in a space-separated format
 * @param v the values to write to file
 * @param filename the name of the output file
 */
void write_to_file(mfem::Vector v, std::string filename)
{
  std::ofstream outfile(filename);
  for (int i = 0; i < v.Size(); i++) {
    outfile << v[i] << std::endl;
  }
  outfile.close();
}

/**
 * @brief write a sparse matrix out to file
 * @param A the matrix to write to file
 * @param filename the name of the output file
 */
void write_to_file(mfem::SparseMatrix A, std::string filename)
{
  std::ofstream outfile(filename);
  A.PrintMM(outfile);
  outfile.close();
}

std::ostream& operator<<(std::ostream& out, DoF dof)
{
  out << "{" << dof.index() << ", " << dof.sign() << ", " << dof.orientation() << "}";
  return out;
}

/**
 * @brief write a 2D array of values out to file, in a space-separated format
 * @tparam T the type of each value in the array
 * @param v the values to write to file
 * @param filename the name of the output file
 */
template <typename T>
void write_to_file(axom::Array<T, 2, serac::detail::host_memory_space> arr, std::string filename)
{
  std::ofstream outfile(filename);

  for (axom::IndexType i = 0; i < arr.shape()[0]; i++) {
    outfile << "{";
    for (axom::IndexType j = 0; j < arr.shape()[1]; j++) {
      outfile << arr(i, j);
      if (j < arr.shape()[1] - 1) outfile << ", ";
    }
    outfile << "}\n";
  }

  outfile.close();
}

/**
 * @brief write a 3D array of values out to file, in a mathematica-compatible format
 * @tparam T the type of each value in the array
 * @param v the values to write to file
 * @param filename the name of the output file
 */
template <typename T>
void write_to_file(axom::Array<T, 3, serac::detail::host_memory_space> arr, std::string filename)
{
  std::ofstream outfile(filename);

  outfile << std::setprecision(16);

  for (axom::IndexType i = 0; i < arr.shape()[0]; i++) {
    outfile << "{";
    for (axom::IndexType j = 0; j < arr.shape()[1]; j++) {
      outfile << "{";
      for (axom::IndexType k = 0; k < arr.shape()[2]; k++) {
        outfile << arr(i, j, k);
        if (k < arr.shape()[2] - 1) outfile << ", ";
      }
      outfile << "}";
      if (j < arr.shape()[1] - 1) outfile << ", ";
    }
    outfile << "}\n";
  }

  outfile.close();
}

#ifdef __CUDACC__
#include <cuda_runtime.h>
/**
 * @brief Helper function that prints usage of global device memory.  Useful for debugging potential
 *        memory or register leaks
 */
#include <iostream>
void printCUDAMemUsage()
{
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  int i = 0;
  cudaSetDevice(i);

  size_t freeBytes, totalBytes;
  cudaMemGetInfo(&freeBytes, &totalBytes);
  size_t usedBytes = totalBytes - freeBytes;

  std::cout << "Device Number: " << i << std::endl;
  std::cout << " Total Memory (MB): " << (totalBytes / 1024.0 / 1024.0) << std::endl;
  std::cout << " Free Memory (MB): " << (freeBytes / 1024.0 / 1024.0) << std::endl;
  std::cout << " Used Memory (MB): " << (usedBytes / 1024.0 / 1024.0) << std::endl;
}
#endif
