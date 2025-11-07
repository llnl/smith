// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include "mfem.hpp"
#include "axom/core.hpp"

#ifdef __GNUG__
#include <cxxabi.h>
#include <cstdlib>
#endif

#include "smith/infrastructure/memory.hpp"
#include "smith/numerics/functional/element_restriction.hpp"

namespace smith {

/**
 * @brief Return string of given parameter's type
 * @tparam T the type to get a string name for
 * @param[in] var the variable to get the type of
 * @return string representation of the type
 */
template <typename T>
std::string typeString(T& var)
{
  // Remove reference, but keep the const/volatile qualifiers.
  const char* name = typeid(var).name();
#ifdef __GNUG__
  int status = -4;  // Arbitrary value to eliminate the compiler warning
  char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  std::string result((status == 0) ? demangled : name);
  std::free(demangled);
  if constexpr (std::is_const_v<T>) {
    result = "const " + result;
  }
  return result;
#else
  // Return name if compiler doesn't support GNU's extensions (most do)
  return name;
#endif
}

/**
 * @brief write an array of values out to file, in a space-separated format
 * @tparam T the type of each value in the array
 * @param v the values to write to file
 * @param filename the name of the output file
 */
template <typename T>
void writeToFile(std::vector<T> v, std::string filename)
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
void writeToFile(mfem::Vector v, std::string filename)
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
void writeToFile(mfem::SparseMatrix A, std::string filename)
{
  std::ofstream outfile(filename);
  A.PrintMM(outfile);
  outfile.close();
}

/**
 * @brief stream output for DoF
 */
std::ostream& operator<<(std::ostream& out, DoF dof)
{
  out << "{" << dof.index() << ", " << dof.sign() << ", " << dof.orientation() << "}";
  return out;
}

/**
 * @brief write a 2D array of values out to file, in a space-separated format
 * @tparam T the type of each value in the array
 * @param arr the array to write to file
 * @param filename the name of the output file
 */
template <typename T>
void writeToFile(axom::Array<T, 2, smith::detail::host_memory_space> arr, std::string filename)
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
 * @param arr the array to write to file
 * @param filename the name of the output file
 */
template <typename T>
void writeToFile(axom::Array<T, 3, smith::detail::host_memory_space> arr, std::string filename)
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

}  // namespace smith
