// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <thread>
#include <fstream>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "axom/CLI11.hpp"
#include "axom/core/utilities/Timer.hpp"

#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/smith_config.hpp"

using timer = axom::utilities::Timer;

// partition the range {0, 1, 2, ... , n - 1} into
// `num_blocks` roughly equal-sized contiguous chunks
std::vector<uint32_t> partition_range(uint32_t n, uint32_t num_blocks)
{
  uint32_t quotient = n / num_blocks;
  uint32_t remainder = n % num_blocks;

  std::vector<uint32_t> blocks(num_blocks + 1);
  if (blocks.size() > 0) {
    blocks[0] = 0;
    for (uint32_t i = 1; i < num_blocks + 1; i++) {
      if (remainder > 0) {
        blocks[i] = blocks[i - 1] + quotient + 1;
        remainder--;
      } else {
        blocks[i] = blocks[i - 1] + quotient;
      }
    }
  }
  return blocks;
}

//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
  axom::CLI::App app{"a tool for partitioning large meshes into smaller parts suitable for MPI"};

  std::string input_mesh;
  app.add_option("-i, --input-mesh", input_mesh, "Input file to use")->required()->check(axom::CLI::ExistingFile);

  std::string output_prefix;
  app.add_option("-o, --output-prefix", output_prefix, "Prefix for output meshes")->required();

  uint32_t num_parts = 4;
  app.add_option("-n, --num_parts", num_parts, "number of partitions to generate");

  uint32_t num_threads = 1;
  // this is disabled temporarily, as it seems mfem's implementation is not thread safe
  // app.add_option("-j, --num_threads", num_threads, "number of partitions to generate");

  CLI11_PARSE(app, argc, argv);

  timer stopwatch;

  std::cout << "reading in mesh file ... ";
  stopwatch.start();
  std::ifstream infile(input_mesh);
  mfem::Mesh mesh(infile);
  stopwatch.stop();
  std::cout << "completed after " << stopwatch.elapsed() * 1000.0 << "ms" << std::endl;

  std::cout << "partitioning mesh into " << num_parts << " pieces ... ";
  stopwatch.start();
  int* partitioning = mesh.GeneratePartitioning(int(num_parts));
  stopwatch.stop();
  std::cout << "completed after " << stopwatch.elapsed() * 1000.0 << "ms" << std::endl;

  mfem::MeshPartitioner partitioner(mesh, int(num_parts), partitioning);

  timer fileio;
  fileio.start();
  std::vector<uint32_t> chunks = partition_range(num_parts, num_threads);

  std::vector<std::thread> threads;
  for (uint32_t k = 0; k < num_threads; k++) {
    threads.push_back(std::thread(
        [&](uint32_t tid) {
          mfem::MeshPart mesh_part;
          for (uint32_t i = chunks[tid]; i < chunks[tid + 1]; i++) {
            std::string filename = mfem::MakeParFilename(output_prefix + ".mesh.", int(i));
            if (num_threads == 1) {
              std::cout << "extracting part " << i << " and writing it to " << filename << " ... ";
            }
            stopwatch.start();
            partitioner.ExtractPart(int(i), mesh_part);
            std::ofstream f(filename);
            f.precision(16);
            mesh_part.Print(f);
            stopwatch.stop();
            if (num_threads == 1) {
              std::cout << "completed after " << stopwatch.elapsed() * 1000.0 << "ms" << std::endl;
            }
          }
        },
        k));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  fileio.stop();
  std::cout << "writing files to disk took " << fileio.elapsed() * 1000.0 << "ms total" << std::endl;

  delete partitioning;
}
