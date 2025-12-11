// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/infrastructure/accelerator.hpp"

#include "mfem.hpp"

#include <memory>

#include "smith/infrastructure/logger.hpp"

namespace smith {

namespace accelerator {

// Restrict global to this file only
namespace {
std::unique_ptr<mfem::Device> device;
}  // namespace

void initializeDevice(ExecutionSpace exec_space)
{
  SLIC_ERROR_ROOT_IF(device, "smith::accelerator::initializeDevice cannot be called more than once");
  device = std::make_unique<mfem::Device>();
  switch (exec_space) {
    case ExecutionSpace::GPU:
#if defined(MFEM_USE_CUDA) && defined(SMITH_USE_CUDA_KERNEL_EVALUATION)
      device->Configure("cuda");
#elif defined(MFEM_USE_HIP)
      device->Configure("hip");
#endif
      break;
    case ExecutionSpace::CPU:
      break;
    case ExecutionSpace::Dynamic:
      break;
  }
}

void terminateDevice()
{
  // Idempotent, no adverse affects if called multiple times
  device.reset();
}

}  // namespace accelerator

}  // namespace smith
