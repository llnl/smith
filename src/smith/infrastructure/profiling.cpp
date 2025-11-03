// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/infrastructure/profiling.hpp"

#ifdef SMITH_USE_CALIPER
#include <optional>
#endif

#include "smith/infrastructure/about.hpp"
#include "smith/infrastructure/logger.hpp"

namespace smith::profiling {

#ifdef SMITH_USE_CALIPER
namespace {
std::optional<cali::ConfigManager> mgr;
}  // namespace
#endif

void initialize([[maybe_unused]] MPI_Comm comm, [[maybe_unused]] std::string options)
{
#ifdef SMITH_USE_ADIAK
  // Initialize Adiak
  adiak::init(&comm);

  adiak::launchdate();
  adiak::executable();
  adiak::cmdline();
  adiak::clustername();
  adiak::jobsize();
  adiak::walltime();
  adiak::cputime();
  adiak::systime();
  SMITH_SET_METADATA("smith_version", smith::version(true));
  SMITH_SET_METADATA("smith_compiler", smith::compiler());
#endif

#ifdef SMITH_USE_CALIPER
  // Initialize Caliper
  mgr = cali::ConfigManager();
  auto check_result = mgr->check(options.c_str());

  if (check_result.empty()) {
    mgr->add(options.c_str());
  } else {
    SLIC_WARNING_ROOT("Caliper options invalid, ignoring: " << check_result);
  }

  // Defaults, should probably always be enabled
  mgr->add("runtime-report,spot");
  mgr->start();
#endif
}

void finalize()
{
#ifdef SMITH_USE_ADIAK
  // Finalize Adiak
  adiak::fini();
#endif

#ifdef SMITH_USE_CALIPER
  // Finalize Caliper
  if (mgr) {
    mgr->stop();
    mgr->flush();
  }

  mgr.reset();
#endif
}

}  // namespace smith::profiling
