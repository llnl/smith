// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/about.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/infrastructure/logger.hpp"

#ifdef SERAC_USE_CALIPER
#include <optional>
#endif

namespace serac::profiling {

#ifdef SERAC_USE_CALIPER
namespace {
std::optional<cali::ConfigManager> mgr;
}  // namespace
#endif

void initialize([[maybe_unused]] MPI_Comm comm, [[maybe_unused]] std::string options)
{
#ifdef SERAC_USE_ADIAK
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
  SERAC_SET_METADATA("serac_version", serac::version(true));
  SERAC_SET_METADATA("serac_compiler", serac::compiler());
#endif

#ifdef SERAC_USE_CALIPER
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
#ifdef SERAC_USE_ADIAK
  // Finalize Adiak
  adiak::fini();
#endif

#ifdef SERAC_USE_CALIPER
  // Finalize Caliper
  if (mgr) {
    mgr->stop();
    mgr->flush();
  }

  mgr.reset();
#endif
}

}  // namespace serac::profiling
