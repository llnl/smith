// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/infrastructure/logger.hpp"

#include <iostream>
#include <string>

#include "mpi.h"

#include "smith/infrastructure/about.hpp"

namespace smith::logger {

bool initialize(MPI_Comm comm)
{
  namespace slic = axom::slic;

  if (!slic::isInitialized()) {
    slic::initialize();
  }

  auto [num_ranks, rank] = getMPIInfo(comm);

  // Mark rank 0 as the root for message filtering macros
  slic::setIsRoot(rank == 0);

  std::string loggerName = num_ranks > 1 ? "smith_parallel_logger" : "smith_serial_logger";
  slic::createLogger(loggerName);
  slic::activateLogger(loggerName);
  if (!slic::activateLogger(loggerName)) {
    // Can't log through SLIC since it just failed to activate
    std::cerr << "Error: Failed to activate logger: " << loggerName << std::endl;
    return false;
  }

  // Console streams, std::cout for info/debug, std::cerr for warnings/errors
  slic::LogStream* i_logstream = nullptr;   // info
  slic::LogStream* d_logstream = nullptr;   // debug
  slic::LogStream* we_logstream = nullptr;  // warnings and errors

  // Stream formatting strings
  std::string i_format_string = "<MESSAGE>\n";
  std::string d_format_string = "[<LEVEL>]: <MESSAGE>\n";
  std::string we_format_string = "[<LEVEL> (<FILE>:<LINE>)]\n<MESSAGE>\n\n";

  // Only create a parallel logger when there is more than one rank
  if (num_ranks > 1) {
    // Add rank to format strings if parallel
    // Note: i_format_string's extra space is on purpose due to no space on above string
    i_format_string = "[<RANK>] " + i_format_string;
    d_format_string = "[<RANK>]" + d_format_string;
    we_format_string = "[<RANK>]" + we_format_string;

    const int RLIMIT = 8;

    i_logstream = new slic::LumberjackStream(&std::cout, comm, RLIMIT, i_format_string);
    d_logstream = new slic::LumberjackStream(&std::cout, comm, RLIMIT, d_format_string);
    we_logstream = new slic::LumberjackStream(&std::cerr, comm, RLIMIT, we_format_string);
  } else {
    i_logstream = new slic::GenericOutputStream(&std::cout, i_format_string);
    d_logstream = new slic::GenericOutputStream(&std::cout, d_format_string);
    we_logstream = new slic::GenericOutputStream(&std::cerr, we_format_string);
  }

  slic::setLoggingMsgLevel(slic::message::Debug);

  // Add message levels to streams
  addStreamToMsgLevel(i_logstream, slic::message::Info);
  addStreamToMsgLevel(d_logstream, slic::message::Debug);
  addStreamToMsgLevel(we_logstream, slic::message::Warning);
  addStreamToMsgLevel(we_logstream, slic::message::Error);

  // Set SLIC abort functionality
  // NOTE: Do not set a collective abort function via `slic::setAbortFunction`.
  // This can cause runs to hang. SLIC flushs locally on the node that fails,
  // so that the error message is not lost.
  slic::setAbortOnError(true);
  slic::setAbortOnWarning(false);

  std::string msg = axom::fmt::format("Logger activated: '{0}'", loggerName);
  SLIC_INFO_ROOT(msg);
  smith::logger::flush();

  return true;
}

void finalize() { axom::slic::finalize(); }

void flush() { axom::slic::flushStreams(); }

}  // namespace smith::logger
