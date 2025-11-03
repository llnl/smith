// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/infrastructure/cli.hpp"

#include <cstdlib>
#include <utility>
#include <vector>

#include "axom/CLI11.hpp"

#include "smith/infrastructure/input.hpp"
#include "smith/infrastructure/logger.hpp"

namespace smith::cli {

//------- Command Line Interface -------

std::unordered_map<std::string, std::string> defineAndParse(int argc, char* argv[], std::string app_description)
{
  // NOTE: When adding/removing command line options remember to update the following places as well:
  //   src/docs/sphinx/user_guide/command_line_options.rst
  //   smith::cli::printGiven() (in this file)

  // specify all input arguments
  axom::CLI::App app{app_description};
  std::string input_file_path;
  app.add_option("-i, --input-file", input_file_path, "Input file to use")->check(axom::CLI::ExistingFile);
  int restart_cycle;
  auto restart_opt =
      app.add_option("-c, --restart-cycle", restart_cycle, "Cycle to restart from")->check(axom::CLI::PositiveNumber);
  bool create_input_file_docs{false};
  app.add_flag("-d, --create-input-file-docs", create_input_file_docs,
               "Writes Sphinx documentation for input file, then exits");
  std::string output_directory;
  app.add_option("-o, --output-directory", output_directory, "Directory to put outputted files");
  bool enable_paraview{false};
  app.add_flag("-p, --paraview", enable_paraview, "Enable ParaView output");
  bool print_unused{false};
  app.add_flag("-u, --print-unused", print_unused, "Prints unused entries in input file, then exits");
  bool version{false};
  app.add_flag("-v, --version", version, "Print version and provenance information, then exits");

  // Parse the arguments and check if they are good
  try {
    app.parse(argc, argv);
  } catch (const axom::CLI::ParseError& e) {
    smith::logger::flush();
    if (e.get_name() == "CallForHelp") {
      auto msg = app.help();
      SLIC_INFO_ROOT(msg);
      exit(0);
    } else {
      auto err_msg = axom::CLI::FailureMessage::simple(&app, e);
      SLIC_ERROR_ROOT(err_msg);
    }
  }

  // Store found values and set defaults if not set above
  std::unordered_map<std::string, std::string> cli_opts;
  if (version) {
    // If version is on the command line ignore all others and do not require anything
    cli_opts.insert({"version", {}});
  } else {
    if (input_file_path.empty()) {
      SLIC_ERROR_ROOT("No input file given. Use '--help' for command line options.");
    }

    cli_opts.insert({std::string("input-file"), input_file_path});
    // If a restart cycle was specified
    if (restart_opt->count() > 0) {
      cli_opts["restart-cycle"] = std::to_string(restart_cycle);
    }
    if (create_input_file_docs) {
      cli_opts.insert({"create-input-file-docs", {}});
    }
    if (print_unused) {
      cli_opts.insert({"print-unused", {}});
    }
    if (output_directory == "") {
      // if given by user use that otherwise use input file's basename minus extension
      output_directory = smith::input::getInputFileName(input_file_path);
    }
    cli_opts.insert({"output-directory", output_directory});
    if (enable_paraview) {
      cli_opts.insert({"paraview", {}});
      cli_opts.insert({"paraview-directory", output_directory + "_paraview"});
    }
  }

  return cli_opts;
}

namespace detail {

std::string cliValueToString(std::string value) { return value; }

std::string cliValueToString(bool value) { return value ? "true" : "false"; }

std::string cliValueToString(int value) { return std::to_string(value); }

}  // namespace detail

void printGiven(std::unordered_map<std::string, std::string>& cli_opts)
{
  // Add header
  std::string optsMsg = axom::fmt::format("\n{:*^80}\n", "Command Line Options");

  // Create options map
  // clang-format off
  std::vector<std::pair<std::string, std::string>> opts_output_map{
    {"create-input-file-docs", "Create Input File Docs"},
    {"input-file", "Input File"},
    {"output-directory", "Output Directory"},
    {"paraview", "Enable ParaView output"},
    {"restart-cycle", "Restart Cycle"},
    {"version", "Print version"}};
  // clang-format on

  // Add options to string
  for (auto output_pair : opts_output_map) {
    auto search = cli_opts.find(output_pair.first);
    if (search != cli_opts.end()) {
      optsMsg += axom::fmt::format("{0}: {1}\n", output_pair.second, detail::cliValueToString(search->second));
    }
  }

  // Add footer
  optsMsg += axom::fmt::format("{:*^80}\n", "*");

  SLIC_INFO_ROOT(optsMsg);
  smith::logger::flush();
}

}  // namespace smith::cli
