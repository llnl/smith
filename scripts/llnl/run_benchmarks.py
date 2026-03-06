#!/bin/sh
"exec" "python3" "-u" "-B" "$0" "$@"

# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
 file: run_benchmarks.py

 description: 
  Run benchmarks and update shared (or any desired) location with new Caliper files

"""

from common_build_functions import *

from argparse import ArgumentParser

import grp
import os
import stat


def parse_args():
    "Parses args from command line"
    parser = ArgumentParser()
    parser.add_argument("-e", "--extra-cmake-options",
                      dest="extra_cmake_options",
                      default=os.environ.get("EXTRA_CMAKE_OPTIONS", ""),
                      help="Extra cmake options to add to the cmake configure line. Note that options to enable benchmarks, " +
                           "disable docs, and build as Release are always appended.")
    parser.add_argument("-hc", "--host-config",
                    dest="host_config",
                    default=os.environ.get("HOST_CONFIG", None),
                    help="Specific host-config filename to build (defaults to HOST_CONFIG environment variable)")
    parser.add_argument("-sd", "--spot-directory",
                      dest="spot_dir",
                      default=get_shared_spot_dir(),
                      help="Where to put all resulting caliper files to use for SPOT analysis (defaults to a shared location)")
    parser.add_argument("-t", "--timestamp",
                    dest="timestamp",
                    default=get_timestamp(),
                    help="Set timestamp manually for debugging")

    # Parse args
    args, extra_args = parser.parse_known_args()
    args = vars(args)

    # Verify args
    if args["host_config"] is None:
        print("[ERROR: Both host_config argument and HOST_CONFIG environment variable unset!]")
        sys.exit(1)

    return args


def main():
    args = parse_args()

    if not ensure_on_lc_and_group_permissions():
        return 1

    # Args
    cmake_options = args["extra_cmake_options"] + " -DENABLE_BENCHMARKS=ON -DENABLE_DOCS=OFF -DCMAKE_BUILD_TYPE=Release"
    host_config = args["host_config"]
    spot_dir = os.path.abspath(args["spot_dir"])
    timestamp = args["timestamp"]

    # Vars
    repo_dir = get_repo_dir()
    test_root = get_build_and_test_root(repo_dir, timestamp)
    host_config_path = get_host_config_path(repo_dir, host_config)
    host_config_root = get_host_config_root(host_config)
    benchmarks_output_file = os.path.join(test_root, "output.log.%s.benchmarks.txt" % host_config_root)
    build_dir = os.path.join(test_root, "build-%s" % host_config_root)

    # Build Smith
    os.chdir(repo_dir)
    os.makedirs(test_root, exist_ok=True)
    build_and_test_host_config(test_root=test_root, host_config=host_config_path,
                               report_to_stdout=True, extra_cmake_options=cmake_options,
                               skip_install=True, skip_tests=True)

    # Go to build location
    os.chdir(build_dir)

    # Run benchmarks
    result = shell_exec("make run_benchmarks", echo=True, print_output=True, output_file=benchmarks_output_file)

    # Move resulting .cali files to specified directory
    os.makedirs(spot_dir, exist_ok=True)
    cali_files = glob.glob(pjoin(build_dir, "*.cali"))
    for cali_file in cali_files:
        if os.path.exists(cali_file):
            shutil.copy2(cali_file, spot_dir)
            # Update group and permissions of new caliper file
            cali_file = os.path.join(spot_dir, os.path.basename(cali_file))
            group_info = grp.getgrnam("smithdev")
            os.chown(cali_file, -1, group_info.gr_gid)
            os.chmod(cali_file,
                     stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)

    # Print SPOT url
    if on_rz():
        print("[View SPOT directory here: https://rzlc.llnl.gov/spot2/?sf={0}]".format(spot_dir))
    else:
        print("[View SPOT directory here: https://lc.llnl.gov/spot2/?sf={0}]".format(spot_dir))

    return result; 


if __name__ == "__main__":
    sys.exit(main())
