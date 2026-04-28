#!/bin/sh
"exec" "python3" "-u" "-B" "$0" "$@"

# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
 file: build_tpls.py

 description: 
  uses uberenv to install tpls for the set of compilers we want
  for current machine.

"""

from common_build_functions import *

from argparse import ArgumentParser

import os


def parse_args():
    "Parses args from command line"
    parser = ArgumentParser()
    # Directory to do all the building
    parser.add_argument("-d", "--directory",
                      dest="directory",
                      default="",
                      help="Location to build all TPL's, sys_type/timestamp directory will be created (Defaults to shared location)")
    parser.add_argument("--short-path",
                      action="store_true",
                      dest="short_path",
                      default=False,
                      help="Does not add sys_type or timestamp to tpl directory (useful for CI).")
    # Spack spec to use for the build
    parser.add_argument("-s", "--spec",
                      dest="spec",
                      nargs="+",
                      default="",
                      help="Spack spec to build (defaults to all available on SYS_TYPE)")
    parser.add_argument("-v", "--verbose",
                      action="store_true",
                      dest="verbose",
                      default=False,
                      help="Output logs to screen as well as to files")
    parser.add_argument("-m", "--mirror",
                      dest="mirror",
                      default="",
                      help="Mirror location to use (defaults to shared location)")
    parser.add_argument("-j", "--jobs",
                      dest="jobs",
                      default="",
                      help="Allow N jobs at once for any `make` commands (empty string means max system amount)")

    ###############
    # parse args
    ###############
    args, _ = parser.parse_known_args()
    # we want a dict b/c the values could 
    # be passed without using argparse
    args = vars(args)
    return args


def main():
    args = parse_args()

    if not ensure_on_lc_and_group_permissions():
        return 1

    # Handle case where spec is a List (i.e. spec contained spaces and wasn't wrapped in quotes)
    spec = args["spec"]
    if len(spec) > 0:
        spec = " ".join(spec)

    # Determine location to do all the building
    if args["directory"] != "":
        builds_dir = args["directory"]
        if not os.path.exists(builds_dir):
            os.makedirs(builds_dir)
    else:
        builds_dir = get_shared_libs_dir()
    builds_dir = os.path.abspath(builds_dir)

    repo_dir = get_repo_dir()

    try:
        original_wd = os.getcwd()
        os.chdir(repo_dir)

        timestamp = get_timestamp()
        res = full_build_and_test_of_tpls(builds_dir, timestamp, spec, args["verbose"], args["short_path"], args["mirror"], args["jobs"])
    finally:
        os.chdir(original_wd)

    return res


if __name__ == "__main__":
    sys.exit(main())
