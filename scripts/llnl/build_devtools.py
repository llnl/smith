#!/bin/sh
"exec" "python3" "-u" "-B" "$0" "$@"

# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
 file: build_devtools.py

 description: 
  Builds all Smith Devtools

"""

from common_build_functions import *

from argparse import ArgumentParser


def parse_args():
    "Parses args from command line"
    parser = ArgumentParser()
    # Location of source directory to build
    parser.add_argument("-d", "--directory",
                      dest="directory",
                      default="",
                      help="Location to build all TPL's, timestamp directory will be created (Defaults to shared location)")
    parser.add_argument("--short-path",
                      action="store_true",
                      dest="short_path",
                      default=False,
                      help="Does not add sys_type or timestamp to tpl directory (useful for CI and debugging).")
    parser.add_argument("-v", "--verbose",
                      action="store_true",
                      dest="verbose",
                      default=False,
                      help="Output logs to screen as well as to files")
    ###############
    # parse args
    ###############
    args, extra_args = parser.parse_known_args()
    # we want a dict b/c the values could 
    # be passed without using argparse
    args = vars(args)
    return args


def main():
    args = parse_args()

    if not ensure_on_lc_and_group_permissions():
        return 1

    # Determine location to do all the building
    if args["directory"] != "":
        build_dir = args["directory"]
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
    else:
        build_dir = get_shared_devtool_dir()
    build_dir = os.path.abspath(build_dir)

    repo_dir = get_repo_dir()

    try:
        original_wd = os.getcwd()
        os.chdir(repo_dir)

        res = build_devtools(build_dir, get_timestamp(), args["short_path"], args["verbose"])
    finally:
        os.chdir(original_wd)

    return res


if __name__ == "__main__":
    sys.exit(main())
