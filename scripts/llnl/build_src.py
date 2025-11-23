#!/bin/sh
"exec" "python3" "-u" "-B" "$0" "$@"

# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
 file: build_src.py

 description: 
  Builds Smith with the host-configs for the current machine.

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
                      help="Directory of source to be built (Defaults to current)")
    # Whether to build a specific hostconfig
    parser.add_argument("--host-config",
                      dest="hostconfig",
                      default="",
                      help="Specific host-config file to build (Tries multiple known paths to locate given file)")
    # Extra cmake options to pass to config build
    parser.add_argument("--extra-cmake-options",
                      dest="extra_cmake_options",
                      default="",
                      help="Extra cmake options to add to the cmake configure line")
    parser.add_argument("--automation-mode",
                      action="store_true",
                      dest="automation",
                      default=False,
                      help="Toggle automation mode which uses env $HOST_CONFIG then $SYS_TYPE/$COMPILER if found")
    parser.add_argument("--skip-install",
                      action="store_true",
                      dest="skip_install",
                      default=False,
                      help="Skip testing install target which does not work in some configurations (codevelop)")
    parser.add_argument("--skip-tests",
                      action="store_true",
                      dest="skip_tests",
                      default=False,
                      help="Skip unit tests which will not work in some configurations (e.g. CUDA on GitHub Actions)")
    parser.add_argument("-v", "--verbose",
                      action="store_true",
                      dest="verbose",
                      default=False,
                      help="Output logs to screen as well as to files")
    parser.add_argument("-j", "--jobs",
                      dest="jobs",
                      default="",
                      help="Allow N jobs at once for any `make` commands (empty string means max system amount)")
    ###############
    # parse args
    ###############
    args, extra_args = parser.parse_known_args()
    # we want a dict b/c the values could 
    # be passed without using argparse
    args = vars(args)

    # Ensure correctness
    if args["automation"] and args["hostconfig"] != "":
        print("[ERROR: automation and host-config modes are mutually exclusive]")
        sys.exit(1)

    return args


def main():
    args = parse_args()

    if not ensure_on_lc_and_group_permissions(allow_non_lc_builds=True):
        return 1

    # Determine source directory to be built
    if os.environ.get("UBERENV_PREFIX") != None:
        repo_dir = os.environ["UBERENV_PREFIX"]
        if not os.path.isdir(repo_dir):
            print("[ERROR: Given environment variable 'UBERENV_PREFIX' is not a valid directory]")
            print("[    'UBERENV_PREFIX' = %s]" % repo_dir)
            return 1
    if args["directory"] != "":
        repo_dir = args["directory"]
        if not os.path.isdir(repo_dir):
            print("[ERROR: Given command line variable '--directory' is not a valid directory]")
            print("[    '--directory' = %s]" % repo_dir)
            return 1
    else:
        repo_dir = get_repo_dir()

    try:
        original_wd = os.getcwd()
        os.chdir(repo_dir)
        timestamp = get_timestamp()

        # Default to build all SYS_TYPE's host-configs in host-config/
        build_all = not args["hostconfig"] and not args["automation"]
        if build_all:
            res = build_and_test_host_configs(repo_dir, timestamp, False,
                                              args["verbose"], args["extra_cmake_options"],
                                              args["skip_install"], args["jobs"])
        # Otherwise try to build a specific host-config
        else:
            # Command-line arg has highest priority
            if args["hostconfig"]:
                hostconfig = args["hostconfig"]
            
            # Otherwise try to reconstruct host-config path from SYS_TYPE and COMPILER
            elif args["automation"]:
                if not "SYS_TYPE" in os.environ:
                    print("[ERROR: Automation mode required 'SYS_TYPE' environment variable]")
                    return 1
                if not "COMPILER" in os.environ:
                    print("[ERROR: Automation mode required 'COMPILER' environment variable]")
                    return 1
                import socket
                hostname = socket.gethostname()
                # Remove any numbers after the end
                hostname = hostname.rstrip('0123456789')
                sys_type = os.environ["SYS_TYPE"]
                # Remove everything including and after the last hyphen
                sys_type = sys_type.rsplit('-', 1)[0]
                compiler = os.environ["COMPILER"]
                compiler = compiler.rsplit('-', 1)[0]
                hostconfig = "%s-%s-%s.cmake" % (hostname, sys_type, compiler)

            hostconfig_path = get_host_config_path(repo_dir, hostconfig)

            test_root = get_build_and_test_root(repo_dir, timestamp)
            os.mkdir(test_root)
            res = build_and_test_host_config(test_root, hostconfig_path, args["verbose"], args["extra_cmake_options"],
                                             args["skip_install"], args["skip_tests"], args["jobs"])

    finally:
        os.chdir(original_wd)

    return res

if __name__ == "__main__":
    sys.exit(main())
