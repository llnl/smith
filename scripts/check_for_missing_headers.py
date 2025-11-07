#!/bin/sh
"exec" "python3" "-u" "-B" "$0" "$@"
##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

"""
 file: check_for_missing_headers.py

 description:
  This script takes a Smith install and source directory and checks to see
  if install includes the same header files.

"""

import os
import sys
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--install-dir",
                        default="",
                        dest="install_dir",
                        help="specify path of the install directory.")
    parser.add_argument("-s",
                        "--src-dir",
                        default="",
                        dest="src_dir",
                        help="specify path of the src directory.")
    return parser.parse_known_args()

# return list of relative paths to header files
def get_headers_from(dir):
    headers = []
    for (dirpath, dirnames, filenames) in os.walk(dir, topdown=True):
        # skip tests directories
        if "tests" in dirnames:
            dirnames.remove("tests")
        for f in filenames:
            if ".hpp" in f and ".in" not in f:
                relative_header_path = dirpath.replace(dir + "/", "")
                headers.append({"path": relative_header_path, "headerfile": f})
    return headers

def main():
    args, unknown_args = parse_arguments()

    # ensure args are valid
    install_dir = args.install_dir
    if not os.path.isdir(install_dir):
        print("Error: install_dir is not a directory or does not exist: {}".format(install_dir))
        return 1
    install_dir = os.path.abspath(install_dir)

    src_dir = args.src_dir
    if not os.path.isdir(src_dir):
        print("Error: src_dir is not a directory or does not exist: {}".format(src_dir))
        return 1
    src_dir = os.path.abspath(src_dir)

    print("============================================================")
    print("check_for_missing_headers.py args")
    print("install_dir: {0}".format(install_dir))
    print("src_dir:     {0}".format(src_dir))
    print("============================================================")

    # grab headers from install and src
    install_headers = get_headers_from(os.path.join(install_dir, "include", "smith"))
    src_headers = get_headers_from(os.path.join(src_dir, "smith"))
    
    # check if each header in src is in install as well
    res = 0
    for sh in src_headers:
        found = False
        for ih in install_headers:
            src_relative_header = "{0}/{1}".format(sh["path"], sh["headerfile"])
            install_relative_header = "{0}/{1}".format(ih["path"], ih["headerfile"])
            if src_relative_header == install_relative_header:
                found = True
                break
        if not found:
            cmakelists_path = os.path.join(src_dir, sh["path"], "CMakeLists.txt")
            print("Header '{0}' is missing; it should probably be listed in {1}".format(sh["headerfile"], cmakelists_path))
            res = 1

    if res == 0:
        print("No missing headers found.")

    return res

if __name__ == "__main__":
    sys.exit(main())
