#!/bin/sh
"exec" "python3" "-u" "-B" "$0" "$@"
##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

import argparse
import os
import re
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Checks log for warnings and errors")

    parser.add_argument("-l", "--log", type=str, required=True, help="Path to log file to be checked")
    parser.add_argument("-i", "--ignore", type=str, required=False, help="Path to file that includes regex's to ignore lines")

    args = parser.parse_args()

    print("~~~~~~~ Given Command line Arguments ~~~~~~~")
    print("Ignore file Path: {0}".format(args.ignore))
    print("Log Path:         {0}".format(args.log))
    print("")

    return args


def main():
    args = parse_args()

    # read lines from log
    with open(args.log, "r") as f:
        log_lines = f.readlines()

    # read ignore regex's
    ignore_regexs = []
    if args.ignore:
        with open(args.ignore, "r") as f:
            # Only save lines that aren't comments ("#")
            for currline in f.readlines():
                stripped = currline.strip()
                if stripped != "" and not stripped.startswith("#"):
                    # Add regex w/o trailing newline characters
                    ignore_regexs.append(currline.rstrip())

    # Get warnings/errors out of log file and print ignored as you find them
    # with what regex matched
    warnings = []
    errors = []
    ignored_count = 0
    print("~~~~~~~~~~~~~~~ Ignored ~~~~~~~~~~~~~~~~")
    for log_line in log_lines:
        # First, check if it is a warning/error
        lowered = log_line.lower()
        has_warning = "warning:" in lowered
        has_error = "error:" in lowered

        # Then, check if it matches any ignore regex's
        matches_ignore = False
        if has_error or has_warning:
            for ignore_regex in ignore_regexs:
                if re.search(ignore_regex, log_line.rstrip()):
                    print("line : {0}".format(log_line.rstrip()))
                    print("regex: {0}\n".format(ignore_regex))
                    matches_ignore = True
                    ignored_count += 1
                    break

        if not matches_ignore:
            if has_error:
                errors.append(log_line)
            elif has_warning:
                warnings.append(log_line)

    # Print warnings/errors that are found
    print("~~~~~~~~~~~~~~~ Warnings ~~~~~~~~~~~~~~~")
    for warning in warnings:
        print(warning)
    print("")

    print("~~~~~~~~~~~~~~~ Errors ~~~~~~~~~~~~~~~~~")
    for error in errors:
        print(error)
    print("")

    # Print summary info
    print("~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~")
    print("Warning Count: {0}".format(len(warnings)))
    print("Error Count:   {0}".format(len(errors)))
    print("Ignored Count: {0}".format(ignored_count))

    # Error out if any found
    return len(warnings) + len(errors)

if __name__ == "__main__":
    sys.exit(main())
