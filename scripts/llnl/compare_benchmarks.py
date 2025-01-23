#!/bin/sh
"exec" "python3" "-u" "-B" "$0" "$@"

# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# Original Source: https://llnl-hatchet.readthedocs.io/en/latest/llnl.html#id3

from common_build_functions import *

import sys
import os
import platform
import datetime as dt
from IPython.display import HTML, display
from argparse import ArgumentParser

input_deploy_dir_str = "/usr/gapps/spot/live/"
machine = platform.uname().machine
sys.path.append(input_deploy_dir_str + "/hatchet-venv/" + machine + "/lib/python3.7/site-packages")
sys.path.append(input_deploy_dir_str + "/hatchet/" + machine)
sys.path.append(input_deploy_dir_str + "/spotdb")
import hatchet
import spotdb

def parse_args():
    "Parses args from command line"
    parser = ArgumentParser()
    parser.add_argument("-bd", "--build-directory",
                      dest="build_dir",
                      help="Path to a Serac build containing caliper files (make sure it's Release and benchmarks are enabled!)")
    parser.add_argument("-sd", "--spot-directory",
                      dest="spot_dir",
                      default=get_shared_spot_dir(),
                      help="Where to put all resulting caliper files to use for SPOT analysis (defaults to a shared location)")

    # Parse args
    args, _ = parser.parse_known_args()
    args = vars(args)

    return args

def get_benchmark_name(gf):
    """Get benchmark name from a graph frame"""
    executable = str(gf.metadata.get("executable", 1))
    job_size = int(gf.metadata.get("jobsize", 1)) 
    return "{0}_{1}".format(executable, job_size)

def main():
    # setup
    args = parse_args()

    build_dir = args["build_dir"]
    spot_dir = args["spot_dir"]

    # TODO
    gfs_current = [] # build_dir, locally generated caliper files 

    # Setup baseline (shared SPOT) graph frames
    # Only take caliper files from the previous week
    baseline_calis = os.listdir(spot_dir)
    baseline_calis.sort(reverse=True)
    baseline_cali_date = baseline_calis[0].split('-')[0]
    delete_index = 0
    for i in range(len(baseline_calis)):
        if baseline_cali_date in baseline_calis[i]:
            baseline_calis[i] = os.path.join(spot_dir, baseline_calis[0])
        else:
            delete_index = i
            break
    del baseline_calis[delete_index:]
    db = spotdb.connect(spot_dir)
    gfs_baseline = hatchet.GraphFrame.from_spotdb(db, baseline_calis)

    # TODO Setup current (local build dir) graph frames

if __name__ == "__main__":
    sys.exit(main())
