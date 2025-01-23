#!/bin/sh
"exec" "python3" "-u" "-B" "$0" "$@"

# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# Original Source: https://llnl-hatchet.readthedocs.io/en/latest/llnl.html#id3

import sys
import os
import platform
import datetime as dt
from argparse import ArgumentParser
from common_build_functions import *


# Setup SPOT db and hatchet (LC systems only)
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
                      required=True,
                      help="Path to a Serac build containing caliper files (make sure it's Release and benchmarks are enabled!)")
    parser.add_argument("-sd", "--spot-directory",
                      dest="spot_dir",
                      default=get_shared_spot_dir(),
                      help="Where to put all resulting caliper files to use for SPOT analysis (defaults to a shared location)")
    parser.add_argument("-ma", "--max-allowance",
                      dest="max_allowance",
                      default=10,
                      help="Maximum difference (in seconds) current benchmarks are allowed to be from associated baseline")
    parser.add_argument("-v", "--verbose",
                      dest="verbose",
                      action="store_true",
                      default=False,
                      help="Additionally print graph frames")

    # Parse args
    args, _ = parser.parse_known_args()
    args = vars(args)

    return args


def get_benchmark_id(gf):
    """Get unique benchmark id from a graph frame"""
    cluster = str(gf.metadata.get("cluster", 1))
    compiler = str(gf.metadata.get("serac_compiler", 1)).replace(" version ", "_")
    executable = str(gf.metadata.get("executable", 1))
    job_size = int(gf.metadata.get("jobsize", 1)) 
    return "{0}_{1}_{2}_{3}".format(cluster, compiler, executable, job_size)


def get_max(min_max):
    """Given a string containing a min and max, return the max value"""
    return float(min_max.split()[3])


def main():
    # setup
    args = parse_args()

    build_dir = os.path.abspath(args["build_dir"])
    spot_dir = os.path.abspath(args["spot_dir"])
    max_allowance = args["max_allowance"]
    verbose = args["verbose"]

    # Dictionary of summaries for each benchmark
    # key = benchmark id
    # val = summary string
    min_maxes = dict()

    # Setup baseline (shared SPOT) graph frames
    # Only take caliper files from the previous week
    baseline_calis = os.listdir(spot_dir)
    baseline_calis.sort(reverse=True)
    last_weekly_benchmark_date = baseline_calis[0].split('-')[0]
    delete_index = 0
    for i in range(len(baseline_calis)):
        if last_weekly_benchmark_date in baseline_calis[i]:
            baseline_calis[i] = os.path.join(spot_dir, baseline_calis[i])
        else:
            delete_index = i
            break
    del baseline_calis[delete_index:]
    db = spotdb.connect(spot_dir)
    gfs_baseline = hatchet.GraphFrame.from_spotdb(db, baseline_calis)

    # Setup current (local build dir) graph frames
    current_calis = list()
    for file in os.listdir(build_dir):
        if ".cali" in file:
            current_calis.append(os.path.join(build_dir, file))
    db = spotdb.connect(build_dir)
    gfs_current = hatchet.GraphFrame.from_spotdb(db, current_calis)

    # Only keep graph frames that match the current cluster/ machine name
    gfs_baseline = [gf for gf in gfs_baseline if get_machine_name() == str(gf.metadata.get("cluster"))] 
    gfs_current = [gf for gf in gfs_current if get_machine_name() == str(gf.metadata.get("cluster"))] 

    # Create dictionary of current graph frames for fast look-ups
    gfs_current_dict = dict()
    for gf in gfs_current:
        id = get_benchmark_id(gf)
        gfs_current_dict[id] = gf

    # Generate graph frames from the difference between associating current and baseline benchmarks
    for gf_baseline in gfs_baseline:
        id = get_benchmark_id(gf_baseline)
        gf_current = gfs_current_dict.get(id)
        if gf_current == None:
            print("Warning: Failed to find benchmark in build dir with the following id {0}".format(id))
            continue

        gf_diff = gf_current - gf_baseline

        # Print difference tree. Higher difference means local build is X seconds slower.
        if verbose:
            print("Hatchet diff tree for {0}:".format(id))
            print(gf_diff.tree())

        shallow_tree_str = gf_diff.tree(depth=1).splitlines()
        for line in shallow_tree_str:
            pos = line.find("Min:")
            if pos != -1:
                min_max_str = line[pos:].strip(")")
                min_maxes[id] = min_max_str
                break

    # Calculate number of "failed" benchmarks
    num_failed = 0
    num_passed = 0
    num_benchmarks = len(min_maxes)
    print(f"{'Status':<10} {'Benchmark ID':<60} {'Min/Max (seconds)':<20}")
    for id, min_max in min_maxes.items():
        max = get_max(min_max)
        status_str = ""
        if max >= max_allowance:
            num_failed += 1
            status_str = "❌ Failed"
        else:
            num_passed += 1
            status_str = "✅ Passed"

        # Print whether an individual benchmark passed or failed
        print(f"{status_str:<10} {id:<60} {min_max:<20}")

    # Print summary
    print("\n{0} out of {1} benchmarks passed given a max allowance of {2}".format(
        num_passed, num_benchmarks, max_allowance))

    return num_failed


if __name__ == "__main__":
    sys.exit(main())
