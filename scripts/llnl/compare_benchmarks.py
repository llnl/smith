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
import re
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
                      help="Shared caliper files to used for SPOT analysis (defaults to a shared location)")
    parser.add_argument("-ma", "--max-allowance",
                      dest="max_allowance",
                      default=10,
                      help="Maximum difference (in seconds) current benchmarks are allowed to be from associated baseline")
    parser.add_argument("-v", "--verbose",
                      dest="verbose",
                      action="store_true",
                      default=False,
                      help="Additionally print graph frames")
    parser.add_argument("-d", "--depth",
                      dest="depth",
                      default=10000,
                      help="Depth of graph frames (if verbose is on). The default shows the full graph.")
    parser.add_argument("-mc", "--metric-column",
                      dest="metric_column",
                      default="Avg time/rank (inc)",
                      help="Set the metric column to display")

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


def remove_color_codes(text):
    """Remove color codes from a string"""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[mG]')
    return ansi_escape.sub('', text)


def get_clean_gf_tree(gf, metric_column, depth, keep_color=False):
    """Clean up a graph frame tree and return as string. Keeping the color
       will also keep the color legend."""
    gf_tree = gf.tree(depth=depth, render_header=False, metric_column=metric_column)
    if not keep_color:
        gf_tree = remove_color_codes(gf_tree)
        gf_tree = gf_tree.split("\nLegend")[0]
    else:
        gf_tree = gf_tree.split("\n\x1b[4mLegend\x1b[0m")[0]
    return gf_tree


def get_gf_tree_sum(gf, metric_column, info_type):
    """Get the sum of the graph tree depth=1. Info type can be diff, baseline, or current."""
    shallow_tree_str = get_clean_gf_tree(gf, metric_column, 1).splitlines()
    sum = 0 
    for line in shallow_tree_str:
        pos = line.find(" ")
        if pos != -1:
           sum += float(line[:pos])
    return sum


def main():
    # setup
    args = parse_args()

    build_dir = os.path.abspath(args["build_dir"])
    spot_dir = os.path.abspath(args["spot_dir"])
    max_allowance = args["max_allowance"]
    verbose = args["verbose"]
    depth = int(args["depth"])
    metric_column = args["metric_column"]

    # Dictionary of summaries for each benchmark
    benchmark_times = dict()

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

    # Only keep graph frames that match the current cluster and compiler
    gfs_baseline = [gf for gf in gfs_baseline if get_machine_name() == str(gf.metadata.get("cluster"))] 
    gfs_current = [gf for gf in gfs_current if get_machine_name() == str(gf.metadata.get("cluster"))] 

    # Filter by compiler for baseline as well, based on the local build
    compiler_current = str(gfs_current[0].metadata.get("serac_compiler", 1))
    gfs_baseline = [gf for gf in gfs_baseline if compiler_current == str(gf.metadata.get("serac_compiler"))] 

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
            print("=" * 80)
            print("Hatchet diff tree for {0}:".format(id))
            print("=" * 80)
            print(get_clean_gf_tree(gf_diff, metric_column, depth, keep_color=True))

        # Store time info between baseline and current main (or total of depth=1)
        benchmark_times[id] = {
            "diff": get_gf_tree_sum(gf_diff, metric_column, "diff"),
            "current": get_gf_tree_sum(gf_current, metric_column, "current"),
            "baseline": get_gf_tree_sum(gf_baseline, metric_column, "baseline"),
        }

    # Print metric column info
    if verbose:
        print(f"Using metric columns of '{metric_column}'. Other metric column options are:")
        print(gfs_baseline[0].show_metric_columns())
        print()

    # Print whether an individual benchmark passed or failed
    num_failed = 0
    num_passed = 0
    num_benchmarks = len(benchmark_times)
    print(f"{'Status':<10} {'Benchmark ID':<60} {'Diff (seconds)':<20} {'Baseline (seconds)':<20} {'Current (seconds)':<20}")
    for id, benchmark_time in benchmark_times.items():
        status_str = ""
        if benchmark_time["diff"] >= max_allowance:
            num_failed += 1
            status_str = "❌ Failed"
        else:
            num_passed += 1
            status_str = "✅ Passed"

        print(f"{status_str:<10} {id:<60} {benchmark_time['diff']:<20.2f} {benchmark_time['baseline']:<20.2f} {benchmark_time['current']:<20.2f}")

    # Print summary
    print(f"\n{num_passed} out of {num_benchmarks} benchmarks passed given a max allowance of {max_allowance} seconds")

    return num_failed


if __name__ == "__main__":
    sys.exit(main())
