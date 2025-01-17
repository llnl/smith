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
from IPython.display import HTML, display

input_deploy_dir_str = "/usr/gapps/spot/live/"
machine = platform.uname().machine

sys.path.append(input_deploy_dir_str + "/hatchet-venv/" + machine + "/lib/python3.7/site-packages")
sys.path.append(input_deploy_dir_str + "/hatchet/" + machine)
sys.path.append(input_deploy_dir_str + "/spotdb")

import hatchet
import spotdb

# Find benchmarks run on this branch `your_benchmarks`
# OR
# Use given caliper file (to compare benchmarks one at a time)

# Find latest benchmarks in shared spot location `latest_shared_benchmarks`

# Iterate latest benchmarks list
    # Find associated branch benchmark
    # diff = PR - develop, if the `main` number is bigger, it means it took X more seconds

input_db_uri_str = "/usr/workspace/meemee/serac/repo/build-ruby-toss_4_x86_64_ib-gcc@10.3.1-release"
input_run_ids_str = "{0},{1}".format(
    os.path.join(input_db_uri_str, "physics-benchmarks-functional-baseline.cali"),
    os.path.join(input_db_uri_str, "physics-benchmarks-functional-PR.cali"))

db = spotdb.connect(input_db_uri_str)
runs = input_run_ids_str.split(',')

gfs = hatchet.GraphFrame.from_spotdb(db, runs)

gf_diff = gfs[1] - gfs[0]
#gf_diff.drop_index_levels()

# print(gf_diff.dataframe)
# print(gf_diff.tree(rank=0))

print(gf_diff.tree(depth=1))

# for idx, gf in enumerate(gfs):
#     launchdate = dt.datetime.fromtimestamp(int(gf.metadata["launchdate"]))
#     jobsize = int(gf.metadata.get("jobsize", 1))
#     print("launchdate: {}, jobsize: {}".format(launchdate, jobsize))
#     print(gf.tree())
#     display(HTML(gf.dataframe.to_html()))
