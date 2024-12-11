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

# Find latest benchmarks in shared spot location `latest_shared_benchmarks`

# Find benchmarks run on this branch `your_benchmarks`

# Iterate latest benchmarks list
    # Find associated branch benchmark

input_db_uri_str = "/usr/workspace/smithdev/califiles"
input_run_ids_str = "{0},{1},{2}".format(
    os.path.join(input_db_uri_str, "241207-013256_172562_MrFHk9gvGQ9q.cali"),
    os.path.join(input_db_uri_str, "241207-013250_172139_sSyXuwBLSRyG.cali"),
    os.path.join(input_db_uri_str, "241207-013241_171769_ErlTtmmkBAAY.cali"))
print(input_run_ids_str)

db = spotdb.connect(input_db_uri_str)
runs = input_run_ids_str.split(',')

gfs = hatchet.GraphFrame.from_spotdb(db, runs)

for idx, gf in enumerate(gfs):
    launchdate = dt.datetime.fromtimestamp(int(gf.metadata["launchdate"]))
    jobsize = int(gf.metadata.get("jobsize", 1))
    print("launchdate: {}, jobsize: {}".format(launchdate, jobsize))
    print(gf.tree())
    display(HTML(gf.dataframe.to_html()))
