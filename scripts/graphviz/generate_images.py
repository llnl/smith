#!/bin/sh

# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"exec" "python3" "-u" "-B" "$0" "$@"

#
# usage:
# python3 ./config-build.py --graphviz=graphviz/graph.dot
# python3 ./scripts/graphviz/generate_images.py --graphviz=/path/to/build/graphviz
#

import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g",
                        "--graphviz",
                        type=str,
                        default="",
                        help="specify path of the graphviz directory.")
    return parser.parse_known_args()

def main():
    args, unknown_args = parse_arguments()

    # go to graphviz dir
    graphviz_dir = args.graphviz
    if os.path.exists(graphviz_dir):
        os.chdir(graphviz_dir)
    else:
        print("Error: graphviz directory does not exist {}".format(graphviz_dir))
        return 1

    # create image directory if doesn't exist
    image_dir = "images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # scan all dot files and create image for them
    for filename in os.listdir():
        if ".dot" in filename:
            cmd = "dot -T png -o {0}/{1}.png {1}".format(image_dir, filename)
            os.system(cmd)

    return 0

if __name__ == '__main__':
    exit(main())
