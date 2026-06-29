#!/bin/bash
# Override version installed from https://github.com/llnl/radiuss-docker/blob/main/scripts/install-cmake-binary.sh
# TODO remove this one CMake version has been updated in radiuss-docker repo

set -euo pipefail
set -x
: ${CMAKE:=4.3.4}
curl -s -L https://github.com/Kitware/CMake/releases/download/v$CMAKE/cmake-$CMAKE-linux-x86_64.sh > cmake.sh
sh cmake.sh --prefix=/usr/local --skip-license
rm cmake.sh
