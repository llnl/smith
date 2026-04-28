#!/bin/bash
##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

set -x

function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

echo HOST_CONFIG
echo $HOST_CONFIG

echo "~~~~~~ RUNNING CMAKE ~~~~~~~~"
cmake_args="-DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=ON -DENABLE_CLANGTIDY=OFF -DSMITH_ENABLE_CODE_CHECKS=ON"

if [[ "$CHECK_TYPE" == "coverage" ]] ; then
    # Alias llvm-cov to gcov so it acts like gcov
    ln -s `which llvm-cov` /home/serac/gcov
    cmake_args="$cmake_args -DENABLE_COVERAGE=ON -DGCOV_EXECUTABLE=/home/serac/gcov"
fi

if [[ "$CHECK_TYPE" == "docs" ]] ; then
    SPHINX_EXECUTABLE=/usr/bin/sphinx-build
    if [[ ! -f "$SPHINX_EXECUTABLE" ]]; then
        echo "sphinx not found: $SPHINX_EXECUTABLE"
        exit 1
    fi    
    DOXYGEN_EXECUTABLE=/usr/local/bin/doxygen
    if [[ ! -f "$DOXYGEN_EXECUTABLE" ]]; then
        echo "doxygen not found: $DOXYGEN_EXECUTABLE"
        exit 1
    fi    
    cmake_args="$cmake_args -DENABLE_DOCS=ON -DSPHINX_EXECUTABLE=$SPHINX_EXECUTABLE -DDOXYGEN_EXECUTABLE=$DOXYGEN_EXECUTABLE"
fi

if [[ "$CHECK_TYPE" == "style" ]] ; then
    CLANGFORMAT_EXECUTABLE=/usr/bin/clang-format
    if [[ ! -f "$CLANGFORMAT_EXECUTABLE" ]]; then
        echo "clang-format not found: $CLANGFORMAT_EXECUTABLE"
        exit 1
    fi    
    cmake_args="$cmake_args -DENABLE_CLANGFORMAT=ON -DCLANGFORMAT_EXECUTABLE=$CLANGFORMAT_EXECUTABLE"
fi

or_die ./config-build.py -hc host-configs/docker/${HOST_CONFIG} -bp build-check-debug -ip install-check-debug $cmake_args
or_die cd build-check-debug

if [[ "$CHECK_TYPE" == "coverage" ]] ; then
    or_die make -j4
    or_die make smith_coverage
    # Move cov report to repo dir, so that Github Actions can find it
    mv smith_coverage.info.cleaned ..
fi

if [[ "$CHECK_TYPE" == "docs" ]] ; then
    or_die make VERBOSE=1 docs 2>&1 | tee docs_output
    or_die ../scripts/check_log.py -l docs_output -i ../scripts/github-actions/docs_ignore_regexs.txt
fi

if [[ "$CHECK_TYPE" == "style" ]] ; then
    or_die make VERBOSE=1 clangformat_check
fi

if [[ "$CHECK_TYPE" == "header" ]] ; then
    or_die make -j4
    or_die make install -j4
    or_die ../scripts/check_for_missing_headers.py -i ../install-check-debug -s ../src
fi

exit 0
