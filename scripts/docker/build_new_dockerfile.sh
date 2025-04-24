#!/bin/bash
##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

function usage() {
    echo "Usage:          ./build_new_dockerfile.sh <compiler_name> <compiler_full_version> <optional_cuda_version>"
    echo "Example:        ./build_new_dockerfile.sh gcc 13.1.0"
    echo "Example (CUDA): ./build_new_dockerfile.sh gcc 12.3.0 12-3"
}

# Must be between two and three args
if [ "$#" -eq 2 ] ; then
    using_cuda=false
elif [ "$#" -eq 3 ] ; then
    using_cuda=true
else
    usage
    exit 1
fi

name=$1
ver=$2
cuda_ver=$3

maj_ver="${ver%%\.*}"

if [[ ! "$ver" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ; then
    echo "Error: specify full compiler version in the format X.Y.Z"
    usage
    exit 1
fi

if [ $using_cuda = true ] ; then
    if [[ ! "$cuda_ver" =~ ^[0-9]+-[0-9]+$ ]] ; then
        echo "Error: specify full CUDA version in the format X-Y"
        usage
        exit 1
    fi

    cuda_maj_ver="${cuda_ver%-*}"
    tag_name="cuda-${cuda_maj_ver}"
    image="ghcr.io/llnl/radiuss:cuda-${cuda_ver}-ubuntu-22.04"
    spec="%${name}@${ver}+cuda+raja+umpire+shared cuda_arch=70"
else
    tag_name="${name}-${maj_ver}"
    image="ghcr.io/llnl/radiuss:${name}-${maj_ver}-ubuntu-22.04"
    spec="%${name}@${ver}+shared"
fi

dockerfile_name="dockerfile_$tag_name"

sed -e "s/<VER>/$ver/g" \
    -e "s/<MAJ_VER>/$maj_ver/g" \
    -e "s/<NAME>/$name/g" \
    -e "s/<SPEC>/$spec/g" \
    -e "s@<IMAGE>@$image@g" dockerfile.in > "$dockerfile_name"

# Remove extra sections to save disk space and not `make test`, since runners don't have GPUs
if [ $using_cuda = true ] ; then
    sed -i "s/make -j4 test && //g" "$dockerfile_name"
    sed -i "s/texlive-full //g" "$dockerfile_name"
fi
