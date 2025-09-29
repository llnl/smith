#!/bin/bash
##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

echo "=== Setup Serac Variables ==="
echo "spec: $spec"
echo "============================="

# Build/install TPLs via spack and then remove the temporary build directory on success
cd serac_repo
python3 ./scripts/uberenv/uberenv.py --spack-env-file=./scripts/spack/configs/docker/ubuntu24/spack.yaml \
                                     --project-json=.uberenv_config.json \
                                     --spec="$spec" --prefix=../serac_tpls -k
rm -rf ../serac_tpls/build_stage ../serac_tpls/spack

mkdir -p ../export_hostconfig
cp ./serac_repo/*.cmake ../export_hostconfig

# Make sure the new hostconfig worked
# Note: having high job slots causes build log to disappear and job to fail
cd serac_repo
python3 config-build.py -hc *.cmake -bp build -DBUILD_SHARED_LIBS=ON
cd build
make -j4 VERBOSE=1
make -j4 test
cd /home/serac
rm -rf serac_repo
