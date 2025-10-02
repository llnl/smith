#!/bin/bash
# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

echo "=============================="
echo "setup-serac.sh script env vars"
echo "spec: $spec"
echo "=============================="

# Build/install TPLs via spack 
cd /home/serac/serac_repo
python3 ./scripts/uberenv/uberenv.py --spack-env-file=./scripts/spack/configs/docker/ubuntu24/spack.yaml \
                                     --project-json=.uberenv_config.json --spec="$spec" --prefix=../serac_tpls -k -j4

# Remove the temporary build directory on success
rm -rf ../serac_tpls/build_stage ../serac_tpls/spack

# Store hostconfig
mkdir -p ../export_hostconfig
cp *.cmake ../export_hostconfig

# Make sure the new hostconfig worked
# Note: having high job slots causes build log to disappear and job to fail
python3 config-build.py -hc *.cmake -bp build -DBUILD_SHARED_LIBS=ON
cd build
make -j4 VERBOSE=1
if [[ "$spec" != *+cuda* ]] || [[ "$spec" != *+rocm* ]]; then
  make -j4 test
fi

# Remove repo to save space
# NOTE: this script is self-destructive
cd /home/serac
rm -rf serac_repo
