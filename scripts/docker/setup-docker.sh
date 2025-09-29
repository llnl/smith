#!/bin/bash
##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

spec=$1

echo "=== Setup Docker Args ==="
echo "spec:   $spec"
echo "branch: $branch"
echo "========================="

sudo apt-get update -y
sudo apt-get install gettext gfortran-$(gcc -dumpversion) graphviz libomp-dev libopenblas-dev \
                     lsb-release lua5.2 lua5.2-dev mpich python3-sphinx ssh texlive-full -fy
sudo useradd -m -s /bin/bash -G sudo serac

echo "Installing proper doxygen version (should match version in LC host configs)"
sudo wget https://github.com/doxygen/doxygen/releases/download/Release_1_9_8/doxygen-1.9.8.linux.bin.tar.gz
sudo tar -xf doxygen-1.9.8.linux.bin.tar.gz
cd doxygen-1.9.8 && sudo make && sudo make install && doxygen --version

echo "Removing flags from mpich"
sudo sed -i "s/ -flto=auto//g" /usr/bin/mpicxx.mpich
sudo sed -i "s/ -flto=auto//g" /usr/bin/mpicc.mpich
sudo sed -i "s/ -flto=auto//g" /usr/bin/mpifort.mpich
sudo sed -i "s/ -ffat-lto-objects//g" /usr/bin/mpicxx.mpich
sudo sed -i "s/ -ffat-lto-objects//g" /usr/bin/mpicc.mpich
sudo sed -i "s/ -ffat-lto-objects//g" /usr/bin/mpifort.mpich
sudo sed -i "s/ -fallow-invalid-boz//g" /usr/bin/mpifort.mpich
sudo sed -i "s/ -fallow-argument-mismatch//g" /usr/bin/mpifort.mpich

# Become Serac User
su - serac

cd /home/serac
git clone --recursive --branch $branch --single-branch --depth 1 https://github.com/LLNL/serac.git serac_repo

# Build/install TPLs via spack and then remove the temporary build directory on success
cd serac_repo
python3 ./scripts/uberenv/uberenv.py --spack-env-file=./scripts/spack/configs/docker/ubuntu22/spack.yaml \
                                     --project-json=.uberenv_config.json \
                                     --spec="$spec" --prefix=/home/serac/serac_tpls -k
rm -rf /home/serac/serac_tpls/build_stage /home/serac/serac_tpls/spack

mkdir -p /home/serac/export_hostconfig
cp ./serac_repo/*.cmake /home/serac/export_hostconfig

# Make sure the new hostconfig worked
# Note: having high job slots causes build log to disappear and job to fail
cd serac_repo
python3 config-build.py -hc *.cmake -bp build -DBUILD_SHARED_LIBS=ON
cd build
make -j4 VERBOSE=1
make -j4 test
cd /home/serac
rm -rf serac_repo
