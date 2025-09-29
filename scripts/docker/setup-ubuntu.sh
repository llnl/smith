#!/bin/bash
##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

skip_devtools=$1

echo "=== Setup Ubuntu Args ==="
echo "skip_devtools: $skip_devtools"
echo "========================="

sudo apt-get update -y
sudo apt-get install gettext gfortran-$(gcc -dumpversion) libomp-dev libopenblas-dev \
                     lsb-release lua5.2 lua5.2-dev mpich  ssh -fy
if $skip_devtools; then
  # Install devtool-related packages
  # NOTE: Skipping this can significantly save disk space
  sudo apt-get install graphviz python3-sphinx texlive-full -fy
  echo "Installing proper doxygen version (should match version in LC host configs)"
  sudo wget https://github.com/doxygen/doxygen/releases/download/Release_1_9_8/doxygen-1.9.8.linux.bin.tar.gz
  sudo tar -xf doxygen-1.9.8.linux.bin.tar.gz
  cd doxygen-1.9.8 && sudo make && sudo make install && doxygen --version
else
  echo "~~ Skipping devtool-related packages"
fi

echo "Removing flags from mpich"
sudo sed -i "s/ -flto=auto//g" /usr/bin/mpicxx.mpich
sudo sed -i "s/ -flto=auto//g" /usr/bin/mpicc.mpich
sudo sed -i "s/ -flto=auto//g" /usr/bin/mpifort.mpich
sudo sed -i "s/ -ffat-lto-objects//g" /usr/bin/mpicxx.mpich
sudo sed -i "s/ -ffat-lto-objects//g" /usr/bin/mpicc.mpich
sudo sed -i "s/ -ffat-lto-objects//g" /usr/bin/mpifort.mpich
sudo sed -i "s/ -fallow-invalid-boz//g" /usr/bin/mpifort.mpich
sudo sed -i "s/ -fallow-argument-mismatch//g" /usr/bin/mpifort.mpich

echo "Setup new Serac user"
sudo useradd -m -s /bin/bash -G sudo serac
