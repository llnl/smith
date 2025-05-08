.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _ubuntu_build-label:

==================
Serac Ubuntu Build
==================

------------------
Basic System Setup
------------------

We recommend installing some basic system-level development packages to minimize the
amount of packages that Spack will build.

Install clang version 14 and make it the default compiler:

.. code-block:: bash

    sudo apt install -y --no-install-recommends clang-14 clang-format-14 llvm-14 libomp-14-dev gfortran-13
    # Set clang-14 as the default clang
    sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-14 100 \
    && sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100 \
    && sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-14 100

Install required build packages to minimize what Spack will build:

.. code-block:: bash

    sudo apt-get -qq install -y --no-install-recommends build-essential bzip2 cmake libopenblas-dev lua5.2 lua5.2-dev mpich unzip
    # Remove problematic flags from mpich
    sudo sed -i "s/ -flto=auto//g" /usr/bin/mpicxx.mpich \
    && sudo sed -i "s/ -flto=auto//g" /usr/bin/mpicc.mpich \
    && sudo sed -i "s/ -flto=auto//g" /usr/bin/mpifort.mpich \
    && sudo sed -i "s/ -ffat-lto-objects//g" /usr/bin/mpicxx.mpich \
    && sudo sed -i "s/ -ffat-lto-objects//g" /usr/bin/mpicc.mpich \
    && sudo sed -i "s/ -ffat-lto-objects//g" /usr/bin/mpifort.mpich \
    && sudo sed -i "s/ -fallow-invalid-boz//g" /usr/bin/mpifort.mpich \
    && sudo sed -i "s/ -fallow-argument-mismatch//g" /usr/bin/mpifort.mpich

Optionally install packages to generate documenation:

.. code-block:: bash

    sudo apt-get -qq install -y --no-install-recommends graphviz python3-sphinx texlive-full
    sudo wget https://github.com/doxygen/doxygen/releases/download/Release_1_9_8/doxygen-1.9.8.linux.bin.tar.gz
    sudo tar -xf doxygen-1.9.8.linux.bin.tar.gz
    cd doxygen-1.9.8 && sudo make && sudo make install && doxygen --version


-------------------------------
Generate Spack Environment File
-------------------------------

Spack uses an environment file to describe where system level packages are to minimize what it builds.
The following command will download the specific Spack version we use and run a minimal set of commands to
generate an environment file for you. This should be a good starting point and should be used in following
Spack builds.

.. code-block:: bash

    scripts/uberenv/uberenv.py --prefix=<path/outside/repository> --setup-and-env-only

