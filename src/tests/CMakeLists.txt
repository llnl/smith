# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

#
# Language tests
#

set(language_tests
    copy_elision.cpp
    mfem_array_std_algo.cpp)

set(language_depends
    gtest
    mfem
    blt::mpi
    ${serac_device_depends})

blt_list_append(TO language_depends ELEMENTS blt::openmp IF SERAC_ENABLE_OPENMP)

serac_add_tests( SOURCES    ${language_tests}
                 DEPENDS_ON ${language_depends})

#
# Enzyme test
#

if(ENZYME_FOUND)
    serac_add_tests( SOURCES    enzyme_smoketest.cpp
                     DEPENDS_ON ClangEnzymeFlags gtest)
endif()

#
# GPU tests
#

if(SERAC_ENABLE_CUDA)
    serac_add_library( NAME       cuda_smoketest_kernel
                     SOURCES    cuda_smoketest_kernel.cpp
                     DEPENDS_ON ${serac_device_depends})

    serac_add_tests( SOURCES    cuda_smoketest.cpp
                     DEPENDS_ON cuda_smoketest_kernel gtest)
endif()

if(SERAC_ENABLE_HIP)
    serac_add_tests( SOURCES    hip_smoketest.cpp
                     DEPENDS_ON ${serac_device_depends} gtest)
endif()

#
# TPL regression and smoke tests
#

serac_add_tests( SOURCES       axom_smoketest.cpp
                 DEPENDS_ON    axom::inlet)

if(PETSC_FOUND)
    serac_add_tests(SOURCES       mfem_petsc_smoketest.cpp
                    DEPENDS_ON    gtest serac_infrastructure
                    NUM_MPI_TASKS 4)

    if(SLEPC_FOUND)
        serac_add_tests(SOURCES       mfem_slepc_smoketest.cpp
                        DEPENDS_ON    gtest serac_infrastructure
                        NUM_MPI_TASKS 4)
    endif()
endif()
