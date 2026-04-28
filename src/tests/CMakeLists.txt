# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
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
    blt::mpi)

blt_list_append(TO language_depends ELEMENTS blt::openmp IF SMITH_ENABLE_OPENMP)

smith_add_tests( SOURCES    ${language_tests}
                 DEPENDS_ON ${language_depends})

#
# Enzyme test
#

if(ENZYME_FOUND)
    smith_add_tests( SOURCES    enzyme_smoketest.cpp
                     DEPENDS_ON ClangEnzymeFlags gtest)
endif()

#
# GPU tests
#

if(SMITH_ENABLE_CUDA)
    smith_add_library( NAME       cuda_smoketest_kernel
                       SOURCES    cuda_smoketest_kernel.cpp)

    smith_add_tests( SOURCES    cuda_smoketest.cpp
                     DEPENDS_ON cuda_smoketest_kernel gtest)
endif()

if(SMITH_ENABLE_HIP)
    smith_add_tests( SOURCES    hip_smoketest.cpp
                     DEPENDS_ON gtest)
endif()

#
# TPL regression and smoke tests
#

smith_add_tests( SOURCES       axom_smoketest.cpp
                 DEPENDS_ON    axom::inlet)

if(PETSC_FOUND)
    smith_add_tests(SOURCES       mfem_petsc_smoketest.cpp
                    DEPENDS_ON    gtest smith_infrastructure
                    NUM_MPI_TASKS 4)

    if(SLEPC_FOUND)
        smith_add_tests(SOURCES       mfem_slepc_smoketest.cpp
                        DEPENDS_ON    gtest smith_infrastructure
                        NUM_MPI_TASKS 4)
    endif()
endif()
