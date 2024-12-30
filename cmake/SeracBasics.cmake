# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Options
#------------------------------------------------------------------------------
option(ENABLE_ASAN "Enable AddressSanitizer for memory checking (Clang or GCC only)" OFF)
if(ENABLE_ASAN)
    if(NOT (C_COMPILER_FAMILY_IS_CLANG OR C_COMPILER_FAMILY_IS_GNU))
        message(FATAL_ERROR "ENABLE_ASAN only supports Clang and GCC")
    endif()
endif()

option(SERAC_ENABLE_CODEVELOP
       "Enable Serac's codevelop build (MFEM and Axom included as CMake subdirectories)"
       OFF)

# Only enable Serac's code checks by default if it is the top-level project
# or a user overrides it
if("${CMAKE_PROJECT_NAME}" STREQUAL "serac")
    set(_enable_serac_code_checks ON)
else()
    set(_enable_serac_code_checks OFF)
endif()
option(SERAC_ENABLE_CODE_CHECKS "Enable Serac's code checks" ${_enable_serac_code_checks})

cmake_dependent_option(SERAC_ENABLE_TESTS "Enables Serac Tests" ON "ENABLE_TESTS" OFF)
cmake_dependent_option(SERAC_ENABLE_CUDA "Enables Serac with CUDA support" ON "ENABLE_CUDA" OFF)
cmake_dependent_option(SERAC_ENABLE_HIP "Enables Serac with HIP support" ON "ENABLE_HIP" OFF)
cmake_dependent_option(SERAC_ENABLE_MPI "Enables Serac with MPI support" ON "ENABLE_MPI" OFF)
cmake_dependent_option(SERAC_ENABLE_OPENMP "Enables Serac with OPENMP support" ON "ENABLE_OPENMP" OFF)

#------------------------------------------------------------------------------
# Profiling options
#------------------------------------------------------------------------------
# User turned on benchmarking but didn't turn on profiling
if ((ENABLE_BENCHMARKS OR SERAC_ENABLE_BENCHMARKS) AND NOT DEFINED SERAC_ENABLE_PROFILING)
    set(SERAC_ENABLE_PROFILING ON)
endif()

option(SERAC_ENABLE_PROFILING "Enable profiling functionality" OFF)

# ENABLE_BENCHMARKS must be ON in order to modify SERAC_ENABLE_BENCHMARKS.
# SERAC_ENABLE_BENCHMARKS will automatically be set to ON if ENABLE_BENCHMARKS is ON.
# SERAC_ENABLE_BENCHMARKS is an option to allow external projects to disable Serac benchmarks while enabling theirs.
cmake_dependent_option(SERAC_ENABLE_BENCHMARKS "Enable benchmark executables" ON "ENABLE_BENCHMARKS" OFF)

# User turned on benchmarking but explicitly turned off profiling. Error out.
if ((ENABLE_BENCHMARKS OR SERAC_ENABLE_BENCHMARKS) AND NOT SERAC_ENABLE_PROFILING)
    message(FATAL_ERROR
            "Both ENABLE_BENCHMARKS and SERAC_ENABLE_BENCHMARKS require SERAC_ENABLE_PROFILING to be turned on")
endif()

#------------------------------------------------------------------------------
# Create symlink in installed bin
#------------------------------------------------------------------------------
if(GLVIS_EXECUTABLE)
    add_custom_target(glvis_symlink ALL
                      COMMAND ${CMAKE_COMMAND}
                      -E create_symlink ${GLVIS_EXECUTABLE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/glvis)
endif()

#------------------------------------------------------------------------------
# Set ordering configuration
#------------------------------------------------------------------------------
option(SERAC_USE_VDIM_ORDERING "Use mfem::Ordering::byVDIM for DOF vectors (faster for algebraic multigrid)" ON)
if (SERAC_USE_VDIM_ORDERING)
  message(STATUS "Using byVDIM degree-of-freedom vector ordering.")
else()
  message(STATUS "Using byNODES degree-of-freedom vector ordering.")
endif()
