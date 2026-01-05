# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if (NOT SMITH_BASICS_SETUP)
    # Prevent this file from being called twice in the same scope
    set(SMITH_BASICS_SETUP TRUE)

    #------------------------------------------------------------------------------
    # Options
    #------------------------------------------------------------------------------
    option(ENABLE_ASAN "Enable AddressSanitizer for memory checking (Clang or GCC only)" OFF)
    if(ENABLE_ASAN)
        if(NOT (C_COMPILER_FAMILY_IS_CLANG OR C_COMPILER_FAMILY_IS_GNU))
            message(FATAL_ERROR "ENABLE_ASAN only supports Clang and GCC")
        endif()
    endif()

    option(SMITH_ENABLE_CODEVELOP
        "Enable Smith's codevelop build (MFEM and Axom included as CMake subdirectories)"
        OFF)

    # Only enable Smith's code checks by default if it is the top-level project
    # or a user overrides it
    if("${CMAKE_PROJECT_NAME}" STREQUAL "smith")
        set(_enable_smith_code_checks ON)
    else()
        set(_enable_smith_code_checks OFF)
    endif()
    option(SMITH_ENABLE_CODE_CHECKS "Enable Smith's code checks" ${_enable_smith_code_checks})

    cmake_dependent_option(SMITH_ENABLE_TESTS "Enables Smith Tests" ON "ENABLE_TESTS" OFF)
    cmake_dependent_option(SMITH_ENABLE_CUDA "Enables Smith with CUDA support" ON "ENABLE_CUDA" OFF)
    cmake_dependent_option(SMITH_ENABLE_HIP "Enables Smith with HIP support" ON "ENABLE_HIP" OFF)
    cmake_dependent_option(SMITH_ENABLE_MPI "Enables Smith with MPI support" ON "ENABLE_MPI" OFF)
    cmake_dependent_option(SMITH_ENABLE_OPENMP "Enables Smith with OPENMP support" ON "ENABLE_OPENMP" OFF)

    # Options for builtin TPLs
    option(SMITH_ENABLE_GRETL "Enables Smith with Gretl Support" ON)
    option(SMITH_ENABLE_CONTINUATION "Enables Smith with Continuation Solver support" ON)

    if (SMITH_ENABLE_HIP OR SMITH_ENABLE_CUDA)
        message(STATUS "Disabling Smith's Continuation Solver support due to currently non-supported GPU build")
        set(SMITH_ENABLE_CONTINUATION FALSE CACHE BOOL "" FORCE)
    endif()

    #------------------------------------------------------------------------------
    # Profiling options
    #------------------------------------------------------------------------------
    # User turned on benchmarking but didn't turn on profiling
    if ((ENABLE_BENCHMARKS OR SMITH_ENABLE_BENCHMARKS) AND NOT DEFINED SMITH_ENABLE_PROFILING)
        set(SMITH_ENABLE_PROFILING ON)
    endif()

    option(SMITH_ENABLE_PROFILING "Enable profiling functionality" OFF)

    # ENABLE_BENCHMARKS must be ON in order to modify SMITH_ENABLE_BENCHMARKS.
    # SMITH_ENABLE_BENCHMARKS will automatically be set to ON if ENABLE_BENCHMARKS is ON.
    # SMITH_ENABLE_BENCHMARKS is an option to allow external projects to disable Smith benchmarks while enabling theirs.
    cmake_dependent_option(SMITH_ENABLE_BENCHMARKS "Enable benchmark executables" ON "ENABLE_BENCHMARKS" OFF)

    # User turned on benchmarking but explicitly turned off profiling. Error out.
    if ((ENABLE_BENCHMARKS OR SMITH_ENABLE_BENCHMARKS) AND NOT SMITH_ENABLE_PROFILING)
        message(FATAL_ERROR
                "Both ENABLE_BENCHMARKS and SMITH_ENABLE_BENCHMARKS require SMITH_ENABLE_PROFILING to be turned on")
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
    option(SMITH_USE_VDIM_ORDERING "Use mfem::Ordering::byVDIM for DOF vectors (faster for algebraic multigrid)" ON)
    if (SMITH_USE_VDIM_ORDERING)
        message(STATUS "Using byVDIM degree-of-freedom vector ordering.")
    else()
        message(STATUS "Using byNODES degree-of-freedom vector ordering.")
    endif()

    #------------------------------------------------------------------------------
    # CMake GraphViz
    #------------------------------------------------------------------------------
    configure_file(${PROJECT_SOURCE_DIR}/cmake/CMakeGraphVizOptions.cmake
                   ${CMAKE_BINARY_DIR}/CMakeGraphVizOptions.cmake
                   COPYONLY)

endif()
