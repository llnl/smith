# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Basic CMake Setup
#------------------------------------------------------------------------------

# Do not bump this over 3.14 due to it changing CMake policies and breaking
# the CUDA build. We check the required version specifically for this below
cmake_minimum_required(VERSION 3.14) 

# Check for minimum CMake version required w/o changing policies like cmake_minimum_required
if(ENABLE_CUDA AND ${CMAKE_VERSION} VERSION_LESS 3.18.0)
    message(FATAL_ERROR "Serac requires CMake version 3.18.0+ when CUDA is enabled.")
endif()

project(serac LANGUAGES CXX C)

# MPI is required in Serac.
# Note: for the style only mode, which doesn't need to actually build, we need to ignore this
if (NOT SERAC_STYLE_CI_ONLY)
    set(ENABLE_MPI ON CACHE BOOL "")

    if (NOT MPI_C_COMPILER OR NOT MPI_CXX_COMPILER)
        message(FATAL_ERROR 
                "Serac requires MPI.  It is required to provide the MPI C/C++ "
                "compiler wrappers via the CMake variables, "
                "MPI_C_COMPILER and MPI_CXX_COMPILER.")
    endif()
endif()


# Save off "raw" flags before they are modified by serac/BLT
set(RAW_CMAKE_C_FLAGS   ${CMAKE_C_FLAGS}   CACHE STRING "")
set(RAW_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} CACHE STRING "")

if (NOT DEFINED SERAC_ENABLE_CODEVELOP)
   set(SERAC_ENABLE_CODEVELOP OFF)
endif()
message(STATUS "Serac Codevelop Build: ${SERAC_ENABLE_CODEVELOP}")

if(NOT DEFINED SERAC_SOURCE_DIR)
  set(SERAC_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "")
endif()
message(STATUS "Serac Source Dir: ${SERAC_SOURCE_DIR}")

# For HIP case, this flag prevents the following Strumpack linker error:
# undefined symbol: mpi_abort_
if(SERAC_ENABLE_CODEVELOP OR (ENABLE_HIP AND STRUMPACK_DIR))
    set(ENABLE_FORTRAN ON CACHE BOOL "")
endif()

# Prevent deprecated warnings from C++17 in GTest on TOSS4Cray
if(DEFINED ENV{SYS_TYPE} AND "$ENV{SYS_TYPE}" STREQUAL "toss_4_x86_64_ib_cray")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated-declarations")
endif()

#------------------------------------------------------------------------------
# Setup BLT
#------------------------------------------------------------------------------

if(DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR
    if(NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt" CACHE PATH "")
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR
            "The BLT is not present. "
            "Either run the following two commands in your git repository: \n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif()
endif()

# Tune BLT to our needs
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++17" CACHE STRING "")
endif()

# These BLT tools are not used in Serac, turn them off
set(_unused_blt_tools
    CLANGQUERY
    VALGRIND
    ASTYLE
    CMAKEFORMAT
    UNCRUSTIFY
    YAPF)
foreach(_tool ${_unused_blt_tools})
    set(ENABLE_${_tool} OFF CACHE BOOL "")
endforeach()

# These BLT tools are only used by Serac developers, so turn them off
# unless an explicit executable path is given
set(_used_blt_tools
    CLANGFORMAT
    CLANGTIDY
    CPPCHECK
    DOXYGEN
    SPHINX)
foreach(_tool ${_used_blt_tools})
    if(NOT ${_tool}_EXECUTABLE)
        set(ENABLE_${_tool} OFF CACHE BOOL "")
    else()
        set(ENABLE_${_tool} ON CACHE BOOL "")
    endif()
endforeach()

set(BLT_REQUIRED_CLANGFORMAT_VERSION  "14" CACHE STRING "")

set(ENABLE_ALL_WARNINGS ON CACHE BOOL "")
set(ENABLE_WARNINGS_AS_ERRORS ON CACHE BOOL "")

set(BLT_EXPORT_THIRDPARTY OFF CACHE BOOL "")
if (BLT_EXPORT_THIRDPARTY)
    message(FATAL_ERROR "Serac does not use BLT's exporting of TPLs, it conflicts with the new blt_install_tpl_setups macro.")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

#------------------------------------------------------------------------------
# Setup Macros/Basics
#------------------------------------------------------------------------------

include(${PROJECT_SOURCE_DIR}/cmake/SeracMacros.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/SeracBasics.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/SeracCompilerFlags.cmake)

#------------------------------------------------------------------------------
# Add Code Checks
#------------------------------------------------------------------------------

message(STATUS "Serac Code Checks: ${SERAC_ENABLE_CODE_CHECKS}")

# Add extra file extensions for code styling
# Note: Add them also to serac_add_code_checks in cmake/SeracMacros.cmake

# CUDA
list(APPEND BLT_C_FILE_EXTS ".cuh")
# Configured C++ files
list(APPEND BLT_C_FILE_EXTS ".cpp.in" ".hpp.in")

if (SERAC_STYLE_CI_ONLY OR SERAC_ENABLE_CODE_CHECKS)
    serac_add_code_checks(PREFIX serac)
endif()

if (SERAC_STYLE_CI_ONLY)
    # Exit processing the rest of the build in style only build to avoid any possible
    # CMake configuration issues outside of just enabling `make style`. This build,
    # is not capable of building Serac and should not be advertised. It is for CI 
    # purposes only.
    return()
endif()

#------------------------------------------------------------------------------
# Setup Dependencies
#------------------------------------------------------------------------------

# Restore raw compiler flags so subprojects don't end up using Serac's flags
set(SERAC_CMAKE_C_FLAGS   ${CMAKE_C_FLAGS})
set(SERAC_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_C_FLAGS         ${RAW_CMAKE_C_FLAGS} CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS       ${RAW_CMAKE_CXX_FLAGS} CACHE STRING "" FORCE)

include(${PROJECT_SOURCE_DIR}/cmake/thirdparty/SetupSeracThirdParty.cmake)

set(CMAKE_C_FLAGS   ${SERAC_CMAKE_C_FLAGS})
set(CMAKE_CXX_FLAGS ${SERAC_CMAKE_CXX_FLAGS})

include(${PROJECT_SOURCE_DIR}/cmake/SeracConfigHeader.cmake)

#------------------------------------------------------------------------------
# Add subdirectories
#------------------------------------------------------------------------------

add_subdirectory(src)

add_subdirectory(examples)

#------------------------------------------------------------------------------
# Generate helper script for running ATS
#------------------------------------------------------------------------------
configure_file(
    ${PROJECT_SOURCE_DIR}/scripts/testing/ats.sh.in
    ${CMAKE_BINARY_DIR}/ats.sh)

#------------------------------------------------------------------------------
# Export Targets
#------------------------------------------------------------------------------
set(exported_targets
    serac_infrastructure
    serac_mesh
    serac_numerics
    serac_physics)

add_library(serac INTERFACE)
target_link_libraries(serac INTERFACE ${exported_targets})
install(TARGETS              serac
        EXPORT               serac-targets
        DESTINATION          lib
        )

if (SERAC_ENABLE_CODEVELOP)
    # This really shouldn't be needed but it keeps getting dropped from axom.
    # It certainly shouldn't go here either.
    target_include_directories(axom SYSTEM INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/axom/src>
    )

    install(TARGETS              mfem ${axom_exported_targets} ${tribol_exported_targets}
            EXPORT               serac-targets
            DESTINATION          lib
            )
endif()

install(EXPORT serac-targets 
        NAMESPACE serac::
        DESTINATION lib/cmake
        )
