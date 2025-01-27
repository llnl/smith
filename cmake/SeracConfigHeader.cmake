# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Version information that go into the generated config header
#------------------------------------------------------------------------------
set(SERAC_VERSION_MAJOR 0)
set(SERAC_VERSION_MINOR 0)
set(SERAC_VERSION_PATCH 1)
string(CONCAT SERAC_VERSION_FULL
    "v${SERAC_VERSION_MAJOR}"
    ".${SERAC_VERSION_MINOR}"
    ".${SERAC_VERSION_PATCH}" )

if (Git_FOUND)
  ## check to see if we are building from a Git repo or an exported tarball
  blt_is_git_repo( OUTPUT_STATE is_git_repo )

  if(${is_git_repo})
    blt_git_hashcode(HASHCODE sha1 RETURN_CODE rc)
    if(NOT ${rc} EQUAL 0)
      message(FATAL_ERROR "blt_git_hashcode failed!")
    endif()

    set(SERAC_GIT_SHA ${sha1})
  endif()

endif()

message(STATUS "Configuring Serac version ${SERAC_VERSION_FULL}")


#------------------------------------------------------------------------------
# Create variable for every TPL
#------------------------------------------------------------------------------
set(TPL_DEPS ADIAK AXOM CALIPER CAMP CONDUIT CUDA FMT HDF5 HIP LUA MFEM MPI PETSC RAJA SLEPC STRUMPACK SUNDIALS TRIBOL UMPIRE)
foreach(dep ${TPL_DEPS})
    if( ${dep}_FOUND OR ENABLE_${dep} )
        set(SERAC_USE_${dep} TRUE)
    endif()
endforeach()


#--------------------------------------------------------------------------
# Add define we can use when debug builds are enabled
#--------------------------------------------------------------------------
set(SERAC_DEBUG FALSE)
if(CMAKE_BUILD_TYPE MATCHES "(Debug|RelWithDebInfo)")
    set(SERAC_DEBUG TRUE)

    # Controls various behaviors in Axom, like turning off/on SLIC debug and assert macros
    set(AXOM_DEBUG TRUE)
endif()


#------------------------------------------------------------------------------
# General Build Info
#------------------------------------------------------------------------------
serac_convert_to_native_escaped_file_path(${PROJECT_SOURCE_DIR} SERAC_REPO_DIR)
serac_convert_to_native_escaped_file_path(${CMAKE_BINARY_DIR}   SERAC_BINARY_DIR)

#------------------------------------------------------------------------------
# Create Config Header
#------------------------------------------------------------------------------
serac_configure_file(
    ${PROJECT_SOURCE_DIR}/src/serac/serac_config.hpp.in
    ${PROJECT_BINARY_DIR}/include/serac/serac_config.hpp
)

install(FILES ${PROJECT_BINARY_DIR}/include/serac/serac_config.hpp DESTINATION include/serac)

#------------------------------------------------------------------------------
# Generate serac-config.cmake for importing serac into other CMake packages
#------------------------------------------------------------------------------

# Set up some paths, preserve existing cache values (if present)
set(SERAC_INSTALL_INCLUDE_DIR "include" CACHE STRING "")
set(SERAC_INSTALL_CONFIG_DIR "lib" CACHE STRING "")
set(SERAC_INSTALL_LIB_DIR "lib" CACHE STRING "")
set(SERAC_INSTALL_BIN_DIR "bin" CACHE STRING "")
set(SERAC_INSTALL_CMAKE_MODULE_DIR "${SERAC_INSTALL_CONFIG_DIR}/cmake" CACHE STRING "")

set(SERAC_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE STRING "" FORCE)


include(CMakePackageConfigHelpers)

# Add version helper
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/serac-config-version.cmake
    VERSION ${SERAC_VERSION_FULL}
    COMPATIBILITY AnyNewerVersion
)

# Set up cmake package config file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/serac-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/serac-config.cmake
  INSTALL_DESTINATION
    ${SERAC_INSTALL_CONFIG_DIR}
  PATH_VARS
    SERAC_INSTALL_INCLUDE_DIR
    SERAC_INSTALL_LIB_DIR
    SERAC_INSTALL_BIN_DIR
    SERAC_INSTALL_CMAKE_MODULE_DIR
  )

# Install config files
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/serac-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/serac-config-version.cmake
  DESTINATION
    ${SERAC_INSTALL_CMAKE_MODULE_DIR}
)

# Install BLT files that recreate BLT targets in downstream projects
blt_install_tpl_setups(DESTINATION ${SERAC_INSTALL_CMAKE_MODULE_DIR})
