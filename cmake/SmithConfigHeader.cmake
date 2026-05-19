# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Create variable for every TPL
#------------------------------------------------------------------------------
set(TPL_DEPS ADIAK AXOM CALIPER CAMP CONDUIT CONTINUATION CUDA ENZYME GRETL HDF5 HIP LUA MFEM MPI PETSC RAJA SLEPC STRUMPACK SUNDIALS TRIBOL UMPIRE)
foreach(dep ${TPL_DEPS})
    if( ${dep}_FOUND OR ENABLE_${dep} )
        set(SMITH_USE_${dep} TRUE)
    endif()
endforeach()


#--------------------------------------------------------------------------
# Add define we can use when debug builds are enabled
#--------------------------------------------------------------------------
set(SMITH_DEBUG FALSE)
if(CMAKE_BUILD_TYPE MATCHES "(Debug|RelWithDebInfo)")
    set(SMITH_DEBUG TRUE)

    # Controls various behaviors in Axom, like turning off/on SLIC debug and assert macros
    set(AXOM_DEBUG TRUE)
endif()


#------------------------------------------------------------------------------
# General Build Info
#------------------------------------------------------------------------------
smith_convert_to_native_escaped_file_path(${PROJECT_SOURCE_DIR} SMITH_REPO_DIR)
smith_convert_to_native_escaped_file_path(${CMAKE_BINARY_DIR}   SMITH_BINARY_DIR)

#------------------------------------------------------------------------------
# Create Config Header
#------------------------------------------------------------------------------
smith_configure_file(
    ${PROJECT_SOURCE_DIR}/src/smith/smith_config.hpp.in
    ${PROJECT_BINARY_DIR}/include/smith/smith_config.hpp
)

install(FILES ${PROJECT_BINARY_DIR}/include/smith/smith_config.hpp DESTINATION include/smith)

#------------------------------------------------------------------------------
# Generate smith-config.cmake for importing Smith into other CMake packages
#------------------------------------------------------------------------------

# Set up some paths, preserve existing cache values (if present)
set(SMITH_INSTALL_INCLUDE_DIR "include" CACHE STRING "")
set(SMITH_INSTALL_CONFIG_DIR "lib" CACHE STRING "")
set(SMITH_INSTALL_LIB_DIR "lib" CACHE STRING "")
set(SMITH_INSTALL_BIN_DIR "bin" CACHE STRING "")
set(SMITH_INSTALL_CMAKE_MODULE_DIR "${SMITH_INSTALL_CONFIG_DIR}/cmake" CACHE STRING "")

set(SMITH_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE STRING "" FORCE)


include(CMakePackageConfigHelpers)

# Add version helper
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/smith-config-version.cmake
    VERSION ${SMITH_VERSION_FULL}
    COMPATIBILITY AnyNewerVersion
)

# Set up cmake package config file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/smith-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/smith-config.cmake
  INSTALL_DESTINATION
    ${SMITH_INSTALL_CONFIG_DIR}
  PATH_VARS
    SMITH_INSTALL_INCLUDE_DIR
    SMITH_INSTALL_LIB_DIR
    SMITH_INSTALL_BIN_DIR
    SMITH_INSTALL_CMAKE_MODULE_DIR
  )

# Install config files
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/smith-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/smith-config-version.cmake
  DESTINATION
    ${SMITH_INSTALL_CMAKE_MODULE_DIR}
)

# Install BLT files that recreate BLT targets in downstream projects
blt_install_tpl_setups(DESTINATION ${SMITH_INSTALL_CMAKE_MODULE_DIR})
