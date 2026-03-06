# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# Defines the following:
# ARPACK_FOUND
# ARPACK_INCLUDE_DIRS
# ARPACK_LIBRARIES

# Set ARPACK_DIR to specify the prefix of ARPACK installation

if (NOT EXISTS "${ARPACK_DIR}")
    message(FATAL_ERROR "Given ARPACK_DIR does not exist: ${ARPACK_DIR}")
endif()

if (NOT IS_DIRECTORY "${ARPACK_DIR}")
    message(FATAL_ERROR "Given ARPACK_DIR is not a directory: ${ARPACK_DIR}")
endif()

# Find include dirs
find_path(ARPACK_INCLUDE_DIRS
  NAMES arpackdef.h
  PATHS ${ARPACK_DIR}/include/arpack ${ARPACK_DIR}/include/arpack-ng
  NO_DEFAULT_PATH
  NO_CMAKE_ENVIRONMENT_PATH
  NO_CMAKE_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  NO_CMAKE_SYSTEM_PATH
)

# Find libraries
find_library(ARPACK_LIBRARIES
  NAMES parpack
  PATHS ${ARPACK_DIR}/lib
  NO_DEFAULT_PATH
  NO_CMAKE_ENVIRONMENT_PATH
  NO_CMAKE_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  NO_CMAKE_SYSTEM_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ARPACK DEFAULT_MSG ARPACK_LIBRARIES ARPACK_INCLUDE_DIRS)
