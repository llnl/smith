# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Setup MFEM
#
# This file defines:
#  MFEM_FOUND            - If MFEM was found
#  mfem                  - BLT Registered Library 
#  MFEM_BUILT_WITH_CMAKE - If MFEM was built with CMake
#------------------------------------------------------------------------------

if(NOT MFEM_DIR)
    message(FATAL_ERROR "MFEM support needs explicit MFEM_DIR")
endif()
message(STATUS "Looking for MFEM using MFEM_DIR = ${MFEM_DIR}")
smith_assert_is_directory(DIR_VARIABLE MFEM_DIR)

set(_MFEM_DIR ${MFEM_DIR}) # Save MFEM_DIR as a non-cache variable
find_package(MFEM CONFIG NO_DEFAULT_PATH PATHS "${MFEM_DIR}/lib/cmake/mfem")
# find_package will overwrite MFEM_DIR, so restore it here
set(MFEM_DIR ${_MFEM_DIR} CACHE PATH "" FORCE)

if(MFEM_FOUND)
    # MFEM was built with CMake so use that config file
    message(STATUS "Using MFEM's CMake config file")
    set(MFEM_BUILT_WITH_CMAKE TRUE)
    # It looks like include directories are not always built into the target
    target_include_directories(mfem INTERFACE ${MFEM_INCLUDE_DIRS})
else()
    set(MFEM_BUILT_WITH_CMAKE FALSE)
    find_path(
        MFEM_INCLUDE_DIRS mfem.hpp
        PATHS ${MFEM_DIR}/include
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH
    )

    find_library(
        MFEM_LIBRARIES NAMES mfem
        PATHS ${MFEM_DIR}/lib
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH )


    # when MFEM is built w/o cmake, we can get the details
    # of deps from its config.mk file
    find_path(
        MFEM_CFG_DIR config.mk
        PATHS ${MFEM_DIR}/share/mfem/
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH
    )

    if(NOT MFEM_CFG_DIR)
        message(FATAL_ERROR "Failed to find any MFEM build configuration files in ${MFEM_DIR}")
    else()
        message(STATUS "Using MFEM's GNU Make config file: ${MFEM_CFG_DIR}/config.mk")
    endif()

    # read config.mk file
    file(READ "${MFEM_CFG_DIR}/config.mk" mfem_cfg_file_txt)

    # parse include flags
    string(REGEX MATCHALL "MFEM_TPLFLAGS [^\n]+\n" mfem_tpl_inc_flags ${mfem_cfg_file_txt})
    if(${CMAKE_VERSION} VERSION_GREATER 3.15.0)
        message(VERBOSE "Content of variable mfem_tpl_inc_flags: ${mfem_tpl_inc_flags}")
    endif()
    string(REGEX REPLACE  "MFEM_TPLFLAGS +=" "" mfem_tpl_inc_flags ${mfem_tpl_inc_flags})
    string(FIND  "${mfem_tpl_inc_flags}" "\n" mfem_tpl_inc_flags_end_pos)
    string(SUBSTRING "${mfem_tpl_inc_flags}" 0 ${mfem_tpl_inc_flags_end_pos} mfem_tpl_inc_flags)
    string(STRIP "${mfem_tpl_inc_flags}" mfem_tpl_inc_flags)

    # remove the " -I" and add them to the include dir list
    separate_arguments(mfem_tpl_inc_flags)
    foreach(_include_flag ${mfem_tpl_inc_flags})
        string(FIND "${_include_flag}" "-I" _pos)
        if(_pos EQUAL 0)
            string(SUBSTRING "${_include_flag}" 2 -1 _include_dir)
            list(APPEND MFEM_INCLUDE_DIRS ${_include_dir})
        endif()
    endforeach()

    # parse link flags
    string(REGEX MATCHALL "MFEM_EXT_LIBS [^\n]+\n" mfem_tpl_lnk_flags "${mfem_cfg_file_txt}")
    if(${CMAKE_VERSION} VERSION_GREATER 3.15.0)
        message(VERBOSE "Content of variable mfem_tpl_lnk_flags: ${mfem_tpl_lnk_flags}")
    endif()
    if(NOT mfem_tpl_lnk_flags EQUAL "")
        string(REGEX REPLACE  "MFEM_EXT_LIBS +=" "" mfem_tpl_lnk_flags "${mfem_tpl_lnk_flags}")
        string(FIND  "${mfem_tpl_lnk_flags}" "\n" mfem_tpl_lnl_flags_end_pos )
        string(SUBSTRING "${mfem_tpl_lnk_flags}" 0 ${mfem_tpl_lnl_flags_end_pos} mfem_tpl_lnk_flags)
        string(STRIP "${mfem_tpl_lnk_flags}" mfem_tpl_lnk_flags)

        # filter out items containing "Xlinker"
        set(_mfem_tpl_list ${mfem_tpl_lnk_flags})
        separate_arguments(_mfem_tpl_list)
        list(FILTER _mfem_tpl_list EXCLUDE REGEX Xlinker)
        list(JOIN _mfem_tpl_list " " mfem_tpl_lnk_flags)
    else()
        message(WARNING "No third party library flags found in ${MFEM_CFG_DIR}/config.mk")
    endif()

    list(APPEND MFEM_LIBRARIES ${mfem_tpl_lnk_flags})

    if(mfem_cfg_file_txt MATCHES "MFEM_USE_CUDA += YES")
        if(NOT SMITH_ENABLE_CUDA)
            message(WARNING "MFEM was built with CUDA but CUDA is not enabled")
        endif()
        list(APPEND MFEM_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
        list(APPEND MFEM_LIBRARIES ${CMAKE_CUDA_LINK_FLAGS})
        list(APPEND MFEM_LIBRARIES ${CUDA_LIBRARIES})
        list(APPEND MFEM_LIBRARIES ${CUDA_CUBLAS_LIBRARIES})
        list(APPEND MFEM_LIBRARIES ${CUDA_cusolver_LIBRARY})
    endif()

    blt_import_library(
        NAME          mfem
        INCLUDES      ${MFEM_INCLUDE_DIRS}
        LIBRARIES     ${MFEM_LIBRARIES}
        TREAT_INCLUDES_AS_SYSTEM ON
        EXPORTABLE    ON)

    install(TARGETS          mfem
        EXPORT               smith-targets
        DESTINATION          lib
        )
endif()

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set MFEM_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(MFEM DEFAULT_MSG
                                  MFEM_LIBRARIES
                                  MFEM_INCLUDE_DIRS )

if(NOT MFEM_FOUND)
    message(FATAL_ERROR "MFEM_DIR is not a path to a valid MFEM install")
endif()

message(STATUS "MFEM Includes: ${MFEM_INCLUDE_DIRS}")
message(STATUS "MFEM Libraries: ${MFEM_LIBRARIES}")
