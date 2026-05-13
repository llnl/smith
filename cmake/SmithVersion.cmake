# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Version information that go into the generated config header
#------------------------------------------------------------------------------
set(SMITH_VERSION_MAJOR 0)
set(SMITH_VERSION_MINOR 1)
set(SMITH_VERSION_PATCH 0)
string(CONCAT SMITH_VERSION_FULL
    "v${SMITH_VERSION_MAJOR}"
    ".${SMITH_VERSION_MINOR}"
    ".${SMITH_VERSION_PATCH}" )

if (Git_FOUND)
  ## check to see if we are building from a Git repo or an exported tarball
  blt_is_git_repo( OUTPUT_STATE is_git_repo )

  if(${is_git_repo})
    blt_git_hashcode(HASHCODE sha1 RETURN_CODE rc)
    if(NOT ${rc} EQUAL 0)
      message(FATAL_ERROR "blt_git_hashcode failed!")
    endif()

    set(SMITH_GIT_SHA ${sha1})
  endif()

endif()

message(STATUS "Configuring Smith version ${SMITH_VERSION_FULL}")
