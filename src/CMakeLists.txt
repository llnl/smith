# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(smith)
add_subdirectory(drivers)
add_subdirectory(tools)

if (ENABLE_DOCS)
    add_subdirectory(docs)
endif()

if(SMITH_ENABLE_TESTS)
    add_subdirectory(tests)
endif()

# Create single unified header of component-level unified headers
smith_write_unified_header(
    NAME    smith
    HEADERS ${_smith_unified_headers}
    NO_PATH_MODIFICATION ON
    )
