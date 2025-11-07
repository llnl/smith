# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

# Mesh Utilities

set(mesh_utils_headers
    mesh_utils.hpp
    mesh_utils_base.hpp
    )

set(mesh_utils_sources
    mesh_utils.cpp
    )

set(mesh_utils_depends smith_infrastructure)

smith_add_library(
    NAME        smith_mesh_utils
    HEADERS     ${mesh_utils_headers}
    SOURCES     ${mesh_utils_sources}
    DEPENDS_ON  ${mesh_utils_depends}
    )

smith_write_unified_header(
    NAME    mesh_utils
    HEADERS ${mesh_utils_headers}
    )

install(FILES ${mesh_utils_headers} DESTINATION include/smith/mesh_utils )

install(TARGETS              smith_mesh_utils
        EXPORT               smith-targets
        DESTINATION          lib
        )

if(SMITH_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
