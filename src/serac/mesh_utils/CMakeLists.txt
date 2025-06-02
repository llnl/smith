# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
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

set(mesh_utils_depends serac_infrastructure)

blt_add_library(
    NAME        serac_mesh_utils
    HEADERS     ${mesh_utils_headers}
    SOURCES     ${mesh_utils_sources}
    DEPENDS_ON  ${mesh_utils_depends}
    )

serac_write_unified_header(
    NAME    mesh_utils
    HEADERS ${mesh_utils_headers}
    )

install(FILES ${mesh_utils_headers} DESTINATION include/serac/mesh_utils )

install(TARGETS              serac_mesh_utils
        EXPORT               serac-targets
        DESTINATION          lib
        )

if(SERAC_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
