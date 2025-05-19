# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(mesh_headers
    mesh_utils.hpp
    mesh_utils_base.hpp
    )

set(mesh_sources
    mesh_utils.cpp
    )

set(mesh_depends serac_infrastructure)

blt_add_library(
    NAME        serac_mesh
    HEADERS     ${mesh_headers}
    SOURCES     ${mesh_sources}
    DEPENDS_ON  ${mesh_depends}
    )

serac_write_unified_header(
    NAME mesh
    HEADERS ${mesh_headers}
    )

install(FILES ${mesh_headers} DESTINATION include/serac/mesh )

install(TARGETS              serac_mesh
        EXPORT               serac-targets
        DESTINATION          lib
        )

if(SERAC_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
