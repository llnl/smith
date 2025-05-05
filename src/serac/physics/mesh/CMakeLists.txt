# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

# Serac MFEM Mesh Wrapper

set(mesh_headers
    mesh.hpp
    )

set(mesh_sources
    mesh.cpp
    )

set(mesh_depends serac_functional serac_state serac_mesh_utils)

blt_add_library(
    NAME        serac_mesh
    HEADERS     ${mesh_headers}
    SOURCES     ${mesh_sources}
    DEPENDS_ON  ${mesh_depends}
    )

install(FILES ${mesh_headers} DESTINATION include/serac/physics/mesh)

install(TARGETS              serac_mesh
        EXPORT               serac-targets
        DESTINATION          lib
        )
