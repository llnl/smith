# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(containers)

set(refactor_headers

    elements/h1_edge.hpp
    elements/h1_hexahedron.hpp
    elements/h1_quadrilateral.hpp
    elements/h1_tetrahedron.hpp
    elements/h1_triangle.hpp
    elements/h1_vertex.hpp

    elements/hcurl_edge.hpp
    elements/hcurl_hexahedron.hpp
    elements/hcurl_quadrilateral.hpp
    elements/hcurl_tetrahedron.hpp
    elements/hcurl_triangle.hpp
    elements/hcurl_vertex.hpp

    common.hpp
    evaluate.hpp
    integrate.hpp

)

set(refactor_sources
    common.cpp
    evaluate.cpp
    integrate_diag.cpp
    integrate_spmat.cpp
    integrate_residual.cpp
)

set(refactor_depends serac_infrastructure serac_functional serac_refactor_containers)

blt_add_library(
    NAME        serac_refactor
    HEADERS     ${refactor_headers}
    SOURCES     ${refactor_sources}
    DEPENDS_ON  ${refactor_depends}
    )

install(FILES ${refactor_headers} DESTINATION include/serac/refactor )

install(TARGETS              serac_refactor
        EXPORT               serac-targets
        DESTINATION          lib
        )

if(SERAC_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
