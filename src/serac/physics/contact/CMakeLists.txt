# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(contact_headers
    contact_config.hpp
    contact_data.hpp
    contact_interaction.hpp
    )

set(contact_sources
    contact_data.cpp
    contact_interaction.cpp
    )

set(contact_depends serac_infrastructure)
blt_list_append(TO contact_depends ELEMENTS tribol IF TRIBOL_FOUND)
blt_list_append(TO contact_depends ELEMENTS redecomp IF TRIBOL_FOUND AND SERAC_ENABLE_MPI)

blt_add_library(
    NAME        serac_contact
    HEADERS     ${contact_headers}
    SOURCES     ${contact_sources}
    DEPENDS_ON  ${contact_depends}
    )

install(FILES ${contact_headers} DESTINATION include/serac/physics/contact )

install(TARGETS              serac_contact
        EXPORT               serac-targets
        DESTINATION          lib
        )
