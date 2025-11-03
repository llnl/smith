# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
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

set(contact_depends smith_infrastructure smith_state)
blt_list_append(TO contact_depends ELEMENTS tribol IF TRIBOL_FOUND)
if (TRIBOL_FOUND AND SMITH_ENABLE_MPI)
    list(APPEND contact_depends redecomp)
endif()

smith_add_library(
    NAME        smith_contact
    HEADERS     ${contact_headers}
    SOURCES     ${contact_sources}
    DEPENDS_ON  ${contact_depends}
    )

# Unified header is not needed in this component

install(FILES ${contact_headers} DESTINATION include/smith/physics/contact )

install(TARGETS              smith_contact
        EXPORT               smith-targets
        DESTINATION          lib
        )
