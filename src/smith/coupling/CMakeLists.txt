# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(coupling_headers
    external_interface.hpp
    )

set(coupling_sources
    external_interface.cpp
    )

set(coupling_depends
    smith_functional
    )

smith_add_library(
    NAME        smith_coupling
    HEADERS     ${coupling_headers}
    SOURCES     ${coupling_sources}
    DEPENDS_ON  ${coupling_depends}
    )

install(FILES ${coupling_headers} DESTINATION include/smith/coupling )

install(TARGETS              smith_coupling
        EXPORT               smith-targets
        DESTINATION          lib
        )
