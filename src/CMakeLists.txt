# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(serac)
add_subdirectory(drivers)
add_subdirectory(tools)

if (ENABLE_DOCS)
    add_subdirectory(docs)
endif()

if(SERAC_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
