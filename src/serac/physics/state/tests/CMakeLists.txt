# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(state_test_depends serac_mesh serac_state gtest)

set(state_test_sources
    test_finite_element_state.cpp)

serac_add_tests(SOURCES    ${state_test_sources}
                DEPENDS_ON ${state_test_depends})
