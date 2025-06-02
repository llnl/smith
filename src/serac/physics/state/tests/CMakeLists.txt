# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(state_test_depends serac_state serac_physics_materials serac_mesh_utils gtest)

set(state_test_sources
    test_finite_element_dual.cpp
    test_finite_element_state.cpp
    test_state_manager.cpp)

serac_add_tests(SOURCES    ${state_test_sources}
                DEPENDS_ON ${state_test_depends})
