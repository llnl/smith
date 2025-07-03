# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_custom_target(run_examples
                  COMMAND ctest -C Example -L Example -VV)

configure_file(
    using-with-cmake/host-config.cmake.in
    ${CMAKE_BINARY_DIR}/examples/using-with-cmake/host-config.cmake)

install(
    FILES
        using-with-cmake/CMakeLists.txt
        using-with-cmake/serac_example.cpp
        ${CMAKE_BINARY_DIR}/examples/using-with-cmake/host-config.cmake
    DESTINATION
        examples/serac/using-with-cmake
)

# Add the conduction examples
add_subdirectory(conduction)

# Add the contact examples
add_subdirectory(contact)

# Add the buckling examples
add_subdirectory(buckling)

# Add the uniaxial examples
add_subdirectory(uniaxial)

add_subdirectory(korner_examples)
