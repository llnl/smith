# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_custom_target(run_examples
                  COMMAND ctest -C Example -L Example -VV)

configure_file(
    using-with-cmake/host-config.cmake.in
    ${PROJECT_BINARY_DIR}/examples/using-with-cmake/host-config.cmake)

install(
    FILES
        using-with-cmake/CMakeLists.txt
        using-with-cmake/smith_example.cpp
        ${PROJECT_BINARY_DIR}/examples/using-with-cmake/host-config.cmake
    DESTINATION
        examples/smith/using-with-cmake
)

# Add the conduction examples
add_subdirectory(conduction)

# Add the contact examples
add_subdirectory(contact)

# Add the contact-homotopy examples
add_subdirectory(contact/homotopy)

# Add the buckling examples
add_subdirectory(buckling)

# Add the explicit dynamics examples
add_subdirectory(explicit)

# Add the uniaxial examples
add_subdirectory(uniaxial)

# Add the inertia relief example
add_subdirectory(inertia_relief)
