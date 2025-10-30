# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer is ON (ENABLE_ASAN)")
    foreach(_flagvar CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS)
        string(APPEND ${_flagvar} " -fsanitize=address -fno-omit-frame-pointer")
    endforeach()
endif()

# Need to add symbols to dynamic symtab in order to be visible from stacktraces
string(APPEND CMAKE_EXE_LINKER_FLAGS " -rdynamic")

# Prevent unused -Xlinker arguments on Lassen Clang-10
if(DEFINED ENV{SYS_TYPE} AND "$ENV{SYS_TYPE}" STREQUAL "blueos_3_ppc64le_ib_p9")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wno-unused-command-line-argument")
endif()

# Enable warnings for things not covered by -Wall -Wextra
set(_extra_flags "-Wshadow -Wdouble-promotion -Wconversion -Wundef -Wnull-dereference -Wold-style-cast")
blt_append_custom_compiler_flag(FLAGS_VAR CMAKE_CXX_FLAGS DEFAULT ${_extra_flags})

# Only clang has fine-grained control over the designated initializer warnings
# This can be added to the GCC flags when C++20 is available
# This should be compatible with Clang 8 through Clang 12
blt_append_custom_compiler_flag(FLAGS_VAR CMAKE_CXX_FLAGS CLANG "-Wpedantic -Wno-c++2a-extensions -Wunused-private-field")
