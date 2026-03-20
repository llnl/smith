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

# Apple ld warns about duplicate -l flags when the same library is reachable
# via multiple dependency paths (common with Spack-built CMake targets that use
# raw -l strings instead of imported targets). Suppress the spurious warning.
if(APPLE)
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,-no_warn_duplicate_libraries")
endif()

# Prevent unused -Xlinker arguments on Lassen Clang-10
if(DEFINED ENV{SYS_TYPE} AND "$ENV{SYS_TYPE}" STREQUAL "blueos_3_ppc64le_ib_p9")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wno-unused-command-line-argument")
endif()

# Enable warnings for things not covered by -Wall -Wextra
set(_extra_flags "-Wshadow -Wdouble-promotion -Wconversion -Wundef -Wnull-dereference -Wold-style-cast -Wpedantic -Wunused-private-field")
blt_append_custom_compiler_flag(FLAGS_VAR CMAKE_CXX_FLAGS DEFAULT ${_extra_flags})

