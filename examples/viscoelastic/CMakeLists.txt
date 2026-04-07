# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

smith_add_executable( NAME        viscoelastic_buckling
                      SOURCES     viscoelastic_buckling.cpp
                      OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                      DEPENDS_ON  smith
                      )

if(SMITH_ENABLE_HIP)
    blt_add_target_compile_flags(TO viscoelastic_buckling FLAGS "-fgpu-rdc")
    blt_add_target_link_flags(TO viscoelastic_buckling FLAGS "-fgpu-rdc")
endif()

install(
    FILES
        viscoelastic_buckling.cpp
    DESTINATION
        examples/smith/viscoelastic
)
