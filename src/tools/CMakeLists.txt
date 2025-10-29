# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

serac_add_executable(NAME       partitioner
                     SOURCES    partitioner.cpp
                     DEPENDS_ON serac_mesh_utils)
