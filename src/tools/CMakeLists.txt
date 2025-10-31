# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

smith_add_executable(NAME       partitioner
                     SOURCES    partitioner.cpp
                     DEPENDS_ON smith_mesh_utils)
