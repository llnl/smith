# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if (SPHINX_FOUND)
  blt_add_sphinx_target(smith_docs)
endif()

if (DOXYGEN_FOUND)
  add_subdirectory(doxygen)
endif()

