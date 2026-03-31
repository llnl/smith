################################################################################
# Copyright (c) 2022-23, Lawrence Livermore National Security, LLC and RADIUSS
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
################################################################################

if (SPHINX_FOUND)
  blt_add_sphinx_target(radiuss_spack_configs_docs)
endif ()

if ( NOT SPHINX_FOUND)
  message(WARNING "ENABLE_DOCS=On, but Sphinx or Doxygen not found. \
    Documentation won't be built.")
endif ()
