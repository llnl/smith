#! /bin/bash
##############################################################################
# Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
# RADIUSS project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

hostname

. ${MY_SPACK_PARENT_DIR}/spack/share/spack/setup-env.sh
export SPACK_DISABLE_LOCAL_CONFIG=""
export SPACK_USER_CACHE_PATH="${MY_SPACK_USER_CACHE}"
spack --version
spack ${MY_SPACK_DEBUG} env activate --without-view ${SPACK_CONCRETE_ENV_DIR}
spack ${MY_SPACK_DEBUG} config blame repos
spack ${MY_SPACK_DEBUG} repo update
spack ${MY_SPACK_DEBUG} config blame repos
spack ${MY_SPACK_DEBUG} config blame mirrors
spack ${MY_SPACK_DEBUG} mirror rm buildcache-destination
spack ${MY_SPACK_DEBUG} mirror add --oci-username-variable CI_REGISTRY_USER --oci-password-variable CI_REGISTRY_PASSWORD buildcache-destination oci://${CI_REGISTRY_IMAGE}/${SPACK_TARGET}
spack ${MY_SPACK_DEBUG} config blame mirrors
spack ${MY_SPACK_DEBUG} ci rebuild
echo Building ${SPACK_JOB_SPEC_PKG_NAME} /${SPACK_JOB_SPEC_DAG_HASH}
spack --color=always --backtrace install --include-build-deps --no-check-signature --use-buildcache=package:never,dependencies:only /${SPACK_JOB_SPEC_DAG_HASH}
echo Pushing ${SPACK_JOB_SPEC_PKG_NAME} /${SPACK_JOB_SPEC_DAG_HASH}
spack buildcache push buildcache-destination /${SPACK_JOB_SPEC_DAG_HASH}
