#!/bin/bash

##############################################################################
# Copyright (c) 2025, Lawrence Livermore National Security, LLC and
# RADIUSS project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

# This script lists the default and latest versions of tce packages, available
# through Lmod modules, and the version of the system packages and libraries.
# It is meant to help with maintaining an accurate description of external
# packages in Spack configurations.
# The list of packages corresponds to what we require on Toss4.
# TODO: Add ability to specify system (different package list per system).
# TODO: Add ability to update Spack packages.yaml directly through script.

###############################################################################
# MODULES

tce_packages=("cmake" "cuda" "rocm" "papi" "python")

get_module_versions() {
  local module_name="${1}"

  if [[ -z "${module_name}" ]]; then
      echo "Error: Module name is required"
      echo "Usage: get_module_versions <module_name>"
      return 1
  fi

  echo ""
  echo "${module_name}"
  echo "Default:"
  module --default show "${module_name}" 2>&1 | grep "prepend_path(\"PATH\"," | grep -o "/[^\"]*"
  echo "Latest:"
  module --latest show "${module_name}" 2>&1 | grep "prepend_path(\"*PATH\"," | grep -o "/[^\"]*"
}

while IFS= read -r module; do
  get_module_versions "${module}"
done < <(printf "%s\n" "${tce_packages[@]}" | sort)

###############################################################################
# SYSTEM PACKAGES

get_system_package_version() {
  echo ""
  dnf info "${1}" | grep -E "Name|Version"
}

system_lib_packages=("lapack" "libepoxy" "libunwind" "libX11" "libyogrt" "readline" "zlib")
system_bin_packages=("autoconf" "automake" "binutils" "bzip2" "curl" "diffutils" "elfutils" "findutils" "gettext" "ghostscript" "graphviz" "groff" "libtool" "lua" "make" "m4" "perl" "pdsh" "pkgconf" "tar" "unzip")

packages_union=("${system_lib_packages[@]}" "${system_bin_packages[@]}")

while IFS= read -r package; do
  get_system_package_version "${package}"
done < <(printf "%s\n" "${packages_union[@]}" | sort)
