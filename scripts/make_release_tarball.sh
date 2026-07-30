#!/usr/bin/env bash

# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

##-----------------------------------------------------------------------------
## GLOBAL DEFINITIONS
##-----------------------------------------------------------------------------
bold=$(tput bold)
reset=$(tput sgr0)

TAR_CMD=`which tar`
TAR_VERSION=`$TAR_CMD --version |head -n 1`
VERSION=`git describe --tags`


##-----------------------------------------------------------------------------
## HELPER FUNCTIONS 
##-----------------------------------------------------------------------------
function show_help() {
  echo
  echo -e "$bold SYNOPSIS $reset"
  echo -e "\t Generates a release tarball."
  echo 
  echo -e "$bold Usage:$reset ./scripts/make_release_tarball.sh [options]"
  echo -e
  echo -e "$bold OPTIONS $reset"
  
  echo -e "\t$bold-h$reset, $bold--help$reset"
  echo -e "\t\t Displays this help information and exits."

  echo -e "\t$bold--with-data$reset"
  echo -e "\t\t Generate a tarball consisting of the data"
}

##-----------------------------------------------------------------------------
function info() {
  echo "$bold[INFO]:$reset $1"
}

##-----------------------------------------------------------------------------
function error() {
  echo "$bold[ERROR]:$reset $1"
  exit -1
}

##-----------------------------------------------------------------------------
## MAIN 
##-----------------------------------------------------------------------------

## parse arguments
while [ "$#" -gt 0 ]
do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
    esac
    shift
done

info "using tar command [$TAR_CMD]"
info "detected tar version: $TAR_VERSION"

if [[ $TAR_VERSION != *GNU* ]]; then
  error "This script requires GNU tar!"
fi

info "creating archive for version [$VERSION]" 
git archive --prefix=Smith-${VERSION}/ -o Smith-${VERSION}.tar HEAD 2> /dev/null

info "processing submodules..."

p=`pwd` && (echo .; git submodule foreach --recursive) | while read entering path; do
    temp="${path%\'}";
    temp="${temp#\'}";
    path=$temp;

    if [[ -n $path && "$path" != "data" ]]; then
      info "archiving [$path] submodule..."
      (cd $path && git archive --prefix=Smith-${VERSION}/$path/ HEAD > $p/tmp.tar && ${TAR_CMD} --concatenate --file=$p/Smith-${VERSION}.tar $p/tmp.tar && rm $p/tmp.tar);
    fi

done

gzip Smith-${VERSION}.tar

info "done."
