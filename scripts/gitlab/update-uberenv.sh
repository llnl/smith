#!/bin/bash
#
# Uberenv is used as a submodule. This script updates uberenv to a given ref.
# This is used in CI context so that we can test that an update of Uberenv
# has no side-effect.


if [[ ! ${1} ]]
then
    echo "ERROR: expecting reference for uberenv repo" >&2
else
    uberenv_ref="${1}"
fi

git submodule update --init

cd scripts/uberenv

git fetch origin
git checkout -b testing/uberenv $uberenv_ref
