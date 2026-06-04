#!/bin/bash
# Redirect Jacamar CI from your HOME directory to your LC workspace and periodically cleanup CI jobs to avoid exceeding
# disk quota
#
# Some things to note:
# - Running this script on oslic is recommended, because its best suited for file IO
# - It's important to understand that Cron jobs are installed per node (not per machine!), so do not run this script
#   more than once unless you know what you're doing.
# - For more info on Cron, see https://myconfluence.llnl.gov/spaces/RAM/pages/417729120/Cron+-+how+it+can+be+useful+for+triggering+work+at+a+given+time

set -euo pipefail

USER_NAME="$(whoami)"
HOST_NAME="$(hostname)"
WORKSPACE_CI_DIR="/usr/workspace/${USER_NAME}/.jacamar-ci"
HOME_CI_LINK="${HOME}/.jacamar-ci"
CRON_DIR="${HOME}/crontabs"
CRON_SCRIPT_PATH="${CRON_DIR}/${HOST_NAME}Cron"

echo "Starting setup for user: ${USER_NAME}"
echo "Host: ${HOST_NAME}"

# Error checking. Guard against running this script more than once.
if [ -e "${WORKSPACE_CI_DIR}" ]; then
  echo "Workspace CI directory already exists at: ${WORKSPACE_CI_DIR}"
  exit 1
# fi
if [ -e "${CRON_DIR}" ]; then
  echo "Cron directory already exists at: ${CRON_DIR}."
  echo "If you want to regenerate your Cron job, first log into the specific node that has the job, then delete"
  echo "it with 'crontab -r'. After that you can safely remove the crontab directory in your HOME directory."
  exit 1
fi

# Setup symlink
echo "Creating symlink: ${HOME_CI_LINK} -> ${WORKSPACE_CI_DIR}"
rm -rf "${HOME_CI_LINK}"
mkdir "${WORKSPACE_CI_DIR}"
ln -s "${WORKSPACE_CI_DIR}" "${HOME_CI_LINK}"

# Setup Cron job
cron_script_contents() {
  cat <<EOF
# For more information about using Cron on LC:
# https://myconfluence.llnl.gov/spaces/RAM/pages/417729120/Cron+-+how+it+can+be+useful+for+triggering+work+at+a+given+time

# Allows you to update your Cron job without having to go to the specific node
0,20,40 * * * * crontab ${CRON_SCRIPT_PATH}

# Remove contents of Jacamar CI
0 0 * * * srun -N1 rm -rf ${WORKSPACE_CI_DIR}/*
EOF
}

echo "Writing Cron file to: ${CRON_SCRIPT_PATH}"
mkdir "${CRON_DIR}"
cron_script_contents > "${CRON_SCRIPT_PATH}"
crontab "${CRON_SCRIPT_PATH}"
