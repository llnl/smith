#!/bin/bash

# Build and test Smith on team's shared MacMini, then report results to a set of emails

# Immediately fail upon error and update environment
set -e
source ~/.bash_profile

# Variables
CI_ROOT_DIR="/Users/chapman39/dev/.ci/smith"
PROJECT_DIR="$CI_ROOT_DIR/repo"
OUTPUT_LOG="$CI_ROOT_DIR/logs/macmini-build-and-test-$(date +"%Y_%m_%d_%H_%M_%S").log"
HOST_CONFIG="$PROJECT_DIR/host-configs/other/firion-*.cmake"
RECIPIENTS="chapman39@llnl.gov,white238@llnl.gov,talamini1@llnl.gov"
EMAIL_SUBJECT="Smith Failed! MacMini build and test report $(date)"
EMAIL_BODY="This is automatic weekly report of Smith's MacMini build. See attached for log."

function send_email() {
    echo "$EMAIL_BODY" | print_run_log mutt -a "$OUTPUT_LOG" -s "$EMAIL_SUBJECT" -- "$RECIPIENTS"
}

# Send email before exiting from an error
trap "send_email; exit 1" ERR

# Print command and its output into a log file
print_run_log(){
    echo "####################" >> "$OUTPUT_LOG"
    echo "# $@" >> "$OUTPUT_LOG"
    echo "####################" >> "$OUTPUT_LOG"
    "$@" >> "$OUTPUT_LOG" 2>&1
    echo >> "$OUTPUT_LOG"
}

# Go to project directory
print_run_log cd $PROJECT_DIR

# Update repo
print_run_log git checkout develop
print_run_log git pull
print_run_log git submodule update --init --recursive

# Clear previous build(s)
print_run_log rm -rf _smith_build_and_test*

# Build and test project
print_run_log python3 ./scripts/llnl/build_src.py --host-config $HOST_CONFIG -v -j16

# Update email subject to indicate success and send
EMAIL_SUBJECT="Smith Succeeded! MacMini build and test report $(date)"
send_email
