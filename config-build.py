#!/bin/sh

# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"exec" "python3" "-u" "-B" "$0" "$@"

# Python wrapper script for generating the correct cmake line with the options specified by the user.
#
# Please keep parser option names as close to possible as the names of the cmake options they are wrapping.

import sys
import os
import subprocess
import argparse
import platform
import shutil
import socket

_host_configs_map = {"rzgenie"   : "rzwhippet-toss_4_x86_64_ib-llvm@19.1.3.cmake",
                     "rzwhippet" : "rzwhippet-toss_4_x86_64_ib-llvm@19.1.3.cmake",
                     "rzvernal"  : "rzvernal-toss_4_x86_64_ib_cray-rocmcc@6.2.1_hip.cmake",
                     "rzadams"   : "rzadams-toss_4_x86_64_ib_cray-llvm-amdgpu@6.4.2_hip.cmake",
                     "dane"      : "dane-toss_4_x86_64_ib-llvm@19.1.3.cmake",
                     "tioga"     : "tuolumne-toss_4_x86_64_ib_cray-llvm-amdgpu@6.4.2_hip.cmake",
                     "tuolumne"  : "tuolumne-toss_4_x86_64_ib_cray-llvm-amdgpu@6.4.2_hip.cmake"}

def get_machine_name():
    return socket.gethostname().rstrip('1234567890')

def get_default_host_config():
    machine_name = get_machine_name()

    if machine_name in _host_configs_map.keys():
        return _host_configs_map[machine_name]
    else:
        return ""

def extract_cmake_location(file_path):
    # print "Extracting cmake entry from host config file ", file_path
    if os.path.exists(file_path):
        cmake_line_prefix = "# cmake executable path: "
        file_handle = open(file_path, "r")
        content = file_handle.readlines()
        for line in content:
            if line.lower().startswith(cmake_line_prefix):
                return line.split(" ")[4].strip()
        print("Could not find a cmake entry in host config file.")
    return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure cmake build.",
                                     epilog="Note: Additional or unrecognized parameters will be passed directly to cmake."
                                            " For example, append '-DENABLE_OPENMP=ON' to enable OpenMP."
                                            )

    parser.add_argument("-bp",
                        "--buildpath",
                        type=str,
                        default="",
                        help="specify path for build directory.  If not specified, will create in current directory.")

    parser.add_argument("-ip",
                        "--installpath",
                        type=str,
                        default="",
                        help="specify path for install directory. If not specified, will create in current directory.")

    parser.add_argument("-bt",
                        "--buildtype",
                        type=str,
                        choices=["Release", "Debug", "RelWithDebInfo", "MinSizeRel"],
                        help="build type. defaults to Release")

    parser.add_argument("-e",
                        "--eclipse",
                        action='store_true',
                        help="create an eclipse project file.")

    parser.add_argument("-ecc",
                        "--exportcompilercommands",
                        action='store_true',
                        help="generate a compilation database.  Can be used by the clang tools such as clang-modernize.  Will create a 'compile_commands.json' file in build directory.")

    parser.add_argument("-hc",
                        "--hostconfig",
                        default="",
                        type=str,
                        help="select a specific host-config file to initalize CMake's cache")

    parser.add_argument("--print-default-host-config",
                        action='store_true',
                        help="print the default host config for this system and exit")

    parser.add_argument("--print-machine-name",
                        action='store_true',
                        help="print the machine name for this system and exit")

    parser.add_argument("-n", 
                        "--ninja",
                        action='store_true',
                        help="use ninja generator to build Smith instead of make")

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("[config-build]: Passing the following arguments directly to cmake... %s" % unknown_args)

    return args, unknown_args


########################
# Find CMake Cache File
########################
def find_host_config(args, repodir):
    if args.hostconfig != "":
        hostconfigpath = os.path.abspath(args.hostconfig)
    else:
        hostconfigpath = get_default_host_config()
        if hostconfigpath == "":
            print("[config-build]: Error could not find default host-config for this platform.")
            print("   Either set one in this script or use the command line argument '-hc'.")
        else:
           hostconfigpath = os.path.join(repodir, "host-configs", hostconfigpath)
    assert os.path.exists( hostconfigpath ), "Could not find CMake host config file '%s'." % hostconfigpath
    print("Using host config file: '%s'." % hostconfigpath)
    return hostconfigpath


########################
# Get Platform information from host config name
########################
def get_platform_info(hostconfigpath):
    platform_info = ""
    platform_info = os.path.split(hostconfigpath)[1]
    if platform_info.endswith(".cmake"):
        platform_info = platform_info[:-6]
    return platform_info
        

#####################
# Setup Build Dir
#####################
def setup_build_dir(args, platform_info):
    if args.buildpath != "":
        # use explicit build path
        buildpath = args.buildpath
    else:
        # use platform info & build type
        buildpath = "-".join(["build", platform_info, args.buildtype.lower()])

    buildpath = os.path.abspath(buildpath)

    if os.path.exists(buildpath):
        print("Build directory '%s' already exists.  Deleting..." % buildpath)
        shutil.rmtree(buildpath)

    print("Creating build directory '%s'..." % buildpath)
    os.makedirs(buildpath)
    return buildpath


#####################
# Setup Install Dir
#####################
def setup_install_dir(args, platform_info):
    # For install directory, we will clean up old ones, but we don't need to create it, cmake will do that.
    if args.installpath != "":
        installpath = os.path.abspath(args.installpath)
    else:
        # use platform info & build type
        installpath = "-".join(
            ["install", platform_info, args.buildtype.lower()]
        )

    installpath = os.path.abspath(installpath)

    if os.path.exists(installpath):
        print(
            "Install directory '%s' already exists, deleting..." % installpath
        )
        shutil.rmtree(installpath)

    print("Creating install path '%s'..." % installpath)
    os.makedirs(installpath)
    return installpath


############################
# Check if executable exists 
############################
def executable_exists(path):
    if path == "cmake":
        return True
    return os.path.isfile(path) and os.access(path, os.X_OK)


############################
# Build CMake command line
############################
def create_cmake_command_line(args, unknown_args, buildpath, installpath, hostconfigpath):

    import stat

    cmakeline = extract_cmake_location(hostconfigpath)
    assert cmakeline, "Host config file doesn't contain valid cmake location, value was %s" % cmakeline
    assert executable_exists( cmakeline ), "['%s'] invalid path to cmake executable or file does not have execute permissions" % cmakeline

    # create the ccmake command for convenience
    cmakedir   = os.path.dirname(cmakeline)
    ccmake_cmd = cmakedir + "/ccmake"
    if executable_exists( ccmake_cmd ):
        # write the ccmake command to a file to use for convenience
        with open( "%s/ccmake_cmd" % buildpath, "w" ) as ccmakefile:
            ccmakefile.write("#!/usr/bin/env bash\n")
            ccmakefile.write(ccmake_cmd)
            ccmakefile.write(" $@")
            ccmakefile.write("\n")

        st = os.stat("%s/ccmake_cmd" % buildpath)
        os.chmod("%s/ccmake_cmd" % buildpath, st.st_mode | stat.S_IEXEC)

    # Add cache file option
    cmakeline += " -C %s" % hostconfigpath

    # Add build type (opt or debug)
    cmakeline += " -DCMAKE_BUILD_TYPE=" + args.buildtype

    # Set install dir
    cmakeline += " -DCMAKE_INSTALL_PREFIX=%s" % installpath

    if args.exportcompilercommands:
        cmakeline += " -DCMAKE_EXPORT_COMPILE_COMMANDS=on"

    if args.eclipse:
        cmakeline += ' -G "Eclipse CDT4 - Unix Makefiles"'

    if args.ninja:
        cmakeline += ' -G Ninja'

    if unknown_args:
        cmakeline += " " + " ".join( unknown_args )

    rootdir = os.path.dirname( os.path.abspath(sys.argv[0]) )
    cmakeline += " %s " % rootdir

    # Dump the cmake command to file for convenience
    with open("%s/cmake_cmd" % buildpath, "w") as cmdfile:
       cmdfile.write(cmakeline)
       cmdfile.write("\n")

    st = os.stat("%s/cmake_cmd" % buildpath)
    os.chmod("%s/cmake_cmd" % buildpath, st.st_mode | stat.S_IEXEC)
    return cmakeline


############################
# Run CMake
############################
def run_cmake(buildpath, cmakeline):
    print("Changing to build directory...")
    os.chdir(buildpath)
    print("Executing CMake line: '%s'" % cmakeline)
    print()
    returncode = subprocess.call(cmakeline, shell=True)
    if not returncode == 0:
        print("Error: CMake command failed with return code: {0}".format(returncode))
        return False
    return True


############################
# Main
############################
def main():
    repodir = os.path.abspath(os.path.dirname(__file__))
    assert os.path.abspath(os.getcwd())==repodir, "config-build must be run from %s" % repodir

    args, unknown_args = parse_arguments()

    if args.print_machine_name:
       machine_name = get_machine_name()
       print(machine_name)
       return True

    if args.print_default_host_config:
       default_hc = get_default_host_config()
       if default_hc != "":
          print(os.path.splitext(default_hc)[0])
          return True
       else:
          return False

    if args.buildtype == None:
        # Set default CMake build type
        args.buildtype = "Release"
        # If CMAKE_BUILD_TYPE was passed in as an argument, use that option instead
        for unknown_arg in unknown_args:
            if "-DCMAKE_BUILD_TYPE" in unknown_arg:
                args.buildtype = unknown_arg.split("=")[1]
                break

    basehostconfigpath = find_host_config(args, repodir)
    platform_info = get_platform_info(basehostconfigpath)
    buildpath = setup_build_dir(args, platform_info)
    installpath = setup_install_dir(args, platform_info)

    cmakeline = create_cmake_command_line(args, unknown_args, buildpath, installpath, basehostconfigpath)
    return run_cmake(buildpath, cmakeline)

if __name__ == '__main__':
    exit(0 if main() else 1)
