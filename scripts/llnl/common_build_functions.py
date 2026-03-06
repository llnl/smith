# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
 file: common_build_functions.py

 description: 
  helpers for installing src and tpls on llnl lc systems.

"""

import os
import socket
import sys
import subprocess
import datetime
import glob
import json
import getpass
import shutil
import time
import grp
import pwd

from os.path import join as pjoin


def shell_exec(cmd,
               echo = False,
               return_output = False,
               print_output = False,
               output_file = None,
               error_prefix = "ERROR:"):
    """
    Helper for executing shell commands.

    args:
        echo (bool): Whether to print command to screen
        return_output (bool): Whether to return output in addition to return code
        print_output (bool): Whether to print output to screen
        output_file (path): Path to write output to
        error_prefix (string): Prefix message when non-zero return code

    return:
        return code and optionally command output
    """
    if echo:
        print("[exe: %s]" % cmd)

    p = subprocess.Popen(cmd,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

    if return_output:
        full_output = ""

    if output_file != None:
        output_file_stream = open(output_file, "w")

    for line in p.stdout:
        if isinstance(line, bytes):
            line = line.decode()
        if return_output:
            full_output = full_output + line
        if print_output:
            sys.stdout.write(line)
        if output_file != None:
            output_file_stream.write(line)

    p.wait()

    if p.returncode != 0:
        print("[{0} [return code: {1}] from command: {2}]".format(error_prefix, p.returncode, cmd))

    if return_output:
        return p.returncode, full_output
    else:
        return p.returncode


def get_timestamp(t=None,sep="_"):
    """ Creates a timestamp that can easily be included in a filename. """
    if t is None:
        t = datetime.datetime.now()
    sargs = (t.year,t.month,t.day,t.hour,t.minute,t.second)
    sbase = "".join(["%04d",sep,"%02d",sep,"%02d",sep,"%02d",sep,"%02d",sep,"%02d"])
    return  sbase % sargs


def build_info():
    res = {}
    res["built_by"] = os.environ["USER"]
    res["built_from_branch"] = "unknown"
    res["built_from_sha1"]   = "unknown"
    res["platform"] = get_platform()
    rc, out = shell_exec('git branch -a | grep \"*\"',return_output=True,error_prefix="WARNING:")
    out = out.strip()
    if rc == 0 and out != "":
        res["built_from_branch"]  = out.split()[1]
    rc,out = shell_exec('git rev-parse --verify HEAD',return_output=True,error_prefix="WARNING:")
    out = out.strip()
    if rc == 0 and out != "":
        res["built_from_sha1"] = out
    return res


def write_build_info(ofile):
    print("[build info]")
    binfo_str = json.dumps(build_info(),indent=2)
    print(binfo_str)
    open(ofile,"w").write(binfo_str)


def log_success(prefix, msg, timestamp=""):
    """
    Called at the end of the process to signal success.
    """
    print(msg)
    info = {}
    info["prefix"] = prefix
    info["platform"] = get_platform()
    info["status"] = "success"
    info["message"] = msg
    if timestamp == "":
        info["timestamp"] = get_timestamp()
    else:
        info["timestamp"] = timestamp
    json.dump(info,open(pjoin(prefix,"success.json"),"w"),indent=2)


def log_failure(prefix, msg, timestamp=""):
    """
    Called when the process failed.
    """
    print(msg)
    info = {}
    info["prefix"] = prefix
    info["platform"] = get_platform()
    info["status"] = "failed"
    info["message"] = msg
    if timestamp == "":
        info["timestamp"] = get_timestamp()
    else:
        info["timestamp"] = timestamp
    json.dump(info,open(pjoin(prefix,"failed.json"),"w"),indent=2)


def assertUberenvExists():
    if not os.path.exists(get_uberenv_path()):
        print("[ERROR: {0} does not exist".format(get_uberenv_path()))
        print("  run 'git submodule update --init'")
        print("]")
        sys.exit(1)


def uberenv_create_mirror(prefix, spec, project_file, mirror_path, report_to_stdout = False):
    """
    Calls uberenv to create a spack mirror.
    """
    assertUberenvExists()
    cmd  = "{0} {1} --create-mirror -k ".format(sys.executable, get_uberenv_path())
    cmd += "--prefix=\"{0}\" --mirror=\"{1}\" ".format(prefix, mirror_path)
    cmd += "--spec=\"{0}\" ".format(spec)
    if project_file:
        cmd += "--project-json=\"{0}\" ".format(project_file)

    print("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]")
    print("[ It is expected for 'spack --create-mirror' to throw warnings.                ]")
    print("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]")
    res = shell_exec(cmd, echo=True, print_output=report_to_stdout, error_prefix="WARNING:")
    print("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]")
    print("[ End of expected warnings from 'spack --create-mirror'                        ]")
    print("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]")
    set_group_and_perms(mirror_path)
    return res


def uberenv_build(prefix, spec, project_file, mirror_path, report_to_stdout = False):
    """
    Calls uberenv to install tpls for a given spec to given prefix.
    """
    assertUberenvExists()
    cmd  = "{0} {1} -k ".format(sys.executable, get_uberenv_path())
    cmd += "--prefix=\"{0}\" --spec=\"{1}\" ".format(prefix, spec)
    cmd += "--mirror=\"{0}\" ".format(mirror_path)
    if project_file:
        cmd += "--project-json=\"{0}\" ".format(project_file)

    spack_tpl_build_log = pjoin(prefix,"output.log.spack.tpl.build.%s.txt" % spec.replace(" ", "_"))
    print("[starting tpl install of spec %s]" % spec)
    print("[log file: %s]" % spack_tpl_build_log)
    res = shell_exec(cmd,
                     echo=True,
                     print_output=report_to_stdout,
                     output_file = spack_tpl_build_log)

    # Move files generated by spack in source directory to TPL install directory
    print("[Moving spack generated files to TPL build directory]")
    repo_dir = get_repo_dir()
    for file in ["spack-build-env.txt", "spack-build-out.txt", "spack-configure-args.txt"]:
        src = pjoin(repo_dir, file)
        dst = pjoin(prefix, "{0}-{1}".format(spec.replace(" ", "_"),file))
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)

    if res != 0:
        log_failure(prefix,"[ERROR: uberenv/spack build of spec: %s failed]" % spec)
    return res


def test_examples(host_config, build_dir, install_dir, report_to_stdout = False, job_count = ""):
    print("[starting to build examples]")

    # Install
    log_file =  pjoin(build_dir,"output.log.make.install.txt")
    print("[log file: %s]" % log_file)
    res = shell_exec(f"cd {build_dir} && make VERBOSE=1 install -j {job_count}",
                     output_file = log_file,
                     print_output = report_to_stdout,
                     echo=True)

    if res != 0:
        print("[ERROR: error code={0}: Install for host-config: {1} failed]\n".format(res, host_config))
        return res

    # Configure examples
    log_file =  pjoin(build_dir,"output.log.configure.examples.txt")
    print("[log file: %s]" % log_file)
    example_dir = pjoin(install_dir, "examples", get_project_name(), "using-with-cmake")
    res = shell_exec("cd {0} && mkdir build && cd build && cmake -C {0}/host-config.cmake {0}".format(example_dir),
                     output_file = log_file,
                     print_output = report_to_stdout,
                     echo=True)

    if res != 0:
        print("[ERROR: error code={0}: Configure examples for host-config: {1} failed]\n".format(res, host_config))
        return res

    # Make examples
    log_file =  pjoin(build_dir,"output.log.make.examples.txt")
    print("[log file: %s]" % log_file)
    install_build_dir = pjoin(example_dir, "build")
    res = shell_exec("cd {0} && make && ls -al && make test ARGS=\"-VV\"".format(install_build_dir),
                     output_file = log_file,
                     print_output = report_to_stdout,
                     echo=True)

    if res != 0:
        print("[ERROR: error code={0}: Make and test examples for host-config: {1} failed]\n".format(res, host_config))
        return res

    return 0

def build_and_test_host_config(test_root, host_config, report_to_stdout=False, extra_cmake_options="", skip_install=False, skip_tests=False, job_count=""):
    host_config_root = get_host_config_root(host_config)
    # setup build and install dirs
    build_dir   = pjoin(test_root,"build-%s"   % host_config_root)
    install_dir = pjoin(test_root,"install-%s"   % host_config_root)
    print("[Testing build, test, and docs of host config file: %s]" % host_config)
    print("[ build dir: %s]"   % build_dir)
    print("[ install dir: %s]"   % install_dir)

    # configure
    cfg_output_file = pjoin(test_root,"output.log.%s.configure.txt" % host_config_root)
    print("[starting configure of %s]" % host_config)
    print("[log file: %s]" % cfg_output_file)
    res = shell_exec("%s config-build.py -bp %s -hc %s -ip %s %s" % (sys.executable, build_dir, host_config, install_dir, extra_cmake_options),
                     output_file = cfg_output_file,
                     print_output = report_to_stdout,
                     echo=True)

    if res != 0:
        print("[ERROR: Configure for host-config: %s failed]\n" % host_config)
        return res

    ####
    # build, test, and install
    ####

    # build the code
    start_time = time.time()
    bld_output_file =  pjoin(build_dir,"output.log.make.txt")
    print("[starting build]")
    print("[log file: %s]" % bld_output_file)
    res = shell_exec("cd %s && make -j %s VERBOSE=1 " % (build_dir, job_count),
                     output_file = bld_output_file,
                     print_output = report_to_stdout,
                     echo=True)

    if res != 0:
        print("[ERROR: Build for host-config: %s failed]\n" % host_config)
        return res
    end_time = time.time()
    print("[Smith build time: {0}]\n".format(convertSecondsToReadableTime(end_time - start_time)))

    # test the code
    start_time = time.time()
    if not skip_tests:
        tst_output_file = pjoin(build_dir,"output.log.make.test.txt")
        print("[starting unit tests]")
        print("[log file: %s]" % tst_output_file)

        # Use a maximum of 8 job slots for unit tests due to extra parallelism from OpenMP/MPI
        test_job_count = 8
        if job_count:
            test_job_count = min(int(job_count), 8)

        tst_cmd = "cd %s && make CTEST_OUTPUT_ON_FAILURE=1 test ARGS=\"--no-compress-output -T Test -VV -j %s\"" % (build_dir, test_job_count)
        res = shell_exec(tst_cmd,
                        output_file = tst_output_file,
                        print_output = report_to_stdout,
                        echo=True)

        # Convert CTest output to JUnit, do not overwrite previous res
        print("[Checking to see if xsltproc exists...]")
        test_xsltproc_res = shell_exec("xsltproc --version", echo=True)
        if test_xsltproc_res != 0:
            print("[WARNING: xsltproc does not exist skipping JUnit conversion]")
        else:
            junit_file = pjoin(build_dir, "junit.xml")
            xsl_file = pjoin(get_blt_dir(), "tests/ctest-to-junit.xsl")
            ctest_file = pjoin(build_dir, "Testing/*/Test.xml")

            print("[Converting CTest XML to JUnit XML]")
            convert_cmd  = "xsltproc -o {0} {1} {2}".format(junit_file, xsl_file, ctest_file)
            convert_res = shell_exec(convert_cmd, echo=True)
            if convert_res != 0:
                print("[WARNING: Converting to JUnit failed.]")

        if res != 0:
            print("[ERROR: Tests for host-config: %s failed]\n" % host_config)
            return res
    else:
        print("[skipping unit tests]")
    end_time = time.time()
    print("[unit test time: {0}]\n".format(convertSecondsToReadableTime(end_time - start_time)))

    # build the docs
    start_time = time.time()
    docs_output_file = pjoin(build_dir,"output.log.make.docs.txt")
    print("[starting docs generation]")
    print("[log file: %s]" % docs_output_file)

    res = shell_exec("cd %s && make docs " % build_dir,
                     output_file = docs_output_file,
                     print_output = report_to_stdout,
                     echo=True)

    if res != 0:
        print("[ERROR: Docs generation for host-config: %s failed]\n\n" % host_config)
        return res
    end_time = time.time()
    print("[docs time: {0}]\n".format(convertSecondsToReadableTime(end_time - start_time)))

    # Install and test examples
    start_time = time.time()
    if skip_install:
        print("[Skipping 'make install']\n")
    else:
        res = test_examples(host_config, build_dir, install_dir, report_to_stdout, job_count)

        if res != 0:
            print("[ERROR: Building examples for host-config: %s failed]\n\n" % host_config)
            return res
    end_time = time.time()
    print("[install time: {0}]\n".format(convertSecondsToReadableTime(end_time - start_time)))

    print("[SUCCESS: Build, test, and install for host-config: {0} complete]\n".format(host_config))

    set_group_and_perms(build_dir)
    set_group_and_perms(install_dir)

    return 0


def build_and_test_host_configs(prefix, timestamp, use_generated_host_configs, report_to_stdout = False,
                                extra_cmake_options = "", skip_install=False, skip_tests=False, job_count=""):
    host_configs = get_host_configs_for_current_machine(prefix, use_generated_host_configs)
    if len(host_configs) == 0:
        log_failure(prefix,"[ERROR: No host configs found at %s]" % prefix)
        return 1
    print("Found Host-configs:")
    for host_config in host_configs:
        print("    " + host_config)
    print("\n")

    test_root =  get_build_and_test_root(prefix, timestamp)
    os.mkdir(test_root)
    write_build_info(pjoin(test_root,"info.json")) 
    ok  = []
    bad = []
    for host_config in host_configs:
        build_dir = get_build_dir(test_root, host_config)

        start_time = time.time()
        if build_and_test_host_config(test_root, host_config, report_to_stdout, extra_cmake_options, skip_install, skip_tests, job_count) == 0:
            ok.append(host_config)
            log_success(build_dir, "[Success: Built host-config: {0}]".format(host_config), timestamp)
        else:
            bad.append(host_config)
            log_failure(build_dir, "[Error: Failed to build host-config: {0}]".format(host_config), timestamp)
        end_time = time.time()
        print("[host config build time: {0}]\n".format(convertSecondsToReadableTime(end_time - start_time)))

    # Log overall job success/failure
    if len(bad) != 0:
        log_failure(test_root, "[Error: Failed to build host-configs: {0}]".format(bad), timestamp)
    else:
        log_success(test_root,"[Success: Built all host-configs: {0}]".format(ok), timestamp)

    # Output summary of failure/succesful builds
    if len(ok) > 0:
        print("Succeeded:")
        for host_config in ok:
            print("    " + host_config)

    if len(bad) > 0:
        print("Failed:")
        for host_config in bad:
            print("    " + host_config)
        print("\n")
        return 1

    print("\n")

    return 0


def set_group_and_perms(directory):
    """
    Sets the proper group and access permissions of given input
    directory. 
    """

    skip = True
    shared_dirs = [get_shared_base_dir()]
    for shared_dir in shared_dirs:
        if directory.startswith(shared_dir):
            skip = False
            break

    if skip:
        print("[Skipping update of group and access permissions. Provided directory was not a known shared location: {0}]".format(directory))
    else:
        print("[changing group and access perms of: %s]" % directory)
        print("[changing group to smithdev]")
        shell_exec("chgrp -f -R smithdev %s" % (directory),echo=True,error_prefix="WARNING:")
        print("[changing perms for smithdev members to 'rwX' and all to 'rX']")
        shell_exec("chmod -f -R g+rwX,a+rX %s" % (directory),echo=True,error_prefix="WARNING:")
        print("[done setting perms for: %s]" % directory)
    return 0


def full_build_and_test_of_tpls(builds_dir, timestamp, spec, report_to_stdout = False, short_path = False, mirror_location = '', job_count=""):
    if spec:
        specs = [spec]
    else:
        specs = get_specs_for_current_machine()
    print("[Building and testing tpls for specs: ")
    for spec in specs:
        print("{0}".format(spec))
    print("]\n")

    # Use shared network mirror location otherwise create local one
    if mirror_location:
        mirror_dir = mirror_location
    else:
        mirror_dir = get_shared_mirror_dir()
        if not os.path.exists(mirror_dir):
            mirror_dir = pjoin(builds_dir,"mirror")
        print("[using mirror location: %s]" % mirror_dir)

    # unique install location
    prefix = builds_dir
    if not short_path:
        prefix = pjoin(prefix, get_system_type())
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    if not short_path:
        prefix = pjoin(prefix, timestamp)
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    # create a mirror
    uberenv_create_mirror(prefix, spec, "", mirror_dir, report_to_stdout)
    # write info about this build
    write_build_info(pjoin(prefix, "info.json"))

    repo_dir = get_repo_dir()
    # Clean previously generated host-configs into TPL install directory
    print("[Cleaning previously generated host-configs if they exist]")
    host_configs = get_host_configs_for_current_machine(repo_dir, True)
    for host_config in host_configs:
        os.remove(host_config)

    # use uberenv to install for all specs
    tpl_build_failed = False
    for spec in specs:
        start_time = time.time()
        fullspec = "{0}".format(spec)
        res = uberenv_build(prefix, fullspec, "", mirror_dir, report_to_stdout)
        end_time = time.time()
        print("[uberenv build time: {0}]".format(convertSecondsToReadableTime(end_time - start_time)))
        if res != 0:
            print("[ERROR: Failed build of tpls for spec %s]\n" % spec)
            tpl_build_failed = True
            break
        else:
            print("[SUCCESS: Finished build tpls for spec %s]\n" % spec)

    # Copy generated host-configs into TPL install directory
    print("[Copying spack generated host-configs to TPL build directory]")
    host_configs = get_host_configs_for_current_machine(repo_dir, True)
    for host_config in host_configs:
        dst = pjoin(prefix, os.path.basename(host_config))
        if os.path.exists(host_config) and not os.path.exists(dst):
            shutil.copy2(host_config, dst)

    src_build_failed = False
    if not tpl_build_failed:
        # build the src against the new tpls
        res = build_and_test_host_configs(prefix, timestamp, True, report_to_stdout, "", False, False, job_count)
        if res != 0:
            print("[ERROR: Build and test of src vs tpls test failed.]\n")
            src_build_failed = True
        else:
            print("[SUCCESS: Build and test of src vs tpls test passed.]\n")

    # set proper perms for installed tpls
    set_group_and_perms(prefix)

    if tpl_build_failed:
        print("[ERROR: Failed to build all specs of third party libraries]")
    if src_build_failed:
        print("[ERROR: Failed to build all specs of source code against new host-configs]")
    return res


def build_devtools(builds_dir, timestamp, short_path, report_to_stdout = False):
    sys_type = get_system_type()
    project_file = "scripts/spack/devtools.json"

    if "toss_4" in sys_type:
        compiler_spec = "%gcc_13"
    elif "blueos" in sys_type:
        compiler_spec = "%gcc@8.3.1"

    print("[Building devtools using compiler spec: {0}]".format(compiler_spec))

    if short_path:
        prefix = builds_dir
    else:
        # unique install location
        prefix = pjoin(builds_dir, sys_type, timestamp)

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Use shared mirror
    mirror_dir = get_shared_mirror_dir()
    print("[Using mirror location: {0}]".format(mirror_dir))
    uberenv_create_mirror(prefix, compiler_spec, project_file, mirror_dir, report_to_stdout)

    # write info about this build
    write_build_info(pjoin(prefix,"info.json"))

    # use uberenv to install devtools
    start_time = time.time()
    res = uberenv_build(prefix, compiler_spec, project_file, mirror_dir, report_to_stdout)
    end_time = time.time()

    print("[Build time: {0}]".format(convertSecondsToReadableTime(end_time - start_time)))
    # Only update the latest symlink if successful and short_path is not set
    if res == 0 and not short_path:
        link_path = pjoin(builds_dir, sys_type, "latest")
        view_dir = pjoin(prefix, "view")
        print("[Creating symlink to latest devtools view:\n{0}\n->\n{1}]".format(link_path, view_dir))
        if os.path.exists(link_path) or os.path.islink(link_path):
            if not os.path.islink(link_path):
                print("[ERROR: Latest devtools link path exists and is not a link: {0}".format(link_path))
                return 1
            os.unlink(link_path)
        os.symlink(view_dir, link_path)

    if res == 0:
        print("[SUCCESS: Finished build devtools for spec %s]\n" % compiler_spec)
    else:
        print("[ERROR: Failed build of devtools for spec %s]\n" % compiler_spec)

    # set proper perms for installed devtools
    set_group_and_perms(prefix)

    return res


def ensure_on_lc_and_group_permissions(allow_non_lc_builds=False):
    user = getpass.getuser()
    group_name = "smithdev"
    ok = True

    # Check group membership
    try:
        # Supplementary groups
        groups = [g.gr_name for g in grp.getgrall() if user in g.gr_mem]
        # Primary group
        primary_group = grp.getgrgid(pwd.getpwnam(user).pw_gid).gr_name
        in_smithdev = group_name in groups or primary_group == group_name
    except Exception as e:
        print(f"Error: exception when checking group membership: {e}")
        ok = False

    # --- Check SYS_TYPE environment variable ---
    is_sys_type_defined = False
    if 'SYS_TYPE' in os.environ:
        is_sys_type_defined = True
        sys_type = os.environ['SYS_TYPE']
        if not sys_type.strip():
            print("Error: This script requires to you be on LC. The 'SYS_TYPE' environment variable is set but empty.")
            ok = False

    if allow_non_lc_builds:
        if is_sys_type_defined:
            # Only check smithdev in this case if on LC
            if not in_smithdev:
                print(f"Error: You are on LC and this script requires you to be in the '{group_name}' group.\n User '{user}' is not in the required '{group_name}' group.")
                ok = False
    else:
        if not is_sys_type_defined:
            print("Error: This script requires to you be on LC. The 'SYS_TYPE' environment variable is not set.")
            if not in_smithdev:
                print(f"Error: User '{user}' is not in the required '{group_name}' group.")
                ok = False
        else:
            if not in_smithdev:
                print(f"Error: This script requires you to be on LC and be in the '{group_name}' group.\n User '{user}' is not in the required '{group_name}' group.")
                ok = False
    return ok


def get_specs_for_current_machine():
    repo_dir = get_repo_dir()
    specs_json_path = pjoin(repo_dir, "scripts/spack/specs.json")

    with open(specs_json_path, 'r') as f:
        specs_json = json.load(f)

    sys_type = get_system_type()
    machine_name = get_machine_name()

    specs = []
    if machine_name in specs_json.keys():
        specs = specs_json[machine_name]
    else:
        specs = specs_json[sys_type]

    specs = [spec for spec in specs]

    return specs


def get_host_configs_for_current_machine(src_dir, use_generated_host_configs):
    host_configs = []

    # Generated host-configs will be at the base of the source repository
    host_configs_dir = src_dir
    if not use_generated_host_configs:
        host_configs_dir = pjoin(src_dir, "host-configs")

    hostname_base = get_machine_name()
    host_configs = glob.glob(pjoin(host_configs_dir, hostname_base + "*.cmake"))

    return host_configs


def get_host_config_root(host_config):
    return os.path.splitext(os.path.basename(host_config))[0]


def get_blt_dir():
    _path = "cmake/blt"
    if os.path.exists(_path):
        return _path
    _path = pjoin("smith", _path)
    return _path


def get_build_dir(prefix, host_config):
    host_config_root = get_host_config_root(host_config)
    return pjoin(prefix, "build-" + host_config_root)


def get_repo_dir():
    return os.path.abspath(pjoin(get_script_dir(), "../.."))


def get_build_and_test_root(prefix, timestamp):
    dirname = "_{0}_build_and_test_{1}".format(get_project_name(), timestamp)
    return pjoin(prefix, dirname)


def get_machine_name():
    return socket.gethostname().rstrip('1234567890')


def get_system_type():
    return os.environ["SYS_TYPE"]


def get_platform():
    return get_system_type() if "SYS_TYPE" in os.environ else get_machine_name()


def get_username():
    return getpass.getuser()


def get_shared_base_dir():
    return "/usr/WS2/smithdev"


def get_shared_devtool_dir():
    dir = pjoin(get_shared_base_dir(), "devtools")
    return dir


def get_shared_mirror_dir():
    return pjoin(get_shared_base_dir(), "mirror")


def get_shared_libs_dir():
    return pjoin(get_shared_base_dir(), "libs", get_project_name())


def get_shared_spot_dir():
    return pjoin(get_shared_base_dir(), "califiles", get_project_name())


def get_uberenv_path():
    return pjoin(get_repo_dir(), "scripts/uberenv/uberenv.py")


def on_rz():
    machine_name = get_machine_name()
    if machine_name.startswith("rz"):
        return True
    return False


def get_script_dir():
    # NOTE: sys.argv[0] is based-on the script that is calling get_script_dir (e.g. build_tpls.py). This is to properly
    # handle the case where this `scripts/llnl` directory is being symlinked by Smith. 
    return os.path.dirname(os.path.abspath(sys.argv[0]))


_project_name = ""
def get_project_name():
    global _project_name
    if not _project_name:
        uberenv_config_path = pjoin(get_repo_dir(), ".uberenv_config.json")
        _project_name = "UNKNOWN_PROJECT"
        if os.path.exists(uberenv_config_path):
            with open(uberenv_config_path) as json_file:
                data = json.load(json_file)
                if "package_name" in data:
                    _project_name = data["package_name"]
    return _project_name


def convertSecondsToReadableTime(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def get_host_config_path(repo_dir, host_config):
    # First try with where uberenv generates host-configs.
    host_config_path = os.path.join(repo_dir, host_config)
    if not os.path.isfile(host_config_path):
        print("[INFO: Looking for host_config at %s]" % host_config_path)
        print("[WARNING: Spack generated host-config not found, trying with predefined]")

        # Then look into project predefined host-configs.
        host_config_path = os.path.join(repo_dir, "host-configs", host_config)
        if not os.path.isfile(host_config_path):
            print("[INFO: Looking for host_config at %s]" % host_config_path)
            print("[WARNING: Predefined host-config not found, trying with Docker]")

            # Otherwise look into project predefined Docker host-configs.
            host_config_path = os.path.join(repo_dir, "host-configs", "docker", host_config)
            if not os.path.isfile(host_config_path):
                print("[INFO: Looking for host_config at %s]" % host_config_path)
                print("[WARNING: Predefined Docker host-config not found]")
                print("[ERROR: Could not find any host-configs in any known path. Try giving fully qualified path.]")
                sys.exit(1)

    return host_config_path
