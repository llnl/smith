#!/usr/bin/env python3

##############################################################################
# Copyright (c) 2025, Lawrence Livermore National Security, LLC and
# RADIUSS project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

# This script add the software stack for the specified rocm version to Spack
# packages.yaml.

import argparse
import os
import sys
from pathlib import Path

try:
    from ruamel.yaml import YAML
except ImportError:
    print("Error: ruamel.yaml is required. Install with: pip install ruamel.yaml")
    sys.exit(1)

def find_yaml_files(directory):
    """Find all packages.yaml files in the directory tree."""
    yaml_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "packages.yaml":
                yaml_files.append(os.path.join(root, file))
    return yaml_files

def get_rocm_packages():
    """Return list of ROCm-related packages that need updating."""
    return [
        "hip",
        "hipsparse", 
        "hipblas",
        "hipblas-common",
        "hipsolver",
        "llvm-amdgpu",
        "hsa-rocr-dev",
        "rocminfo",
        "rocm-device-libs",
        "rocprim",
        "rocthrust",
        "rocsparse",
        "rocrand",
        "rocblas",
        "rocsolver",
        "rocm-core",
        "cray-mpich"  # Has ROCm compiler variants
    ]

def get_prefix_pattern(package_name, version):
    """Generate the expected prefix path for a package and version."""
    if package_name == "hip":
        if version in ["5.7.1"]:
            return f"/opt/rocm-{version}/hip"
        else:
            return f"/opt/rocm-{version}"
    elif package_name == "llvm-amdgpu":
        return f"/opt/rocm-{version}/llvm"
    elif package_name == "cray-mpich":
        return f"/usr/tce/packages/cray-mpich/cray-mpich-8.1.31-rocmcc-{version}-magic"
    else:
        return f"/opt/rocm-{version}"

def version_key(version_str):
    """Convert version string to tuple for proper sorting."""
    return tuple(int(x) for x in version_str.split('.'))

def prepare_yaml_for_processing(content):
    """
    Temporarily modify YAML content to make it parseable by ruamel.yaml.
    Returns (modified_content, has_quoted_packages_key).
    """
    has_quoted_packages = content.strip().startswith("'packages:'")

    if has_quoted_packages:
        # Temporarily remove quotes around packages: for processing
        modified_content = content.replace("'packages:'", "packages", 1)
        return modified_content, True

    return content, False

def restore_yaml_format(content, had_quoted_packages):
    """Restore the original YAML format with quoted packages key if needed."""
    if had_quoted_packages:
        # Restore quotes around packages:
        content = content.replace("packages:", "'packages:':", 1)

    return content

def debug_yaml_structure(data, file_path):
    """Print the structure of the YAML file for debugging."""
    print(f"  YAML structure in {file_path}:")
    if isinstance(data, dict):
        for key in data.keys():
            print(f"    - {key}")
            if key == "packages" and isinstance(data[key], dict):
                print(f"      Found {len(data[key])} packages")
    else:
        print(f"    Root is {type(data)}, not a dictionary")

def add_rocm_version_to_package(packages_data, package_name, new_version):
    """Add a new ROCm version to a specific package in the YAML data."""

    if package_name not in packages_data:
        print(f"  Package {package_name} not found in file")
        return False

    package = packages_data[package_name]
    print(f"  Processing {package_name}")

    # Special handling for cray-mpich - it might not have a version list
    if package_name == "cray-mpich":
        return add_rocm_version_to_cray_mpich(package, new_version)

    # Check if package has version list
    if 'version' not in package:
        print(f"  No version list found for {package_name}")
        return False

    # Get existing versions
    existing_versions = package['version']
    if not isinstance(existing_versions, list):
        print(f"  Version field is not a list for {package_name}")
        return False

    # Skip if version already exists
    if new_version in existing_versions:
        print(f"  Version {new_version} already exists for {package_name}")
        return False

    # Add new version and sort
    existing_versions.append(new_version)
    existing_versions.sort(key=version_key)

    # Check if package has externals
    if 'externals' not in package:
        print(f"  No externals found for {package_name}")
        return False

    externals = package['externals']
    if not isinstance(externals, list):
        print(f"  Externals field is not a list for {package_name}")
        return False

    # Generate new external entry
    prefix = get_prefix_pattern(package_name, new_version)
    new_spec = f"{package_name}@{new_version}%rocmcc@{new_version}"

    new_external = {
        'spec': new_spec,
        'prefix': prefix
    }

    # Add new external entry
    externals.append(new_external)

    # Sort externals by version
    def extract_version_from_spec(external):
        spec = external.get('spec', '')
        # Extract version from spec like "package@version%compiler"
        if '@' in spec:
            parts = spec.split('@')[1]  # Get everything after package@
            if '%' in parts:
                version_part = parts.split('%')[0]  # Get version before %compiler
            else:
                version_part = parts
            try:
                return version_key(version_part)
            except:
                return (0, 0, 0)  # fallback for unparseable versions
        return (0, 0, 0)

    externals.sort(key=extract_version_from_spec)

    print(f"  Added version {new_version} to {package_name}")
    return True

def add_rocm_version_to_cray_mpich(package, new_version):
    """Special handling for cray-mpich package."""
    print(f"  Special handling for cray-mpich")
    
    # Check if package has externals
    if 'externals' not in package:
        print(f"  No externals found for cray-mpich")
        return False

    externals = package['externals']
    if not isinstance(externals, list):
        print(f"  Externals field is not a list for cray-mpich")
        return False

    # Check if this ROCm version already exists
    new_spec_pattern = f"cray-mpich@8.1.31%rocmcc@{new_version}"
    for external in externals:
        if external.get('spec', '') == new_spec_pattern:
            print(f"  Version {new_version} already exists for cray-mpich")
            return False

    # Generate new external entry
    prefix = get_prefix_pattern("cray-mpich", new_version)
    new_spec = f"cray-mpich@8.1.31%rocmcc@{new_version}"

    new_external = {
        'spec': new_spec,
        'prefix': prefix
    }

    # Add new external entry
    externals.append(new_external)

    # Sort externals by ROCm compiler version
    def extract_rocmcc_version_from_spec(external):
        spec = external.get('spec', '')
        # Look for %rocmcc@version pattern
        if '%rocmcc@' in spec:
            rocmcc_part = spec.split('%rocmcc@')[1]
            # Get version (might have other stuff after it)
            version_part = rocmcc_part.split()[0]  # Take first part before any spaces
            try:
                return version_key(version_part)
            except:
                return (0, 0, 0)
        # Non-ROCm entries go to the beginning
        return (0, 0, 0)

    # Sort to keep ROCm entries together and in version order
    externals.sort(key=lambda x: (
        0 if '%rocmcc@' not in x.get('spec', '') else 1,  # Non-ROCm first, then ROCm
        extract_rocmcc_version_from_spec(x)  # Then by ROCm version
    ))

    print(f"  Added version {new_version} to cray-mpich")
    return True

def update_yaml_file(file_path, new_version):
    """Update a single YAML file with the new ROCm version."""
    print(f"Updating {file_path}")

    try:
        # Read the original file
        with open(file_path, 'r') as f:
            original_content = f.read()

        # Prepare content for YAML processing
        content_for_processing, had_quoted_packages = prepare_yaml_for_processing(original_content)

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.map_indent = 2
        yaml.sequence_indent = 2
        yaml.sequence_dash_offset = 0

        # Parse the modified content
        data = yaml.load(content_for_processing)

        if not data:
            print(f"  Empty or invalid YAML file: {file_path}")
            return

        # Debug: show the structure
        debug_yaml_structure(data, file_path)

        # Try different possible structures
        packages_data = None

        if 'packages' in data:
            packages_data = data['packages']
        elif isinstance(data, dict):
            # Check if the root level contains package definitions directly
            # Look for common package names to determine if this is a packages structure
            rocm_packages = get_rocm_packages()
            found_packages = [pkg for pkg in rocm_packages if pkg in data]
            if found_packages:
                print(f"  Found packages at root level: {found_packages}")
                packages_data = data

        if not packages_data:
            print(f"  No packages section found in {file_path}")
            return

        if not isinstance(packages_data, dict):
            print(f"  Packages section is not a dictionary in {file_path}")
            return

        rocm_packages = get_rocm_packages()

        changes_made = False
        for package in rocm_packages:
            if add_rocm_version_to_package(packages_data, package, new_version):
                changes_made = True

        # Write back if changes were made
        if changes_made:
            # Write to string buffer first
            from io import StringIO
            buffer = StringIO()
            yaml.dump(data, buffer)

            # Get the processed content and restore original format
            processed_content = buffer.getvalue()
            final_content = restore_yaml_format(processed_content, had_quoted_packages)

            # Write final content to file
            with open(file_path, 'w') as f:
                f.write(final_content)

            print(f"  Updated {file_path}")
        else:
            print(f"  No changes needed for {file_path}")

    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="Add a new ROCm version to all relevant packages in packages.yaml files"
    )
    parser.add_argument("rocm_version", help="ROCm version to add (e.g., 6.4.0)")
    parser.add_argument(
        "--directory", 
        default=".", 
        help="Directory to search for packages.yaml files (default: current directory)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be changed without making changes"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Show debug information about YAML structure"
    )

    args = parser.parse_args()

    # Validate version format
    try:
        version_key(args.rocm_version)
    except:
        print(f"Error: Invalid version format '{args.rocm_version}'. Expected format: X.Y.Z")
        sys.exit(1)

    # Find all packages.yaml files
    yaml_files = find_yaml_files(args.directory)

    if not yaml_files:
        print("No packages.yaml files found")
        sys.exit(1)

    print(f"Found {len(yaml_files)} packages.yaml files")
    print(f"Adding ROCm version {args.rocm_version}")

    if args.dry_run:
        print("DRY RUN - No files will be modified")
        # For dry run, we'd need to implement a preview mode
        # For now, just show which files would be processed
        for yaml_file in yaml_files:
            print(f"Would update: {yaml_file}")
    else:
        for yaml_file in yaml_files:
            update_yaml_file(yaml_file, args.rocm_version)

if __name__ == "__main__":
    main()
