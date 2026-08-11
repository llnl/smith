[comment]: # (#################################################################)
[comment]: # (Copyright (c) Lawrence Livermore National Security, LLC and)
[comment]: # (other Smith Project Developers. See the top-level LICENSE file)
[comment]: # (for details.)
[comment]: #
[comment]: # (SPDX-License-Identifier: (BSD-3-Clause))
[comment]: # (#################################################################)


# Smith Software Release Notes

Notes describing significant changes in each Smith release are documented in this file.

The format of this file is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

The Smith project release numbers follow [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased] - Release date yyyy-mm-dd

### Added

- Added this release notes file to track changes in project
- Added composable differentiable-numerics systems for solid mechanics, thermal mechanics, internal variables, and
  thermo-mechanics coupling through shared field stores.
- Added `combineSystems` and `SystemSolver` support for monolithic and staggered coupled solves, including coupled relaxation algorithms, and fixed-sweep modes.
- Added typed coupling and parameter-field registration APIs that interpolate each coupled physics pack with its own
  time-integration rule before invoking material, source, traction, and objective callbacks.
- Added post-solve and cycle-zero auxiliary systems for stress projection, initial acceleration solves, and other
  derived-field updates in differentiable multiphysics time integration.
- Added composable solid-mechanics and thermo-mechanics examples, tutorials, and regression tests covering coupled
  sensitivities, finite-difference checks, field parameters, and solves.
- Added axisymmetric solid mechanics materials and loads for 2D `(r, z)` meshes.

### Removed

### Deprecated

### Changed

### Fixed

## [Version 0.1.0] - Release date 2026-04-28

## Legend for sections

###  Added
- Use this section for new features

###  Changed
- Use this section for changes in existing functionality

###  Deprecated
- Use this section for soon-to-be removed features

###  Removed
- Use this section for now removed features

###  Fixed
- Use this section for any bug fixes

###  Security
- Use this section in case of vulnerabilities

[Version 0.1.0]: https://github.com/llnl/smith/releases/tag/v0.1.0
