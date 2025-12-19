# <img src="/share/smith/logo/smith-logo-orange.png?raw=true" width="150" alt="Smith"/>

[![Build Status](https://github.com/LLNL/smith/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/LLNL/smith/actions/workflows/unit-tests.yml)
[![Documentation Status](https://readthedocs.org/projects/serac/badge/?version=latest)](https://llnlsmith.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/LLNL/serac/branch/develop/graph/badge.svg?token=DO4KFMPNM0)](https://codecov.io/gh/LLNL/serac)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](./LICENSE)

Smith is a 3D implicit nonlinear thermal-structural simulation code. Its primary purpose is to investigate 
abstraction strategies and implicit finite element-based algorithm development for emerging computing architectures. 
It also heavily leverages the [MFEM finite element library](https://mfem.org/).

> This project is under heavy development and is currently a pre-alpha release. Functionality and interfaces may change rapidly
> as development progresses.

Documentation
------

Build, run, and design documentation can be found at [readthedocs](https://llnlsmith.readthedocs.io).

Source documentation can be found [here](https://llnlsmith.readthedocs.io/en/latest/doxygen/html/index.html).

Contributions
-------------

We welcome all kinds of contributions: new features, bug fixes, and documentation edits.

For more information, see the [contributing guide](./CONTRIBUTING.md).

License
-------

Copyright (c) Lawrence Livermore National Security, LLC. 
Produced at the Lawrence Livermore National Laboratory.

Copyrights and patents in the Smith project are retained by contributors.
No copyright assignment is required to contribute to Smith.

See [LICENSE](./LICENSE) for details.

Unlimited Open Source - BSD 3-clause Distribution  
`LLNL-CODE-805541`

SPDX usage
-----------

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

Files that are licensed as BSD 3-Clause contain the following
text in the license header:

    SPDX-License-Identifier: (BSD-3-Clause)

External Packages
-----------------

Smith bundles some of its external dependencies in its repository.  These
packages are covered by various permissive licenses.  A summary listing
follows.  See the license included with each package for full details.


[//]: # (Note: The spaces at the end of each line below add line breaks)

PackageName: Axom  
PackageHomePage: https://github.com/LLNL/axom  
PackageLicenseDeclared: BSD-3-Clause  

PackageName: BLT  
PackageHomePage: https://github.com/LLNL/blt  
PackageLicenseDeclared: BSD-3-Clause  

PackageName: ContinuationSolvers  
PackageHomePage: https://github.com/LLNL/ContinuationSolvers  
PackageLicenseDeclared: BSD-3-Clause  

PackageName: MFEM  
PackageHomePage: https://github.com/mfem/mfem  
PackageLicenseDeclared: BSD-3-Clause  

PackageName: radiuss-spack-configs  
PackageHomePage: https://github.com/LLNL/radiuss-spack-configs  
PackageLicenseDeclared: MIT License

PackageName: Tribol  
PackageHomePage: https://github.com/LLNL/tribol  
PackageLicenseDeclared: MIT License

PackageName: uberenv  
PackageHomePage: https://github.com/LLNL/uberenv  
PackageLicenseDeclared: BSD-3-Clause  
