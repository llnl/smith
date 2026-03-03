# Building Smith

## Compute-node requirement (Codex)

Only run compilation and test commands on a compute node. Determine this by running:

```bash
./skills/building/scripts/is_compute_node
```

If it prints `login`, do **not** run `./config-build.py`, `cmake --build ...`, or `ctest ...`. Stop and ask the user to switch to/allocate a compute node, then continue once `is_compute_node` prints `compute`.

This repository supports two build workflows:

- **Smith-only (default):** builds Smith against externally provided TPLs (e.g., via Spack/uberenv + a host-config).
- **Co-develop Smith (opt-in):** additionally builds the `mfem`, `axom`, and `tribol` git submodules as CMake subdirectories.

## Smith-only build (default)

1) (Recommended) Initialize required submodules (at least `cmake/blt`):

```bash
git submodule update --init --recursive
```

2) Configure (recommended wrapper around CMake):

```bash
./config-build.py -bp build -ip install -hc host-configs/<file>.cmake -bt Debug --exportcompilercommands
```

3) Build and test:

```bash
cmake --build build -j
ctest --test-dir build
```

## Co-develop Smith (MFEM + Axom + Tribol)

Use this only when you intend to modify/build Smith against the submodules.

1) Initialize submodules (required):

```bash
git submodule update --init --recursive
```

2) Configure with co-develop enabled:

```bash
./config-build.py -bp build-codevelop -ip install-codevelop -hc host-configs/<file>.cmake -bt Debug -DSMITH_ENABLE_CODEVELOP=ON --exportcompilercommands
```

3) Build and test:

```bash
cmake --build build-codevelop -j
ctest --test-dir build-codevelop
```

## Common build options

Common CMake options (and their defaults) live in `cmake/SmithBasics.cmake`. Pass them at configure time as `-D<OPTION>=ON|OFF`, for example:

```bash
./config-build.py -hc host-configs/<file>.cmake -DENABLE_ASAN=ON
```

### AddressSanitizer (`ENABLE_ASAN`)

AddressSanitizer is available via the `ENABLE_ASAN` CMake option (default: `OFF`). It is supported with GCC or Clang.

```bash
./config-build.py -hc host-configs/<file>.cmake -bt Debug -DENABLE_ASAN=ON
```
