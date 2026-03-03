# Smith Unit Test Guidance

This repo’s unit tests:
- Use GoogleTest (`TEST`, `TEST_F`, etc.).
- Live close to the code they cover in a `tests/` subdirectory.
- Are registered with CTest via the CMake macro `smith_add_tests(...)`.

## Where tests go

Place unit tests next to the component they cover:
- `src/smith/<component>/tests/` (recommended for library code)
- `src/tests/` (project-wide smoke/regression tests)

Each `tests/` directory should contain:
- One or more `*.cpp` test files
- A `CMakeLists.txt` that registers those tests with `smith_add_tests`

Also ensure the parent component adds the tests subdirectory when tests are enabled:

```cmake
if(SMITH_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
```

## Writing GoogleTest unit tests

- Include GoogleTest: `#include <gtest/gtest.h>`
- Prefer fast, deterministic, CPU-only tests by default.
- Use fixtures (`TEST_F`) when setup/teardown is shared.
- Avoid depending on external files unless the test is explicitly a data-driven/integration test.

## Registering tests with `smith_add_tests`

`smith_add_tests` creates **one test executable per source file** and registers it with CTest.
The executable/test name is the source basename (e.g., `test_state_manager.cpp` → `test_state_manager`).

Minimal `tests/CMakeLists.txt` template:

```cmake
set(my_component_test_depends
  smith_my_component
  gtest)

set(my_component_test_sources
  test_feature_a.cpp
  test_feature_b.cpp)

smith_add_tests(SOURCES    ${my_component_test_sources}
                DEPENDS_ON ${my_component_test_depends}
                NUM_MPI_TASKS 1)
```

Notes:
- Keep test source basenames unique across the project (CMake targets are global).
- Add only the libraries you need to `DEPENDS_ON` (the library under test + `gtest` + any required Smith/TPL targets).

### MPI / OpenMP tests

If a test requires MPI, add the MPI dependency and set `NUM_MPI_TASKS`:

```cmake
smith_add_tests(SOURCES       my_parallel_test.cpp
                DEPENDS_ON    gtest blt::mpi smith_my_component
                NUM_MPI_TASKS 4)
```

If a test requires OpenMP, set `NUM_OMP_THREADS` accordingly.

### CUDA-enabled unit tests

If you need a CUDA-enabled unit test, gate it behind `SMITH_ENABLE_CUDA` and pass `USE_CUDA`:

```cmake
if(SMITH_ENABLE_CUDA)
  smith_add_tests(SOURCES    my_cuda_test.cpp
                  DEPENDS_ON gtest smith_my_component
                  USE_CUDA   TRUE)
endif()
```

## Running tests

- Configure with tests enabled: `-DENABLE_TESTS=ON -DSMITH_ENABLE_TESTS=ON`
- Run all tests: `ctest`
- Run a single test by name: `ctest -R test_state_manager`

## Examples in this repo

- Macro definition: `cmake/SmithMacros.cmake`
- Typical component tests: `src/smith/physics/state/tests/CMakeLists.txt`
- Project-wide tests: `src/tests/CMakeLists.txt`

