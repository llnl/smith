# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(physics_benchmark_depends smith_physics)

set(physics_benchmarks
    physics_benchmark_functional
    physics_benchmark_solid_nonlinear_solve
    physics_benchmark_thermal
    )

# Create executable for each benchmark
foreach(physics_benchmark_exec ${physics_benchmarks})
    smith_add_executable(NAME ${physics_benchmark_exec}
                         SOURCES ${physics_benchmark_exec}.cpp
                         DEPENDS_ON ${physics_benchmark_depends}
                         OUTPUT_DIR ${PROJECT_BINARY_DIR}/benchmarks
                         FOLDER smith/benchmarks
                         )

    # Add benchmarks with various task counts
    foreach(task_count 1 4 16)
        set(physics_benchmark_name ${physics_benchmark_exec}_${task_count}_task_count)
        blt_add_benchmark(NAME ${physics_benchmark_name}
                          COMMAND ${physics_benchmark_exec}
                          NUM_MPI_TASKS ${task_count}
                          WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                          )

        # Increase benchmark time limit
        set_tests_properties(${physics_benchmark_name} PROPERTIES TIMEOUT 4800)
    endforeach()
endforeach()
