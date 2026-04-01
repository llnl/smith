# set (korner_examples
#   01_test
#   02_implicit_velocity
#   03_implicit_velocity_but_better
#   04_nominal_lattice
#   05_optimal_lattice
#   06_snap_array_nominal
#   07_snap_array_optimal
#   08_quasistatic_thermomechanics
#   09_viscoelastic_cyclic_tests
# )
# set (korner_example_depends
#   smith_differentiable_numerics
#   smith_physics 
# )

# set(KORNER_EXAMPLES_DIR $CMAKE_BINARY_DIR)

# foreach(kor_ex IN LISTS korner_examples)
#   smith_add_executable( NAME       ${kor_ex} 
#     SOURCES     ${kor_ex}.cpp
#     OUTPUT_DIR  ${CMAKE_BINARY_DIR}/korner_examples
#     DEPENDS_ON  ${korner_example_depends}
#   )
#   # install(
#   #   FILES
#   #   ${kor_ex}.cpp
#   #   DESTINATION
#   #   korner_examples/${kor_ex}
#   # )
# endforeach()


