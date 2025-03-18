set(gretl_test_depends smith_gretl gtest)

set(gretl_tests
    test_gretl_checkpoint.cpp
    test_gretl_dynamics.cpp
    test_gretl_graph)

serac_add_tests( SOURCES       ${gretl_tests}
                 DEPENDS_ON    ${gretl_test_depends}
                 NUM_MPI_TASKS 1)
