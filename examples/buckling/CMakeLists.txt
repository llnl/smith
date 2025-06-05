# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(TRIBOL_FOUND AND PETSC_FOUND)
    blt_add_executable( NAME        buckling_cylinder
                        SOURCES     cylinder.cpp
                        OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                        DEPENDS_ON  serac_physics
                        )

    message("\n\n\n\n env $ENV{SERAC_CI_WORKFLOW_TYPE}")

    # Run tests with the fast option, unless running nightly CI.
    if("$ENV{SERAC_CI_WORKFLOW_TYPE}" STREQUAL "full")
        set(_fast "")
    else()
        set(_fast "--fast")
    endif()

    blt_add_test(NAME           buckling_cylinder
                 COMMAND        buckling_cylinder ${_fast}
                 NUM_MPI_TASKS  4)

    # TODO Fix this test
    # `InvIso: Newtons method did not converge.`
    # configure_file(cylinder_petsc_options.yml
    #                ${PROJECT_BINARY_DIR}/examples/buckling/cylinder_petsc_options.yml
    #                COPYONLY)
    #
    # blt_add_test(NAME            buckling_cylinder_mortar_contact
    #              COMMAND         buckling_cylinder ${_fast} --contact --contact-type 1 --preconditioner 6 -options_file examples/buckling/cylinder_petsc_options.yml
    #              NUM_MPI_TASKS   4)

    blt_add_test(NAME           buckling_cylinder_no_contact
                 COMMAND        buckling_cylinder ${_fast} --no-contact
                 NUM_MPI_TASKS  4)

    unset(_fast)

    install(
        FILES
            cylinder.cpp
        DESTINATION
            examples/serac/buckling
    )

endif()
