blt_add_executable( NAME inertia_relief_example
	            SOURCES inertia_relief_example.cpp
		    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
		    DEPENDS_ON serac_physics continuation_solver)

install(
    FILES
        inertia_relief_example.cpp
    DESTINATION
        examples/serac/inertia_relief
)
