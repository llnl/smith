blt_add_executable( NAME homotopy_example
	            SOURCES homotopy_example.cpp
		    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
		    DEPENDS_ON serac_physics serac_mesh serac_functional serac_continuation_solvers)

install(
    FILES
        homotopy_example.cpp
    DESTINATION
        examples/serac/homotopy
)
