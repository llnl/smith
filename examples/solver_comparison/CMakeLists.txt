smith_add_executable(NAME       challenge_suite
                     SOURCES    challenge_suite.cpp
                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                     DEPENDS_ON smith)

if(SMITH_ENABLE_HIP)
  blt_add_target_compile_flags(TO challenge_suite FLAGS "-fgpu-rdc")
  blt_add_target_link_flags(TO challenge_suite FLAGS "-fgpu-rdc")
endif()
