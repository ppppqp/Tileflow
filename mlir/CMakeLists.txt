add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

option(TILEFLOW_ENABLE_PYTHON_EXTENSION "Build tileflow_mlir Python extension" ON)
if(TILEFLOW_ENABLE_PYTHON_EXTENSION)
  add_subdirectory(python)
endif()

