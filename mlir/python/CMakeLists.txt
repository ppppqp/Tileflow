find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
  message(WARNING "pybind11 not found; skipping tileflow_mlir Python extension")
  return()
endif()

get_filename_component(LLVM_BUILD_ROOT "${LLVM_TOOLS_BINARY_DIR}" DIRECTORY)
find_library(MLIR_PYTHON_CAPI_LIBRARY
  NAMES MLIRPythonCAPI
  PATHS "${LLVM_BUILD_ROOT}/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs"
  NO_DEFAULT_PATH
  REQUIRED
)

pybind11_add_module(tileflow_mlir TileFlowPasses.cpp)
target_sources(tileflow_mlir PRIVATE
  $<TARGET_OBJECTS:obj.MLIRTileFlowDialect>
  $<TARGET_OBJECTS:obj.MLIRTileFlowTransforms>
)
set_target_properties(tileflow_mlir PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/tileflow/_mlir"
)
target_link_libraries(tileflow_mlir PRIVATE
  "${MLIR_PYTHON_CAPI_LIBRARY}"
)
