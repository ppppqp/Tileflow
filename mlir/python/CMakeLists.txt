find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
  message(WARNING "pybind11 not found; skipping tileflow_mlir Python extension")
  return()
endif()

pybind11_add_module(tileflow_mlir TileFlowPasses.cpp)
set_target_properties(tileflow_mlir PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/tileflow/_mlir"
)
target_link_libraries(tileflow_mlir PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRRegisterAllPasses
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTileFlowDialect
  MLIRTileFlowTransforms
)
