find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
  message(WARNING "pybind11 not found; skipping tileflow_mlir Python extension")
  return()
endif()

pybind11_add_module(tileflow_mlir TileFlowPasses.cpp)
target_link_libraries(tileflow_mlir PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTileFlowDialect
  MLIRTileFlowTransforms
)

