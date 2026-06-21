add_llvm_executable(tileflow-opt tileflow-opt.cpp)
llvm_update_compile_flags(tileflow-opt)

target_link_libraries(tileflow-opt PRIVATE
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
  MLIRTools
)

mlir_check_all_link_libraries(tileflow-opt)
