set(LLVM_TARGET_DEFINITIONS TileFlowOps.td)
mlir_tablegen(TileFlowOps.h.inc -gen-op-decls)
mlir_tablegen(TileFlowOps.cpp.inc -gen-op-defs)
mlir_tablegen(TileFlowDialect.h.inc -gen-dialect-decls -dialect=tileflow)
mlir_tablegen(TileFlowDialect.cpp.inc -gen-dialect-defs -dialect=tileflow)
add_public_tablegen_target(TileFlowDialectIncGen)

