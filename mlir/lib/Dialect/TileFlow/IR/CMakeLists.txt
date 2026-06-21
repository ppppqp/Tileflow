add_mlir_dialect_library(MLIRTileFlowDialect
  TileFlowDialect.cpp

  DEPENDS
  TileFlowDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

