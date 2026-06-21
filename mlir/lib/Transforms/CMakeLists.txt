add_mlir_library(MLIRTileFlowTransforms
  FrontendVerify.cpp

  DEPENDS
  TileFlowPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTileFlowDialect
  MLIRTransforms
)

