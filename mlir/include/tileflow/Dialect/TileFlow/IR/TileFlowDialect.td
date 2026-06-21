#ifndef TILEFLOW_DIALECT_TD
#define TILEFLOW_DIALECT_TD

include "mlir/IR/OpBase.td"

def TileFlow_Dialect : Dialect {
  let name = "tileflow";
  let cppNamespace = "::tileflow";
  let summary = "TileFlow frontend dialect";
  let description = [{
    TileFlow captures TileLang-compatible frontend constructs before they are
    verified, optimized, and lowered through native MLIR passes.
  }];
}

#endif // TILEFLOW_DIALECT_TD

