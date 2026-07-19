#ifndef TILEFLOW_DIALECT_TD
#define TILEFLOW_DIALECT_TD

include "mlir/IR/OpBase.td"

def TileFlow_Dialect : Dialect {
  let name = "tileflow";
  let cppNamespace = "::tileflow";
  let summary = "Logical tiled GPU computation dialect";
  let description = [{
    TileFlow represents typed SSA tile computations and structured logical GPU
    execution before physical CTA, warp, lane, register, and shared-memory
    layouts are assigned.
  }];
}

#endif // TILEFLOW_DIALECT_TD
