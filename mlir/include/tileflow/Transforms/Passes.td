#ifndef TILEFLOW_TRANSFORMS_PASSES_TD
#define TILEFLOW_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TileFlowFrontendVerify : Pass<"tileflow-frontend-verify", "::mlir::ModuleOp"> {
  let summary = "Verify TileFlow frontend IR invariants";
  let description = [{
    Initial native pass hook for TileFlow frontend verification. This pass is
    intentionally conservative while operation semantics move from Python into
    MLIR.
  }];
}

#endif // TILEFLOW_TRANSFORMS_PASSES_TD
