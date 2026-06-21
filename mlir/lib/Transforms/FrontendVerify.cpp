#include "tileflow/Transforms/Passes.h"

#include "tileflow/Dialect/TileFlow/IR/TileFlowOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace tileflow {
#define GEN_PASS_DEF_TILEFLOWFRONTENDVERIFY
#include "tileflow/Transforms/Passes.h.inc"
} // namespace tileflow

using namespace mlir;

namespace {
struct FrontendVerifyPass
    : public tileflow::impl::TileFlowFrontendVerifyBase<FrontendVerifyPass> {
  void runOnOperation() override {
    // Placeholder native hook. Real invariants should be added here as the
    // TileFlow dialect gains typed ops and region structure.
    getOperation()->walk([](Operation *op) {
      (void)op;
    });
  }
};
} // namespace

namespace tileflow {
} // namespace tileflow
