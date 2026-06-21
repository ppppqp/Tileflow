#ifndef TILEFLOW_TRANSFORMS_PASSES_H
#define TILEFLOW_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace tileflow {

#define GEN_PASS_DECL
#include "tileflow/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "tileflow/Transforms/Passes.h.inc"

} // namespace tileflow

#endif // TILEFLOW_TRANSFORMS_PASSES_H
