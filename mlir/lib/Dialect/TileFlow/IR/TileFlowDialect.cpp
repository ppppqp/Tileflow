#include "tileflow/Dialect/TileFlow/IR/TileFlowOps.h"

#include "mlir/IR/Builders.h"

#include "tileflow/Dialect/TileFlow/IR/TileFlowDialect.cpp.inc"

#define GET_OP_CLASSES
#include "tileflow/Dialect/TileFlow/IR/TileFlowOps.cpp.inc"

void tileflow::TileFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tileflow/Dialect/TileFlow/IR/TileFlowOps.cpp.inc"
      >();
}

