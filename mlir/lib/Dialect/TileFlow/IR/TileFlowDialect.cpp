#include "tileflow/Dialect/TileFlow/IR/TileFlowOps.h"

#include "mlir/IR/Builders.h"

#include "tileflow/Dialect/TileFlow/IR/TileFlowDialect.cpp.inc"

#define GET_OP_CLASSES
#include "tileflow/Dialect/TileFlow/IR/TileFlowOps.cpp.inc"

using namespace mlir;

namespace {
bool compatibleDim(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}
} // namespace

LogicalResult tileflow::KernelOp::verify() {
  if (getGridRank() == 0 || getGridRank() > 3)
    return emitOpError("requires grid_rank in the range [1, 3]");
  if (getThreads() && *getThreads() == 0)
    return emitOpError("requires a positive thread count");
  if (getGrid().size() != getGridRank())
    return emitOpError("requires one grid extent per grid dimension");

  Block &entry = getBody().front();
  if (entry.getNumArguments() != getGridRank())
    return emitOpError("requires one body program identifier per grid dimension");
  if (llvm::any_of(entry.getArgumentTypes(),
                   [](Type type) { return !type.isIndex(); }))
    return emitOpError("requires index-typed body arguments");
  return success();
}

LogicalResult tileflow::ParallelOp::verify() {
  size_t rank = getLowerBounds().size();
  if (rank == 0)
    return emitOpError("requires at least one parallel dimension");
  if (getUpperBounds().size() != rank || getSteps().size() != rank)
    return emitOpError("requires equally sized lower, upper, and step lists");

  Block &entry = getBody().front();
  if (entry.getNumArguments() != rank)
    return emitOpError("requires one body index argument per dimension");
  if (llvm::any_of(entry.getArgumentTypes(),
                   [](Type type) { return !type.isIndex(); }))
    return emitOpError("requires index-typed body arguments");
  return success();
}

LogicalResult tileflow::DotOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto acc = getAccumulator().getType();
  auto result = getResult().getType();
  if (lhs.getRank() != 2 || rhs.getRank() != 2 || acc.getRank() != 2 ||
      result.getRank() != 2)
    return emitOpError("requires rank-2 lhs, rhs, accumulator, and result tiles");
  if (!compatibleDim(lhs.getDimSize(1), rhs.getDimSize(0)) ||
      !compatibleDim(lhs.getDimSize(0), acc.getDimSize(0)) ||
      !compatibleDim(rhs.getDimSize(1), acc.getDimSize(1)))
    return emitOpError("has incompatible matrix dimensions");
  if (acc != result)
    return emitOpError("requires accumulator and result to have the same type");
  return success();
}

void tileflow::TileFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tileflow/Dialect/TileFlow/IR/TileFlowOps.cpp.inc"
      >();
}
