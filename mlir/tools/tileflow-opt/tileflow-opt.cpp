#include "tileflow/Dialect/TileFlow/IR/TileFlowDialect.h"
#include "tileflow/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  tileflow::TileFlowDialect>();

  mlir::registerAllPasses();
  tileflow::registerTileFlowPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TileFlow optimizer driver\n", registry));
}
