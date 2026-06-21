#include "tileflow/Dialect/TileFlow/IR/TileFlowDialect.h"
#include "tileflow/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {
void populateRegistry(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  tileflow::TileFlowDialect>();
}

struct RegisterPassesOnLoad {
  RegisterPassesOnLoad() {
    mlir::registerAllPasses();
    tileflow::registerTileFlowPasses();
  }
} registerPassesOnLoad;

class PassPipeline {
public:
  void add(std::string pipelineText) { elements.push_back(std::move(pipelineText)); }

  std::string run(const std::string &mlirText) const {
    mlir::DialectRegistry registry;
    populateRegistry(registry);

    mlir::MLIRContext context(registry);
    mlir::ParserConfig config(&context);
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(mlirText, config);
    if (!module) {
      throw std::runtime_error("failed to parse TileFlow MLIR");
    }

    mlir::PassManager pm(&context, "builtin.module");
    for (const auto &element : elements) {
      if (mlir::failed(mlir::parsePassPipeline(element, pm, llvm::errs()))) {
        throw std::runtime_error("failed to parse pass pipeline element: " + element);
      }
    }
    if (mlir::failed(pm.run(*module))) {
      throw std::runtime_error("TileFlow native pass pipeline failed");
    }

    std::string output;
    llvm::raw_string_ostream os(output);
    module->print(os, mlir::OpPrintingFlags().useLocalScope());
    os.flush();
    return output;
  }

private:
  std::vector<std::string> elements;
};
} // namespace

PYBIND11_MODULE(tileflow_mlir, m) {
  m.doc() = "TileFlow native MLIR pass pipeline bindings";
  py::class_<PassPipeline>(m, "PassPipeline")
      .def(py::init<>())
      .def("add", &PassPipeline::add, py::arg("pipeline_text"))
      .def("run", &PassPipeline::run, py::arg("mlir_text"));
}
