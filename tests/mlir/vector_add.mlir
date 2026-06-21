// RUN: tileflow-opt --tileflow-frontend-verify --canonicalize --cse %s | FileCheck %s

module {
  func.func @vector_add(%A: memref<*xf32>, %B: memref<*xf32>) {
    "tileflow.tensor_decl"() {name = "A", shape = "(1024,)", dtype = "T.float32"} : () -> ()
    "tileflow.tensor_decl"() {name = "B", shape = "(1024,)", dtype = "T.float32"} : () -> ()
    "tileflow.empty"() {arg0 = "(1024,)", arg1 = "T.float32", name = "C", callee = "T.empty"} : () -> ()
    "tileflow.kernel_launch"() {arg0 = "T.ceildiv(1024, 128)", threads = "128", bindings = ["bx"]} : () -> ()
    "tileflow.parallel_for"() {targets = ["tx"], iter = "T.Parallel(128)", arg0 = "128", callee = "T.Parallel"} : () -> ()
    "tileflow.assign"() {target = "i", value = "bx * 128 + tx"} : () -> ()
    "tileflow.store"() {target = "C[bx * 128 + tx]", value = "A[bx * 128 + tx] + B[bx * 128 + tx]"} : () -> ()
    "tileflow.end_parallel_for"() : () -> ()
    "tileflow.kernel_end"() : () -> ()
    "tileflow.return_value"() {value = "C"} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @vector_add
// CHECK: tileflow.tensor_decl
// CHECK: tileflow.store

