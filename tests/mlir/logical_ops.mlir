// RUN: tileflow-opt %s --tileflow-frontend-verify | FileCheck %s

module {
  func.func @logical_tiles(%src: memref<128x128xf16>,
                           %dst: memref<128x128xf16>) {
    "tileflow.kernel"(%src, %dst) <{grid_rank = 2 : i64}> ({
    ^bb0(%kernel_src: memref<128x128xf16>,
         %kernel_dst: memref<128x128xf16>):
      %pid0 = tileflow.program_id 0 : index
      %pid1 = tileflow.program_id 1 : index
      %tile = "tileflow.load"(%kernel_src, %pid0, %pid1)
          <{operandSegmentSizes = array<i32: 1, 2, 0, 0>}> :
          (memref<128x128xf16>, index, index) -> tensor<16x16xf16>
      %zero = arith.constant dense<0.0> : tensor<16x16xf32>
      %acc = "tileflow.dot"(%tile, %tile, %zero) :
          (tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf32>)
          -> tensor<16x16xf32>
      "tileflow.store"(%acc, %kernel_dst, %pid0, %pid1)
          <{operandSegmentSizes = array<i32: 1, 1, 2, 0>}> :
          (tensor<16x16xf32>, memref<128x128xf16>, index, index) -> ()
      tileflow.yield
    }) : (memref<128x128xf16>, memref<128x128xf16>) -> ()
    return
  }
}

// CHECK: tileflow.kernel
// CHECK: tileflow.program_id 0
// CHECK: tileflow.load
// CHECK: tileflow.dot
// CHECK: tileflow.store
