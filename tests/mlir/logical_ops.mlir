// RUN: tileflow-opt %s --tileflow-frontend-verify | FileCheck %s

module {
  func.func @logical_tiles(%src: memref<128x128xf16>,
                           %dst: memref<128x128xf16>) {
    %grid0 = arith.constant 8 : index
    %grid1 = arith.constant 8 : index
    "tileflow.kernel"(%grid0, %grid1) <{grid_rank = 2 : i64}> ({
    ^bb0(%pid0: index, %pid1: index):
      %tile = "tileflow.load"(%src, %pid0, %pid1)
          <{operandSegmentSizes = array<i32: 1, 2, 0, 0>}> :
          (memref<128x128xf16>, index, index) -> tensor<16x16xf16>
      %zero = arith.constant dense<0.0> : tensor<16x16xf32>
      %acc = "tileflow.dot"(%tile, %tile, %zero) :
          (tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf32>)
          -> tensor<16x16xf32>
      "tileflow.store"(%acc, %dst, %pid0, %pid1)
          <{operandSegmentSizes = array<i32: 1, 1, 2, 0>}> :
          (tensor<16x16xf32>, memref<128x128xf16>, index, index) -> ()
      tileflow.yield
    }) : (index, index) -> ()
    return
  }
}

// CHECK: tileflow.kernel
// CHECK: ^bb0(%{{.*}}: index, %{{.*}}: index)
// CHECK: tileflow.load
// CHECK: tileflow.dot
// CHECK: tileflow.store
