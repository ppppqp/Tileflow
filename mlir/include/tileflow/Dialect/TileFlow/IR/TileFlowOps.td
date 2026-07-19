#ifndef TILEFLOW_OPS_TD
#define TILEFLOW_OPS_TD

include "tileflow/Dialect/TileFlow/IR/TileFlowDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TileFlow_Op<string mnemonic, list<Trait> traits = []>
    : Op<TileFlow_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Structured kernel and execution operations
//===----------------------------------------------------------------------===//

def TileFlow_YieldOp : TileFlow_Op<"yield", [Pure, Terminator]> {
  let summary = "terminate a TileFlow region";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

def TileFlow_KernelOp : TileFlow_Op<"kernel", [
    SingleBlockImplicitTerminator<"YieldOp">,
    RecursiveMemoryEffects
  ]> {
  let summary = "logical TileFlow GPU kernel";
  let description = [{
    Defines the body of one logical GPU kernel. Operands are its grid extents,
    and the entry block exposes the corresponding logical program identifiers
    as index arguments. Values defined outside the operation remain visible in
    the body. Block mapping is materialized by later GPU lowering passes.
  }];
  let arguments = (ins Variadic<Index>:$grid,
                       I64Attr:$grid_rank,
                       OptionalAttr<I64Attr>:$threads,
                       OptionalAttr<DenseI64ArrayAttr>:$cluster_dims);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def TileFlow_ProgramIdOp : TileFlow_Op<"program_id", [Pure]> {
  let summary = "logical program identifier for one grid dimension";
  let arguments = (ins I64Attr:$axis);
  let results = (outs Index:$result);
  let assemblyFormat = "$axis attr-dict `:` type($result)";
}

def TileFlow_ParallelOp : TileFlow_Op<"parallel", [
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">,
    RecursiveMemoryEffects
  ]> {
  let summary = "parallel iteration with unresolved GPU mapping";
  let description = [{
    Represents one or more logical parallel dimensions. Each lower bound,
    upper bound, and step list has the same length. The body entry block has
    one index argument per dimension. Layout and GPU mapping passes decide how
    these dimensions map to CTAs, warps, and lanes.
  }];
  let arguments = (ins Variadic<Index>:$lower_bounds,
                       Variadic<Index>:$upper_bounds,
                       Variadic<Index>:$steps,
                       OptionalAttr<DictionaryAttr>:$mapping);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Logical tile operations
//===----------------------------------------------------------------------===//

def TileFlow_LoadOp : TileFlow_Op<"load", [AttrSizedOperandSegments]> {
  let summary = "load a logical tile from an addressable buffer";
  let arguments = (ins Arg<AnyMemRef, "source buffer", [MemRead]>:$source,
                       Variadic<Index>:$indices,
                       Optional<AnyType>:$mask,
                       Optional<AnyType>:$other);
  let results = (outs AnyRankedTensor:$result);
}

def TileFlow_StoreOp : TileFlow_Op<"store", [AttrSizedOperandSegments]> {
  let summary = "store a logical tile to an addressable buffer";
  let arguments = (ins AnyRankedTensor:$value,
                       Arg<AnyMemRef, "destination buffer", [MemWrite]>:$target,
                       Variadic<Index>:$indices,
                       Optional<AnyType>:$mask);
}

def TileFlow_BroadcastOp : TileFlow_Op<"broadcast", [Pure]> {
  let summary = "broadcast a logical tile to a new shape";
  let arguments = (ins AnyRankedTensor:$source);
  let results = (outs AnyRankedTensor:$result);
}

def TileFlow_TransposeOp : TileFlow_Op<"transpose", [Pure]> {
  let summary = "permute logical tile dimensions";
  let arguments = (ins AnyRankedTensor:$source,
                       DenseI64ArrayAttr:$permutation);
  let results = (outs AnyRankedTensor:$result);
}

def TileFlow_ReduceOp : TileFlow_Op<"reduce", [Pure]> {
  let summary = "reduce one or more logical tile dimensions";
  let arguments = (ins AnyRankedTensor:$source,
                       StrAttr:$kind,
                       DenseI64ArrayAttr:$axes);
  let results = (outs AnyRankedTensor:$result);
}

def TileFlow_DotOp : TileFlow_Op<"dot", [Pure]> {
  let summary = "logical matrix multiply-accumulate";
  let arguments = (ins AnyRankedTensor:$lhs,
                       AnyRankedTensor:$rhs,
                       AnyRankedTensor:$accumulator);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def TileFlow_ConvertLayoutOp : TileFlow_Op<"convert_layout", [Pure]> {
  let summary = "convert between explicitly assigned tile layouts";
  let arguments = (ins AnyRankedTensor:$source);
  let results = (outs AnyRankedTensor:$result);
}

//===----------------------------------------------------------------------===//
// Addressable device storage
//===----------------------------------------------------------------------===//

def TileFlow_SharedAllocOp : TileFlow_Op<"shared_alloc", [
    MemoryEffects<[MemAlloc<DefaultResource>]>
  ]> {
  let summary = "allocate workgroup-shared storage";
  let arguments = (ins Variadic<Index>:$dynamic_sizes,
                       OptionalAttr<I64Attr>:$alignment);
  let results = (outs AnyMemRef:$result);
}

#endif // TILEFLOW_OPS_TD
