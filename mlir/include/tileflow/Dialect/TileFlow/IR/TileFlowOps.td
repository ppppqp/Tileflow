#ifndef TILEFLOW_OPS_TD
#define TILEFLOW_OPS_TD

include "tileflow/Dialect/TileFlow/IR/TileFlowDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TileFlow_Op<string mnemonic, list<Trait> traits = []>
    : Op<TileFlow_Dialect, mnemonic, traits>;

class TileFlow_MetadataOp<string mnemonic>
    : TileFlow_Op<mnemonic, [Pure]> {
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def TileFlow_TensorDeclOp : TileFlow_MetadataOp<"tensor_decl">;
def TileFlow_EmptyOp : TileFlow_MetadataOp<"empty">;
def TileFlow_AllocSharedOp : TileFlow_MetadataOp<"alloc_shared">;
def TileFlow_AllocLocalOp : TileFlow_MetadataOp<"alloc_local">;
def TileFlow_AllocFragmentOp : TileFlow_MetadataOp<"alloc_fragment">;
def TileFlow_AllocGlobalOp : TileFlow_MetadataOp<"alloc_global">;
def TileFlow_AllocVarOp : TileFlow_MetadataOp<"alloc_var">;
def TileFlow_AllocTmemOp : TileFlow_MetadataOp<"alloc_tmem">;
def TileFlow_KernelLaunchOp : TileFlow_MetadataOp<"kernel_launch">;
def TileFlow_KernelEndOp : TileFlow_MetadataOp<"kernel_end">;
def TileFlow_ForOp : TileFlow_MetadataOp<"for">;
def TileFlow_EndForOp : TileFlow_MetadataOp<"end_for">;
def TileFlow_ParallelForOp : TileFlow_MetadataOp<"parallel_for">;
def TileFlow_EndParallelForOp : TileFlow_MetadataOp<"end_parallel_for">;
def TileFlow_SerialForOp : TileFlow_MetadataOp<"serial_for">;
def TileFlow_EndSerialForOp : TileFlow_MetadataOp<"end_serial_for">;
def TileFlow_PipelinedForOp : TileFlow_MetadataOp<"pipelined_for">;
def TileFlow_EndPipelinedForOp : TileFlow_MetadataOp<"end_pipelined_for">;
def TileFlow_AssignOp : TileFlow_MetadataOp<"assign">;
def TileFlow_StoreOp : TileFlow_MetadataOp<"store">;
def TileFlow_CopyOp : TileFlow_MetadataOp<"copy">;
def TileFlow_AsyncCopyOp : TileFlow_MetadataOp<"async_copy">;
def TileFlow_GemmOp : TileFlow_MetadataOp<"gemm">;
def TileFlow_ClearOp : TileFlow_MetadataOp<"clear">;
def TileFlow_SymbolOp : TileFlow_MetadataOp<"symbol">;
def TileFlow_ReturnValueOp : TileFlow_MetadataOp<"return_value">;
def TileFlow_IfOp : TileFlow_MetadataOp<"if">;
def TileFlow_ElseOp : TileFlow_MetadataOp<"else">;
def TileFlow_EndIfOp : TileFlow_MetadataOp<"endif">;
def TileFlow_ExprOp : TileFlow_MetadataOp<"expr">;
def TileFlow_UnsupportedStmtOp : TileFlow_MetadataOp<"unsupported_stmt">;
def TileFlow_UnsupportedAssignOp : TileFlow_MetadataOp<"unsupported_assign">;
def TileFlow_UnsupportedAnnAssignOp : TileFlow_MetadataOp<"unsupported_ann_assign">;
def TileFlow_UnsupportedWithOp : TileFlow_MetadataOp<"unsupported_with">;

#endif // TILEFLOW_OPS_TD
