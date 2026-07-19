"""Direct emission of upstream MLIR operations from the TileFlow frontend IR."""

from __future__ import annotations

from typing import Any

from tileflow.compiler.mlir_types import _load_ir, to_mlir_type
from tileflow.language.ir import (
    BoolType,
    FloatType,
    IndexType,
    IntType,
    KernelIR,
    OpName,
    Operation,
    Region,
    Value,
)


class MLIREmissionError(RuntimeError):
    pass


class MLIREmitter:
    """Lower the upstream-compatible subset of frontend IR to real MLIR."""

    def __init__(self, context: Any):
        self.ir = _load_ir()
        self.context = context
        self.values: dict[Value, Any] = {}

    def emit(self, kernel: KernelIR) -> Any:
        ir = self.ir
        with self.context, ir.Location.unknown():
            module = ir.Module.create()
            signature = [*kernel.params, *kernel.outputs]
            argument_types = [to_mlir_type(item.value.type) for item in signature]
            function_type = ir.FunctionType.get(argument_types, [])
            function = ir.Operation.create(
                "func.func",
                attributes={
                    "sym_name": ir.StringAttr.get(kernel.name),
                    "function_type": ir.TypeAttr.get(function_type),
                },
                regions=1,
            )
            module.body.append(function)
            entry = function.regions[0].blocks.append(*argument_types)
            for item, argument in zip(signature, entry.arguments, strict=True):
                self.values[item.value] = argument

            with ir.InsertionPoint(entry):
                self._emit_region_contents(kernel.body)
                ir.Operation.create("func.return")
            return module

    def _emit_region_contents(self, region: Region) -> None:
        block = region.entry
        for op in block.ops:
            self._emit_op(op)
        if block.terminator is not None and block.terminator.name != OpName.RETURN:
            self._emit_op(block.terminator)

    def _emit_op(self, op: Operation) -> None:
        handlers = {
            OpName.CONST: self._emit_constant,
            OpName.CAST: self._emit_cast,
            OpName.ADD: self._emit_binary,
            OpName.SUB: self._emit_binary,
            OpName.MUL: self._emit_binary,
            OpName.DIV: self._emit_binary,
            OpName.FLOORDIV: self._emit_binary,
            OpName.MOD: self._emit_binary,
            OpName.NEG: self._emit_binary,
            OpName.BITAND: self._emit_binary,
            OpName.BITOR: self._emit_binary,
            OpName.BITXOR: self._emit_binary,
            OpName.SHL: self._emit_binary,
            OpName.SHR: self._emit_binary,
            OpName.EQ: self._emit_compare,
            OpName.NE: self._emit_compare,
            OpName.LT: self._emit_compare,
            OpName.LE: self._emit_compare,
            OpName.GT: self._emit_compare,
            OpName.GE: self._emit_compare,
            OpName.AND: self._emit_binary,
            OpName.OR: self._emit_binary,
            OpName.NOT: self._emit_binary,
            OpName.SELECT: self._emit_select,
            OpName.ALLOC: self._emit_alloc,
            OpName.TILE_EMPTY: self._emit_tile_empty,
            OpName.LOAD: self._emit_load,
            OpName.STORE: self._emit_store,
            OpName.SERIAL_FOR: self._emit_serial_for,
            OpName.IF: self._emit_if,
            OpName.WHILE: self._emit_while,
        }
        try:
            handler = handlers[op.name]
        except KeyError as exc:
            raise MLIREmissionError(f"direct MLIR emission is not implemented for {op.name}") from exc
        handler(op)

    def _lookup(self, value: Value) -> Any:
        try:
            return self.values[value]
        except KeyError as exc:
            raise MLIREmissionError(f"frontend value {value.ir_name} has not been emitted") from exc

    def _bind_results(self, op: Operation, emitted: Any) -> None:
        results = list(emitted.results)
        if len(results) != len(op.results):
            raise MLIREmissionError(
                f"{op.name} produced {len(results)} MLIR results for {len(op.results)} frontend results"
            )
        for frontend, mlir_value in zip(op.results, results, strict=True):
            self.values[frontend] = mlir_value

    def _emit_constant(self, op: Operation) -> None:
        ir = self.ir
        result_type = op.results[0].type
        mlir_type = to_mlir_type(result_type)
        value = op.attrs["value"]
        if isinstance(result_type, FloatType):
            attribute = ir.FloatAttr.get(mlir_type, value)
        else:
            attribute = ir.IntegerAttr.get(mlir_type, value)
        emitted = ir.Operation.create(
            "arith.constant", results=[mlir_type], attributes={"value": attribute}
        )
        self._bind_results(op, emitted)

    def _emit_cast(self, op: Operation) -> None:
        source = op.operands[0].type
        target = op.results[0].type
        if source == target:
            self.values[op.results[0]] = self._lookup(op.operands[0])
            return
        if isinstance(source, FloatType) and isinstance(target, FloatType):
            name = "arith.extf" if source.bits < target.bits else "arith.truncf"
        elif isinstance(source, (IndexType, IntType, BoolType)) and isinstance(
            target, (IndexType, IntType, BoolType)
        ):
            name = "arith.index_cast" if isinstance(source, IndexType) != isinstance(
                target, IndexType
            ) else "arith.extsi"
        elif isinstance(source, (IndexType, IntType, BoolType)) and isinstance(target, FloatType):
            name = "arith.sitofp"
        elif isinstance(source, FloatType) and isinstance(target, (IndexType, IntType, BoolType)):
            name = "arith.fptosi"
        else:
            raise MLIREmissionError(f"unsupported cast from {source} to {target}")
        emitted = self.ir.Operation.create(
            name,
            operands=[self._lookup(op.operands[0])],
            results=[to_mlir_type(target)],
        )
        self._bind_results(op, emitted)

    def _emit_binary(self, op: Operation) -> None:
        type_ = op.operands[0].type
        is_float = isinstance(type_, FloatType)
        float_names = {
            OpName.ADD: "arith.addf",
            OpName.SUB: "arith.subf",
            OpName.MUL: "arith.mulf",
            OpName.DIV: "arith.divf",
            OpName.MOD: "arith.remf",
            OpName.NEG: "arith.negf",
        }
        integer_names = {
            OpName.ADD: "arith.addi",
            OpName.SUB: "arith.subi",
            OpName.MUL: "arith.muli",
            OpName.DIV: "arith.divsi",
            OpName.FLOORDIV: "arith.floordivsi",
            OpName.MOD: "arith.remsi",
            OpName.BITAND: "arith.andi",
            OpName.BITOR: "arith.ori",
            OpName.BITXOR: "arith.xori",
            OpName.SHL: "arith.shli",
            OpName.SHR: "arith.shrsi",
            OpName.AND: "arith.andi",
            OpName.OR: "arith.ori",
            OpName.NOT: "arith.xori",
        }
        names = float_names if is_float else integer_names
        try:
            name = names[op.name]
        except KeyError as exc:
            raise MLIREmissionError(f"unsupported {type_} operation {op.name}") from exc
        operands = [self._lookup(value) for value in op.operands]
        if op.name == OpName.NOT:
            true_attr = self.ir.IntegerAttr.get(to_mlir_type(BoolType()), 1)
            true_op = self.ir.Operation.create(
                "arith.constant",
                results=[to_mlir_type(BoolType())],
                attributes={"value": true_attr},
            )
            operands.append(true_op.results[0])
        emitted = self.ir.Operation.create(
            name, operands=operands, results=[to_mlir_type(op.results[0].type)]
        )
        self._bind_results(op, emitted)

    def _emit_compare(self, op: Operation) -> None:
        is_float = isinstance(op.operands[0].type, FloatType)
        predicates = {
            OpName.EQ: (1, 0),
            OpName.NE: (6, 1),
            OpName.LT: (4, 2),
            OpName.LE: (5, 3),
            OpName.GT: (2, 4),
            OpName.GE: (3, 5),
        }
        float_predicate, integer_predicate = predicates[op.name]
        predicate = self.ir.IntegerAttr.get(
            self.ir.IntegerType.get_signless(64),
            float_predicate if is_float else integer_predicate,
        )
        emitted = self.ir.Operation.create(
            "arith.cmpf" if is_float else "arith.cmpi",
            operands=[self._lookup(value) for value in op.operands],
            results=[to_mlir_type(BoolType())],
            attributes={"predicate": predicate},
        )
        self._bind_results(op, emitted)

    def _emit_select(self, op: Operation) -> None:
        emitted = self.ir.Operation.create(
            "arith.select",
            operands=[self._lookup(value) for value in op.operands],
            results=[to_mlir_type(op.results[0].type)],
        )
        self._bind_results(op, emitted)

    def _emit_alloc(self, op: Operation) -> None:
        emitted = self.ir.Operation.create(
            "memref.alloca",
            results=[to_mlir_type(op.results[0].type)],
        )
        self._bind_results(op, emitted)

    def _emit_tile_empty(self, op: Operation) -> None:
        emitted = self.ir.Operation.create(
            "tensor.empty",
            results=[to_mlir_type(op.results[0].type)],
        )
        self._bind_results(op, emitted)

    def _emit_load(self, op: Operation) -> None:
        emitted = self.ir.Operation.create(
            "memref.load",
            operands=[self._lookup(value) for value in op.operands],
            results=[to_mlir_type(op.results[0].type)],
        )
        self._bind_results(op, emitted)

    def _emit_store(self, op: Operation) -> None:
        self.ir.Operation.create(
            "memref.store", operands=[self._lookup(value) for value in op.operands]
        )

    def _emit_serial_for(self, op: Operation) -> None:
        if op.attrs.get("rank") != 1 or len(op.operands) != 3:
            raise MLIREmissionError("scf.for emission requires a rank-1 serial loop")
        emitted = self.ir.Operation.create(
            "scf.for",
            operands=[self._lookup(value) for value in op.operands],
            regions=1,
        )
        body = emitted.regions[0].blocks.append(to_mlir_type(IndexType()))
        frontend_body = op.regions[0].entry
        self.values[frontend_body.args[0]] = body.arguments[0]
        with self.ir.InsertionPoint(body):
            self._emit_region_contents(op.regions[0])
            self.ir.Operation.create("scf.yield")

    def _emit_if(self, op: Operation) -> None:
        emitted = self.ir.Operation.create(
            "scf.if",
            operands=[self._lookup(op.operands[0])],
            regions=2,
        )
        for frontend_region, mlir_region in zip(op.regions, emitted.regions, strict=False):
            block = mlir_region.blocks.append()
            with self.ir.InsertionPoint(block):
                self._emit_region_contents(frontend_region)
                self.ir.Operation.create("scf.yield")

    def _emit_while(self, op: Operation) -> None:
        if len(op.regions) != 2:
            raise MLIREmissionError("scf.while emission requires condition and body regions")
        emitted = self.ir.Operation.create("scf.while", regions=2)

        condition_region = op.regions[0]
        condition_block = emitted.regions[0].blocks.append()
        with self.ir.InsertionPoint(condition_block):
            for nested in condition_region.entry.ops:
                self._emit_op(nested)
            terminator = condition_region.entry.terminator
            if terminator is None or terminator.name != OpName.YIELD or len(terminator.operands) != 1:
                raise MLIREmissionError("while condition must yield exactly one condition")
            self.ir.Operation.create(
                "scf.condition", operands=[self._lookup(terminator.operands[0])]
            )

        body_region = op.regions[1]
        body_block = emitted.regions[1].blocks.append()
        with self.ir.InsertionPoint(body_block):
            self._emit_region_contents(body_region)
            self.ir.Operation.create("scf.yield")


def emit_upstream_mlir(kernel: KernelIR, *, context: Any | None = None) -> Any:
    """Emit the upstream-compatible subset into a real ``mlir.ir.Module``."""

    ir = _load_ir()
    context = context or ir.Context()
    context.allow_unregistered_dialects = False
    return MLIREmitter(context).emit(kernel)
