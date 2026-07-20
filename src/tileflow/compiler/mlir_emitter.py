"""Direct emission of upstream MLIR operations from the TileFlow frontend IR."""

from __future__ import annotations

from typing import Any

from tileflow.compiler.mlir_types import _load_ir, to_mlir_type
from tileflow.language.ir import (
    BoolType,
    BufferType,
    FloatType,
    IndexType,
    IntType,
    KernelIR,
    OpName,
    Operation,
    Region,
    TileType,
    Value,
)


class MLIREmissionError(RuntimeError):
    pass


def create_mlir_context() -> Any:
    """Create an MLIR context with the native TileFlow dialect registered."""

    ir = _load_ir()
    try:
        from tileflow._mlir import tileflow_mlir
    except ImportError as exc:
        raise MLIREmissionError(
            "the native TileFlow MLIR extension is unavailable; build the tileflow_mlir target"
        ) from exc
    registry = ir.DialectRegistry()
    tileflow_mlir.register_dialects(registry)
    context = ir.Context()
    context.append_dialect_registry(registry)
    context.load_all_available_dialects()
    context.allow_unregistered_dialects = False
    return context


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
            OpName.CEILDIV: self._emit_ceildiv,
            OpName.SELECT: self._emit_select,
            OpName.ALLOC: self._emit_alloc,
            OpName.TILE_EMPTY: self._emit_tile_empty,
            OpName.LOAD: self._emit_load,
            OpName.STORE: self._emit_store,
            OpName.SERIAL_FOR: self._emit_serial_for,
            OpName.PIPELINED_FOR: self._emit_pipelined_for,
            OpName.PARALLEL: self._emit_parallel,
            OpName.KERNEL: self._emit_kernel,
            OpName.IF: self._emit_if,
            OpName.WHILE: self._emit_while,
        }
        try:
            handler = handlers[op.name]
        except KeyError as exc:
            raise MLIREmissionError(
                f"direct MLIR emission is not implemented for {op.name}"
            ) from exc
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

    def _coerce(self, value: Value, target: Any) -> Any:
        """Coerce a SSA value to type required by MLIR operation consuming it.
        We should maybe emit CAST before MLIR emission.
        """
        source = value.type
        emitted_value = self._lookup(value)
        if source == target:
            return emitted_value
        if isinstance(source, FloatType) and isinstance(target, FloatType):
            name = "arith.extf" if source.bits < target.bits else "arith.truncf"
        elif isinstance(source, (IndexType, IntType, BoolType)) and isinstance(
            target, (IndexType, IntType, BoolType)
        ):
            if isinstance(source, IndexType) != isinstance(target, IndexType):
                name = "arith.index_cast"
            elif getattr(source, "bits", 64) < getattr(target, "bits", 64):
                name = "arith.extsi"
            else:
                name = "arith.trunci"
        elif isinstance(source, IndexType) and isinstance(target, FloatType):
            i64 = self.ir.IntegerType.get_signless(64)
            integer = self.ir.Operation.create(
                "arith.index_cast", operands=[emitted_value], results=[i64]
            ).results[0]
            return self.ir.Operation.create(
                "arith.sitofp", operands=[integer], results=[to_mlir_type(target)]
            ).results[0]
        elif isinstance(source, (IntType, BoolType)) and isinstance(target, FloatType):
            name = "arith.sitofp"
        elif isinstance(source, FloatType) and isinstance(target, IndexType):
            i64 = self.ir.IntegerType.get_signless(64)
            integer = self.ir.Operation.create(
                "arith.fptosi", operands=[emitted_value], results=[i64]
            ).results[0]
            return self.ir.Operation.create(
                "arith.index_cast", operands=[integer], results=[to_mlir_type(target)]
            ).results[0]
        elif isinstance(source, FloatType) and isinstance(target, (IntType, BoolType)):
            name = "arith.fptosi"
        else:
            raise MLIREmissionError(f"cannot coerce {source} to {target}")
        return self.ir.Operation.create(
            name, operands=[emitted_value], results=[to_mlir_type(target)]
        ).results[0]

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
            name = (
                "arith.index_cast"
                if isinstance(source, IndexType) != isinstance(target, IndexType)
                else "arith.extsi"
            )
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
        operands = [self._coerce(value, op.results[0].type) for value in op.operands]
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
            operands=[self._coerce(value, op.operands[0].type) for value in op.operands],
            results=[to_mlir_type(BoolType())],
            attributes={"predicate": predicate},
        )
        self._bind_results(op, emitted)

    def _emit_select(self, op: Operation) -> None:
        target = op.results[0].type
        emitted = self.ir.Operation.create(
            "arith.select",
            operands=[
                self._lookup(op.operands[0]),
                self._coerce(op.operands[1], target),
                self._coerce(op.operands[2], target),
            ],
            results=[to_mlir_type(target)],
        )
        self._bind_results(op, emitted)

    def _emit_ceildiv(self, op: Operation) -> None:
        emitted = self.ir.Operation.create(
            "arith.ceildivui",
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
        name = "tileflow.load" if isinstance(op.results[0].type, TileType) else "memref.load"
        emitted = self.ir.Operation.create(
            name,
            operands=[self._lookup(value) for value in op.operands],
            results=[to_mlir_type(op.results[0].type)],
        )
        self._bind_results(op, emitted)

    def _emit_store(self, op: Operation) -> None:
        name = "tileflow.store" if isinstance(op.operands[0].type, TileType) else "memref.store"
        value = self._lookup(op.operands[0])
        target_type = op.operands[1].type
        if isinstance(target_type, BufferType):
            value = self._coerce(op.operands[0], target_type.element_type)
        self.ir.Operation.create(
            name,
            operands=[value, *[self._lookup(item) for item in op.operands[1:]]],
        )

    def _i64_attr(self, value: int) -> Any:
        return self.ir.IntegerAttr.get(self.ir.IntegerType.get_signless(64), value)

    def _emit_kernel(self, op: Operation) -> None:
        rank = op.attrs.get("rank")
        if not isinstance(rank, int) or rank != len(op.operands):
            raise MLIREmissionError("kernel rank must match its grid extent count")
        attributes = {"grid_rank": self._i64_attr(rank)}
        if "threads" in op.attrs:
            attributes["threads"] = self._i64_attr(op.attrs["threads"])
        if "cluster_dims" in op.attrs:
            attributes["cluster_dims"] = self.ir.DenseI64ArrayAttr.get(
                list(op.attrs["cluster_dims"])
            )
        emitted = self.ir.Operation.create(
            "tileflow.kernel",
            operands=[self._lookup(value) for value in op.operands],
            attributes=attributes,
            regions=1,
        )
        body = emitted.regions[0].blocks.append(*[to_mlir_type(IndexType()) for _ in range(rank)])
        self._emit_structured_body(op, body, "tileflow.yield")

    def _emit_parallel(self, op: Operation) -> None:
        rank = op.attrs.get("rank")
        if not isinstance(rank, int) or rank < 1 or len(op.operands) != 3 * rank:
            raise MLIREmissionError("parallel loop requires equally sized bound and step lists")
        attributes = {"operandSegmentSizes": self.ir.DenseI32ArrayAttr.get([rank, rank, rank])}
        mapping = op.attrs.get("mapping")
        if mapping is not None:
            if not isinstance(mapping, dict):
                raise MLIREmissionError("parallel mapping must be a dictionary")
            attributes["mapping"] = self.ir.DictAttr.get(
                {key: self.ir.StringAttr.get(str(value)) for key, value in mapping.items()}
            )
        emitted = self.ir.Operation.create(
            "tileflow.parallel",
            operands=[self._lookup(value) for value in op.operands],
            attributes=attributes,
            regions=1,
        )
        body = emitted.regions[0].blocks.append(*[to_mlir_type(IndexType()) for _ in range(rank)])
        self._emit_structured_body(op, body, "tileflow.yield")

    def _emit_structured_body(self, op: Operation, body: Any, terminator: str) -> None:
        frontend_body = op.regions[0].entry
        if len(frontend_body.args) != len(body.arguments):
            raise MLIREmissionError(f"{op.name} region argument count does not match its rank")
        for frontend, mlir_value in zip(frontend_body.args, body.arguments, strict=True):
            self.values[frontend] = mlir_value
        with self.ir.InsertionPoint(body):
            self._emit_region_contents(op.regions[0])
            self.ir.Operation.create(terminator)

    def _emit_serial_for(self, op: Operation) -> None:
        num_iter_args = len(op.operands) - 3
        if op.attrs.get("rank") != 1 or len(op.operands) != 3 + num_iter_args:
            raise MLIREmissionError("scf.for emission requires a rank-1 serial loop")
        emitted = self.ir.Operation.create(
            "scf.for",
            operands=[self._lookup(value) for value in op.operands],
            results=[to_mlir_type(result.type) for result in op.results],
            regions=1,
        )
        body = emitted.regions[0].blocks.append(
            to_mlir_type(IndexType()),
            *[to_mlir_type(value.type) for value in op.operands[3:]],
        )
        frontend_body = op.regions[0].entry
        if len(frontend_body.args) != len(body.arguments):
            raise MLIREmissionError("serial loop body arguments do not match its iter_args")
        for frontend, mlir_value in zip(frontend_body.args, body.arguments, strict=True):
            self.values[frontend] = mlir_value
        with self.ir.InsertionPoint(body):
            for nested in frontend_body.ops:
                self._emit_op(nested)
            yielded = [] if frontend_body.terminator is None else [
                self._lookup(value) for value in frontend_body.terminator.operands
            ]
            self.ir.Operation.create("scf.yield", operands=yielded)
        self._bind_results(op, emitted)

    def _emit_pipelined_for(self, op: Operation) -> None:
        self._emit_serial_for(op)

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
            if (
                terminator is None
                or terminator.name != OpName.YIELD
                or len(terminator.operands) != 1
            ):
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

    context = context or create_mlir_context()
    return MLIREmitter(context).emit(kernel)
