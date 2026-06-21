"""AST frontend for TileLang-compatible TileFlow programs."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any

from tileflow.ir import IRBuilder, KernelIR
import logging


# FIXME: this is not robust. we should not assume user always import tileflow.language as T, and we should support more flexible patterns for symbol definition and buffer declaration.
# e.g. currently only `A: T.Tensor(...)` is recognized for buffer declaration, but users may also want to write `A = T.empty(...)` or even `A = T.Tensor(...)


@dataclass
class ParseContext:
    symbols: dict[str, Any]
    buffers: dict[str, dict[str, Any]] = field(default_factory=dict)
    returns: list[str] = field(default_factory=list)


def parse_jit_function(jit_fn, params: dict[str, Any] | None = None) -> KernelIR:
    params = params or {}
    try:
        source = textwrap.dedent(inspect.getsource(jit_fn.fn))
    except OSError as exc:
        raise RuntimeError(
            "TileFlow's TileLang-compatible frontend parses Python source with inspect.getsource. "
            "Define @tileflow.jit kernels in a .py file instead of an interactive stdin/eval block."
        ) from exc
    module = ast.parse(source)

    logging.debug("Parsing function %s with params %s", jit_fn.name, params)
    # logging.debug(("AST for function %s:\n%s", jit_fn.name, ast.dump(module, indent=4)))
    fn_node = next(node for node in module.body if isinstance(node, ast.FunctionDef))
    builder = IRBuilder(jit_fn.name)
    ctx = ParseContext(symbols=dict(params))

    for arg_index, arg in enumerate(fn_node.args.args):
        if arg.arg in params:
            continue
        # Shape/dtype may be supplied later by an in-body `A: T.Tensor(...)`.
        builder.argument(arg.arg, arg_index, "unknown", None)
        ctx.buffers[arg.arg] = {"kind": "arg", "dtype": "unknown", "shape": None}

    _parse_block(fn_node.body, builder, ctx)
    for name in ctx.returns:
        builder.emit("return_value", attrs={"value": name}, has_result=False)
    return builder.ir


def _parse_block(stmts: list[ast.stmt], builder: IRBuilder, ctx: ParseContext) -> None:
    for stmt in stmts:
        if isinstance(stmt, ast.Assign):
            _parse_assign(stmt, builder, ctx)
        elif isinstance(stmt, ast.AnnAssign):
            _parse_ann_assign(stmt, builder, ctx)
        elif isinstance(stmt, ast.With):
            _parse_with(stmt, builder, ctx)
        elif isinstance(stmt, ast.For):
            _parse_for(stmt, builder, ctx)
        elif isinstance(stmt, ast.Expr):
            _parse_expr_stmt(stmt.value, builder, ctx)
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None:
                ctx.returns.append(_expr(stmt.value, ctx))
        elif isinstance(stmt, ast.If):
            builder.emit("if", attrs={"test": _expr(stmt.test, ctx)}, has_result=False)
            _parse_block(stmt.body, builder, ctx)
            if stmt.orelse:
                builder.emit("else", has_result=False)
                _parse_block(stmt.orelse, builder, ctx)
            builder.emit("endif", has_result=False)
        elif isinstance(stmt, ast.Pass):
            continue
        else:
            builder.emit("unsupported_stmt", attrs={"ast": type(stmt).__name__}, has_result=False)


def _parse_assign(stmt: ast.Assign, builder: IRBuilder, ctx: ParseContext) -> None:
    if len(stmt.targets) != 1:
        builder.emit("unsupported_assign", attrs={"reason": "multi-target"}, has_result=False)
        return
    target = stmt.targets[0]
    value = stmt.value

    if _is_call(value, "T", "const"):
        # T.const for symbol definition
        assert isinstance(value, ast.Call)
        names = _literal_const_names(value)
        logging.debug("Defining symbols name: %s from T.const call", names)
        targets = _target_names(target)
        logging.debug("Defining targets name: %s from T.const call", targets)
        for name, target_name in zip(names, targets, strict=False):
            resolved = ctx.symbols.get(target_name, name)
            ctx.symbols[target_name] = resolved
            builder.emit("symbol", attrs={"name": target_name, "value": resolved}, has_result=False)
        return

    target_name = _target_name(target)
    if target_name and isinstance(value, ast.Call):
        callee = _callee(value)
        if callee in {
            "T.empty",
            "T.alloc_shared",
            "T.alloc_local",
            "T.alloc_fragment",
            "T.alloc_global",
            "T.alloc_var",
            "T.alloc_tmem",
        }:
            # assignment/allocation for buffers
            attrs = _call_attrs(value, ctx)
            attrs["name"] = target_name
            attrs["callee"] = callee
            ctx.buffers[target_name] = attrs
            builder.emit(_op_name_for_alloc(callee), attrs=attrs, has_result=False)
            return
        # the else case fall through to emit unsupported_assign

    if isinstance(target, ast.Subscript):
        # a[i] = ... style assignment
        builder.emit(
            "store",
            attrs={"target": _expr(target, ctx), "value": _expr(value, ctx)},
            has_result=False,
        )
        return

    if target_name:
        value_expr = _expr(value, ctx)
        ctx.symbols[target_name] = value_expr
        builder.emit("assign", attrs={"target": target_name, "value": value_expr}, has_result=False)
        return

    builder.emit("unsupported_assign", attrs={"target": _expr(target, ctx)}, has_result=False)


def _parse_ann_assign(stmt: ast.AnnAssign, builder: IRBuilder, ctx: ParseContext) -> None:
    name = _target_name(stmt.target)
    if not name:
        return
    if isinstance(stmt.annotation, ast.Call) and _callee(stmt.annotation) == "T.Tensor":
        attrs = _call_attrs(stmt.annotation, ctx)
        dtype = str(attrs.get("arg1", "unknown"))
        shape = attrs.get("arg0")
        ctx.buffers[name] = {"kind": "tensor", "shape": shape, "dtype": dtype}
        for arg in builder.ir.args:
            if arg.name == name:
                arg.dtype = dtype
                arg.rank = _rank_from_shape(shape)
                break
        builder.emit(
            "tensor_decl", attrs={"name": name, "shape": shape, "dtype": dtype}, has_result=False
        )
        return
    builder.emit(
        "unsupported_ann_assign",
        attrs={"target": name, "annotation": _expr(stmt.annotation, ctx)},
        has_result=False,
    )


def _parse_with(stmt: ast.With, builder: IRBuilder, ctx: ParseContext) -> None:
    if len(stmt.items) != 1:
        builder.emit("unsupported_with", attrs={"reason": "multi-item"}, has_result=False)
        return
    item = stmt.items[0]
    if isinstance(item.context_expr, ast.Call) and _callee(item.context_expr) == "T.Kernel":
        attrs = _call_attrs(item.context_expr, ctx)
        attrs["bindings"] = _target_names(item.optional_vars)
        builder.emit("kernel_launch", attrs=attrs, has_result=False)
        _parse_block(stmt.body, builder, ctx)
        builder.emit("kernel_end", has_result=False)
        return
    builder.emit(
        "unsupported_with", attrs={"context": _expr(item.context_expr, ctx)}, has_result=False
    )


def _parse_for(stmt: ast.For, builder: IRBuilder, ctx: ParseContext) -> None:
    loop_kind = "for"
    attrs = {"targets": _target_names(stmt.target), "iter": _expr(stmt.iter, ctx)}
    if isinstance(stmt.iter, ast.Call):
        callee = _callee(stmt.iter)
        attrs.update(_call_attrs(stmt.iter, ctx))
        if callee == "T.Pipelined":
            loop_kind = "pipelined_for"
        elif callee == "T.Parallel":
            loop_kind = "parallel_for"
        elif callee == "T.Sequential":
            loop_kind = "sequential_for"
        attrs["callee"] = callee
    builder.emit(loop_kind, attrs=attrs, has_result=False)
    _parse_block(stmt.body, builder, ctx)
    builder.emit(f"end_{loop_kind}", has_result=False)


def _parse_expr_stmt(expr: ast.expr, builder: IRBuilder, ctx: ParseContext) -> None:
    if isinstance(expr, ast.Call):
        callee = _callee(expr)
        if callee.startswith("T."):
            builder.emit(callee.removeprefix("T."), attrs=_call_attrs(expr, ctx), has_result=False)
            return
    builder.emit("expr", attrs={"value": _expr(expr, ctx)}, has_result=False)


def _call_attrs(call: ast.Call, ctx: ParseContext) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for index, arg in enumerate(call.args):
        attrs[f"arg{index}"] = _expr(arg, ctx)
    for keyword in call.keywords:
        if keyword.arg is not None:
            attrs[keyword.arg] = _expr(keyword.value, ctx)
    return attrs


def _op_name_for_alloc(callee: str) -> str:
    return {
        "T.empty": "empty",
        "T.alloc_shared": "alloc_shared",
        "T.alloc_local": "alloc_local",
        "T.alloc_fragment": "alloc_fragment",
        "T.alloc_global": "alloc_global",
        "T.alloc_var": "alloc_var",
        "T.alloc_tmem": "alloc_tmem",
    }[callee]


def _callee(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return f"{_callee(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ast.unparse(node)


def _is_call(node: ast.AST, root: str, attr: str) -> bool:
    return isinstance(node, ast.Call) and _callee(node) == f"{root}.{attr}"


def _literal_const_names(call: ast.Call) -> list[str]:
    if not call.args:
        return []
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return [name.strip() for name in first.value.split(",")]
    return []


def _target_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _target_names(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [elt.id for elt in node.elts if isinstance(elt, ast.Name)]
    return [ast.unparse(node)]


def _rank_from_shape(shape: Any) -> int | None:
    # Heuristic to determine rank from shape expression. This is not robust and can be improved by better shape parsing.
    # e.g. it can handle literal tuples/lists like (N, M) or [N, M], but not more complex expressions.
    if shape is None:
        return None
    text = str(shape)
    if text.startswith("(") and text.endswith(")"):
        body = text[1:-1].strip()
        if not body:
            return 0
        return len([part for part in body.split(",") if part.strip()])
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        if not body:
            return 0
        return len([part for part in body.split(",") if part.strip()])
    return 1


def _expr(node: ast.AST | None, ctx: ParseContext | None = None) -> str:
    if node is None:
        return ""
    if ctx is not None:
        node = _substitute_symbols(node, ctx.symbols)
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _substitute_symbols(node: ast.AST, symbols: dict[str, Any]) -> ast.AST:
    # e.g. turn `N` in `T.ceildiv(N, 128)` into a constant if `N=1024` is supplied as a parameter.
    expr = ast.parse(ast.unparse(node), mode="eval").body

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id not in symbols:
                return node
            value = symbols[node.id]
            if isinstance(value, (int, float)):
                return ast.copy_location(ast.Constant(value=value), node)
            if isinstance(value, str):
                try:
                    replacement = ast.parse(value, mode="eval").body
                except SyntaxError:
                    return node
                return ast.copy_location(replacement, node)
            return node

    return ast.fix_missing_locations(Substitute().visit(expr))
