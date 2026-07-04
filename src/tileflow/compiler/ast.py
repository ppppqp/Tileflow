from __future__ import annotations
import ast
from collections.abc import Callable
from dataclasses import dataclass
from email.quoprimime import quote
import inspect
from typing import Any, Sequence, cast
from tileflow.dsl.ir import KernelIR
from tileflow.dsl.jit import JitFunction
import textwrap

"""
AST Rewrite to builder pattern
"""


def parse_jit_function(func: Callable, params: dict) -> JitFunction:
    """Parse a JIT function into IR builder pattern.

    Args:
        kernel: The JIT function to parse.
        params: A dictionary of parameters to pass to the kernel.

    Returns:
        A JitFunction object representing the parsed kernel.
    """
    # Implementation of parsing logic goes here
    # This is a placeholder for the actual parsing logic

    pass


def mutate(func: Callable):
    # get AST
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    # TODO: closures here


class DSLMutator(ast.NodeTransformer):
    def __init__(self):
        self.tmp_counter = 0

    def get_tmp(self) -> str:
        # temporary variable name generator
        name = f"__{self.tmp_counter}"
        self.tmp_counter += 1
        return name

    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        br = self.get_tmp()
        if len(node.orelse) == 0:
            return quote(
                f"""
for {br} in __tb.ctx_if(cond):
    for _ in __tb.ctx_then({br}):
        pass
    for _ in __tb.ctx_else({br}):
        pass
                """,
                cond=node.test,
                passes=[node.body, node.orelse],
                span=node,
            )

    def visit_Expr(self, node: ast.Expr):
        node = self.generic_visit(node)
        return quote("__tb.eval(value)", value=node.value, span=node)

    def visit_For(self, node: ast.For):
        node = self.generic_visit(node)
        tmp = self.get_tmp()
        var = ast.Name(tmp, ctx=ast.Load())
        ast_set_span(var, ast_get_span(node.target))
        stmts = self._emit_assign_target(node.target, var)
        return quote(
            f"""
for {tmp} in __tb.ctx_for(range):
    pass
            """,
            range=node.iter,
            passes=list(stmts) + node.body,
            span=node,
        )

    def _emit_assign_target(
        self, target: ast.expr, value: ast.expr, annot: ast.expr = None
    ) -> Sequence[ast.AST]:
        """
        Emit assignment statements for a given target and value.
        This function handles different types of assignment targets, including
        simple names, tuples, and lists. It recursively generates assignment statements for each element in the target if it is a tuple or list.
        """
        # TODO: annotate
        if isinstance(target, ast.Name):
            return quote(
                f"name = __tb.bind('{target.id}', value)", name=target, value=value, span=target
            )
        elif isinstance(target, ast.Attribute):
            s = ast.unparse(target)
            raise NotImplementedError(f"Attribute assignment not implemented: {s}")
        elif isinstance(target, ast.Subscript):
            return quote(
                "__tb.assign_slice(lval, slice, value)",
                lval=target.value,
                slice=target.slice,
                value=value,
                span=target,
            )
        else:
            # tuple
            # example: a, b = f()
            unpacked = []

            def _visit_target(target: ast.expr) -> ast.expr:
                if isinstance(target, (ast.Name, ast.Subscript)):
                    tmp = self.get_tmp()
                    unpacked.append((tmp, target))
                    res = ast.Name(id=tmp, ctx=target.ctx)
                    ast_set_span(res, ast_get_span(target))
                    return res
                elif isinstance(target, ast.Tuple):
                    elts = [_visit_target(elt) for elt in target.elts]
                    res = ast.Tuple(elts=elts, ctx=target.ctx)
                    ast_set_span(res, ast_get_span(target))
                    return res
                else:
                    s = ast.unparse(target)
                    raise NotImplementedError(f"Unsupported assignment target: {s}")

            unpack_stmt = ast.Assign(
                targets=[_visit_target(target)],
                value=quote_expr("__tb.unwrap_value(rval)", rval=value, span=value),
            )
            ast_set_span(unpack_stmt, ast_get_span(target))
            stmts = [unpack_stmt]
            bind_lvals = []
            bind_rvals = []

            def flush_binds():
                if bind_lvals:
                    stmts.append(
                        cast(
                            ast.Assign,
                            quote1(
                                f"{', '.join(bind_lvals)}, = {', '.join(bind_rvals)}, ", span=target
                            ),
                        )
                    )
                    bind_lvals.clear()
                    bind_rvals.clear()

            # to support swap like semantics
            for tmp, _ in unpacked:
                # tmp_0, tmp_1 = a ,b
                # tmp_0 = __tb.bind("_", tmp_0)
                # tmp_1 = __tb.bind("_", tmp_1)
                # "_" is a special name that means "discard the value"
                bind_lvals.append(tmp)
                bind_rvals.append(f'__tb.bind("_", {tmp})')
            flush_binds()

            for tmp, target in unpacked:
                # a = __tb.bind("a", tmp_0)
                # b = __tb.bind("b", tmp_1)
                if isinstance(target, ast.Name):
                    stmts.append(
                        cast(
                            ast.Assign,
                            quote1(f"{target.id} = __tb.bind('{target.id}', {tmp})", span=target),
                        )
                    )
                elif isinstance(target, ast.Subscript):
                    stmts.append(
                        cast(
                            ast.Assign,
                            quote1(
                                f"__tb.assign_slice({target.value}, {target.slice}, {tmp})",
                                lval=target.value,
                                slice=target.slice,
                                span=target,
                            ),
                        )
                    )
                else:
                    s = ast.unparse(target)
                    raise NotImplementedError(f"Unsupported assignment target: {s}")

            flush_binds()
            return stmts

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        rval = node.value
        if len(node.targets) == 1:
            return self._emit_assign_target(node.targets[0], rval)
        else:
            # x = y, in AST-wise
            # x = ast.Name(id="x", ctx=ast.Store())
            # y = ast.Name(id="y", ctx=ast.Load())
            tmp_name = self.get_tmp()
            tmp_store = ast.Name(tmp_name, ctx=ast.Store())
            tmp_load = ast.Name(tmp_name, ctx=ast.Load())
            ast_set_span(tmp_store, ast_get_span(node.targets[0]))
            ast_set_span(tmp_load, ast_get_span(node.targets[0]))
            stmt = list(self._emit_assign_target(tmp_store, rval))
            for target in node.targets:
                stmt.extend(self._emit_assign_target(target, tmp_load))
            return stmt


"""
span: for source code location tracking
"""

type Span = tuple[int, int, int, int]
_span_attrs = ["lineno", "col_offset", "end_lineno", "end_col_offset"]


def ast_set_span(ast: ast.AST, span: Span):
    for attr, value in zip(_span_attrs, span):
        setattr(ast, attr, value)


def ast_get_span(ast: ast.AST) -> Span:
    return tuple(getattr(ast, attr) for attr in _span_attrs)


"""
quote: convert a string of Python code into an AST node
"""


def quote(expr: str, *, passes: list[Any] | None = None, span=None, **kws) -> list[ast.AST]:
    """Quote a string of Python code into an ast.AST node."""
    tree = ast.parse(expr)
    tree = QuoteVisitor(kws, passes, span).visit(tree)
    return tree.body


def quote1(expr: str, *, passes: list[Any] | None = None, span=None, **kws) -> ast.AST:
    """Quote a string of Python code into an ast.AST node."""
    tree = ast.parse(expr)
    tree = QuoteVisitor(kws, passes, span).visit(tree)
    if len(tree.body) != 1:
        raise ValueError(f"Expected a single statement, got {len(tree.body)}")
    return tree.body[0]


def quote_expr(expr: str, *, passes: list[Any] | None = None, span=None, **kws) -> ast.expr:
    res = quote1(expr, **kws)
    assert isinstance(res, ast.Expr)
    return res.value


class QuoteVisitor(ast.NodeTransformer):
    def __init__(self, names: dict[str, ast.AST], passes: list[Any] | None = None, span=None):
        self.names = names
        self.passes = passes
        self.span = span

    def generic_visit(self, node: ast.AST):
        if self.span is not None:
            ast_set_span(node, self.span)
        return super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # perform placeholder substitution for names in the AST
        if node.id in self.names:
            return self.names[node.id]
        return node
