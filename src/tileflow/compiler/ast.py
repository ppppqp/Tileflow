from __future__ import annotations
import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import inspect
from typing import Any, Literal, cast
from tileflow.dsl.ir import KernelIR
from tileflow.dsl.jit import JitFunction
import textwrap
from tileflow.dsl import dtypes

"""
AST Rewrite to builder pattern
"""


class _empty:
    pass


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
    def __init__(self, nonlocals: dict[str, Any], globals: dict[str, Any]):
        self.tmp_counter = 0
        self.extra_type_hints: dict[str, Any] = {}
        self.non_locals = nonlocals
        self.globals = globals

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

    def visit_AugAssign(self, node: ast.AugAssign):
        node = self.generic_visit(node)
        target, rval = node.target, node.value
        op = get_operator_name(node.op)
        if isinstance(target, ast.Name):
            # a += b
            target_load = ast.Name(target.id, ctx=ast.Load())
            ast_set_span(target_load, ast_get_span(target))
            return quote(
                f"__tl_lhs = __tb.aug_assign('{op}', __tl_target, __tl_aug_value, name='{target.id}')",
                __tl_lhs=target,
                __tl_target=target_load,
                __tl_aug_value=rval,
                span=node,
            )
        elif isinstance(target, ast.Subscript):
            # a[i] += b
            return quote(
                f"__tb.aug_assign_slice('{op}', lval, slice, rval)",
                lval=target.value,
                slice=target.slice,
                rval=rval,
                span=node,
            )
        else:
            return node

    def visit_AnnAssign(self, node: ast.AnnAssign):
        node = self.generic_visit(node)
        # a: int = 1
        # a: int
        rval = node.value or quote_expr("__tb.empty", span=node, annot=node)
        return self._emit_assign_target(node.target, rval, annot=node.annotation)

    def visit_While(self, node: ast.While):
        node = self.generic_visit(node)
        # using lambda for lazy evaluation of the condition
        # using the for protocol to enter a context
        return quote(
            "for _ in __tb.ctx_while(lambda: cond):\n pass",
            cond=node.test,
            passes=[node.body],
            span=node,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        stmts = []
        arg_names = set()
        # TODO: support args, kwargs, defaults
        all_args = node.args.args
        for arg in all_args:
            name = arg.arg
            arg_names.add(name)
            arg_stmt = quote1(f'{name} = __tb.bind("{name}", {name})', span=arg)
            stmts.append(arg_stmt)

        for stmt in node.body:
            self._parse_arg_annot(stmt, arg_names)

    def _try_eval(self, node: ast.expr) -> Any:
        # execute python code and get result
        try:
            code = f"lambda {','.join(self.non_locals.keys())}: {ast.unparse(node)}"
            return eval(code, self.globals)(**self.non_locals)
        except Exception:
            return _empty

    def _parse_arg_annot(self, stmt: ast.stmt, arg_names: set[str]):
        if not isinstance(stmt, ast.AnnAssign):
            return
        if not isinstance(stmt.target, ast.Name):
            return
        if stmt.value is not None:
            return
        name = stmt.target.id
        if name not in arg_names:
            return
        annot = stmt.annotation

        # case 1: T.float32
        # TODO: dtype?
        if isinstance(annot, ast.Attribute) and annot.attr in dtypes._all_dtypes:
            eval_res = self._try_eval(annot)
            if isinstance(eval_res, dtypes.dtype):
                self.extra_type_hints[name] = eval_res
                return
        # case 2: T.float32[...] or T.Tensor(...)
        inner = None
        if isinstance(annot, ast.Subscript) and isinstance(annot.value, ast.Attribute):
            inner = annot.value
        if isinstance(annot, ast.Call) and isinstance(annot.func, ast.Attribute):
            inner = annot.func
        if inner and inner.attr in ["Tensor", "StridedTensor", "ptr"]:
            eval_res = self._try_eval(annot)
            # pass
            # TODO: TensorProxy stuff

    def visit_BoolOp(self, node: ast.BoolOp):
        node = self.generic_visit(node)
        op_name = get_boolop_name(node.op)
        last = node.values[-1]
        # e.g. a and b and c
        # reverse order so we first build c in the inner most evaluation, and then b, and then a
        # so that short circuiting works correctly
        for i in reversed(range(len(node.values) - 1)):
            last = quote_expr(
                expr=f"__tb.boolop('{op_name}', left, lambda:right)",
                left=node.values[i],
                right=last,
                span=node,
            )
        return last

    def visit_Compare(self, node: ast.Compare):
        node = self.generic_visit(node)
        left = node.left
        split = []
        # e.g. a < b < c
        for op, comp in zip(node.ops, node.comparators, strict=True):
            cmp = ast.Compare(left=left, ops=[op], comparators=[comp])
            ast_set_span(cmp, ast_get_span(node))
            split.append(cmp)
            left = comp
        last = split[-1]
        for i in reversed(range(len(split) - 1)):
            last = quote_expr(
                expr="__tb.boolop('And', left, lambda:right)",
                left=split[i],
                right=last,
                span=node,
            )
        return last

    def visit_IfExp(self, node: ast.IfExp):
        node = self.generic_visit(node)
        return quote_expr(
            expr="__tb.ifexp(cond, lambda:then_, lambda:else_)",
            cond=node.test,
            then_=node.body,
            else_=node.orelse,
            span=node,
        )

    def visit_Return(self, node: ast.Return):
        node = self.generic_visit(node)
        return quote_expr(
            expr="__tb.ret(value)",
            value=node.value,
            span=node,
        )

    def visit_With(self, node: ast.With):
        is_kernel_ctx = False
        for expr in node.items:
            cexpr = expr.context_expr
            # with Kernel(...) as T:
            if (
                isinstance(cexpr, ast.Call)
                and isinstance(cexpr.func, ast.Attribute)
                and cexpr.func.attr == "Kernel"
            ):
                eval_res = self._try_eval(cexpr.func)
                from tileflow.dsl.language import Kernel

                if eval_res is Kernel:
                    is_kernel_ctx = True
                    break
        node = self.generic_visit(node)
        for expr in node.items:
            expr.context_expr = quote_expr("__tb.ctx_with(e)", e=expr.context_expr, span=node)
        if is_kernel_ctx:
            return [quote1("if __tb.skip_kernel_ctx(): return"), node]
        return node

    def visit_Assert(self, node: ast.Assert):
        node = self.generic_visit(node)
        return quote("__tb.assert_expr(cond, msg)", cond=node.test, msg=node.msg, span=node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            return quote_expr(f"__tb.rval('{node.id}', node)", span=node)
        return node


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


"""
Operator
"""

Operator = Literal[
    "Add",
    "Sub",
    "Mult",
    "MatMult",
    "Div",
    "Mod",
    "Pow",
    "LShift",
    "RShift",
    "BitOr",
    "BitXor",
    "BitAnd",
    "FloorDiv",
]


def get_operator_name(operator: ast.operator) -> Operator:
    return cast(Operator, operator.__class__.__name__)
