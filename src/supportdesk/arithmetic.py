"""Restricted arithmetic. The model proposes an expression. This module evaluates it.

The allowed tree is numbers, parentheses, unary plus and minus, and the operators
+ - * /. Parentheses are grouping in the parser, not a node we have to allow.
Names, calls, and every other operator are rejected. The builtin eval function is not called.
"""

import ast
import math

_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_UNARY = (ast.UAdd, ast.USub)


def calculate(expression: str) -> int | float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ArithmeticError("invalid expression") from exc
    value = _eval_node(tree.body)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ArithmeticError("invalid expression")
    if isinstance(value, float) and not math.isfinite(value):
        raise ArithmeticError("invalid expression")
    return value


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ArithmeticError("invalid expression")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY):
        operand = _eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        return -operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, _BINOPS):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ArithmeticError("division by zero")
        return left / right
    raise ArithmeticError("invalid expression")
