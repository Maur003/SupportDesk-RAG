import ast
from pathlib import Path

import pytest

from supportdesk.arithmetic import calculate


def test_arithmetic_uses_operator_precedence_and_parentheses() -> None:
    assert calculate("2 + 3 * 4") == 14
    assert calculate("(2 + 3) * 4") == 20
    assert calculate("-3 + 4") == 1
    assert calculate("8 / 2") == 4


def test_arithmetic_rejects_anything_outside_the_four_operators() -> None:
    for expression in (
        "__import__('os')",
        "abs(2)",
        "2 ** 3",
        "9 // 2",
        "9 % 2",
        "2 and 3",
        "True",
        "'2'",
        "[1, 2]",
        "nine",
    ):
        with pytest.raises(ArithmeticError):
            calculate(expression)


def test_division_by_zero_is_an_error() -> None:
    with pytest.raises(ArithmeticError, match="division by zero"):
        calculate("1 / 0")


def test_arithmetic_module_does_not_call_eval() -> None:
    source = Path(calculate.__code__.co_filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "eval" not in called
    assert "exec" not in called
