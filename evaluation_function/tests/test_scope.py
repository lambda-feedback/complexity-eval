"""
Pytest tests for interpreter scoping behavior.
"""

import pytest

from ..analyzer.interpreter import Interpreter
from ..schemas.ast_nodes import (
    ProgramNode,
    FunctionNode,
    BlockNode,
    AssignmentNode,
    VariableNode,
    LiteralNode,
    BinaryOpNode,
    OperatorType,
    FunctionCallNode,
    ReturnNode,
)


def test_global_modification():
    """
    Functions without parameters should be able to
    modify global variables.
    """

    increment = FunctionNode(
        name="increment",
        parameters=[],
        body=BlockNode(statements=[
            AssignmentNode(
                target=VariableNode(name="x"),
                value=BinaryOpNode(
                    operator=OperatorType.ADD,
                    left=VariableNode(name="x"),
                    right=LiteralNode(value=1),
                ),
            )
        ]),
    )

    program = ProgramNode(
        functions=[increment],
        global_statements=BlockNode(statements=[
            AssignmentNode(
                target=VariableNode(name="x"),
                value=LiteralNode(value=5),
            ),
            AssignmentNode(
                target=VariableNode(name="result"),
                value=FunctionCallNode(
                    function_name="increment",
                    arguments=[],
                ),
            ),
        ]),
    )

    interp = Interpreter()
    result = interp.run(program)

    assert result["variables"]["x"] == 6
    assert result["variables"]["result"] is None


def test_parameter_isolation():
    """
    Function parameters should not modify
    the global variable of the same name.
    """

    set_local = FunctionNode(
        name="set_local",
        parameters=[VariableNode(name="x")],
        body=BlockNode(statements=[
            AssignmentNode(
                target=VariableNode(name="x"),
                value=LiteralNode(value=99),
            ),
            ReturnNode(value=VariableNode(name="x")),
        ]),
    )

    program = ProgramNode(
        functions=[set_local],
        global_statements=BlockNode(statements=[
            AssignmentNode(
                target=VariableNode(name="x"),
                value=LiteralNode(value=5),
            ),
            AssignmentNode(
                target=VariableNode(name="y"),
                value=FunctionCallNode(
                    function_name="set_local",
                    arguments=[VariableNode(name="x")],
                ),
            ),
        ]),
    )

    interp = Interpreter()
    result = interp.run(program)

    assert result["variables"]["x"] == 5
    assert result["variables"]["y"] == 99


def test_mixed_scoping():
    """
    Function parameters should be isolated,
    but other globals should remain shared.
    """

    func = FunctionNode(
        name="func",
        parameters=[VariableNode(name="a")],
        body=BlockNode(statements=[
            AssignmentNode(
                target=VariableNode(name="a"),
                value=LiteralNode(value=100),  # Local param
            ),
            AssignmentNode(
                target=VariableNode(name="b"),
                value=LiteralNode(value=200),  # Global
            ),
        ]),
    )

    program = ProgramNode(
        functions=[func],
        global_statements=BlockNode(statements=[
            AssignmentNode(
                target=VariableNode(name="a"),
                value=LiteralNode(value=1),
            ),
            AssignmentNode(
                target=VariableNode(name="b"),
                value=LiteralNode(value=2),
            ),
            FunctionCallNode(
                function_name="func",
                arguments=[VariableNode(name="a")],
            ),
        ]),
    )

    interp = Interpreter()
    result = interp.run(program)
    print(result)

    assert result["variables"]["a"] == 1
    assert result["variables"]["b"] == 200
