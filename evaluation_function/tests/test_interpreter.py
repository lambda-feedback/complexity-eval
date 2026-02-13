import pytest

from ..analyzer.interpreter import Interpreter
from ..schemas.ast_nodes import *


# -----------------------------------------
# Helper
# -----------------------------------------

def run_program(statements=None, functions=None):
    program = ProgramNode(
        functions=functions or [],
        global_statements=BlockNode(statements=statements or [])
    )
    return Interpreter().run(program)


# -----------------------------------------
# Basic Assignment
# -----------------------------------------

def test_assignment_literal():
    result = run_program([
        AssignmentNode(
            target=VariableNode(name="x"),
            value=LiteralNode(value=10)
        )
    ])

    assert result["variables"]["x"] == 10


def test_assignment_expression():
    result = run_program([
        AssignmentNode(
            target=VariableNode(name="x"),
            value=BinaryOpNode(
                operator=OperatorType.ADD,
                left=LiteralNode(value=5),
                right=LiteralNode(value=7)
            )
        )
    ])

    assert result["variables"]["x"] == 12


# -----------------------------------------
# Unary & Comparison
# -----------------------------------------

def test_unary_negation():
    result = run_program([
        AssignmentNode(
            target=VariableNode(name="x"),
            value=UnaryOpNode(
                operator=OperatorType.SUBTRACT,
                operand=LiteralNode(value=5)
            )
        )
    ])

    assert result["variables"]["x"] == -5


def test_comparison():
    result = run_program([
        AssignmentNode(
            target=VariableNode(name="x"),
            value=BinaryOpNode(
                operator=OperatorType.GREATER_THAN,
                left=LiteralNode(value=10),
                right=LiteralNode(value=3)
            )
        )
    ])

    assert result["variables"]["x"] is True


# -----------------------------------------
# If / Else
# -----------------------------------------

def test_if_true_branch():
    result = run_program([
        ConditionalNode(
            condition=LiteralNode(value=True),
            then_branch=BlockNode(statements=[
                AssignmentNode(
                    target=VariableNode(name="x"),
                    value=LiteralNode(value=1)
                )
            ]),
            else_branch=BlockNode(statements=[
                AssignmentNode(
                    target=VariableNode(name="x"),
                    value=LiteralNode(value=2)
                )
            ])
        )
    ])

    assert result["variables"]["x"] == 1


def test_if_false_branch():
    result = run_program([
        ConditionalNode(
            condition=LiteralNode(value=False),
            then_branch=BlockNode(statements=[
                AssignmentNode(
                    target=VariableNode(name="x"),
                    value=LiteralNode(value=1)
                )
            ]),
            else_branch=BlockNode(statements=[
                AssignmentNode(
                    target=VariableNode(name="x"),
                    value=LiteralNode(value=2)
                )
            ])
        )
    ])

    assert result["variables"]["x"] == 2


# -----------------------------------------
# While Loop
# -----------------------------------------

def test_while_loop_sum():
    result = run_program([
        AssignmentNode(target=VariableNode(name="i"), value=LiteralNode(value=1)),
        AssignmentNode(target=VariableNode(name="sum"), value=LiteralNode(value=0)),
        LoopNode(
            loop_type=LoopType.WHILE,
            condition=BinaryOpNode(
                operator=OperatorType.LESS_EQUAL,
                left=VariableNode(name="i"),
                right=LiteralNode(value=5)
            ),
            body=BlockNode(statements=[
                AssignmentNode(
                    target=VariableNode(name="sum"),
                    value=BinaryOpNode(
                        operator=OperatorType.ADD,
                        left=VariableNode(name="sum"),
                        right=VariableNode(name="i")
                    )
                ),
                AssignmentNode(
                    target=VariableNode(name="i"),
                    value=BinaryOpNode(
                        operator=OperatorType.ADD,
                        left=VariableNode(name="i"),
                        right=LiteralNode(value=1)
                    )
                )
            ])
        )
    ])

    assert result["variables"]["sum"] == 15


# -----------------------------------------
# For Loop
# -----------------------------------------

def test_for_loop():
    result = run_program([
        AssignmentNode(target=VariableNode(name="sum"), value=LiteralNode(value=0)),
        LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="i"),
            start=LiteralNode(value=1),
            end=LiteralNode(value=4),
            body=BlockNode(statements=[
                AssignmentNode(
                    target=VariableNode(name="sum"),
                    value=BinaryOpNode(
                        operator=OperatorType.ADD,
                        left=VariableNode(name="sum"),
                        right=VariableNode(name="i")
                    )
                )
            ])
        )
    ])

    assert result["variables"]["sum"] == 10


# -----------------------------------------
# Function Call
# -----------------------------------------

# def test_function_return():
#     func = FunctionNode(
#         name="add",
#         parameters=[VariableNode(name="a"), VariableNode(name="b")],
#         body=BlockNode(statements=[
#             ReturnNode(
#                 value=BinaryOpNode(
#                     operator=OperatorType.ADD,
#                     left=VariableNode(name="a"),
#                     right=VariableNode(name="b")
#                 )
#             )
#         ])
#     )

#     result = run_program(
#         functions=[func],
#         statements=[
#             AssignmentNode(
#                 target=VariableNode(name="x"),
#                 value=FunctionCallNode(
#                     function_name="add",
#                     arguments=[LiteralNode(value=3), LiteralNode(value=4)]
#                 )
#             )
#         ]
#     )

#     assert result["variables"]["x"] == 7


# -----------------------------------------
# Recursion (Factorial)
# -----------------------------------------

# def test_recursive_factorial():
#     factorial = FunctionNode(
#         name="fact",
#         parameters=[VariableNode(name="n")],
#         body=BlockNode(statements=[
#             ConditionalNode(
#                 condition=BinaryOpNode(
#                     operator=OperatorType.LESS_EQUAL,
#                     left=VariableNode(name="n"),
#                     right=LiteralNode(value=1)
#                 ),
#                 then_branch=BlockNode(statements=[
#                     ReturnNode(value=LiteralNode(value=1))
#                 ]),
#                 else_branch=BlockNode(statements=[
#                     ReturnNode(
#                         value=BinaryOpNode(
#                             operator=OperatorType.MULTIPLY,
#                             left=VariableNode(name="n"),
#                             right=RecursiveCallNode(
#                                 function_name="fact",
#                                 arguments=[
#                                     BinaryOpNode(
#                                         operator=OperatorType.SUBTRACT,
#                                         left=VariableNode(name="n"),
#                                         right=LiteralNode(value=1)
#                                     )
#                                 ]
#                             )
#                         )
#                     )
#                 ])
#             )
#         ])
#     )

#     result = run_program(
#         functions=[factorial],
#         statements=[
#             AssignmentNode(
#                 target=VariableNode(name="result"),
#                 value=FunctionCallNode(
#                     function_name="fact",
#                     arguments=[LiteralNode(value=5)]
#                 )
#             )
#         ]
#     )

#     assert result["variables"]["result"] == 120


# -----------------------------------------
# Scope Isolation
# -----------------------------------------

# def test_function_scope_isolated():
#     func = FunctionNode(
#         name="set_local",
#         parameters=[VariableNode(name="x")],
#         body=BlockNode(statements=[
#             AssignmentNode(
#                 target=VariableNode(name="x"),
#                 value=LiteralNode(value=99)
#             ),
#             ReturnNode(value=VariableNode(name="x"))
#         ])
#     )

#     result = run_program(
#         functions=[func],
#         statements=[
#             AssignmentNode(target=VariableNode(name="x"), value=LiteralNode(value=5)),
#             AssignmentNode(
#                 target=VariableNode(name="y"),
#                 value=FunctionCallNode(
#                     function_name="set_local",
#                     arguments=[VariableNode(name="x")]
#                 )
#             )
#         ]
#     )

#     assert result["variables"]["x"] == 5
#     assert result["variables"]["y"] == 99
