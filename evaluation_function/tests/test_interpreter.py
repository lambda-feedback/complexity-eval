"""
Comprehensive pytest test suite for the fixed interpreter.

All tests use pytest classes and functions.
Tests all fixes with proper fixtures and assertions.
"""

import pytest

from ..analyzer.interpreter import Interpreter, ReturnException
from ..schemas.ast_nodes import *
from ..schemas.input_schema import RuntimeValue


@pytest.fixture
def interpreter():
    """Create a fresh interpreter instance."""
    return Interpreter()


def run_program(statements=None, functions=None, initial_vars=None):
    """Helper to run a program."""
    program = ProgramNode(
        functions=functions or [],
        global_statements=BlockNode(statements=statements or [])
    )
    return Interpreter().run(program, initial_variables=initial_vars)


class TestBasicOperations:
    """Test basic interpreter operations."""
    
    def test_assignment_literal(self):
        """Test simple assignment."""
        result = run_program([
            AssignmentNode(
                target=VariableNode(name="x"),
                value=LiteralNode(value=10)
            )
        ])
        assert result["variables"]["x"] == 10
    
    def test_binary_add(self):
        """Test addition operation."""
        result = run_program([
            AssignmentNode(
                target=VariableNode(name="result"),
                value=BinaryOpNode(
                    operator=OperatorType.ADD, 
                    left=LiteralNode(value=5), 
                    right=LiteralNode(value=3)
                )
            )
        ])
        assert result["variables"]["result"] == 8
    
    def test_binary_subtract(self):
        """Test subtraction operation."""
        result = run_program([
            AssignmentNode(
                target=VariableNode(name="result"),
                value=BinaryOpNode(
                    operator=OperatorType.SUBTRACT,
                    left=LiteralNode(value=10),
                    right=LiteralNode(value=7)
                )
            )
        ])
        assert result["variables"]["result"] == 3
    
    def test_binary_multiply(self):
        """Test multiplication operation."""
        result = run_program([
            AssignmentNode(
                target=VariableNode(name="result"),
                value=BinaryOpNode(
                    operator=OperatorType.MULTIPLY,
                    left=LiteralNode(value=4),
                    right=LiteralNode(value=3)
                )
            )
        ])
        assert result["variables"]["result"] == 12
    
    def test_binary_divide(self):
        """Test division operation."""
        result = run_program([
            AssignmentNode(
                target=VariableNode(name="result"),
                value=BinaryOpNode(
                    operator=OperatorType.DIVIDE,
                    left=LiteralNode(value=15),
                    right=LiteralNode(value=3)
                )
            )
        ])
        assert result["variables"]["result"] == 5.0
    
    def test_comparison_equal(self):
        """Test equality comparison."""
        result = run_program([
            AssignmentNode(
                target=VariableNode(name="result"), 
                value=BinaryOpNode(
                    operator=OperatorType.EQUAL,
                    left=LiteralNode(value=5),
                    right=LiteralNode(value=5)
                )
            )
        ])
        assert result["variables"]["result"] is True
    
    def test_comparison_not_equal(self):
        """Test not equal comparison."""
        result = run_program([
            AssignmentNode(
                target=VariableNode(name="result"),
                value=BinaryOpNode(
                    operator=OperatorType.NOT_EQUAL,
                    left=LiteralNode(value=5),
                    right=LiteralNode(value=3)
                )
            )
        ])
        assert result["variables"]["result"] is True
    
    def test_comparison_less_than(self):
        """Test less than comparison."""
        result = run_program([
            AssignmentNode(
                target=VariableNode(name="result"),
                value=BinaryOpNode(
                    operator=OperatorType.LESS_THAN,
                    left=LiteralNode(value=3),
                    right=LiteralNode(value=5)
                )
            )
        ])
        assert result["variables"]["result"] is True
    
    def test_comparison_greater_than(self):
        """Test greater than comparison."""
        result = run_program([
            AssignmentNode(
                target=VariableNode(name="result"),
                value=BinaryOpNode(
                    operator=OperatorType.GREATER_THAN,
                    left=LiteralNode(value=7),
                    right=LiteralNode(value=5)
                )
            )
        ])
        assert result["variables"]["result"] is True
    
    def test_unary_negation(self):
        """Test unary negation."""
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
    
    def test_unary_not(self):
        """Test unary NOT operation."""
        result = run_program([
            AssignmentNode(
                target=VariableNode(name="x"),
                value=UnaryOpNode(
                    operator=OperatorType.NOT,
                    operand=LiteralNode(value=True)
                )
            )
        ])
        assert result["variables"]["x"] is False


class TestFIX1_ExpressionStatement:
    """Test FIX #1: ExpressionStatementNode handling."""
    
    def test_expression_statement_function_call(self):
        """Test standalone function call wrapped in ExpressionStatement."""
        func = FunctionNode(
            name="increment",
            parameters=[],
            body=BlockNode(statements=[
                AssignmentNode(
                    target=VariableNode(name="x"),
                    value=BinaryOpNode(
                        operator=OperatorType.ADD,
                        left=VariableNode(name="x"),
                        right=LiteralNode(value=1)
                    )
                )
            ])
        )
        
        result = run_program(
            functions=[func],
            statements=[
                AssignmentNode(target=VariableNode(name="x"), value=LiteralNode(value=5)),
                ExpressionStatementNode(
                    expression=FunctionCallNode(function_name="increment", arguments=[])
                )
            ]
        )
        
        assert result["variables"]["x"] == 6


class TestFIX2_ReturnNone:
    """Test FIX #2: Return without value."""
    
    def test_return_without_value(self):
        """Test return statement without expression."""
        func = FunctionNode(
            name="early_exit",
            parameters=[VariableNode(name="x")],
            body=BlockNode(statements=[
                ConditionalNode(
                    condition=BinaryOpNode(
                        operator=OperatorType.LESS_THAN,
                        left=VariableNode(name="x"),
                        right=LiteralNode(value=0)
                    ),
                    then_branch=BlockNode(statements=[
                        ReturnNode(value=None)
                    ])
                ),
                ReturnNode(value=LiteralNode(value=100))
            ])
        )
        
        result = run_program(
            functions=[func],
            statements=[
                AssignmentNode(
                    target=VariableNode(name="result"),
                    value=FunctionCallNode(
                        function_name="early_exit",
                        arguments=[LiteralNode(value=-5)]
                    )
                )
            ]
        )
        
        assert result["variables"]["result"] is None
    
    def test_return_with_value(self):
        """Test normal return with value."""
        func = FunctionNode(
            name="get_value",
            parameters=[],
            body=BlockNode(statements=[
                ReturnNode(value=LiteralNode(value=42))
            ])
        )
        
        result = run_program(
            functions=[func],
            statements=[
                AssignmentNode(
                    target=VariableNode(name="result"),
                    value=FunctionCallNode(function_name="get_value", arguments=[])
                )
            ]
        )
        
        assert result["variables"]["result"] == 42


class TestFIX3_ForEachLoop:
    """Test FIX #3: FOR_EACH loop implementation."""
    
    def test_for_each_list(self):
        """Test FOR EACH over a list."""
        result = run_program(
            statements=[
                AssignmentNode(target=VariableNode(name="sum"), value=LiteralNode(value=0)),
                AssignmentNode(target=VariableNode(name="numbers"), 
                              value=LiteralNode(value=[1, 2, 3, 4, 5])),
                LoopNode(
                    loop_type=LoopType.FOR_EACH,
                    iterator=VariableNode(name="num"),
                    collection=VariableNode(name="numbers"),
                    body=BlockNode(statements=[
                        AssignmentNode(
                            target=VariableNode(name="sum"),
                            value=BinaryOpNode(
                                operator=OperatorType.ADD,
                                left=VariableNode(name="sum"),
                                right=VariableNode(name="num")
                            )
                        )
                    ])
                )
            ]
        )
        
        assert result["variables"]["sum"] == 15
    
    def test_for_each_string(self):
        """Test FOR EACH over a string."""
        result = run_program(
            statements=[
                AssignmentNode(target=VariableNode(name="count"), value=LiteralNode(value=0)),
                LoopNode(
                    loop_type=LoopType.FOR_EACH,
                    iterator=VariableNode(name="char"),
                    collection=LiteralNode(value="hello"),
                    body=BlockNode(statements=[
                        AssignmentNode(
                            target=VariableNode(name="count"),
                            value=BinaryOpNode(
                                operator=OperatorType.ADD,
                                left=VariableNode(name="count"),
                                right=LiteralNode(value=1)
                            )
                        )
                    ])
                )
            ]
        )
        
        assert result["variables"]["count"] == 5
    
    def test_for_each_tuple(self):
        """Test FOR EACH over a tuple."""
        result = run_program(
            statements=[
                AssignmentNode(target=VariableNode(name="product"), value=LiteralNode(value=1)),
                AssignmentNode(target=VariableNode(name="values"), 
                              value=LiteralNode(value=(2, 3, 4))),
                LoopNode(
                    loop_type=LoopType.FOR_EACH,
                    iterator=VariableNode(name="val"),
                    collection=VariableNode(name="values"),
                    body=BlockNode(statements=[
                        AssignmentNode(
                            target=VariableNode(name="product"),
                            value=BinaryOpNode(
                                operator=OperatorType.MULTIPLY,
                                left=VariableNode(name="product"),
                                right=VariableNode(name="val")
                            )
                        )
                    ])
                )
            ]
        )
        
        assert result["variables"]["product"] == 24


class TestFIX4_RepeatUntilLoop:
    """Test FIX #4: REPEAT_UNTIL loop implementation."""
    
    def test_repeat_until_basic(self):
        """Test basic REPEAT...UNTIL loop."""
        result = run_program(
            statements=[
                AssignmentNode(target=VariableNode(name="x"), value=LiteralNode(value=1)),
                LoopNode(
                    loop_type=LoopType.REPEAT_UNTIL,
                    condition=BinaryOpNode(
                        operator=OperatorType.GREATER_EQUAL,
                        left=VariableNode(name="x"),
                        right=LiteralNode(value=10)
                    ),
                    body=BlockNode(statements=[
                        AssignmentNode(
                            target=VariableNode(name="x"),
                            value=BinaryOpNode(
                                operator=OperatorType.MULTIPLY,
                                left=VariableNode(name="x"),
                                right=LiteralNode(value=2)
                            )
                        )
                    ])
                )
            ]
        )
        
        # Should execute: 1 -> 2 -> 4 -> 8 -> 16 (stops when >= 10)
        assert result["variables"]["x"] == 16
    
    def test_repeat_until_executes_once(self):
        """Test REPEAT UNTIL executes at least once."""
        result = run_program(
            statements=[
                AssignmentNode(target=VariableNode(name="counter"), value=LiteralNode(value=0)),
                LoopNode(
                    loop_type=LoopType.REPEAT_UNTIL,
                    condition=LiteralNode(value=True),  # Always true
                    body=BlockNode(statements=[
                        AssignmentNode(
                            target=VariableNode(name="counter"),
                            value=BinaryOpNode(
                                operator=OperatorType.ADD,
                                left=VariableNode(name="counter"),
                                right=LiteralNode(value=1)
                            )
                        )
                    ])
                )
            ]
        )
        
        # Should execute once even though condition is true
        assert result["variables"]["counter"] == 1


class TestFIX5_NoneCondition:
    """Test FIX #5: None condition in conditionals."""
    
    def test_none_condition_skips(self):
        """Test that None condition doesn't crash and skips the conditional."""
        result = run_program(
            statements=[
                AssignmentNode(target=VariableNode(name="x"), value=LiteralNode(value=1)),
                ConditionalNode(
                    condition=None,
                    then_branch=BlockNode(statements=[
                        AssignmentNode(target=VariableNode(name="x"), 
                                     value=LiteralNode(value=99))
                    ])
                ),
                AssignmentNode(target=VariableNode(name="y"), value=LiteralNode(value=2))
            ]
        )
        
        # x should remain 1 (conditional skipped), y should be 2
        assert result["variables"]["x"] == 1
        assert result["variables"]["y"] == 2


class TestFIX6_NoneExpression:
    """Test FIX #6: None expression handling."""
    
    def test_none_expression_returns_none(self, interpreter):
        """Test that None expression returns None."""
        result = interpreter.evaluate_expression(None)
        assert result is None


class TestFIX7_PrintFunction:
    """Test FIX #7 & #11: print() built-in function."""
    
    def test_print_single_value(self):
        """Test print with single argument."""
        result = run_program(
            statements=[
                ExpressionStatementNode(
                    expression=FunctionCallNode(
                        function_name="print",
                        arguments=[LiteralNode(value="Hello")]
                    )
                )
            ]
        )
        
        assert result["output"] == ["Hello"]
    
    def test_print_multiple_values(self):
        """Test print with multiple arguments."""
        result = run_program(
            statements=[
                ExpressionStatementNode(
                    expression=FunctionCallNode(
                        function_name="print",
                        arguments=[
                            LiteralNode(value="x"),
                            LiteralNode(value="="),
                            LiteralNode(value=42)
                        ]
                    )
                )
            ]
        )
        
        assert result["output"] == ["x = 42"]
    
    def test_print_with_variables(self):
        """Test print with variable values."""
        result = run_program(
            statements=[
                AssignmentNode(target=VariableNode(name="x"), value=LiteralNode(value=10)),
                AssignmentNode(target=VariableNode(name="y"), value=LiteralNode(value=20)),
                ExpressionStatementNode(
                    expression=FunctionCallNode(
                        function_name="print",
                        arguments=[VariableNode(name="x"), VariableNode(name="y")]
                    )
                )
            ]
        )
        
        assert result["output"] == ["10 20"]
    
    def test_multiple_prints(self):
        """Test multiple print statements."""
        result = run_program(
            statements=[
                ExpressionStatementNode(
                    expression=FunctionCallNode(
                        function_name="print",
                        arguments=[LiteralNode(value="Line 1")]
                    )
                ),
                ExpressionStatementNode(
                    expression=FunctionCallNode(
                        function_name="print",
                        arguments=[LiteralNode(value="Line 2")]
                    )
                )
            ]
        )
        
        assert result["output"] == ["Line 1", "Line 2"]


class TestFIX8_EmptyFunctionBody:
    """Test FIX #12: Functions with no body."""
    
    def test_function_with_none_body(self):
        """Test function with body=None."""
        func = FunctionNode(
            name="noop",
            parameters=[],
            body=None
        )
        
        result = run_program(
            functions=[func],
            statements=[
                AssignmentNode(
                    target=VariableNode(name="x"),
                    value=FunctionCallNode(function_name="noop", arguments=[])
                )
            ]
        )
        
        # Should return None (following Python logic)
        assert result["variables"]["x"] is None


class TestLoops:
    """Test all loop types."""
    
    def test_for_loop_basic(self):
        """Test basic FOR loop."""
        result = run_program(
            statements=[
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
            ]
        )
        
        assert result["variables"]["sum"] == 10
    
    def test_while_loop_basic(self):
        """Test basic WHILE loop."""
        result = run_program(
            statements=[
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
            ]
        )
        
        assert result["variables"]["sum"] == 15


class TestConditionals:
    """Test conditional statements."""
    
    def test_if_true_branch(self):
        """Test IF statement taking true branch."""
        result = run_program(
            statements=[
                ConditionalNode(
                    condition=LiteralNode(value=True),
                    then_branch=BlockNode(statements=[
                        AssignmentNode(target=VariableNode(name="x"), 
                                     value=LiteralNode(value=1))
                    ]),
                    else_branch=BlockNode(statements=[
                        AssignmentNode(target=VariableNode(name="x"), 
                                     value=LiteralNode(value=2))
                    ])
                )
            ]
        )
        
        assert result["variables"]["x"] == 1
    
    def test_if_false_branch(self):
        """Test IF statement taking false branch."""
        result = run_program(
            statements=[
                ConditionalNode(
                    condition=LiteralNode(value=False),
                    then_branch=BlockNode(statements=[
                        AssignmentNode(target=VariableNode(name="x"), 
                                     value=LiteralNode(value=1))
                    ]),
                    else_branch=BlockNode(statements=[
                        AssignmentNode(target=VariableNode(name="x"), 
                                     value=LiteralNode(value=2))
                    ])
                )
            ]
        )
        
        assert result["variables"]["x"] == 2


class TestFunctions:
    """Test function calls and recursion."""
    
    def test_function_with_parameters(self):
        """Test function with parameters."""
        func = FunctionNode(
            name="add",
            parameters=[VariableNode(name="a"), VariableNode(name="b")],
            body=BlockNode(statements=[
                ReturnNode(
                    value=BinaryOpNode(
                        operator=OperatorType.ADD,
                        left=VariableNode(name="a"),
                        right=VariableNode(name="b")
                    )
                )
            ])
        )
        
        result = run_program(
            functions=[func],
            statements=[
                AssignmentNode(
                    target=VariableNode(name="result"),
                    value=FunctionCallNode(
                        function_name="add",
                        arguments=[LiteralNode(value=3), LiteralNode(value=4)]
                    )
                )
            ]
        )
        
        assert result["variables"]["result"] == 7
    
    def test_recursive_factorial(self):
        """Test recursive function."""
        factorial = FunctionNode(
            name="fact",
            parameters=[VariableNode(name="n")],
            body=BlockNode(statements=[
                ConditionalNode(
                    condition=BinaryOpNode(
                        operator=OperatorType.LESS_EQUAL,
                        left=VariableNode(name="n"),
                        right=LiteralNode(value=1)
                    ),
                    then_branch=BlockNode(statements=[
                        ReturnNode(value=LiteralNode(value=1))
                    ]),
                    else_branch=BlockNode(statements=[
                        ReturnNode(
                            value=BinaryOpNode(
                                operator=OperatorType.MULTIPLY,
                                left=VariableNode(name="n"),
                                right=RecursiveCallNode(
                                    function_name="fact",
                                    arguments=[
                                        BinaryOpNode(
                                            operator=OperatorType.SUBTRACT,
                                            left=VariableNode(name="n"),
                                            right=LiteralNode(value=1)
                                        )
                                    ]
                                )
                            )
                        )
                    ])
                )
            ])
        )
        
        result = run_program(
            functions=[factorial],
            statements=[
                AssignmentNode(
                    target=VariableNode(name="result"),
                    value=FunctionCallNode(
                        function_name="fact",
                        arguments=[LiteralNode(value=5)]
                    )
                )
            ]
        )
        
        assert result["variables"]["result"] == 120


class TestComplexScenarios:
    """Test complex scenarios combining multiple features."""
    
    def test_nested_loops(self):
        """Test nested FOR and WHILE loops."""
        result = run_program(
            statements=[
                AssignmentNode(target=VariableNode(name="sum"), value=LiteralNode(value=0)),
                LoopNode(
                    loop_type=LoopType.FOR,
                    iterator=VariableNode(name="i"),
                    start=LiteralNode(value=1),
                    end=LiteralNode(value=3),
                    body=BlockNode(statements=[
                        AssignmentNode(target=VariableNode(name="j"), 
                                     value=LiteralNode(value=1)),
                        LoopNode(
                            loop_type=LoopType.WHILE,
                            condition=BinaryOpNode(
                                operator=OperatorType.LESS_EQUAL,
                                left=VariableNode(name="j"),
                                right=LiteralNode(value=2)
                            ),
                            body=BlockNode(statements=[
                                AssignmentNode(
                                    target=VariableNode(name="sum"),
                                    value=BinaryOpNode(
                                        operator=OperatorType.ADD,
                                        left=VariableNode(name="sum"),
                                        right=BinaryOpNode(
                                            operator=OperatorType.MULTIPLY,
                                            left=VariableNode(name="i"),
                                            right=VariableNode(name="j")
                                        )
                                    )
                                ),
                                AssignmentNode(
                                    target=VariableNode(name="j"),
                                    value=BinaryOpNode(
                                        operator=OperatorType.ADD,
                                        left=VariableNode(name="j"),
                                        right=LiteralNode(value=1)
                                    )
                                )
                            ])
                        )
                    ])
                )
            ]
        )
        
        # i=1: j=1(1), j=2(2) -> 3
        # i=2: j=1(2), j=2(4) -> 6
        # i=3: j=1(3), j=2(6) -> 9
        # Total: 18
        assert result["variables"]["sum"] == 18


class TestArrayOperations:
    """Test array operations."""
    
    def test_array_access(self):
        """Test array element access."""
        result = run_program(
            statements=[
                AssignmentNode(
                    target=VariableNode(name="arr"),
                    value=LiteralNode(value=[10, 20, 30, 40])
                ),
                AssignmentNode(
                    target=VariableNode(name="x"),
                    value=ArrayAccessNode(
                        array=VariableNode(name="arr"),
                        index=LiteralNode(value=2)
                    )
                )
            ]
        )
        
        assert result["variables"]["x"] == 30
    
    def test_array_assignment(self):
        """Test array element assignment."""
        result = run_program(
            statements=[
                AssignmentNode(
                    target=VariableNode(name="arr"),
                    value=LiteralNode(value=[1, 2, 3])
                ),
                AssignmentNode(
                    target=ArrayAccessNode(
                        array=VariableNode(name="arr"),
                        index=LiteralNode(value=1)
                    ),
                    value=LiteralNode(value=99)
                )
            ]
        )
        
        assert result["variables"]["arr"] == [1, 99, 3]


class TestErrorHandling:
    """Test error handling."""
    
    def test_undefined_variable_error(self, interpreter):
        """Test error when using undefined variable."""
        program = ProgramNode(
            functions=[],
            global_statements=BlockNode(statements=[
                AssignmentNode(
                    target=VariableNode(name="y"),
                    value=VariableNode(name="x")  # x is undefined
                )
            ])
        )
        
        with pytest.raises(NameError, match="not defined"):
            interpreter.run(program)
    
    def test_division_by_zero_error(self, interpreter):
        """Test division by zero error."""
        program = ProgramNode(
            functions=[],
            global_statements=BlockNode(statements=[
                AssignmentNode(
                    target=VariableNode(name="result"),
                    value=BinaryOpNode(
                        operator=OperatorType.DIVIDE,
                        left=LiteralNode(value=10),
                        right=LiteralNode(value=0)
                    )
                )
            ])
        )
        
        with pytest.raises(ZeroDivisionError):
            interpreter.run(program)
    
    def test_undefined_function_error(self, interpreter):
        """Test error when calling undefined function."""
        program = ProgramNode(
            functions=[],
            global_statements=BlockNode(statements=[
                AssignmentNode(
                    target=VariableNode(name="result"),
                    value=FunctionCallNode(
                        function_name="missing_func",
                        arguments=[]
                    )
                )
            ])
        )
        
        with pytest.raises(NameError, match="not defined"):
            interpreter.run(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])