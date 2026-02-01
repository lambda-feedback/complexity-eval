"""
Comprehensive tests for the AST Builder module.

Tests cover:
- AST node creation
- Node type verification
- Tree structure validation
- Expression handling
- Statement handling
"""

import pytest
from ..schemas.ast_nodes import (
    ProgramNode, FunctionNode, BlockNode, LoopNode, ConditionalNode,
    AssignmentNode, ReturnNode, FunctionCallNode, RecursiveCallNode,
    VariableNode, LiteralNode, BinaryOpNode, UnaryOpNode, ArrayAccessNode,
    NodeType, LoopType, OperatorType, SourceLocation
)


class TestASTNodeTypes:
    """Tests for AST node type enumeration."""
    
    def test_node_types_exist(self):
        """Test that all expected node types exist."""
        expected_types = [
            NodeType.PROGRAM, NodeType.FUNCTION, NodeType.BLOCK,
            NodeType.LOOP, NodeType.CONDITIONAL, NodeType.ASSIGNMENT,
            NodeType.RETURN, NodeType.FUNCTION_CALL, NodeType.RECURSIVE_CALL,
            NodeType.VARIABLE, NodeType.LITERAL, NodeType.BINARY_OP,
            NodeType.UNARY_OP, NodeType.ARRAY_ACCESS, NodeType.EXPRESSION,
        ]
        
        for node_type in expected_types:
            assert node_type is not None
    
    def test_loop_types_exist(self):
        """Test that all expected loop types exist."""
        expected_types = [
            LoopType.FOR, LoopType.FOR_EACH, LoopType.WHILE,
            LoopType.DO_WHILE, LoopType.REPEAT_UNTIL, LoopType.UNKNOWN,
        ]
        
        for loop_type in expected_types:
            assert loop_type is not None
    
    def test_operator_types_exist(self):
        """Test that all expected operator types exist."""
        arithmetic_ops = [
            OperatorType.ADD, OperatorType.SUBTRACT, OperatorType.MULTIPLY,
            OperatorType.DIVIDE, OperatorType.MODULO, OperatorType.POWER,
        ]
        comparison_ops = [
            OperatorType.EQUAL, OperatorType.NOT_EQUAL, OperatorType.LESS_THAN,
            OperatorType.LESS_EQUAL, OperatorType.GREATER_THAN, OperatorType.GREATER_EQUAL,
        ]
        logical_ops = [
            OperatorType.AND, OperatorType.OR, OperatorType.NOT,
        ]
        
        for op in arithmetic_ops + comparison_ops + logical_ops:
            assert op is not None


class TestProgramNode:
    """Tests for ProgramNode."""
    
    def test_create_empty_program(self):
        """Test creating empty program node."""
        program = ProgramNode()
        
        assert program.node_type == NodeType.PROGRAM
        assert len(program.functions) == 0
        assert program.global_statements is None
    
    def test_create_program_with_functions(self):
        """Test creating program with functions."""
        func = FunctionNode(name="test", parameters=[], body=None)
        program = ProgramNode(functions=[func])
        
        assert len(program.functions) == 1
        assert program.functions[0].name == "test"
    
    def test_create_program_with_global_statements(self):
        """Test creating program with global statements."""
        stmt = AssignmentNode(
            target=VariableNode(name="x"),
            value=LiteralNode(value=1, literal_type="int")
        )
        block = BlockNode(statements=[stmt])
        program = ProgramNode(global_statements=block)
        
        assert program.global_statements is not None
        assert len(program.global_statements.statements) == 1
    
    def test_program_to_dict(self):
        """Test program serialization to dict."""
        program = ProgramNode()
        result = program.model_dump()
        
        assert "node_type" in result
        assert "functions" in result
        assert result["node_type"] == NodeType.PROGRAM


class TestFunctionNode:
    """Tests for FunctionNode."""
    
    def test_create_simple_function(self):
        """Test creating simple function node."""
        func = FunctionNode(name="test", parameters=[], body=None)
        
        assert func.node_type == NodeType.FUNCTION
        assert func.name == "test"
        assert len(func.parameters) == 0
        assert func.is_recursive == False
    
    def test_create_function_with_parameters(self):
        """Test creating function with parameters."""
        params = [
            VariableNode(name="a"),
            VariableNode(name="b"),
        ]
        func = FunctionNode(name="add", parameters=params, body=None)
        
        assert len(func.parameters) == 2
        assert func.parameters[0].name == "a"
        assert func.parameters[1].name == "b"
    
    def test_create_function_with_body(self):
        """Test creating function with body."""
        return_stmt = ReturnNode(value=LiteralNode(value=1, literal_type="int"))
        body = BlockNode(statements=[return_stmt])
        func = FunctionNode(name="getOne", parameters=[], body=body)
        
        assert func.body is not None
        assert len(func.body.statements) == 1
    
    def test_create_recursive_function(self):
        """Test creating recursive function."""
        func = FunctionNode(name="factorial", parameters=[], body=None, is_recursive=True)
        
        assert func.is_recursive == True
    
    def test_function_to_dict(self):
        """Test function serialization to dict."""
        func = FunctionNode(name="test", parameters=[], body=None)
        result = func.model_dump()
        
        assert result["name"] == "test"
        assert result["node_type"] == NodeType.FUNCTION


class TestBlockNode:
    """Tests for BlockNode."""
    
    def test_create_empty_block(self):
        """Test creating empty block node."""
        block = BlockNode()
        
        assert block.node_type == NodeType.BLOCK
        assert len(block.statements) == 0
    
    def test_create_block_with_statements(self):
        """Test creating block with statements."""
        stmt1 = AssignmentNode(
            target=VariableNode(name="x"),
            value=LiteralNode(value=1, literal_type="int")
        )
        stmt2 = AssignmentNode(
            target=VariableNode(name="y"),
            value=LiteralNode(value=2, literal_type="int")
        )
        block = BlockNode(statements=[stmt1, stmt2])
        
        assert len(block.statements) == 2
    
    def test_nested_blocks(self):
        """Test nested blocks."""
        inner = BlockNode(statements=[])
        outer = BlockNode(statements=[inner])
        
        assert len(outer.statements) == 1


class TestLoopNode:
    """Tests for LoopNode."""
    
    def test_create_for_loop(self):
        """Test creating FOR loop node."""
        loop = LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="i"),
            start=LiteralNode(value=1, literal_type="int"),
            end=VariableNode(name="n"),
            body=BlockNode(statements=[])
        )
        
        assert loop.node_type == NodeType.LOOP
        assert loop.loop_type == LoopType.FOR
        assert loop.iterator.name == "i"
    
    def test_create_for_loop_with_step(self):
        """Test creating FOR loop with step."""
        loop = LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="i"),
            start=LiteralNode(value=1, literal_type="int"),
            end=VariableNode(name="n"),
            step=LiteralNode(value=2, literal_type="int"),
            body=BlockNode(statements=[])
        )
        
        assert loop.step is not None
        assert loop.step.value == 2
    
    def test_create_while_loop(self):
        """Test creating WHILE loop node."""
        condition = BinaryOpNode(
            operator=OperatorType.LESS_THAN,
            left=VariableNode(name="i"),
            right=VariableNode(name="n")
        )
        loop = LoopNode(
            loop_type=LoopType.WHILE,
            condition=condition,
            body=BlockNode(statements=[])
        )
        
        assert loop.loop_type == LoopType.WHILE
        assert loop.condition is not None
    
    def test_create_foreach_loop(self):
        """Test creating FOR-EACH loop node."""
        loop = LoopNode(
            loop_type=LoopType.FOR_EACH,
            iterator=VariableNode(name="item"),
            collection=VariableNode(name="list"),
            body=BlockNode(statements=[])
        )
        
        assert loop.loop_type == LoopType.FOR_EACH
        assert loop.collection is not None
    
    def test_create_repeat_until_loop(self):
        """Test creating REPEAT-UNTIL loop node."""
        condition = BinaryOpNode(
            operator=OperatorType.GREATER_EQUAL,
            left=VariableNode(name="x"),
            right=VariableNode(name="n")
        )
        loop = LoopNode(
            loop_type=LoopType.REPEAT_UNTIL,
            condition=condition,
            body=BlockNode(statements=[])
        )
        
        assert loop.loop_type == LoopType.REPEAT_UNTIL
    
    def test_loop_nesting_level(self):
        """Test loop nesting level."""
        inner_loop = LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="j"),
            nesting_level=1,
            body=BlockNode(statements=[])
        )
        outer_loop = LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="i"),
            nesting_level=0,
            body=BlockNode(statements=[inner_loop])
        )
        
        assert outer_loop.nesting_level == 0
        assert inner_loop.nesting_level == 1
    
    def test_loop_estimated_iterations(self):
        """Test loop estimated iterations."""
        loop = LoopNode(
            loop_type=LoopType.FOR,
            estimated_iterations="n",
            body=BlockNode(statements=[])
        )
        
        assert loop.estimated_iterations == "n"
    
    def test_loop_to_dict(self):
        """Test loop serialization to dict."""
        loop = LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="i"),
            body=BlockNode(statements=[])
        )
        result = loop.model_dump()
        
        assert result["loop_type"] == LoopType.FOR
        assert result["node_type"] == NodeType.LOOP


class TestConditionalNode:
    """Tests for ConditionalNode."""
    
    def test_create_simple_if(self):
        """Test creating simple IF node."""
        condition = BinaryOpNode(
            operator=OperatorType.GREATER_THAN,
            left=VariableNode(name="x"),
            right=LiteralNode(value=0, literal_type="int")
        )
        cond = ConditionalNode(
            condition=condition,
            then_branch=BlockNode(statements=[])
        )
        
        assert cond.node_type == NodeType.CONDITIONAL
        assert cond.condition is not None
        assert cond.then_branch is not None
        assert cond.else_branch is None
    
    def test_create_if_else(self):
        """Test creating IF-ELSE node."""
        condition = BinaryOpNode(
            operator=OperatorType.GREATER_THAN,
            left=VariableNode(name="x"),
            right=LiteralNode(value=0, literal_type="int")
        )
        cond = ConditionalNode(
            condition=condition,
            then_branch=BlockNode(statements=[]),
            else_branch=BlockNode(statements=[])
        )
        
        assert cond.else_branch is not None
    
    def test_create_if_elif_else(self):
        """Test creating IF-ELIF-ELSE node."""
        condition1 = BinaryOpNode(
            operator=OperatorType.GREATER_THAN,
            left=VariableNode(name="x"),
            right=LiteralNode(value=0, literal_type="int")
        )
        condition2 = BinaryOpNode(
            operator=OperatorType.LESS_THAN,
            left=VariableNode(name="x"),
            right=LiteralNode(value=0, literal_type="int")
        )
        elif_branch = ConditionalNode(
            condition=condition2,
            then_branch=BlockNode(statements=[])
        )
        cond = ConditionalNode(
            condition=condition1,
            then_branch=BlockNode(statements=[]),
            elif_branches=[elif_branch],
            else_branch=BlockNode(statements=[])
        )
        
        assert len(cond.elif_branches) == 1


class TestAssignmentNode:
    """Tests for AssignmentNode."""
    
    def test_create_simple_assignment(self):
        """Test creating simple assignment node."""
        assign = AssignmentNode(
            target=VariableNode(name="x"),
            value=LiteralNode(value=1, literal_type="int")
        )
        
        assert assign.node_type == NodeType.ASSIGNMENT
        assert assign.target.name == "x"
        assert assign.value.value == 1
    
    def test_create_array_assignment(self):
        """Test creating array element assignment."""
        assign = AssignmentNode(
            target=ArrayAccessNode(
                array=VariableNode(name="A"),
                index=VariableNode(name="i")
            ),
            value=LiteralNode(value=0, literal_type="int")
        )
        
        assert isinstance(assign.target, ArrayAccessNode)
    
    def test_create_compound_assignment(self):
        """Test creating compound assignment."""
        assign = AssignmentNode(
            target=VariableNode(name="x"),
            value=LiteralNode(value=1, literal_type="int"),
            operator=OperatorType.ADD_ASSIGN
        )
        
        assert assign.operator == OperatorType.ADD_ASSIGN


class TestExpressionNodes:
    """Tests for expression nodes."""
    
    def test_create_variable_node(self):
        """Test creating variable node."""
        var = VariableNode(name="count")
        
        assert var.node_type == NodeType.VARIABLE
        assert var.name == "count"
    
    def test_create_literal_int(self):
        """Test creating integer literal."""
        lit = LiteralNode(value=42, literal_type="int")
        
        assert lit.node_type == NodeType.LITERAL
        assert lit.value == 42
        assert lit.literal_type == "int"
    
    def test_create_literal_float(self):
        """Test creating float literal."""
        lit = LiteralNode(value=3.14, literal_type="float")
        
        assert lit.value == 3.14
        assert lit.literal_type == "float"
    
    def test_create_literal_string(self):
        """Test creating string literal."""
        lit = LiteralNode(value="hello", literal_type="string")
        
        assert lit.value == "hello"
        assert lit.literal_type == "string"
    
    def test_create_literal_bool(self):
        """Test creating boolean literal."""
        lit_true = LiteralNode(value=True, literal_type="bool")
        lit_false = LiteralNode(value=False, literal_type="bool")
        
        assert lit_true.value == True
        assert lit_false.value == False
    
    def test_create_binary_op_arithmetic(self):
        """Test creating arithmetic binary operations."""
        operators = [
            OperatorType.ADD, OperatorType.SUBTRACT,
            OperatorType.MULTIPLY, OperatorType.DIVIDE,
            OperatorType.MODULO, OperatorType.POWER,
        ]
        
        for op in operators:
            node = BinaryOpNode(
                operator=op,
                left=VariableNode(name="a"),
                right=VariableNode(name="b")
            )
            assert node.node_type == NodeType.BINARY_OP
            assert node.operator == op
    
    def test_create_binary_op_comparison(self):
        """Test creating comparison binary operations."""
        operators = [
            OperatorType.EQUAL, OperatorType.NOT_EQUAL,
            OperatorType.LESS_THAN, OperatorType.LESS_EQUAL,
            OperatorType.GREATER_THAN, OperatorType.GREATER_EQUAL,
        ]
        
        for op in operators:
            node = BinaryOpNode(
                operator=op,
                left=VariableNode(name="a"),
                right=VariableNode(name="b")
            )
            assert node.operator == op
    
    def test_create_binary_op_logical(self):
        """Test creating logical binary operations."""
        and_node = BinaryOpNode(
            operator=OperatorType.AND,
            left=VariableNode(name="a"),
            right=VariableNode(name="b")
        )
        or_node = BinaryOpNode(
            operator=OperatorType.OR,
            left=VariableNode(name="a"),
            right=VariableNode(name="b")
        )
        
        assert and_node.operator == OperatorType.AND
        assert or_node.operator == OperatorType.OR
    
    def test_create_unary_op(self):
        """Test creating unary operations."""
        not_node = UnaryOpNode(
            operator=OperatorType.NOT,
            operand=VariableNode(name="flag")
        )
        neg_node = UnaryOpNode(
            operator=OperatorType.SUBTRACT,
            operand=VariableNode(name="x")
        )
        
        assert not_node.node_type == NodeType.UNARY_OP
        assert neg_node.operator == OperatorType.SUBTRACT
    
    def test_create_array_access_simple(self):
        """Test creating simple array access."""
        access = ArrayAccessNode(
            array=VariableNode(name="A"),
            index=VariableNode(name="i")
        )
        
        assert access.node_type == NodeType.ARRAY_ACCESS
        assert access.array.name == "A"
    
    def test_create_array_access_2d(self):
        """Test creating 2D array access."""
        inner = ArrayAccessNode(
            array=VariableNode(name="A"),
            index=VariableNode(name="i")
        )
        outer = ArrayAccessNode(
            array=inner,
            index=VariableNode(name="j")
        )
        
        assert isinstance(outer.array, ArrayAccessNode)
    
    def test_create_complex_expression(self):
        """Test creating complex nested expression."""
        # Build: (a + b) * (c - d)
        add = BinaryOpNode(
            operator=OperatorType.ADD,
            left=VariableNode(name="a"),
            right=VariableNode(name="b")
        )
        sub = BinaryOpNode(
            operator=OperatorType.SUBTRACT,
            left=VariableNode(name="c"),
            right=VariableNode(name="d")
        )
        mul = BinaryOpNode(
            operator=OperatorType.MULTIPLY,
            left=add,
            right=sub
        )
        
        assert mul.operator == OperatorType.MULTIPLY
        assert isinstance(mul.left, BinaryOpNode)
        assert isinstance(mul.right, BinaryOpNode)


class TestFunctionCallNode:
    """Tests for FunctionCallNode."""
    
    def test_create_function_call_no_args(self):
        """Test creating function call with no arguments."""
        call = FunctionCallNode(
            function_name="test",
            arguments=[]
        )
        
        assert call.node_type == NodeType.FUNCTION_CALL
        assert call.function_name == "test"
        assert len(call.arguments) == 0
    
    def test_create_function_call_with_args(self):
        """Test creating function call with arguments."""
        call = FunctionCallNode(
            function_name="add",
            arguments=[
                VariableNode(name="a"),
                VariableNode(name="b")
            ]
        )
        
        assert len(call.arguments) == 2
    
    def test_create_recursive_call(self):
        """Test creating recursive function call."""
        call = RecursiveCallNode(
            function_name="factorial",
            arguments=[
                BinaryOpNode(
                    operator=OperatorType.SUBTRACT,
                    left=VariableNode(name="n"),
                    right=LiteralNode(value=1, literal_type="int")
                )
            ],
            reduction_pattern="n-1",
            branching_factor=1
        )
        
        assert call.node_type == NodeType.RECURSIVE_CALL
        assert call.reduction_pattern == "n-1"
        assert call.branching_factor == 1


class TestReturnNode:
    """Tests for ReturnNode."""
    
    def test_create_return_with_value(self):
        """Test creating return with value."""
        ret = ReturnNode(value=VariableNode(name="result"))
        
        assert ret.node_type == NodeType.RETURN
        assert ret.value is not None
    
    def test_create_return_no_value(self):
        """Test creating return without value."""
        ret = ReturnNode()
        
        assert ret.value is None


class TestSourceLocation:
    """Tests for SourceLocation."""
    
    def test_create_source_location(self):
        """Test creating source location."""
        loc = SourceLocation(line=1, column=0)
        
        assert loc.line == 1
        assert loc.column == 0
    
    def test_create_source_location_with_end(self):
        """Test creating source location with end position."""
        loc = SourceLocation(line=1, column=0, end_line=5, end_column=10)
        
        assert loc.end_line == 5
        assert loc.end_column == 10
    
    def test_source_location_str(self):
        """Test source location string representation."""
        loc1 = SourceLocation(line=1, column=0)
        loc2 = SourceLocation(line=1, column=0, end_line=5)
        
        assert "line 1" in str(loc1)
        assert "lines 1-5" in str(loc2)


class TestASTSerialization:
    """Tests for AST serialization."""
    
    def test_serialize_simple_program(self):
        """Test serializing simple program to dict."""
        assign = AssignmentNode(
            target=VariableNode(name="x"),
            value=LiteralNode(value=1, literal_type="int")
        )
        block = BlockNode(statements=[assign])
        program = ProgramNode(global_statements=block)
        
        result = program.model_dump()
        
        assert isinstance(result, dict)
        assert "node_type" in result
        assert "global_statements" in result
    
    def test_serialize_function(self):
        """Test serializing function to dict."""
        func = FunctionNode(
            name="test",
            parameters=[VariableNode(name="x")],
            body=BlockNode(statements=[ReturnNode(value=VariableNode(name="x"))])
        )
        
        result = func.model_dump()
        
        assert result["name"] == "test"
        assert len(result["parameters"]) == 1
    
    def test_serialize_loop(self):
        """Test serializing loop to dict."""
        loop = LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="i"),
            start=LiteralNode(value=1, literal_type="int"),
            end=VariableNode(name="n"),
            estimated_iterations="n",
            body=BlockNode(statements=[])
        )
        
        result = loop.model_dump()
        
        assert result["loop_type"] == LoopType.FOR
        assert result["estimated_iterations"] == "n"
    
    def test_serialize_complex_ast(self):
        """Test serializing complex AST."""
        # Create a function with loop and conditional
        condition = BinaryOpNode(
            operator=OperatorType.GREATER_THAN,
            left=ArrayAccessNode(
                array=VariableNode(name="A"),
                index=VariableNode(name="i")
            ),
            right=LiteralNode(value=0, literal_type="int")
        )
        
        if_stmt = ConditionalNode(
            condition=condition,
            then_branch=BlockNode(statements=[
                FunctionCallNode(function_name="print", arguments=[VariableNode(name="i")])
            ])
        )
        
        loop = LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="i"),
            start=LiteralNode(value=1, literal_type="int"),
            end=VariableNode(name="n"),
            body=BlockNode(statements=[if_stmt])
        )
        
        func = FunctionNode(
            name="printPositive",
            parameters=[VariableNode(name="A"), VariableNode(name="n")],
            body=BlockNode(statements=[loop])
        )
        
        program = ProgramNode(functions=[func])
        
        result = program.model_dump()
        
        assert len(result["functions"]) == 1
        assert result["functions"][0]["name"] == "printPositive"
