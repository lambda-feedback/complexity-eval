"""
AST Builder - Transforms Lark parse tree to AST nodes.

This module provides the Transformer class that converts Lark's parse tree
into our custom AST node types defined in schemas.ast_nodes.
"""

from typing import List, Optional, Any

# Try to import lark, but don't fail if not available
try:
    from lark import Transformer, v_args, Token, Tree
    LARK_AVAILABLE = True
except ImportError:
    LARK_AVAILABLE = False
    # Dummy classes for when lark is not available
    class Transformer:
        pass
    def v_args(inline=False):
        def decorator(cls):
            return cls
        return decorator
    Token = None
    Tree = None

from ..schemas.ast_nodes import (
    ProgramNode,
    FunctionNode,
    BlockNode,
    LoopNode,
    ConditionalNode,
    AssignmentNode,
    ReturnNode,
    FunctionCallNode,
    ExpressionNode,
    VariableNode,
    LiteralNode,
    BinaryOpNode,
    UnaryOpNode,
    ArrayAccessNode,
    SourceLocation,
    NodeType,
    LoopType,
    OperatorType,
)


@v_args(inline=True)
class ASTBuilder(Transformer):
    """
    Transforms Lark parse tree into AST nodes.
    
    Uses the @v_args(inline=True) decorator to receive children as arguments
    instead of a list.
    """
    
    def __init__(self):
        super().__init__()
        self.current_function: Optional[str] = None
        self.function_names: List[str] = []
    
    # =========================================================================
    # Program Structure
    # =========================================================================
    
    def start(self, *items) -> ProgramNode:
        """Build the root program node."""
        functions = []
        statements = []
        
        for item in items:
            if item is None:
                continue
            if isinstance(item, FunctionNode):
                functions.append(item)
            elif isinstance(item, list):
                statements.extend(item)
            else:
                statements.append(item)
        
        global_block = BlockNode(statements=statements) if statements else None
        
        return ProgramNode(
            functions=functions,
            global_statements=global_block
        )
    
    def function_def(self, keyword, name, *rest) -> FunctionNode:
        """Build a function definition node."""
        params = []
        body = None
        
        for item in rest:
            if isinstance(item, list) and all(isinstance(p, VariableNode) for p in item):
                params = item
            elif isinstance(item, BlockNode):
                body = item
            elif isinstance(item, list):
                body = BlockNode(statements=item)
        
        func_name = str(name)
        self.function_names.append(func_name)
        
        return FunctionNode(
            name=func_name,
            parameters=params,
            body=body
        )
    
    def function_keyword(self, keyword) -> str:
        """Extract function keyword."""
        return str(keyword)
    
    def param_list(self, *params) -> List[VariableNode]:
        """Build parameter list."""
        return list(params)
    
    def param(self, name, *rest) -> VariableNode:
        """Build a parameter node."""
        return VariableNode(name=str(name))
    
    # =========================================================================
    # Statements
    # =========================================================================
    
    def statement(self, stmt) -> Any:
        """Pass through statement."""
        return stmt
    
    def block_body(self, *statements) -> BlockNode:
        """Build a block of statements."""
        stmts = []
        for s in statements:
            if s is None:
                continue
            if isinstance(s, list):
                stmts.extend(s)
            else:
                stmts.append(s)
        return BlockNode(statements=stmts)
    
    def assignment(self, target, value) -> AssignmentNode:
        """Build an assignment node."""
        return AssignmentNode(
            target=target,
            value=value,
            operator=OperatorType.ASSIGN
        )
    
    def lvalue(self, val) -> Any:
        """Extract lvalue (variable or array access)."""
        return val
    
    # =========================================================================
    # Control Flow - Conditionals
    # =========================================================================
    
    def if_stmt(self, condition, *rest) -> ConditionalNode:
        """Build an if statement node."""
        then_branch = None
        else_branch = None
        elif_branches = []
        
        for item in rest:
            if item is None:
                continue
            if isinstance(item, ConditionalNode):
                elif_branches.append(item)
            elif isinstance(item, BlockNode):
                if then_branch is None:
                    then_branch = item
                else:
                    else_branch = item
            elif isinstance(item, list):
                block = BlockNode(statements=item)
                if then_branch is None:
                    then_branch = block
                else:
                    else_branch = block
        
        return ConditionalNode(
            condition=condition,
            then_branch=then_branch,
            else_branch=else_branch,
            elif_branches=elif_branches
        )
    
    def elif_clause(self, condition, *rest) -> ConditionalNode:
        """Build an elif clause as a ConditionalNode."""
        body = None
        for item in rest:
            if isinstance(item, BlockNode):
                body = item
            elif isinstance(item, list):
                body = BlockNode(statements=item)
        
        return ConditionalNode(
            condition=condition,
            then_branch=body
        )
    
    def else_clause(self, *rest) -> BlockNode:
        """Build else clause body."""
        for item in rest:
            if isinstance(item, BlockNode):
                return item
            elif isinstance(item, list):
                return BlockNode(statements=item)
        return BlockNode(statements=[])
    
    def then_clause(self, *args) -> None:
        """Then clause is just syntax, return None."""
        return None
    
    def end_clause(self, *args) -> None:
        """End clause is just syntax, return None."""
        return None
    
    def inline_statement(self, stmt) -> Any:
        """Pass through inline statement."""
        return stmt
    
    # =========================================================================
    # Control Flow - Loops
    # =========================================================================
    
    def for_loop(self, header, body, *rest) -> LoopNode:
        """Build a for loop node."""
        # Header contains the loop details
        loop = header
        
        # Set the body
        if isinstance(body, BlockNode):
            loop.body = body
        elif isinstance(body, list):
            loop.body = BlockNode(statements=body)
        
        return loop
    
    def for_header(self, *args) -> LoopNode:
        """Parse for loop header and build LoopNode."""
        # Extract components from args
        iterator = None
        start = None
        end = None
        step = None
        
        for arg in args:
            if isinstance(arg, Token) and arg.type == 'NAME':
                if iterator is None:
                    iterator = VariableNode(name=str(arg))
            elif isinstance(arg, VariableNode):
                if iterator is None:
                    iterator = arg
            elif isinstance(arg, ExpressionNode) or isinstance(arg, (int, float)):
                if start is None:
                    start = arg if isinstance(arg, ExpressionNode) else LiteralNode(value=arg, literal_type="number")
                elif end is None:
                    end = arg if isinstance(arg, ExpressionNode) else LiteralNode(value=arg, literal_type="number")
                elif step is None:
                    step = arg if isinstance(arg, ExpressionNode) else LiteralNode(value=arg, literal_type="number")
        
        return LoopNode(
            loop_type=LoopType.FOR,
            iterator=iterator,
            start=start,
            end=end,
            step=step
        )
    
    def step_clause(self, value) -> ExpressionNode:
        """Extract step value."""
        return value
    
    def do_clause(self, *args) -> None:
        """Do clause is just syntax, return None."""
        return None
    
    def while_loop(self, condition, *rest) -> LoopNode:
        """Build a while loop node."""
        body = None
        for item in rest:
            if isinstance(item, BlockNode):
                body = item
            elif isinstance(item, list):
                body = BlockNode(statements=item)
        
        return LoopNode(
            loop_type=LoopType.WHILE,
            condition=condition,
            body=body
        )
    
    def repeat_loop(self, *rest) -> LoopNode:
        """Build a repeat-until loop node."""
        body = None
        condition = None
        
        for item in rest:
            if isinstance(item, BlockNode):
                body = item
            elif isinstance(item, list):
                body = BlockNode(statements=item)
            elif isinstance(item, ExpressionNode):
                condition = item
        
        return LoopNode(
            loop_type=LoopType.REPEAT_UNTIL,
            condition=condition,
            body=body
        )
    
    def foreach_loop(self, *args) -> LoopNode:
        """Build a for-each loop node."""
        iterator = None
        collection = None
        body = None
        
        for arg in args:
            if isinstance(arg, Token) and arg.type == 'NAME':
                if iterator is None:
                    iterator = VariableNode(name=str(arg))
                elif collection is None:
                    collection = VariableNode(name=str(arg))
            elif isinstance(arg, VariableNode):
                if iterator is None:
                    iterator = arg
                elif collection is None:
                    collection = arg
            elif isinstance(arg, ExpressionNode):
                collection = arg
            elif isinstance(arg, BlockNode):
                body = arg
            elif isinstance(arg, list):
                body = BlockNode(statements=arg)
        
        return LoopNode(
            loop_type=LoopType.FOR_EACH,
            iterator=iterator,
            collection=collection,
            body=body
        )
    
    # =========================================================================
    # Other Statements
    # =========================================================================
    
    def return_stmt(self, value=None) -> ReturnNode:
        """Build a return statement node."""
        return ReturnNode(value=value)
    
    def function_call_stmt(self, call) -> FunctionCallNode:
        """Function call as statement."""
        return call
    
    def print_stmt(self, *args) -> FunctionCallNode:
        """Build a print statement as function call."""
        arguments = [a for a in args if isinstance(a, ExpressionNode)]
        return FunctionCallNode(
            function_name="print",
            arguments=arguments
        )
    
    def swap_stmt(self, *args) -> FunctionCallNode:
        """Build a swap statement as function call."""
        arguments = [a for a in args if isinstance(a, ExpressionNode)]
        return FunctionCallNode(
            function_name="swap",
            arguments=arguments
        )
    
    def block_stmt(self, body) -> BlockNode:
        """Build explicit block statement."""
        return body if isinstance(body, BlockNode) else BlockNode(statements=[body])
    
    # =========================================================================
    # Expressions - Binary Operations
    # =========================================================================
    
    def or_op(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.OR, left=left, right=right)
    
    def and_op(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.AND, left=left, right=right)
    
    def not_op(self, operand) -> UnaryOpNode:
        return UnaryOpNode(operator=OperatorType.NOT, operand=operand)
    
    def eq(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.EQUAL, left=left, right=right)
    
    def ne(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.NOT_EQUAL, left=left, right=right)
    
    def lt(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.LESS_THAN, left=left, right=right)
    
    def le(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.LESS_EQUAL, left=left, right=right)
    
    def gt(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.GREATER_THAN, left=left, right=right)
    
    def ge(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.GREATER_EQUAL, left=left, right=right)
    
    def add(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.ADD, left=left, right=right)
    
    def sub(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.SUBTRACT, left=left, right=right)
    
    def mul(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.MULTIPLY, left=left, right=right)
    
    def div(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.DIVIDE, left=left, right=right)
    
    def floordiv(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.FLOOR_DIVIDE, left=left, right=right)
    
    def mod(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.MODULO, left=left, right=right)
    
    def pow(self, left, right) -> BinaryOpNode:
        return BinaryOpNode(operator=OperatorType.POWER, left=left, right=right)
    
    def neg(self, operand) -> UnaryOpNode:
        return UnaryOpNode(operator=OperatorType.SUBTRACT, operand=operand)
    
    def pos(self, operand) -> ExpressionNode:
        return operand  # Unary + doesn't change anything
    
    # =========================================================================
    # Expressions - Atoms
    # =========================================================================
    
    def number(self, token) -> LiteralNode:
        """Build a number literal."""
        value = float(token) if '.' in str(token) else int(token)
        return LiteralNode(
            value=value,
            literal_type="float" if isinstance(value, float) else "int"
        )
    
    def string(self, token) -> LiteralNode:
        """Build a string literal."""
        # Remove quotes
        value = str(token)[1:-1]
        return LiteralNode(value=value, literal_type="string")
    
    def true(self) -> LiteralNode:
        return LiteralNode(value=True, literal_type="bool")
    
    def false(self) -> LiteralNode:
        return LiteralNode(value=False, literal_type="bool")
    
    def null(self) -> LiteralNode:
        return LiteralNode(value=None, literal_type="null")
    
    def variable(self, token) -> VariableNode:
        """Build a variable reference."""
        return VariableNode(name=str(token))
    
    def array_access(self, array, index) -> ArrayAccessNode:
        """Build an array access node."""
        if isinstance(array, Token):
            array = VariableNode(name=str(array))
        return ArrayAccessNode(array=array, index=index)
    
    def function_call(self, name, *args) -> FunctionCallNode:
        """Build a function call node."""
        arguments = []
        for arg in args:
            if isinstance(arg, list):
                arguments.extend(arg)
            elif isinstance(arg, ExpressionNode):
                arguments.append(arg)
        
        func_name = str(name)
        is_recursive = func_name in self.function_names
        
        return FunctionCallNode(
            function_name=func_name,
            arguments=arguments,
            is_recursive=is_recursive
        )
    
    def arg_list(self, *args) -> List[ExpressionNode]:
        """Build argument list."""
        return [a for a in args if isinstance(a, ExpressionNode)]
    
    # =========================================================================
    # Utility
    # =========================================================================
    
    def NAME(self, token) -> str:
        """Extract name from token."""
        return str(token)
    
    def NUMBER(self, token) -> LiteralNode:
        """Convert NUMBER token."""
        return self.number(token)
    
    def STRING(self, token) -> LiteralNode:
        """Convert STRING token."""
        return self.string(token)
