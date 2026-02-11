"""
Abstract Syntax Tree (AST) Node Schemas using Pydantic.

This module defines all AST node types for representing pseudocode structure.
The AST is designed to be extensible for future analysis beyond complexity.

Node Hierarchy:
    ASTNode (base)
    ├── ProgramNode (root)
    ├── FunctionNode (function definitions)
    ├── BlockNode (statement blocks)
    ├── StatementNode (base for statements)
    │   ├── LoopNode (for, while, repeat)
    │   ├── ConditionalNode (if, else, elif)
    │   ├── AssignmentNode (variable assignment)
    │   ├── ReturnNode (return statements)
    │   └── FunctionCallNode (function/procedure calls)
    └── ExpressionNode (base for expressions)
        ├── VariableNode (variable references)
        ├── LiteralNode (constants)
        ├── BinaryOpNode (binary operations)
        ├── UnaryOpNode (unary operations)
        ├── ArrayAccessNode (array indexing)
        └── RecursiveCallNode (recursive function calls)
"""

from typing import Optional, List, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Enumeration of all AST node types."""
    PROGRAM = "program"
    FUNCTION = "function"
    BLOCK = "block"
    LOOP = "loop"
    CONDITIONAL = "conditional"
    ASSIGNMENT = "assignment"
    RETURN = "return"
    FUNCTION_CALL = "function_call"
    RECURSIVE_CALL = "recursive_call"
    VARIABLE = "variable"
    LITERAL = "literal"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    ARRAY_ACCESS = "array_access"
    EXPRESSION = "expression"


class LoopType(str, Enum):
    """Types of loop constructs."""
    FOR = "for"              # for i = 1 to n
    FOR_EACH = "for_each"    # for each x in collection
    WHILE = "while"          # while condition
    DO_WHILE = "do_while"    # do ... while condition
    REPEAT_UNTIL = "repeat_until"  # repeat ... until condition
    UNKNOWN = "unknown"


class OperatorType(str, Enum):
    """Types of operators."""
    # Arithmetic
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "%"
    POWER = "^"
    FLOOR_DIVIDE = "//"
    
    # Comparison
    EQUAL = "=="
    NOT_EQUAL = "!="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    
    # Logical
    AND = "and"
    OR = "or"
    NOT = "not"
    
    # Assignment (for compound assignment)
    ASSIGN = "="
    ADD_ASSIGN = "+="
    SUB_ASSIGN = "-="
    MUL_ASSIGN = "*="
    DIV_ASSIGN = "/="


class SourceLocation(BaseModel):
    """Location information for AST nodes."""
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    
    def __str__(self) -> str:
        if self.end_line:
            return f"lines {self.line}-{self.end_line}"
        return f"line {self.line}"

    class Config:
        frozen = True


class ASTNode(BaseModel):
    """
    Base class for all AST nodes.
    
    Attributes:
        node_type: The type of this node
        location: Source location information
        metadata: Additional metadata for analysis
    """
    node_type: NodeType
    location: Optional[SourceLocation] = None
    metadata: dict = Field(default_factory=dict)

    class Config:
        use_enum_values = False
        extra = "allow"


class ExpressionNode(ASTNode):
    """Base class for expression nodes."""
    node_type: NodeType = NodeType.EXPRESSION
    
    def get_complexity_contribution(self) -> str:
        """Get the complexity contribution of this expression."""
        return "O(1)"  # Default: constant time


class VariableNode(ExpressionNode):
    """
    Variable reference node.
    
    Examples:
        - i
        - count
        - array
    """
    node_type: NodeType = Field(default=NodeType.VARIABLE, frozen=True)
    name: str


class LiteralNode(ExpressionNode):
    """
    Literal value node (constants).
    
    Examples:
        - 42
        - "hello"
        - true
        - 3.14
    """
    node_type: NodeType = Field(default=NodeType.LITERAL, frozen=True)
    value: Any = None
    literal_type: str = "unknown"  # int, float, string, bool


class BinaryOpNode(ExpressionNode):
    """
    Binary operation node.
    
    Examples:
        - a + b
        - x * y
        - i < n
    """
    node_type: NodeType = Field(default=NodeType.BINARY_OP, frozen=True)
    operator: OperatorType
    left: Optional["ExpressionNode"] = None
    right: Optional["ExpressionNode"] = None


class UnaryOpNode(ExpressionNode):
    """
    Unary operation node.
    
    Examples:
        - -x
        - not condition
        - !flag
    """
    node_type: NodeType = Field(default=NodeType.UNARY_OP, frozen=True)
    operator: OperatorType
    operand: Optional["ExpressionNode"] = None


class ArrayAccessNode(ExpressionNode):
    """
    Array/list access node.
    
    Examples:
        - A[i]
        - matrix[i][j]
        - list[index + 1]
    """
    node_type: NodeType = Field(default=NodeType.ARRAY_ACCESS, frozen=True)
    array: Optional["ExpressionNode"] = None  # The array being accessed
    index: Optional["ExpressionNode"] = None  # The index expression


class FunctionCallNode(ASTNode):
    """
    Function/procedure call node.
    
    Examples:
        - print(x)
        - swap(A[i], A[j])
        - max(a, b)
    """
    node_type: NodeType = Field(default=NodeType.FUNCTION_CALL, frozen=True)
    function_name: str
    arguments: List["ExpressionNode"] = Field(default_factory=list)
    is_recursive: bool = False  # Set during analysis


class RecursiveCallNode(FunctionCallNode):
    """
    Recursive function call node.
    Inherits from FunctionCallNode with additional recursion metadata.
    
    Examples:
        - fib(n-1) + fib(n-2)
        - mergeSort(arr, left, mid)
    """
    node_type: NodeType = Field(default=NodeType.RECURSIVE_CALL, frozen=True)
    reduction_pattern: Optional[str] = None  # e.g., "n-1", "n/2"
    branching_factor: int = 1  # Number of recursive calls


class AssignmentNode(ASTNode):
    """
    Assignment statement node.
    
    Examples:
        - x = 5
        - A[i] = A[i] + 1
        - sum = sum + value
    """
    node_type: NodeType = Field(default=NodeType.ASSIGNMENT, frozen=True)
    target: Optional[Union[VariableNode, ArrayAccessNode]] = None
    value: Optional[ExpressionNode] = None
    operator: OperatorType = OperatorType.ASSIGN


class ReturnNode(ASTNode):
    """
    Return statement node.
    
    Examples:
        - return x
        - return fib(n-1) + fib(n-2)
        - return (implicit return)
    """
    node_type: NodeType = Field(default=NodeType.RETURN, frozen=True)
    value: Optional[ExpressionNode] = None


class BlockNode(ASTNode):
    """
    Block of statements node.
    
    Represents a sequence of statements (function body, loop body, etc.)
    """
    node_type: NodeType = Field(default=NodeType.BLOCK, frozen=True)
    statements: List["ASTNode"] = Field(default_factory=list)


class LoopNode(ASTNode):
    """
    Loop construct node.
    
    Represents for, while, do-while, and repeat-until loops.
    
    Examples:
        FOR i = 1 TO n DO
            ...
        
        WHILE condition DO
            ...
    """
    node_type: NodeType = Field(default=NodeType.LOOP, frozen=True)
    loop_type: LoopType = LoopType.FOR
    
    # For 'for' loops
    iterator: Optional[VariableNode] = None
    start: Optional[ExpressionNode] = None
    end: Optional[ExpressionNode] = None
    step: Optional[ExpressionNode] = None
    
    # For 'for-each' loops
    collection: Optional[ExpressionNode] = None
    
    # For 'while' and condition-based loops
    condition: Optional[ExpressionNode] = None
    
    # Loop body
    body: Optional["BlockNode"] = None
    
    # Analysis metadata
    estimated_iterations: Optional[str] = None  # e.g., "n", "n/2", "log(n)"
    nesting_level: int = 0


class ConditionalNode(ASTNode):
    """
    Conditional statement node (if/else).
    
    Examples:
        IF condition THEN
            ...
        ELSE
            ...
    """
    node_type: NodeType = Field(default=NodeType.CONDITIONAL, frozen=True)
    condition: Optional[ExpressionNode] = None
    then_branch: Optional["BlockNode"] = None
    else_branch: Optional["BlockNode"] = None  # Can be another ConditionalNode for elif
    elif_branches: List["ConditionalNode"] = Field(default_factory=list)


class FunctionNode(ASTNode):
    """
    Function/procedure definition node.
    
    Examples:
        FUNCTION bubbleSort(A[1..n])
            ...
        
        ALGORITHM quickSort(A, low, high)
            ...
    """
    node_type: NodeType = Field(default=NodeType.FUNCTION, frozen=True)
    name: str
    parameters: List[VariableNode] = Field(default_factory=list)
    body: Optional[BlockNode] = None
    return_type: Optional[str] = None
    is_recursive: bool = False


class ProgramNode(ASTNode):
    """
    Root node of the AST representing the entire program.
    
    Contains all function definitions and global statements.
    """
    node_type: NodeType = Field(default=NodeType.PROGRAM, frozen=True)
    functions: List[FunctionNode] = Field(default_factory=list)
    global_statements: Optional[BlockNode] = None

# Update forward references for Pydantic
BinaryOpNode.model_rebuild()
UnaryOpNode.model_rebuild()
ArrayAccessNode.model_rebuild()
FunctionCallNode.model_rebuild()
RecursiveCallNode.model_rebuild()
AssignmentNode.model_rebuild()
ReturnNode.model_rebuild()
BlockNode.model_rebuild()
LoopNode.model_rebuild()
ConditionalNode.model_rebuild()
FunctionNode.model_rebuild()
ProgramNode.model_rebuild()
