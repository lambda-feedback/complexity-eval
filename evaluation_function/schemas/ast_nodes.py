"""
Abstract Syntax Tree (AST) Node Schemas using Pydantic.

Corrected Design:
- FunctionCallNode is an ExpressionNode
- ExpressionStatementNode wraps expressions used as statements
- Proper separation of StatementNode and ExpressionNode
- Fixed ConditionalNode to have optional condition for parser compatibility
"""

from typing import Optional, List, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


# ============================================================
# ENUMS
# ============================================================

class NodeType(str, Enum):
    PROGRAM = "program"
    FUNCTION = "function"
    BLOCK = "block"

    # Statements
    LOOP = "loop"
    CONDITIONAL = "conditional"
    ASSIGNMENT = "assignment"
    RETURN = "return"
    EXPRESSION_STATEMENT = "expression_statement"

    # Expressions
    VARIABLE = "variable"
    LITERAL = "literal"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    ARRAY_ACCESS = "array_access"
    FUNCTION_CALL = "function_call"
    RECURSIVE_CALL = "recursive_call"


class LoopType(str, Enum):
    FOR = "for"
    FOR_EACH = "for_each"
    WHILE = "while"
    DO_WHILE = "do_while"
    REPEAT_UNTIL = "repeat_until"
    UNKNOWN = "unknown"


class OperatorType(str, Enum):
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

    # Assignment
    ASSIGN = "="
    ADD_ASSIGN = "+="
    SUB_ASSIGN = "-="
    MUL_ASSIGN = "*="
    DIV_ASSIGN = "/="


# ============================================================
# SOURCE LOCATION
# ============================================================

class SourceLocation(BaseModel):
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    class Config:
        frozen = True


# ============================================================
# BASE NODES
# ============================================================

class ASTNode(BaseModel):
    node_type: NodeType
    location: Optional[SourceLocation] = None
    metadata: dict = Field(default_factory=dict)

    class Config:
        extra = "allow"


class StatementNode(ASTNode):
    """Base class for all statements."""
    pass


class ExpressionNode(ASTNode):
    """Base class for all expressions."""
    pass


# ============================================================
# EXPRESSION NODES
# ============================================================

class VariableNode(ExpressionNode):
    node_type: NodeType = Field(default=NodeType.VARIABLE, frozen=True)
    name: str


class LiteralNode(ExpressionNode):
    node_type: NodeType = Field(default=NodeType.LITERAL, frozen=True)
    value: Any = None
    literal_type: str = "unknown"


class BinaryOpNode(ExpressionNode):
    node_type: NodeType = Field(default=NodeType.BINARY_OP, frozen=True)
    operator: OperatorType
    left: ExpressionNode
    right: ExpressionNode


class UnaryOpNode(ExpressionNode):
    node_type: NodeType = Field(default=NodeType.UNARY_OP, frozen=True)
    operator: OperatorType
    operand: ExpressionNode


class ArrayAccessNode(ExpressionNode):
    node_type: NodeType = Field(default=NodeType.ARRAY_ACCESS, frozen=True)
    array: ExpressionNode
    index: ExpressionNode


class FunctionCallNode(ExpressionNode):
    node_type: NodeType = Field(default=NodeType.FUNCTION_CALL, frozen=True)
    function_name: str
    arguments: List[ExpressionNode] = Field(default_factory=list)
    is_recursive: bool = False


class RecursiveCallNode(FunctionCallNode):
    node_type: NodeType = Field(default=NodeType.RECURSIVE_CALL, frozen=True)
    reduction_pattern: Optional[str] = None
    branching_factor: int = 1


# ============================================================
# STATEMENT NODES
# ============================================================

class ExpressionStatementNode(StatementNode):
    node_type: NodeType = Field(default=NodeType.EXPRESSION_STATEMENT, frozen=True)
    expression: ExpressionNode


class AssignmentNode(StatementNode):
    node_type: NodeType = Field(default=NodeType.ASSIGNMENT, frozen=True)
    target: Union[VariableNode, ArrayAccessNode]
    value: ExpressionNode
    operator: OperatorType = OperatorType.ASSIGN


class ReturnNode(StatementNode):
    node_type: NodeType = Field(default=NodeType.RETURN, frozen=True)
    value: Optional[ExpressionNode] = None


class BlockNode(ASTNode):
    node_type: NodeType = Field(default=NodeType.BLOCK, frozen=True)
    statements: List[StatementNode] = Field(default_factory=list)


class LoopNode(StatementNode):
    node_type: NodeType = Field(default=NodeType.LOOP, frozen=True)

    loop_type: LoopType = LoopType.FOR

    # For classic for loops
    iterator: Optional[VariableNode] = None
    start: Optional[ExpressionNode] = None
    end: Optional[ExpressionNode] = None
    step: Optional[ExpressionNode] = None

    # For for-each
    collection: Optional[ExpressionNode] = None

    # For while / do-while
    condition: Optional[ExpressionNode] = None

    body: Optional["BlockNode"] = None

    # Analysis metadata
    estimated_iterations: Optional[str] = None
    nesting_level: int = 0


class ConditionalNode(StatementNode):
    node_type: NodeType = Field(default=NodeType.CONDITIONAL, frozen=True)

    # Made optional to handle parsing edge cases
    condition: Optional[ExpressionNode] = None
    then_branch: BlockNode
    else_branch: Optional[BlockNode] = None
    elif_branches: List["ConditionalNode"] = Field(default_factory=list)


class FunctionNode(ASTNode):
    node_type: NodeType = Field(default=NodeType.FUNCTION, frozen=True)

    name: str
    parameters: List[VariableNode] = Field(default_factory=list)
    body: Optional[BlockNode] = None
    return_type: Optional[str] = None
    is_recursive: bool = False


class ProgramNode(ASTNode):
    node_type: NodeType = Field(default=NodeType.PROGRAM, frozen=True)

    functions: List[FunctionNode] = Field(default_factory=list)
    global_statements: Optional[BlockNode] = None


# ============================================================
# FORWARD REF REBUILD
# ============================================================

BinaryOpNode.model_rebuild()
UnaryOpNode.model_rebuild()
ArrayAccessNode.model_rebuild()
FunctionCallNode.model_rebuild()
RecursiveCallNode.model_rebuild()
ExpressionStatementNode.model_rebuild()
AssignmentNode.model_rebuild()
ReturnNode.model_rebuild()
BlockNode.model_rebuild()
LoopNode.model_rebuild()
ConditionalNode.model_rebuild()
FunctionNode.model_rebuild()
ProgramNode.model_rebuild()