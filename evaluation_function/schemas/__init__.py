"""
Schemas module for the Algorithm Complexity Evaluation Function.

This module contains all Pydantic data schemas organized by domain:
- ast_nodes: Abstract Syntax Tree node definitions
- complexity: Complexity analysis types and results
- input_schema: Request schemas (response, answer, params)
- output_schema: Response schemas (evaluation results)
"""

from .ast_nodes import (
    ASTNode,
    ProgramNode,
    FunctionNode,
    LoopNode,
    ConditionalNode,
    AssignmentNode,
    ExpressionNode,
    FunctionCallNode,
    ReturnNode,
    BlockNode,
    VariableNode,
    LiteralNode,
    BinaryOpNode,
    UnaryOpNode,
    ArrayAccessNode,
    RecursiveCallNode,
    NodeType,
    LoopType,
    OperatorType,
    SourceLocation,
)

from .complexity import (
    ComplexityClass,
    ComplexityExpression,
    ComplexityFactor,
    LoopComplexity,
    RecursionComplexity,
    TimeComplexity,
    SpaceComplexity,
    ComplexityResult,
)

from .input_schema import (
    StudentResponse,
    ExpectedAnswer,
    EvaluationParams,
)

from .output_schema import (
    EvaluationResult,
    ComplexityAnalysis,
    TimeComplexityResult,
    SpaceComplexityResult,
    ConstructAnalysis,
    FeedbackItem,
    ParseResult,
)

__all__ = [
    # AST Nodes
    "ASTNode",
    "ProgramNode",
    "FunctionNode",
    "LoopNode",
    "ConditionalNode",
    "AssignmentNode",
    "ExpressionNode",
    "FunctionCallNode",
    "ReturnNode",
    "BlockNode",
    "VariableNode",
    "LiteralNode",
    "BinaryOpNode",
    "UnaryOpNode",
    "ArrayAccessNode",
    "RecursiveCallNode",
    "NodeType",
    "LoopType",
    "OperatorType",
    "SourceLocation",
    # Complexity
    "ComplexityClass",
    "ComplexityExpression",
    "ComplexityFactor",
    "LoopComplexity",
    "RecursionComplexity",
    "TimeComplexity",
    "SpaceComplexity",
    "ComplexityResult",
    # Input
    "StudentResponse",
    "ExpectedAnswer",
    "EvaluationParams",
    # Output
    "EvaluationResult",
    "ComplexityAnalysis",
    "TimeComplexityResult",
    "SpaceComplexityResult",
    "ConstructAnalysis",
    "FeedbackItem",
    "ParseResult",
]
