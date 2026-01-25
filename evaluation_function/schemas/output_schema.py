"""
Output Schemas for Algorithm Complexity Evaluation using Pydantic.

This module defines the schemas for:
- EvaluationResult: The complete evaluation response
- ComplexityAnalysis: Detailed complexity analysis
- TimeComplexityResult: Time complexity evaluation details
- SpaceComplexityResult: Space complexity evaluation details
- ConstructAnalysis: Analysis of detected code constructs
- FeedbackItem: Individual feedback messages
- ParseResult: Result of pseudocode parsing
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

from .complexity import ComplexityClass, TimeComplexity, SpaceComplexity
from .ast_nodes import ProgramNode


class FeedbackLevel(str, Enum):
    """Severity level for feedback items."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    HINT = "hint"


class FeedbackItem(BaseModel):
    """
    Individual feedback message for the student.
    
    Provides specific, actionable feedback about the submission.
    """
    level: FeedbackLevel = Field(
        default=FeedbackLevel.INFO,
        description="Severity level of the feedback"
    )
    message: str = Field(
        ...,
        description="The feedback message"
    )
    category: Optional[str] = Field(
        default=None,
        description="Category: time_complexity, space_complexity, syntax, etc."
    )
    location: Optional[str] = Field(
        default=None,
        description="Location in code this feedback refers to"
    )
    suggestion: Optional[str] = Field(
        default=None,
        description="Suggested improvement or correction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "level": "warning",
                "message": "Your time complexity answer differs from the expected answer",
                "category": "time_complexity",
                "location": None,
                "suggestion": "Consider the nested loops - each contributes O(n)"
            }
        }


class ConstructAnalysis(BaseModel):
    """
    Analysis of a specific code construct (loop, recursion, etc.).
    """
    construct_type: str = Field(
        ...,
        description="Type of construct: loop, nested_loop, recursion, etc."
    )
    description: str = Field(
        default="",
        description="Human-readable description of the construct"
    )
    location: Optional[str] = Field(
        default=None,
        description="Location in the pseudocode"
    )
    complexity_contribution: ComplexityClass = Field(
        default=ComplexityClass.CONSTANT,
        description="Complexity contribution of this construct"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about the construct"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "construct_type": "nested_loop",
                "description": "Two nested FOR loops iterating from 1 to n",
                "location": "lines 1-4",
                "complexity_contribution": "O(n²)",
                "details": {
                    "outer_loop": {"variable": "i", "iterations": "n"},
                    "inner_loop": {"variable": "j", "iterations": "n"}
                }
            }
        }


class TimeComplexityResult(BaseModel):
    """
    Detailed result for time complexity evaluation.
    """
    is_correct: bool = Field(
        ...,
        description="Whether the student's answer is correct"
    )
    student_answer: Optional[str] = Field(
        default=None,
        description="The student's stated time complexity"
    )
    expected_answer: str = Field(
        ...,
        description="The expected time complexity"
    )
    detected_complexity: Optional[str] = Field(
        default=None,
        description="Complexity detected from pseudocode analysis"
    )
    
    # Normalized values for comparison
    student_normalized: Optional[ComplexityClass] = Field(
        default=None,
        description="Student's answer normalized to ComplexityClass"
    )
    expected_normalized: ComplexityClass = Field(
        ...,
        description="Expected answer normalized to ComplexityClass"
    )
    
    # Detailed analysis
    analysis: Optional[TimeComplexity] = Field(
        default=None,
        description="Detailed time complexity analysis"
    )
    
    feedback: str = Field(
        default="",
        description="Explanation of the result"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_correct": True,
                "student_answer": "O(n^2)",
                "expected_answer": "O(n²)",
                "detected_complexity": "O(n²)",
                "student_normalized": "O(n²)",
                "expected_normalized": "O(n²)",
                "analysis": None,
                "feedback": "Correct! The nested loops give O(n²) time complexity."
            }
        }


class SpaceComplexityResult(BaseModel):
    """
    Detailed result for space complexity evaluation.
    """
    is_correct: bool = Field(
        ...,
        description="Whether the student's answer is correct"
    )
    student_answer: Optional[str] = Field(
        default=None,
        description="The student's stated space complexity"
    )
    expected_answer: str = Field(
        ...,
        description="The expected space complexity"
    )
    detected_complexity: Optional[str] = Field(
        default=None,
        description="Complexity detected from pseudocode analysis"
    )
    
    # Normalized values
    student_normalized: Optional[ComplexityClass] = Field(
        default=None,
        description="Student's answer normalized to ComplexityClass"
    )
    expected_normalized: ComplexityClass = Field(
        ...,
        description="Expected answer normalized to ComplexityClass"
    )
    
    # Detailed analysis
    analysis: Optional[SpaceComplexity] = Field(
        default=None,
        description="Detailed space complexity analysis"
    )
    
    feedback: str = Field(
        default="",
        description="Explanation of the result"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_correct": True,
                "student_answer": "O(1)",
                "expected_answer": "O(1)",
                "detected_complexity": "O(1)",
                "student_normalized": "O(1)",
                "expected_normalized": "O(1)",
                "analysis": None,
                "feedback": "Correct! Only constant extra space is used."
            }
        }


class ParseResult(BaseModel):
    """
    Result of parsing the pseudocode.
    """
    success: bool = Field(
        ...,
        description="Whether parsing was successful"
    )
    ast: Optional[ProgramNode] = Field(
        default=None,
        description="The parsed AST (if successful)"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Parse errors encountered"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Parse warnings"
    )
    normalized_code: Optional[str] = Field(
        default=None,
        description="Preprocessed/normalized pseudocode"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "ast": None,
                "errors": [],
                "warnings": ["Inconsistent indentation detected"],
                "normalized_code": "for i = 1 to n do\n  for j = 1 to n do\n    sum = sum + a[i][j]"
            }
        }


class ComplexityAnalysis(BaseModel):
    """
    Complete complexity analysis from pseudocode.
    """
    time_complexity: TimeComplexity = Field(
        ...,
        description="Time complexity analysis"
    )
    space_complexity: SpaceComplexity = Field(
        ...,
        description="Space complexity analysis"
    )
    
    # Detected constructs
    constructs: List[ConstructAnalysis] = Field(
        default_factory=list,
        description="List of detected code constructs"
    )
    
    # Overall assessment
    algorithm_type: Optional[str] = Field(
        default=None,
        description="Detected algorithm type"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis"
    )
    
    # Parse info
    parse_result: Optional[ParseResult] = Field(
        default=None,
        description="Parsing result details"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "time_complexity": {
                    "overall": "O(n²)",
                    "expression": "O(n²)",
                    "explanation": "Two nested loops each iterating n times"
                },
                "space_complexity": {
                    "overall": "O(1)",
                    "expression": "O(1)",
                    "explanation": "Only scalar variables used"
                },
                "constructs": [
                    {
                        "construct_type": "nested_loop",
                        "description": "Two nested FOR loops",
                        "complexity_contribution": "O(n²)"
                    }
                ],
                "algorithm_type": "iteration",
                "confidence": 0.95
            }
        }


class EvaluationResult(BaseModel):
    """
    Complete evaluation result returned by the evaluation function.
    
    This is the main output schema conforming to Lambda Feedback requirements.
    """
    # Core result
    is_correct: bool = Field(
        ...,
        description="Overall correctness of the submission"
    )
    
    # Detailed results
    time_complexity_result: Optional[TimeComplexityResult] = Field(
        default=None,
        description="Time complexity evaluation details"
    )
    space_complexity_result: Optional[SpaceComplexityResult] = Field(
        default=None,
        description="Space complexity evaluation details"
    )
    
    # Score (0.0 to 1.0)
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Numerical score from 0 to 1"
    )
    
    # Analysis from pseudocode (if enabled)
    analysis: Optional[ComplexityAnalysis] = Field(
        default=None,
        description="Detailed complexity analysis from pseudocode"
    )
    
    # AST (if requested)
    ast: Optional[ProgramNode] = Field(
        default=None,
        description="Parsed AST (only included if show_ast is True)"
    )
    
    # Feedback
    feedback: str = Field(
        default="",
        description="Overall feedback message"
    )
    feedback_items: List[FeedbackItem] = Field(
        default_factory=list,
        description="List of specific feedback items"
    )
    
    # Warnings and errors
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings during evaluation"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Errors encountered during evaluation"
    )
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the evaluation"
    )
    
    def to_lambda_feedback_response(self) -> Dict[str, Any]:
        """
        Convert to Lambda Feedback expected response format.
        
        Returns a dict with is_correct and optional feedback.
        """
        response = {
            "is_correct": self.is_correct,
        }
        
        if self.feedback:
            response["feedback"] = self.feedback
        
        # Add detailed results if available
        if self.time_complexity_result:
            response["time_complexity"] = {
                "is_correct": self.time_complexity_result.is_correct,
                "student_answer": self.time_complexity_result.student_answer,
                "expected_answer": self.time_complexity_result.expected_answer,
                "feedback": self.time_complexity_result.feedback,
            }
        
        if self.space_complexity_result:
            response["space_complexity"] = {
                "is_correct": self.space_complexity_result.is_correct,
                "student_answer": self.space_complexity_result.student_answer,
                "expected_answer": self.space_complexity_result.expected_answer,
                "feedback": self.space_complexity_result.feedback,
            }
        
        if self.analysis:
            response["analysis"] = {
                "detected_time_complexity": self.analysis.time_complexity.expression,
                "detected_space_complexity": self.analysis.space_complexity.expression,
                "constructs": [c.model_dump() for c in self.analysis.constructs],
            }
        
        return response
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_correct": True,
                "time_complexity_result": {
                    "is_correct": True,
                    "student_answer": "O(n^2)",
                    "expected_answer": "O(n²)",
                    "detected_complexity": "O(n²)",
                    "feedback": "Correct!"
                },
                "space_complexity_result": {
                    "is_correct": True,
                    "student_answer": "O(1)",
                    "expected_answer": "O(1)",
                    "detected_complexity": "O(1)",
                    "feedback": "Correct!"
                },
                "score": 1.0,
                "feedback": "Excellent! Both time and space complexity are correct.",
                "feedback_items": [],
                "warnings": [],
                "errors": []
            }
        }
