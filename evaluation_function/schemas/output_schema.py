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
from pydantic import BaseModel, Field, ConfigDict

from ..schemas.input_schema import RuntimeValue
from .complexity import ComplexityClass, TimeComplexity, SpaceComplexity
from .ast_nodes import ProgramNode


class FeedbackLevel(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    HINT = "hint"


class FeedbackItem(BaseModel):
    level: FeedbackLevel = Field(default=FeedbackLevel.INFO)
    message: str
    category: Optional[str] = None
    location: Optional[str] = None
    suggestion: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "level": "warning",
                "message": "Your time complexity answer differs from the expected answer",
                "category": "time_complexity",
                "location": None,
                "suggestion": "Consider the nested loops - each contributes O(n)"
            }
        }
    )


class ConstructAnalysis(BaseModel):
    construct_type: str
    description: str = ""
    location: Optional[str] = None
    complexity_contribution: ComplexityClass = ComplexityClass.CONSTANT
    details: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
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
    )


class TimeComplexityResult(BaseModel):
    is_correct: bool
    student_answer: Optional[str] = None
    expected_answer: str
    detected_complexity: Optional[str] = None

    student_normalized: Optional[ComplexityClass] = None
    expected_normalized: ComplexityClass

    analysis: Optional[TimeComplexity] = None
    feedback: str = ""

    model_config = ConfigDict(
        json_schema_extra={
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
    )


class SpaceComplexityResult(BaseModel):
    is_correct: bool
    student_answer: Optional[str] = None
    expected_answer: str
    detected_complexity: Optional[str] = None

    student_normalized: Optional[ComplexityClass] = None
    expected_normalized: ComplexityClass

    analysis: Optional[SpaceComplexity] = None
    feedback: str = ""

    model_config = ConfigDict(
        json_schema_extra={
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
    )


class InterpreterResult(BaseModel):
    variables: Dict[str, RuntimeValue] = Field(default_factory=dict)
    output: Optional[List[str]] = Field(default_factory=list)


class TestCaseResult(BaseModel):
    input_data: Dict[str, RuntimeValue] = Field(default_factory=dict)
    expected_output: InterpreterResult = Field(default_factory=InterpreterResult)
    actual_output: InterpreterResult = Field(default_factory=InterpreterResult)
    passed: bool
    error_message: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __repr__(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return (
            f"TestCaseResult({status}, "
            f"input={self.input_data}, "
            f"expect={self.expected_output}, "
            f"got={self.actual_output}, "
            f"error={self.error_message})"
        )


class ParseResult(BaseModel):
    success: bool
    ast: Optional[ProgramNode] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    normalized_code: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CodeCorrectnessResult(BaseModel):
    parse_success: bool
    parse_errors: List[str] = Field(default_factory=list)
    parse_warnings: List[str] = Field(default_factory=list)
    normalized_code: Optional[str] = None
    execution_results: List[TestCaseResult] = Field(default_factory=list)
    is_correct: bool
    feedback: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __repr__(self):
        status = "✅ CORRECT" if self.is_correct else "❌ INCORRECT"
        return (
            f"CodeCorrectnessResult({status}, "
            f"parse_success={self.parse_success}, "
            f"tests={len(self.execution_results)})"
        )


class ComplexityAnalysis(BaseModel):
    time_complexity: TimeComplexity
    space_complexity: SpaceComplexity
    constructs: List[ConstructAnalysis] = Field(default_factory=list)
    algorithm_type: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    parse_result: Optional[ParseResult] = None

from pydantic import BaseModel, Field
from typing import List, Optional


class TestCaseFeedback(BaseModel):
    index: int
    passed: bool
    error_message: str = Field(default="")


class SectionFeedback(BaseModel):
    importance: str
    title: str
    content: str


class EvaluationResult(BaseModel):
    is_correct: bool
    overall_message: str
    time_complexity: Optional[TimeComplexityResult] = Field(default=None)
    space_complexity: Optional[SpaceComplexityResult] = Field(default=None)
    test_cases: List[TestCaseFeedback] = Field(default_factory=list)
    detailed_sections: List[SectionFeedback] = Field(default_factory=list)