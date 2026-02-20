"""
Input Schemas for Algorithm Complexity Evaluation using Pydantic.

This module defines the schemas for:
- StudentResponse: What the student submits
- ExpectedAnswer: The correct answer set by instructor
- EvaluationParams: Configuration parameters for evaluation
"""

from typing import Dict, Optional, List, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict

# RuntimeValue can be any JSON-serializable Python value
RuntimeValue = Union[int, float, str, bool, list, dict, None]


class ExecutionTestCase(BaseModel):
    """Test case for pseudocode execution."""
    
    initial_variables: Dict[str, RuntimeValue] = Field(default_factory=dict)
    expected_variables: Dict[str, RuntimeValue] = Field(default_factory=dict)
    expected_output: List[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StudentResponse(BaseModel):
    """
    Schema for student's submission.
    """

    pseudocode: str = Field(..., description="The pseudocode submitted by the student")

    time_complexity: Optional[str] = Field(
        default=None,
        description="Student's answer for time complexity, e.g., 'O(n^2)'"
    )
    space_complexity: Optional[str] = Field(
        default=None,
        description="Student's answer for space complexity, e.g., 'O(1)'"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Student's explanation of their complexity analysis"
    )

    @field_validator("pseudocode")
    @classmethod
    def pseudocode_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Pseudocode cannot be empty")
        return v.strip()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pseudocode": "FOR i = 1 TO n DO\n  FOR j = 1 TO n DO\n    sum = sum + A[i][j]",
                "time_complexity": "O(n^2)",
                "space_complexity": "O(1)",
                "explanation": "Two nested loops each iterating n times gives O(n²)"
            }
        }
    )


class EvaluationParams(BaseModel):
    """
    Configuration parameters for the evaluation.
    """

    require_time_complexity: bool = True
    require_space_complexity: bool = True

    show_detailed_feedback: bool = True

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "require_time_complexity": True,
                "require_space_complexity": True,
                "show_detailed_feedback": True
            }
        }
    )


class ExpectedAnswer(BaseModel):
    """
    Schema for the expected/correct answer set by instructor.
    """

    expected_time_complexity: str = Field(
        ...,
        description="Expected time complexity in Big-O notation"
    )
    expected_space_complexity: str = Field(
        default="O(1)",
        description="Expected space complexity in Big-O notation"
    )

    # algorithm_description: Optional[str] = None
    # algorithm_type: Optional[str] = None

    # maybe a TODO? can check for the presence of constructs
    # expected_constructs: List[str] = Field(default_factory=list)

    test_cases: List[ExecutionTestCase] = Field(
        default_factory=list,
        description="Test cases for code execution correctness (if applicable)"
    )

    eval_options: Optional[EvaluationParams] = None

    # def get_all_acceptable_time(self) -> List[str]:
    #     return [self.expected_time_complexity] + self.acceptable_time_alternatives

    # def get_all_acceptable_space(self) -> List[str]:
    #     return [self.expected_space_complexity] + self.acceptable_space_alternatives

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "expected_time_complexity": "O(n^2)",
                "expected_space_complexity": "O(1)",
                "eval_options": {
                    "require_time_complexity": True,
                    "require_space_complexity": True,
                    "show_detailed_feedback": True
                }
                # "acceptable_time_alternatives": ["O(n*n)", "O(n²)"],
                # "acceptable_space_alternatives": [],
                # "algorithm_description": "Matrix traversal with nested loops",
                # "algorithm_type": "iteration",
                # "expected_constructs": ["nested_loop"],
                # "time_complexity_weight": 0.6,
                # "space_complexity_weight": 0.4
            }
        }
    )
