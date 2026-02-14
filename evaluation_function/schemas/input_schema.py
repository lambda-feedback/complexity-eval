"""
Input Schemas for Algorithm Complexity Evaluation using Pydantic.

This module defines the schemas for:
- StudentResponse: What the student submits
- ExpectedAnswer: The correct answer set by instructor
- EvaluationParams: Configuration parameters for evaluation
"""
from typing import Dict, Optional, List, Any, Union
from pydantic import BaseModel, Field, field_validator

# RuntimeValue can be any JSON-serializable Python value
RuntimeValue = Union[int, float, str, bool, list, dict, None]

class ExecutionTestCase(BaseModel):
    """Test case for pseudocode execution."""
    
    initial_variables: Dict[str, RuntimeValue] = Field(default_factory=dict)
    expected_variables: Optional[Dict[str, RuntimeValue]] = None
    expected_output: Optional[List[str]] = None
    
    class Config:
        arbitrary_types_allowed = True

class StudentResponse(BaseModel):
    """
    Schema for student's submission.
    
    The student provides:
    1. Their pseudocode implementation
    2. Their analysis of time complexity
    3. Their analysis of space complexity
    
    Example:
        {
            "pseudocode": "FOR i = 1 TO n DO\\n  FOR j = 1 TO n DO\\n    sum = sum + A[i][j]",
            "time_complexity": "O(n^2)",
            "space_complexity": "O(1)"
        }
    """
    # Required: The pseudocode submitted by student
    pseudocode: str = Field(..., description="The pseudocode submitted by the student")
    
    # Student's stated complexity analysis
    time_complexity: Optional[str] = Field(
        default=None,
        description="Student's answer for time complexity, e.g., 'O(n^2)'"
    )
    space_complexity: Optional[str] = Field(
        default=None,
        description="Student's answer for space complexity, e.g., 'O(1)'"
    )
    
    # Optional: Student's explanation/reasoning
    explanation: Optional[str] = Field(
        default=None,
        description="Student's explanation of their complexity analysis"
    )
    
    @field_validator('pseudocode')
    @classmethod
    def pseudocode_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Pseudocode cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "pseudocode": "FOR i = 1 TO n DO\n  FOR j = 1 TO n DO\n    sum = sum + A[i][j]",
                "time_complexity": "O(n^2)",
                "space_complexity": "O(1)",
                "explanation": "Two nested loops each iterating n times gives O(n²)"
            }
        }


class ExpectedAnswer(BaseModel):
    """
    Schema for the expected/correct answer set by instructor.
    
    The instructor specifies:
    1. Expected time complexity
    2. Expected space complexity
    3. Acceptable alternative representations
    4. Optional: Description of what algorithm this should be
    
    Example:
        {
            "expected_time_complexity": "O(n^2)",
            "expected_space_complexity": "O(1)",
            "acceptable_time_alternatives": ["O(n*n)", "O(n²)"],
            "algorithm_description": "Nested loop sum calculation"
        }
    """
    # Expected complexities
    expected_time_complexity: str = Field(
        ...,
        description="Expected time complexity in Big-O notation"
    )
    expected_space_complexity: str = Field(
        default="O(1)",
        description="Expected space complexity in Big-O notation"
    )
    
    # Alternative acceptable representations
    acceptable_time_alternatives: List[str] = Field(
        default_factory=list,
        description="Alternative acceptable time complexity representations"
    )
    acceptable_space_alternatives: List[str] = Field(
        default_factory=list,
        description="Alternative acceptable space complexity representations"
    )
    
    # Description/context
    algorithm_description: Optional[str] = Field(
        default=None,
        description="Description of the expected algorithm"
    )
    algorithm_type: Optional[str] = Field(
        default=None,
        description="Type of algorithm: sorting, searching, graph, etc."
    )
    
    # Expected pseudocode patterns (optional, for validation)
    expected_constructs: List[str] = Field(
        default_factory=list,
        description="Expected code constructs: nested_loop, recursion, etc."
    )
    
    # Scoring weights
    time_complexity_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for time complexity in scoring"
    )
    space_complexity_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for space complexity in scoring"
    )
    
    def get_all_acceptable_time(self) -> List[str]:
        """Get all acceptable time complexity representations."""
        return [self.expected_time_complexity] + self.acceptable_time_alternatives
    
    def get_all_acceptable_space(self) -> List[str]:
        """Get all acceptable space complexity representations."""
        return [self.expected_space_complexity] + self.acceptable_space_alternatives
    
    class Config:
        json_schema_extra = {
            "example": {
                "expected_time_complexity": "O(n^2)",
                "expected_space_complexity": "O(1)",
                "acceptable_time_alternatives": ["O(n*n)", "O(n²)"],
                "acceptable_space_alternatives": [],
                "algorithm_description": "Matrix traversal with nested loops",
                "algorithm_type": "iteration",
                "expected_constructs": ["nested_loop"],
                "time_complexity_weight": 0.6,
                "space_complexity_weight": 0.4
            }
        }


class EvaluationParams(BaseModel):
    """
    Configuration parameters for the evaluation.
    
    Controls how the evaluation is performed and what feedback is provided.
    
    Example:
        {
            "analyze_pseudocode": true,
            "require_time_complexity": true,
            "require_space_complexity": true,
            "partial_credit": true,
            "show_detailed_feedback": true,
            "complexity_equivalence": true
        }
    """
    # What to evaluate
    analyze_pseudocode: bool = Field(
        default=True,
        description="Whether to parse and analyze the pseudocode"
    )
    require_time_complexity: bool = Field(
        default=True,
        description="Whether time complexity answer is required"
    )
    require_space_complexity: bool = Field(
        default=True,
        description="Whether space complexity answer is required"
    )
    
    # Scoring options
    partial_credit: bool = Field(
        default=True,
        description="Allow partial marks for partially correct answers"
    )
    time_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for time complexity in total score"
    )
    space_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for space complexity in total score"
    )
    
    # Comparison options
    complexity_equivalence: bool = Field(
        default=True,
        description="Treat equivalent complexities as equal (O(2n) == O(n))"
    )
    case_sensitive: bool = Field(
        default=False,
        description="Case sensitive comparison (O(N) vs O(n))"
    )
    strict_notation: bool = Field(
        default=False,
        description="Require exact Big-O notation format"
    )
    
    # Feedback options
    show_detailed_feedback: bool = Field(
        default=True,
        description="Provide detailed analysis feedback"
    )
    show_correct_answer: bool = Field(
        default=True,
        description="Show expected answer if student is wrong"
    )
    show_detected_complexity: bool = Field(
        default=True,
        description="Show complexity detected from pseudocode analysis"
    )
    show_ast: bool = Field(
        default=False,
        description="Include AST in response (for debugging)"
    )
    
    # Parser options
    pseudocode_style: str = Field(
        default="auto",
        description="Pseudocode style: auto, python, pascal, c"
    )
    strict_parsing: bool = Field(
        default=False,
        description="Fail on parse errors vs. best effort"
    )
    
    # Advanced options
    max_nesting_depth: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum loop nesting depth to analyze"
    )
    timeout_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Analysis timeout in seconds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "analyze_pseudocode": True,
                "require_time_complexity": True,
                "require_space_complexity": True,
                "partial_credit": True,
                "time_weight": 0.5,
                "space_weight": 0.5,
                "complexity_equivalence": True,
                "case_sensitive": False,
                "strict_notation": False,
                "show_detailed_feedback": True,
                "show_correct_answer": True,
                "show_detected_complexity": True,
                "show_ast": False,
                "pseudocode_style": "auto",
                "strict_parsing": False,
                "max_nesting_depth": 10,
                "timeout_seconds": 5.0
            }
        }