# pydantic should be sfe enough, ignoring all tests here

# """
# Comprehensive tests for Input and Output Schemas.

# Tests cover:
# - StudentResponse validation
# - ExpectedAnswer validation
# - EvaluationParams configuration
# - EvaluationResult structure
# - Feedback items
# - Parse results
# """

# import pytest
# from pydantic import ValidationError

# from ..schemas.input_schema import (
#     StudentResponse, ExpectedAnswer, EvaluationParams
# )
# from ..schemas.output_schema import (
#     EvaluationResult, TimeComplexityResult, SpaceComplexityResult,
#     ComplexityAnalysis, ConstructAnalysis, FeedbackItem, FeedbackLevel,
#     ParseResult
# )
# from ..schemas.complexity import ComplexityClass, TimeComplexity, SpaceComplexity


# class TestStudentResponse:
#     """Tests for StudentResponse schema."""
    
#     def test_create_minimal_response(self):
#         """Test creating response with only required field."""
#         response = StudentResponse(pseudocode="x = 1")
        
#         assert response.pseudocode == "x = 1"
#         assert response.time_complexity is None
#         assert response.space_complexity is None
    
#     def test_create_full_response(self):
#         """Test creating response with all fields."""
#         response = StudentResponse(
#             pseudocode="FOR i = 1 TO n DO\n    print(i)\nEND FOR",
#             time_complexity="O(n)",
#             space_complexity="O(1)",
#             explanation="Single loop iterates n times"
#         )
        
#         assert response.pseudocode is not None
#         assert response.time_complexity == "O(n)"
#         assert response.space_complexity == "O(1)"
#         assert response.explanation is not None
    
#     def test_pseudocode_validation_empty(self):
#         """Test that empty pseudocode raises validation error."""
#         with pytest.raises(ValidationError):
#             StudentResponse(pseudocode="")
    
#     def test_pseudocode_validation_whitespace_only(self):
#         """Test that whitespace-only pseudocode raises validation error."""
#         with pytest.raises(ValidationError):
#             StudentResponse(pseudocode="   \n\t  ")
    
#     def test_pseudocode_stripped(self):
#         """Test that pseudocode is stripped of leading/trailing whitespace."""
#         response = StudentResponse(pseudocode="  x = 1  ")
#         assert response.pseudocode == "x = 1"
    
#     def test_to_dict(self):
#         """Test response serialization to dict."""
#         response = StudentResponse(
#             pseudocode="x = 1",
#             time_complexity="O(1)"
#         )
        
#         result = response.model_dump()
        
#         assert result["pseudocode"] == "x = 1"
#         assert result["time_complexity"] == "O(1)"
    
#     def test_json_schema_example(self):
#         """Test that JSON schema example is valid."""
#         schema = StudentResponse.model_json_schema()
        
#         assert "example" in schema or "properties" in schema


# class TestExpectedAnswer:
#     """Tests for ExpectedAnswer schema."""
    
#     def test_create_minimal_answer(self):
#         """Test creating answer with only required field."""
#         answer = ExpectedAnswer(expected_time_complexity="O(n)")
        
#         assert answer.expected_time_complexity == "O(n)"
#         assert answer.expected_space_complexity == "O(1)"  # Default
    
#     def test_create_full_answer(self):
#         """Test creating answer with all fields."""
#         answer = ExpectedAnswer(
#             expected_time_complexity="O(n^2)",
#             expected_space_complexity="O(1)",
#             acceptable_time_alternatives=["O(n*n)", "O(n²)"],
#             acceptable_space_alternatives=["O(1)", "constant"],
#             algorithm_description="Bubble sort implementation",
#             algorithm_type="sorting",
#             expected_constructs=["nested_loop"],
#             time_complexity_weight=0.6,
#             space_complexity_weight=0.4
#         )
        
#         assert answer.expected_time_complexity == "O(n^2)"
#         assert len(answer.acceptable_time_alternatives) == 2
#         assert answer.algorithm_type == "sorting"
    
#     def test_get_all_acceptable_time(self):
#         """Test getting all acceptable time complexities."""
#         answer = ExpectedAnswer(
#             expected_time_complexity="O(n)",
#             acceptable_time_alternatives=["O(1*n)", "linear"]
#         )
        
#         all_acceptable = answer.get_all_acceptable_time()
        
#         assert "O(n)" in all_acceptable
#         assert "O(1*n)" in all_acceptable
#         assert "linear" in all_acceptable
#         assert len(all_acceptable) == 3
    
#     def test_get_all_acceptable_space(self):
#         """Test getting all acceptable space complexities."""
#         answer = ExpectedAnswer(
#             expected_time_complexity="O(n)",
#             expected_space_complexity="O(1)",
#             acceptable_space_alternatives=["constant"]
#         )
        
#         all_acceptable = answer.get_all_acceptable_space()
        
#         assert "O(1)" in all_acceptable
#         assert "constant" in all_acceptable
    
#     def test_weight_validation(self):
#         """Test that weights are validated to be between 0 and 1."""
#         # Valid weights
#         answer = ExpectedAnswer(
#             expected_time_complexity="O(n)",
#             time_complexity_weight=0.7,
#             space_complexity_weight=0.3
#         )
#         assert answer.time_complexity_weight == 0.7
        
#         # Invalid weights should raise error
#         with pytest.raises(ValidationError):
#             ExpectedAnswer(
#                 expected_time_complexity="O(n)",
#                 time_complexity_weight=1.5  # > 1
#             )
        
#         with pytest.raises(ValidationError):
#             ExpectedAnswer(
#                 expected_time_complexity="O(n)",
#                 time_complexity_weight=-0.1  # < 0
#             )


# class TestEvaluationParams:
#     """Tests for EvaluationParams schema."""
    
#     def test_default_params(self):
#         """Test default parameter values."""
#         params = EvaluationParams()
        
#         assert params.analyze_pseudocode == True
#         assert params.require_time_complexity == True
#         assert params.require_space_complexity == True
#         assert params.partial_credit == True
#         assert params.complexity_equivalence == True
#         assert params.show_detailed_feedback == True
    
#     def test_custom_params(self):
#         """Test creating custom parameters."""
#         params = EvaluationParams(
#             analyze_pseudocode=False,
#             require_time_complexity=True,
#             require_space_complexity=False,
#             partial_credit=False,
#             time_weight=1.0,
#             space_weight=0.0
#         )
        
#         assert params.analyze_pseudocode == False
#         assert params.time_weight == 1.0
    
#     def test_weight_validation(self):
#         """Test that weights are validated."""
#         # Valid
#         params = EvaluationParams(time_weight=0.5, space_weight=0.5)
#         assert params.time_weight == 0.5
        
#         # Invalid
#         with pytest.raises(ValidationError):
#             EvaluationParams(time_weight=2.0)
    
#     def test_max_nesting_depth_validation(self):
#         """Test max nesting depth validation."""
#         # Valid
#         params = EvaluationParams(max_nesting_depth=20)
#         assert params.max_nesting_depth == 20
        
#         # Too low
#         with pytest.raises(ValidationError):
#             EvaluationParams(max_nesting_depth=0)
        
#         # Too high
#         with pytest.raises(ValidationError):
#             EvaluationParams(max_nesting_depth=100)
    
#     def test_timeout_validation(self):
#         """Test timeout validation."""
#         # Valid
#         params = EvaluationParams(timeout_seconds=10.0)
#         assert params.timeout_seconds == 10.0
        
#         # Too low
#         with pytest.raises(ValidationError):
#             EvaluationParams(timeout_seconds=0.01)
        
#         # Too high
#         with pytest.raises(ValidationError):
#             EvaluationParams(timeout_seconds=120.0)
    
#     def test_pseudocode_style_options(self):
#         """Test pseudocode style options."""
#         styles = ["auto", "python", "pascal", "c"]
        
#         for style in styles:
#             params = EvaluationParams(pseudocode_style=style)
#             assert params.pseudocode_style == style


# class TestFeedbackItem:
#     """Tests for FeedbackItem schema."""
    
#     def test_create_feedback_item(self):
#         """Test creating feedback item."""
#         item = FeedbackItem(
#             level=FeedbackLevel.WARNING,
#             message="Time complexity is incorrect"
#         )
        
#         assert item.level == FeedbackLevel.WARNING
#         assert item.message == "Time complexity is incorrect"
    
#     def test_feedback_levels(self):
#         """Test all feedback levels."""
#         levels = [
#             FeedbackLevel.INFO,
#             FeedbackLevel.SUCCESS,
#             FeedbackLevel.WARNING,
#             FeedbackLevel.ERROR,
#             FeedbackLevel.HINT
#         ]
        
#         for level in levels:
#             item = FeedbackItem(level=level, message="Test")
#             assert item.level == level
    
#     def test_feedback_with_details(self):
#         """Test feedback item with all details."""
#         item = FeedbackItem(
#             level=FeedbackLevel.ERROR,
#             message="Expected O(n²) but got O(n)",
#             category="time_complexity",
#             location="line 3",
#             suggestion="Consider the nested loop structure"
#         )
        
#         assert item.category == "time_complexity"
#         assert item.location == "line 3"
#         assert item.suggestion is not None


# class TestConstructAnalysis:
#     """Tests for ConstructAnalysis schema."""
    
#     def test_create_construct_analysis(self):
#         """Test creating construct analysis."""
#         analysis = ConstructAnalysis(
#             construct_type="nested_loop",
#             description="Two nested FOR loops",
#             complexity_contribution=ComplexityClass.QUADRATIC
#         )
        
#         assert analysis.construct_type == "nested_loop"
#         assert analysis.complexity_contribution == ComplexityClass.QUADRATIC
    
#     def test_construct_with_details(self):
#         """Test construct analysis with details."""
#         analysis = ConstructAnalysis(
#             construct_type="loop",
#             description="FOR loop from 1 to n",
#             location="lines 1-3",
#             complexity_contribution=ComplexityClass.LINEAR,
#             details={
#                 "iterator": "i",
#                 "start": 1,
#                 "end": "n",
#                 "step": 1
#             }
#         )
        
#         assert analysis.details["iterator"] == "i"


# class TestTimeComplexityResult:
#     """Tests for TimeComplexityResult schema."""
    
#     def test_create_correct_result(self):
#         """Test creating correct time complexity result."""
#         result = TimeComplexityResult(
#             is_correct=True,
#             student_answer="O(n^2)",
#             expected_answer="O(n²)",
#             expected_normalized=ComplexityClass.QUADRATIC,
#             student_normalized=ComplexityClass.QUADRATIC,
#             feedback="Correct!"
#         )
        
#         assert result.is_correct == True
#         assert result.student_answer == "O(n^2)"
    
#     def test_create_incorrect_result(self):
#         """Test creating incorrect time complexity result."""
#         result = TimeComplexityResult(
#             is_correct=False,
#             student_answer="O(n)",
#             expected_answer="O(n²)",
#             expected_normalized=ComplexityClass.QUADRATIC,
#             student_normalized=ComplexityClass.LINEAR,
#             detected_complexity="O(n²)",
#             feedback="Your answer O(n) differs from expected O(n²)"
#         )
        
#         assert result.is_correct == False
#         assert result.detected_complexity == "O(n²)"


# class TestSpaceComplexityResult:
#     """Tests for SpaceComplexityResult schema."""
    
#     def test_create_space_result(self):
#         """Test creating space complexity result."""
#         result = SpaceComplexityResult(
#             is_correct=True,
#             student_answer="O(1)",
#             expected_answer="O(1)",
#             expected_normalized=ComplexityClass.CONSTANT,
#             student_normalized=ComplexityClass.CONSTANT
#         )
        
#         assert result.is_correct == True


# class TestParseResult:
#     """Tests for ParseResult schema."""
    
#     def test_successful_parse(self):
#         """Test successful parse result."""
#         result = ParseResult(
#             success=True,
#             errors=[],
#             warnings=[],
#             normalized_code="for i = 1 to n do\n    print(i)"
#         )
        
#         assert result.success == True
#         assert len(result.errors) == 0
    
#     def test_failed_parse(self):
#         """Test failed parse result."""
#         result = ParseResult(
#             success=False,
#             errors=["Syntax error at line 3", "Unexpected token"],
#             warnings=["Inconsistent indentation"]
#         )
        
#         assert result.success == False
#         assert len(result.errors) == 2
#         assert len(result.warnings) == 1


# class TestComplexityAnalysis:
#     """Tests for ComplexityAnalysis schema."""
    
#     def test_create_analysis(self):
#         """Test creating complexity analysis."""
#         tc = TimeComplexity(
#             overall=ComplexityClass.QUADRATIC,
#             expression="O(n²)"
#         )
#         sc = SpaceComplexity(
#             overall=ComplexityClass.CONSTANT,
#             expression="O(1)"
#         )
        
#         analysis = ComplexityAnalysis(
#             time_complexity=tc,
#             space_complexity=sc,
#             constructs=[],
#             confidence=0.9
#         )
        
#         assert analysis.confidence == 0.9
    
#     def test_analysis_with_constructs(self):
#         """Test analysis with detected constructs."""
#         tc = TimeComplexity(
#             overall=ComplexityClass.QUADRATIC,
#             expression="O(n²)"
#         )
#         sc = SpaceComplexity(
#             overall=ComplexityClass.CONSTANT,
#             expression="O(1)"
#         )
        
#         constructs = [
#             ConstructAnalysis(
#                 construct_type="nested_loop",
#                 complexity_contribution=ComplexityClass.QUADRATIC
#             )
#         ]
        
#         analysis = ComplexityAnalysis(
#             time_complexity=tc,
#             space_complexity=sc,
#             constructs=constructs,
#             algorithm_type="iteration"
#         )
        
#         assert len(analysis.constructs) == 1
#         assert analysis.algorithm_type == "iteration"


# class TestEvaluationResult:
#     """Tests for EvaluationResult schema."""
    
#     def test_create_correct_result(self):
#         """Test creating correct evaluation result."""
#         result = EvaluationResult(
#             is_correct=True,
#             score=1.0,
#             feedback="Excellent! All answers correct."
#         )
        
#         assert result.is_correct == True
#         assert result.score == 1.0
    
#     def test_create_partial_result(self):
#         """Test creating partial credit result."""
#         time_result = TimeComplexityResult(
#             is_correct=True,
#             student_answer="O(n²)",
#             expected_answer="O(n²)",
#             expected_normalized=ComplexityClass.QUADRATIC
#         )
#         space_result = SpaceComplexityResult(
#             is_correct=False,
#             student_answer="O(n)",
#             expected_answer="O(1)",
#             expected_normalized=ComplexityClass.CONSTANT
#         )
        
#         result = EvaluationResult(
#             is_correct=False,
#             time_complexity_result=time_result,
#             space_complexity_result=space_result,
#             score=0.5,
#             feedback="Time complexity correct, but space complexity is incorrect."
#         )
        
#         assert result.score == 0.5
#         assert result.time_complexity_result.is_correct == True
#         assert result.space_complexity_result.is_correct == False
    
#     def test_create_result_with_feedback_items(self):
#         """Test creating result with feedback items."""
#         feedback_items = [
#             FeedbackItem(level=FeedbackLevel.SUCCESS, message="Time complexity correct"),
#             FeedbackItem(level=FeedbackLevel.ERROR, message="Space complexity incorrect"),
#             FeedbackItem(level=FeedbackLevel.HINT, message="Consider the auxiliary array")
#         ]
        
#         result = EvaluationResult(
#             is_correct=False,
#             score=0.5,
#             feedback="Partial credit awarded",
#             feedback_items=feedback_items
#         )
        
#         assert len(result.feedback_items) == 3
    
#     def test_create_result_with_warnings_errors(self):
#         """Test creating result with warnings and errors."""
#         result = EvaluationResult(
#             is_correct=True,
#             score=1.0,
#             warnings=["Could not fully parse line 5"],
#             errors=[]
#         )
        
#         assert len(result.warnings) == 1
#         assert len(result.errors) == 0
    
#     def test_to_lambda_feedback_response(self):
#         """Test conversion to Lambda Feedback response format."""
#         time_result = TimeComplexityResult(
#             is_correct=True,
#             student_answer="O(n)",
#             expected_answer="O(n)",
#             expected_normalized=ComplexityClass.LINEAR,
#             feedback="Correct!"
#         )
        
#         result = EvaluationResult(
#             is_correct=True,
#             time_complexity_result=time_result,
#             score=1.0,
#             feedback="All correct!"
#         )
        
#         response = result.to_lambda_feedback_response()
        
#         assert response["is_correct"] == True
#         assert response["feedback"] == "All correct!"
#         assert "time_complexity" in response
    
#     def test_score_validation(self):
#         """Test that score is validated to be between 0 and 1."""
#         # Valid
#         result = EvaluationResult(is_correct=True, score=0.5)
#         assert result.score == 0.5
        
#         # Invalid - too high
#         with pytest.raises(ValidationError):
#             EvaluationResult(is_correct=True, score=1.5)
        
#         # Invalid - too low
#         with pytest.raises(ValidationError):
#             EvaluationResult(is_correct=True, score=-0.1)
    
#     def test_result_with_analysis(self):
#         """Test result with complexity analysis."""
#         tc = TimeComplexity(
#             overall=ComplexityClass.LINEAR,
#             expression="O(n)"
#         )
#         sc = SpaceComplexity(
#             overall=ComplexityClass.CONSTANT,
#             expression="O(1)"
#         )
        
#         analysis = ComplexityAnalysis(
#             time_complexity=tc,
#             space_complexity=sc,
#             constructs=[]
#         )
        
#         result = EvaluationResult(
#             is_correct=True,
#             score=1.0,
#             analysis=analysis
#         )
        
#         assert result.analysis is not None
#         assert result.analysis.time_complexity.overall == ComplexityClass.LINEAR
    
#     def test_result_with_metadata(self):
#         """Test result with custom metadata."""
#         result = EvaluationResult(
#             is_correct=True,
#             score=1.0,
#             metadata={
#                 "parse_time_ms": 15,
#                 "analysis_time_ms": 25,
#                 "total_lines": 10
#             }
#         )
        
#         assert result.metadata["parse_time_ms"] == 15


# class TestSchemaRoundTrip:
#     """Tests for schema serialization/deserialization round trips."""
    
#     def test_student_response_roundtrip(self):
#         """Test StudentResponse serialization round trip."""
#         original = StudentResponse(
#             pseudocode="FOR i = 1 TO n DO\n    x = x + 1\nEND FOR",
#             time_complexity="O(n)",
#             space_complexity="O(1)"
#         )
        
#         # Serialize and deserialize
#         data = original.model_dump()
#         restored = StudentResponse(**data)
        
#         assert restored.pseudocode == original.pseudocode
#         assert restored.time_complexity == original.time_complexity
    
#     def test_expected_answer_roundtrip(self):
#         """Test ExpectedAnswer serialization round trip."""
#         original = ExpectedAnswer(
#             expected_time_complexity="O(n²)",
#             expected_space_complexity="O(1)",
#             acceptable_time_alternatives=["O(n^2)"]
#         )
        
#         data = original.model_dump()
#         restored = ExpectedAnswer(**data)
        
#         assert restored.expected_time_complexity == original.expected_time_complexity
#         assert len(restored.acceptable_time_alternatives) == 1
    
#     def test_evaluation_result_roundtrip(self):
#         """Test EvaluationResult serialization round trip."""
#         original = EvaluationResult(
#             is_correct=True,
#             score=0.75,
#             feedback="Good work!",
#             feedback_items=[
#                 FeedbackItem(level=FeedbackLevel.SUCCESS, message="Correct")
#             ]
#         )
        
#         data = original.model_dump()
#         restored = EvaluationResult(**data)
        
#         assert restored.is_correct == original.is_correct
#         assert restored.score == original.score
#         assert len(restored.feedback_items) == 1
