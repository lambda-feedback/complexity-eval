"""
Evaluation function for pseudocode complexity analysis.

This module implements the main evaluation pipeline that:
1. Parses the student's pseudocode
2. Analyzes its time and space complexity
3. Compares against the expected complexity bound
4. The code is CORRECT if its complexity is <= the expected bound

The answer specifies a complexity upper bound that the student's code must meet.
"""

from typing import Any, Dict, Optional, Tuple, List, Union
from lf_toolkit.evaluation import Result

from .schemas.output_schema import (
    ParseResult,
    SpaceComplexityResult,
    TestCaseResult,
    TimeComplexityResult,
)

from .schemas.input_schema import ExpectedAnswer, StudentResponse, EvaluationParams
from .analyzer.code_runner import CodeRunner
from .analyzer.interpreter import Interpreter
from .parser.parser import PseudocodeParser
from .analyzer.complexity_analyzer import ComplexityAnalyzer, AnalysisResult
from .analyzer.feedback_generator import FeedbackGenerator, FeedbackLevel
from .schemas.complexity import ComplexityClass


# previous observations seem to indicate that sometimes
# the response, answer, and params are all stuffed into params, this is likely a shimmy issue
# while the params is never here
# we'll just use the FSA solution - make the params a field in params
def evaluation_function(
    response: Any,
    answer: Any,
    params: Any,
) -> Result:
    """
    Evaluate a student's pseudocode complexity.

    The evaluation checks if the student's code has complexity within
    the expected bound. A student's answer is CORRECT if their code's
    complexity is less than or equal to the expected complexity.

    Args:
        response: Student's response containing:
            - pseudocode: The pseudocode string
            - time_complexity (optional): Student's stated time complexity
            - space_complexity (optional): Student's stated space complexity
        answer: Expected answer containing:
            - expected_time_complexity: Upper bound for time complexity (e.g., "O(n^2)")
            - expected_space_complexity (optional): Upper bound for space complexity
        params: Evaluation parameters for customization

    Returns:
        Result with is_correct, score, and detailed feedback
    """
    try:
        # ---------------------------------------------------------
        # Handle shim case
        # ---------------------------------------------------------
        if answer is None or response is None:
            answer = params.get("answer")
            response = params.get("response")
            params = params.get("params", {})

        # ---------------------------------------------------------
        # Strict validation using Pydantic
        # ---------------------------------------------------------
        response_model = (
            StudentResponse.model_validate_json(response)
            if isinstance(response, str)
            else StudentResponse.model_validate(response)
        )

        answer_model = (
            ExpectedAnswer.model_validate_json(answer)
            if isinstance(answer, str)
            else ExpectedAnswer.model_validate(answer)
        )

        params_model = (
            EvaluationParams.model_validate(params)
            if params
            else EvaluationParams()
        )

        # Override params if instructor provided eval_options
        if answer_model.eval_options:
            params_model = answer_model.eval_options

        # ---------------------------------------------------------
        # Extract validated values
        # ---------------------------------------------------------
        pseudocode = response_model.pseudocode
        student_time = response_model.time_complexity
        student_space = response_model.space_complexity

        expected_time = ComplexityClass.from_string(
            answer_model.expected_time_complexity
        )

        expected_space = (
            ComplexityClass.from_string(answer_model.expected_space_complexity)
            if answer_model.expected_space_complexity
            else None
        )

        # ---------------------------------------------------------
        # Validate pseudocode presence
        # ---------------------------------------------------------
        if not pseudocode:
            result = Result(is_correct=False)
            result.add_feedback("error", "No pseudocode provided. Please submit your algorithm.")
            return result

        # ---------------------------------------------------------
        # Parse pseudocode
        # ---------------------------------------------------------
        parser = PseudocodeParser()
        parse_result = parser.parse(pseudocode)

        if not parse_result.success and params_model.strict_parsing:
            result = Result(is_correct=False)
            result.add_feedback("error", _format_parse_error(parse_result))
            return result

        # ---------------------------------------------------------
        # Analyze complexity
        # ---------------------------------------------------------
        analyzer = ComplexityAnalyzer()
        analysis = analyzer.analyze(pseudocode, parse_result.ast)

        # ---------------------------------------------------------
        # Evaluate time complexity
        # ---------------------------------------------------------
        time_result: TimeComplexityResult = _evaluate_complexity(
            detected=analysis.time_complexity,
            expected_bound=expected_time,
            student_stated=student_time,
            complexity_type="time"
        )

        # ---------------------------------------------------------
        # Evaluate space complexity
        # ---------------------------------------------------------
        space_result = None
        if params_model.require_space_complexity and expected_space:
            space_result: SpaceComplexityResult = _evaluate_complexity(
                detected=analysis.space_complexity,
                expected_bound=expected_space,
                student_stated=student_space,
                complexity_type="space"
            )

        # ---------------------------------------------------------
        # Run execution test cases
        # ---------------------------------------------------------
        test_case_results: List[TestCaseResult] = []

        if answer_model.test_cases:
            runner = CodeRunner(None, Interpreter())
            code_correctness_result = runner.run_with_parse_result(
                parse_result,
                answer_model.test_cases
            )
            test_case_results = code_correctness_result.execution_results

        # ---------------------------------------------------------
        # Calculate overall correctness and score
        # ---------------------------------------------------------
        is_correct = _calculate_result(
            time_result=time_result,
            space_result=space_result,
            test_case_results=test_case_results,
            # eval_options=params_model,
        )

        # ---------------------------------------------------------
        # Generate feedback
        # ---------------------------------------------------------
        feedback = _generate_feedback(
            time_result=time_result,
            space_result=space_result,
            test_case_results=test_case_results,
            analysis=analysis,
            is_correct=is_correct,
            eval_options=params_model,
        )

        # ---------------------------------------------------------
        # Build result
        # ---------------------------------------------------------
        result = Result(is_correct=is_correct)
        # result.score = score
        result.add_feedback("complexity", feedback)

        return result

    except Exception as e:
        result = Result(is_correct=False)
        result.add_feedback("error", f"An error occurred during evaluation: {str(e)}")
        return result


from typing import Optional, Union


def _evaluate_complexity(
    detected: ComplexityClass,
    expected_bound: ComplexityClass,
    student_stated: Optional[str],
    complexity_type: str,
) -> Union[TimeComplexityResult, SpaceComplexityResult]:
    """
    Evaluate if detected complexity meets the expected bound.
    Returns a TimeComplexityResult or SpaceComplexityResult
    matching the output schema.
    """

    # ---------------------------------------------------------
    # Compare detected vs expected
    # ---------------------------------------------------------
    comparison = ComplexityClass.compare(detected, expected_bound)
    is_correct = comparison <= 0  # detected is same or better

    # ---------------------------------------------------------
    # Normalize student answer if provided
    # ---------------------------------------------------------
    student_normalized = None
    if student_stated:
        try:
            student_normalized = ComplexityClass.from_string(student_stated)
        except Exception:
            student_normalized = None

    # ---------------------------------------------------------
    # Build feedback message
    # ---------------------------------------------------------
    if is_correct:
        feedback = (
            f"Correct! Detected complexity {detected.value} "
            f"meets the required bound {expected_bound.value}."
        )
    else:
        feedback = (
            f"Detected complexity is {detected.value}, "
            f"which exceeds the required bound {expected_bound.value}."
        )

    # ---------------------------------------------------------
    # Return correct schema object
    # ---------------------------------------------------------
    if complexity_type == "time":
        return TimeComplexityResult(
            is_correct=is_correct,
            student_answer=student_stated,
            expected_answer=expected_bound.value,
            detected_complexity=detected.value,
            student_normalized=student_normalized,
            expected_normalized=expected_bound,
            analysis=None,
            feedback=feedback,
        )

    else:
        return SpaceComplexityResult(
            is_correct=is_correct,
            student_answer=student_stated,
            expected_answer=expected_bound.value,
            detected_complexity=detected.value,
            student_normalized=student_normalized,
            expected_normalized=expected_bound,
            analysis=None,
            feedback=feedback,
        )

# as previously discussed, we shouldnt be returning a score
def _calculate_result(
    time_result: Optional[TimeComplexityResult],
    space_result: Optional[SpaceComplexityResult],
    test_case_results: List[TestCaseResult],
) -> bool: #Tuple[bool, float]:
    """Calculate overall correctness and score."""

    is_correct = True
    if test_case_results:
        is_correct = all(tc.passed for tc in test_case_results)
    if space_result is not None:
        is_correct = space_result.is_correct and is_correct
    if time_result is not None:
        is_correct = time_result.is_correct and is_correct

    return is_correct#, score


def _generate_feedback(
    time_result: Optional[TimeComplexityResult],
    space_result: Optional[SpaceComplexityResult],
    test_case_results: List[TestCaseResult],
    analysis: AnalysisResult,
    is_correct: bool,
    eval_options: EvaluationParams,
) -> str:
    """Generate comprehensive feedback for the student using FeedbackGenerator."""
    feedback_generator = FeedbackGenerator()

    show_detailed = eval_options.show_detailed_feedback
    level = FeedbackLevel.DETAILED if show_detailed and not is_correct else FeedbackLevel.STANDARD

    detailed_feedback = feedback_generator.generate(analysis, level)

    lines = []

    if is_correct:
        lines.append("✓ Correct! Your algorithm meets the complexity requirements.")
    else:
        lines.append("✗ Your algorithm does not meet the complexity requirements.")
    lines.append("")
    if time_result:
        lines.append("Time Complexity:")
        lines.append(f"  • Required: {time_result.expected_answer} or better")
        lines.append(f"  • Detected: {time_result.detected_complexity}")

        if time_result.is_correct:
            lines.append("  ✓ Your algorithm meets the time complexity requirement.")
        else:
            lines.append("  ✗ Your algorithm exceeds the allowed time complexity.")

    if space_result:
        lines.append("")
        lines.append("Space Complexity:")
        lines.append(f"  • Required: {space_result.expected_answer} or better")
        lines.append(f"  • Detected: {space_result.detected_complexity}")

        if space_result.is_correct:
            lines.append("  ✓ Your algorithm meets the space complexity requirement.")
        else:
            lines.append("  ✗ Your algorithm exceeds the allowed space complexity.")

    if test_case_results:
        lines.append("")
        lines.append("Execution Test Cases:")
        for idx, tc in enumerate(test_case_results, 1):
            status = "✓ Passed" if tc.passed else "✗ Failed"
            lines.append(f"  Test Case {idx}: {status}")
            if not tc.passed and tc.error_message:
                lines.append(f"    Error: {tc.error_message}")

    if show_detailed and not is_correct:
        lines.append("")
        lines.append("-" * 50)

        for section in detailed_feedback.sections:
            lines.append(f"[{section.importance.upper()}] {section.title}")
            lines.append(section.content)
            lines.append("")

    return "\n".join(lines)


def _format_parse_error(parse_result: ParseResult) -> str:
    """Format parsing errors for feedback."""
    lines = ["Failed to parse the pseudocode."]

    if parse_result.errors:
        lines.append("\nErrors:")
        for error in parse_result.errors:
            lines.append(f"  • {error}")

    if parse_result.warnings:
        lines.append("\nWarnings:")
        for warning in parse_result.warnings:
            lines.append(f"  • {warning}")

    lines.append("\nPlease check your pseudocode syntax and try again.")
    return "\n".join(lines)