"""
Evaluation function for pseudocode complexity analysis.

This module implements the main evaluation pipeline that:
1. Parses the student's pseudocode
2. Analyzes its time and space complexity
3. Compares against the expected complexity bound
4. The code is CORRECT if its complexity is <= the expected bound

The answer specifies a complexity upper bound that the student's code must meet.
"""

from typing import Any, Dict, Optional, Tuple
from lf_toolkit.evaluation import Result, Params

from .parser.parser import PseudocodeParser
from .analyzer.complexity_analyzer import ComplexityAnalyzer, AnalysisResult
from .analyzer.feedback_generator import FeedbackGenerator, FeedbackLevel
from .schemas.complexity import ComplexityClass


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
    # result = Result(is_correct=False)
    # result.add_feedback("error", f"An error occurred during evaluation: answer:\n\n{answer}\n\nresponse\n\n{response}\n\nparams:\n\n{params}")
    # return result
    try:
        # Parse inputs
        pseudocode, student_time, student_space = _parse_response(response)
        expected_time, expected_space, eval_options = _parse_answer(answer, params)

        # Validate inputs
        if not pseudocode:
            result = Result(is_correct=False)
            result.add_feedback("error", "No pseudocode provided. Please submit your algorithm.")
            return result

        # Parse and analyze the pseudocode
        parser = PseudocodeParser()
        parse_result = parser.parse(pseudocode)

        if not parse_result.success and eval_options.get('strict_parsing', False):
            result = Result(is_correct=False)
            result.add_feedback("error", _format_parse_error(parse_result))
            return result

        # Analyze complexity
        analyzer = ComplexityAnalyzer()
        analysis = analyzer.analyze(pseudocode, parse_result.ast)

        # Evaluate time complexity
        time_result = _evaluate_complexity(
            detected=analysis.time_complexity,
            expected_bound=expected_time,
            student_stated=student_time,
            complexity_type="time"
        )

        # Evaluate space complexity (optional)
        space_result = None
        if expected_space:
            space_result = _evaluate_complexity(
                detected=analysis.space_complexity,
                expected_bound=expected_space,
                student_stated=student_space,
                complexity_type="space"
            )

        # Calculate overall correctness and score
        is_correct, score = _calculate_result(time_result, space_result, eval_options)

        # Generate feedback
        feedback = _generate_feedback(
            time_result=time_result,
            space_result=space_result,
            analysis=analysis,
            is_correct=is_correct,
            eval_options=eval_options
        )

        # Build result
        result = Result(is_correct=is_correct)
        result.add_feedback("complexity", feedback)

        return result

    except Exception as e:
        result = Result(is_correct=False)
        result.add_feedback("error", f"An error occurred during evaluation: {str(e)}")
        return result


def _parse_response(response: Any) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse the student's response to extract pseudocode and stated complexities."""
    if isinstance(response, str):
        return response, None, None

    if isinstance(response, dict):
        pseudocode = response.get('pseudocode', response.get('code', ''))
        time_complexity = response.get('time_complexity')
        space_complexity = response.get('space_complexity')
        return pseudocode, time_complexity, space_complexity

    return '', None, None


def _parse_answer(answer: Any, params: Params) -> Tuple[ComplexityClass, Optional[ComplexityClass], Dict]:
    """Parse the expected answer and evaluation options."""
    eval_options = {}

    # Handle params
    if hasattr(params, '__iter__'):
        for key in params:
            eval_options[key] = params[key]
    elif hasattr(params, 'to_dict'):
        eval_options = params.to_dict()

    # Parse answer
    if isinstance(answer, str):
        expected_time = ComplexityClass.from_string(answer)
        expected_space = None
    elif isinstance(answer, dict):
        expected_time = ComplexityClass.from_string(
            answer.get('expected_time_complexity', answer.get('time_complexity', 'O(n)'))
        )
        expected_space_str = answer.get('expected_space_complexity', answer.get('space_complexity'))
        expected_space = ComplexityClass.from_string(expected_space_str) if expected_space_str else None

        # Merge answer options into eval_options
        for key in ['show_detailed_feedback', 'strict_parsing', 'partial_credit']:
            if key in answer:
                eval_options[key] = answer[key]
    else:
        expected_time = ComplexityClass.LINEAR
        expected_space = None

    return expected_time, expected_space, eval_options


def _evaluate_complexity(
    detected: ComplexityClass,
    expected_bound: ComplexityClass,
    student_stated: Optional[str],
    complexity_type: str
) -> Dict:
    """
    Evaluate if detected complexity meets the expected bound.

    Returns dict with:
        - is_correct: True if detected <= expected_bound
        - detected: The detected complexity
        - expected: The expected bound
        - comparison: -1 (better), 0 (equal), 1 (worse)
        - student_stated_correct: If student's stated answer matches detected
    """
    # Compare complexities: correct if detected <= expected
    comparison = ComplexityClass.compare(detected, expected_bound)
    is_correct = comparison <= 0  # detected is same or better than bound

    # Check if student's stated complexity matches detected
    student_stated_correct = None
    if student_stated:
        stated_class = ComplexityClass.from_string(student_stated)
        student_stated_correct = stated_class == detected

    return {
        'is_correct': is_correct,
        'detected': detected,
        'expected': expected_bound,
        'comparison': comparison,
        'student_stated': student_stated,
        'student_stated_correct': student_stated_correct,
        'type': complexity_type
    }


def _calculate_result(
    time_result: Dict,
    space_result: Optional[Dict],
    eval_options: Dict
) -> Tuple[bool, float]:
    """Calculate overall correctness and score."""
    time_weight = eval_options.get('time_weight', 0.7)
    space_weight = eval_options.get('space_weight', 0.3)
    partial_credit = eval_options.get('partial_credit', True)

    # If no space requirement, only consider time
    if space_result is None:
        is_correct = time_result['is_correct']
        if partial_credit and not is_correct:
            # Give partial credit based on how close they are
            score = _partial_score(time_result['detected'], time_result['expected'])
        else:
            score = 1.0 if is_correct else 0.0
        return is_correct, score

    # Both time and space required
    is_correct = time_result['is_correct'] and space_result['is_correct']

    if partial_credit:
        time_score = 1.0 if time_result['is_correct'] else _partial_score(
            time_result['detected'], time_result['expected']
        )
        space_score = 1.0 if space_result['is_correct'] else _partial_score(
            space_result['detected'], space_result['expected']
        )
        score = time_weight * time_score + space_weight * space_score
    else:
        score = 1.0 if is_correct else 0.0

    return is_correct, score


def _partial_score(detected: ComplexityClass, expected: ComplexityClass) -> float:
    """Calculate partial credit score based on complexity difference."""
    order = ComplexityClass.get_order()

    try:
        detected_idx = order.index(detected)
        expected_idx = order.index(expected)
    except ValueError:
        return 0.0

    if detected_idx <= expected_idx:
        return 1.0  # Met or exceeded requirement

    # Calculate partial credit: decreasing score for each level above expected
    diff = detected_idx - expected_idx
    # Score decreases by 0.2 for each complexity level above expected
    # Max partial credit is 0.5 for being one level above
    return max(0.0, 0.5 - (diff - 1) * 0.15)


def _generate_feedback(
    time_result: Dict,
    space_result: Optional[Dict],
    analysis: AnalysisResult,
    is_correct: bool,
    eval_options: Dict
) -> str:
    """Generate comprehensive feedback for the student using FeedbackGenerator."""
    # Use FeedbackGenerator for the core analysis feedback
    feedback_generator = FeedbackGenerator()

    # Determine feedback level based on options
    show_detailed = eval_options.get('show_detailed_feedback', True)
    level = FeedbackLevel.DETAILED if show_detailed and not is_correct else FeedbackLevel.STANDARD

    # Generate detailed feedback from analysis
    detailed_feedback = feedback_generator.generate(analysis, level)

    # Build the final feedback string
    lines = []

    # Overall result header
    if is_correct:
        lines.append("✓ Correct! Your algorithm meets the complexity requirements.")
    else:
        lines.append("✗ Your algorithm does not meet the complexity requirements.")
    lines.append("")

    # Time complexity feedback
    time_correct = time_result['is_correct']
    lines.append("Time Complexity:")
    lines.append(f"  • Required: {time_result['expected'].value} or better")
    lines.append(f"  • Detected: {time_result['detected'].value}")

    if time_correct:
        if time_result['comparison'] < 0:
            lines.append("  ✓ Excellent! Your algorithm is more efficient than required.")
        else:
            lines.append("  ✓ Your algorithm meets the time complexity requirement.")
    else:
        lines.append("  ✗ Your algorithm exceeds the allowed time complexity.")
        lines.append(f"    Try to optimize your algorithm to achieve {time_result['expected'].value}.")

    # Student's stated complexity feedback
    if time_result.get('student_stated'):
        if time_result.get('student_stated_correct'):
            lines.append(f"  ✓ Your stated complexity ({time_result['student_stated']}) matches the detected complexity.")
        else:
            lines.append(f"  ⚠ Your stated complexity ({time_result['student_stated']}) differs from detected ({time_result['detected'].value}).")

    lines.append("")

    # Space complexity feedback (if applicable)
    if space_result:
        space_correct = space_result['is_correct']
        lines.append("Space Complexity:")
        lines.append(f"  • Required: {space_result['expected'].value} or better")
        lines.append(f"  • Detected: {space_result['detected'].value}")

        if space_correct:
            lines.append("  ✓ Your algorithm meets the space complexity requirement.")
        else:
            lines.append("  ✗ Your algorithm exceeds the allowed space complexity.")
        lines.append("")

    # Add detailed analysis from FeedbackGenerator (if enabled and incorrect)
    if show_detailed and not is_correct:
        lines.append("-" * 50)

        # Add sections from FeedbackGenerator
        for section in detailed_feedback.sections:
            lines.append(f"[{section.importance.upper()}] {section.title}")
            lines.append(section.content)
            lines.append("")

        # Add suggestions from FeedbackGenerator
        if detailed_feedback.suggestions:
            lines.append("Suggestions:")
            for suggestion in detailed_feedback.suggestions:
                lines.append(f"  • {suggestion}")

    return "\n".join(lines)


def _format_parse_error(parse_result) -> str:
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


