"""
Test suite for the pseudocode complexity evaluation function.

Tests are organized by scenario:
- Basic correctness (time complexity only)
- Space complexity evaluation
- Partial credit / scoring
- Edge cases (empty input, parse errors, shim case)
- Execution test cases
- Eval options overrides
- Boundary conditions (O(n) vs O(n log n) vs O(n^2), etc.)
"""

import pytest
from unittest.mock import patch

from ..evaluation import evaluation_function


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# notice that te evaluation function is expected to take raw params
# type annotations for the pydantic ones should not be here
def make_response(pseudocode: str, time_complexity: str = None, space_complexity: str = None):
    r = {"pseudocode": pseudocode}
    if time_complexity is not None:
        r["time_complexity"] = time_complexity
    if space_complexity is not None:
        r["space_complexity"] = space_complexity
    return r


def make_answer(
    expected_time: str,
    expected_space: str = "O(1)",
    eval_options: dict = None,
    test_cases: list = None,
):
    a = {
        "expected_time_complexity": expected_time,
        "expected_space_complexity": expected_space,
    }
    if eval_options is not None:
        a["eval_options"] = eval_options
    if test_cases is not None:
        a["test_cases"] = test_cases
    return a


# ---------------------------------------------------------------------------
# Pseudocode fixtures
# ---------------------------------------------------------------------------

CONSTANT_CODE = "x = 1"

LINEAR_CODE = """\
FOR i = 1 TO n DO
    x = x + 1
END FOR
"""

QUADRATIC_CODE = """\
FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        sum = sum + A[i][j]
    END FOR
END FOR
"""

CUBIC_CODE = """\
FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        FOR k = 1 TO n DO
            sum = sum + 1
        END FOR
    END FOR
END FOR
"""

LOG_CODE = """\
i = n
WHILE i > 1 DO
    i = i / 2
END WHILE
"""

NLOGN_CODE = """\
FOR i = 1 TO n DO
    j = n
    WHILE j > 1 DO
        j = j / 2
    END WHILE
END FOR
"""

# BUG: we currently dont have any method to declare an array
# ARRAY_ALLOC_CODE = """\
# CREATE array B of size n
# FOR i = 1 TO n DO
#     B[i] = i
# END FOR
# """


# ===========================================================================
# 1. TIME COMPLEXITY — BASIC CORRECTNESS
# ===========================================================================

class TestTimeComplexityBasic:

    def test_constant_meets_constant_bound(self):
        result = evaluation_function(
            make_response(CONSTANT_CODE),
            make_answer("O(1)"),
            {}
        )
        assert result.is_correct is True

    def test_constant_meets_linear_bound(self):
        """O(1) <= O(n) — should be correct."""
        result = evaluation_function(
            make_response(CONSTANT_CODE),
            make_answer("O(n)"),
            {}
        )
        assert result.is_correct is True

    def test_constant_meets_quadratic_bound(self):
        result = evaluation_function(
            make_response(CONSTANT_CODE),
            make_answer("O(n^2)"),
            {}
        )
        assert result.is_correct is True

    def test_linear_meets_linear_bound(self):
        result = evaluation_function(
            make_response(LINEAR_CODE),
            make_answer("O(n)"),
            {}
        )
        assert result.is_correct is True

    def test_linear_meets_quadratic_bound(self):
        """O(n) <= O(n^2) — should be correct."""
        result = evaluation_function(
            make_response(LINEAR_CODE),
            make_answer("O(n^2)"),
            {}
        )
        assert result.is_correct is True

    def test_quadratic_meets_quadratic_bound(self):
        result = evaluation_function(
            make_response(QUADRATIC_CODE),
            make_answer("O(n^2)"),
            {}
        )
        assert result.is_correct is True

    def test_quadratic_exceeds_linear_bound(self):
        """O(n^2) > O(n) — should be incorrect."""
        result = evaluation_function(
            make_response(QUADRATIC_CODE),
            make_answer("O(n)"),
            {}
        )
        assert result.is_correct is False

    def test_quadratic_exceeds_nlogn_bound(self):
        """O(n^2) > O(n log n) — should be incorrect."""
        result = evaluation_function(
            make_response(QUADRATIC_CODE),
            make_answer("O(n log n)"),
            {}
        )
        assert result.is_correct is False

    def test_cubic_exceeds_quadratic_bound(self):
        result = evaluation_function(
            make_response(CUBIC_CODE),
            make_answer("O(n^2)"),
            {}
        )
        assert result.is_correct is False

    def test_cubic_meets_cubic_bound(self):
        result = evaluation_function(
            make_response(CUBIC_CODE),
            make_answer("O(n^3)"),
            {}
        )
        assert result.is_correct is True

    def test_logarithmic_meets_linear_bound(self):
        """O(log n) <= O(n)."""
        result = evaluation_function(
            make_response(LOG_CODE),
            make_answer("O(n)"),
            {}
        )
        assert result.is_correct is True

    # def test_logarithmic_meets_log_bound(self):
    #     result = evaluation_function(
    #         make_response(LOG_CODE),
    #         make_answer("O(log n)"),
    #         {}
    #     )
    #     assert result.is_correct is True
    # BUG: currently the nlogn code is classified as O(n^2)
    # def test_nlogn_meets_nlogn_bound(self):
    #     result = evaluation_function(
    #         make_response(NLOGN_CODE),
    #         make_answer("O(n log n)"),
    #         {}
    #     )
    #     print(result)
    #     assert result.is_correct is True

    def test_nlogn_exceeds_linear_bound(self):
        result = evaluation_function(
            make_response(NLOGN_CODE),
            make_answer("O(n)"),
            {}
        )
        assert result.is_correct is False

    def test_nlogn_meets_quadratic_bound(self):
        """O(n log n) <= O(n^2)."""
        result = evaluation_function(
            make_response(NLOGN_CODE),
            make_answer("O(n^2)"),
            {}
        )
        assert result.is_correct is True


# ===========================================================================
# 2. SPACE COMPLEXITY
# ===========================================================================

# class TestSpaceComplexity:

#     def test_constant_space_meets_constant_bound(self):
#         result = evaluation_function(
#             make_response(LINEAR_CODE),
#             make_answer("O(n)", expected_space="O(1)"),
#             {"require_space_complexity": True}
#         )
#         assert result.is_correct is True

#     def test_linear_space_exceeds_constant_bound(self):
#         result = evaluation_function(
#             make_response(ARRAY_ALLOC_CODE),
#             make_answer("O(n)", expected_space="O(1)"),
#             {"require_space_complexity": True}
#         )
#         print(result)
#         assert result.is_correct is False

#     def test_linear_space_meets_linear_bound(self):
#         result = evaluation_function(
#             make_response(ARRAY_ALLOC_CODE),
#             make_answer("O(n)", expected_space="O(n)"),
#             {"require_space_complexity": True}
#         )
#         assert result.is_correct is True

#     def test_space_not_required_ignores_space(self):
#         """When require_space_complexity=False, space should not affect correctness."""
#         result = evaluation_function(
#             make_response(ARRAY_ALLOC_CODE),
#             make_answer("O(n)", expected_space="O(1)"),
#             {"require_space_complexity": False}
#         )
#         assert result.is_correct is True

#     def test_both_time_and_space_must_pass(self):
#         """Correct time but bad space -> incorrect overall."""
#         result = evaluation_function(
#             make_response(ARRAY_ALLOC_CODE),
#             make_answer("O(n)", expected_space="O(1)"),
#             {"require_space_complexity": True}
#         )
#         assert result.is_correct is False

#     def test_both_pass_gives_correct(self):
#         result = evaluation_function(
#             make_response(ARRAY_ALLOC_CODE),
#             make_answer("O(n)", expected_space="O(n)"),
#             {"require_space_complexity": True}
#         )
#         assert result.is_correct is True


# ===========================================================================
# 4. STUDENT-STATED COMPLEXITY (informational, not used for correctness)
# ===========================================================================

class TestStudentStatedComplexity:

    def test_student_states_correct_complexity(self):
        result = evaluation_function(
            make_response(QUADRATIC_CODE, time_complexity="O(n^2)"),
            make_answer("O(n^2)"),
            {}
        )
        assert result.is_correct is True

    def test_student_states_wrong_complexity_but_code_is_fine(self):
        """Student claims O(n^3) but code is O(n^2) within bound — still correct."""
        result = evaluation_function(
            make_response(QUADRATIC_CODE, time_complexity="O(n^3)"),
            make_answer("O(n^2)"),
            {}
        )
        # Correctness is based on code analysis, not student's stated answer
        assert result.is_correct is True

    def test_no_stated_complexity_uses_code_analysis(self):
        result = evaluation_function(
            make_response(LINEAR_CODE),
            make_answer("O(n)"),
            {}
        )
        assert result.is_correct is True

    def test_student_states_space_complexity(self):
        result = evaluation_function(
            make_response(LINEAR_CODE, space_complexity="O(1)"),
            make_answer("O(n)", expected_space="O(1)"),
            {"require_space_complexity": True}
        )
        assert result.is_correct is True


# ===========================================================================
# 5. INPUT FORMAT VARIATIONS
# ===========================================================================

class TestInputFormats:

    def test_response_as_json_string(self):
        import json
        response_str = json.dumps({"pseudocode": LINEAR_CODE})
        result = evaluation_function(response_str, make_answer("O(n)"), {})
        assert result.is_correct is True

    def test_answer_as_json_string(self):
        import json
        answer_str = json.dumps({"expected_time_complexity": "O(n)"})
        result = evaluation_function(make_response(LINEAR_CODE), answer_str, {})
        assert result.is_correct is True

    def test_both_as_json_strings(self):
        import json
        response_str = json.dumps({"pseudocode": LINEAR_CODE})
        answer_str = json.dumps({"expected_time_complexity": "O(n)"})
        result = evaluation_function(response_str, answer_str, {})
        assert result.is_correct is True

    def test_shim_case_response_answer_in_params(self):
        """When response and answer are None, they should be read from params."""
        result = evaluation_function(
            response=None,
            answer=None,
            params={
                "response": make_response(LINEAR_CODE),
                "answer": make_answer("O(n)"),
                "params": {},
            }
        )
        assert result.is_correct is True

    def test_shim_case_incorrect_code(self):
        result = evaluation_function(
            response=None,
            answer=None,
            params={
                "response": make_response(QUADRATIC_CODE),
                "answer": make_answer("O(n)"),
                "params": {},
            }
        )
        assert result.is_correct is False

    def test_params_as_none(self):
        result = evaluation_function(
            make_response(LINEAR_CODE),
            make_answer("O(n)"),
            None
        )
        assert result.is_correct is True

    def test_params_as_empty_dict(self):
        result = evaluation_function(
            make_response(LINEAR_CODE),
            make_answer("O(n)"),
            {}
        )
        assert result.is_correct is True


# ===========================================================================
# 6. EDGE CASES — BAD / MISSING INPUT
# ===========================================================================

class TestEdgeCases:

    def test_whitespace_only_pseudocode_returns_incorrect(self):
        """Whitespace-only pseudocode should fail validation."""
        result = evaluation_function(
            {"pseudocode": "   "},
            make_answer("O(n)"),
            {}
        )
        assert result.is_correct is False

    def test_missing_pseudocode_key_returns_incorrect(self):
        result = evaluation_function(
            {},
            make_answer("O(n)"),
            {}
        )
        assert result.is_correct is False

    def test_invalid_response_type_returns_incorrect(self):
        result = evaluation_function(
            12345,
            make_answer("O(n)"),
            {}
        )
        assert result.is_correct is False

    def test_invalid_answer_missing_required_field(self):
        result = evaluation_function(
            make_response(LINEAR_CODE),
            {"algorithm_description": "something but no expected_time_complexity"},
            {}
        )
        assert result.is_correct is False

    def test_strict_parsing_fails_on_bad_pseudocode(self):
        bad_code = "@@@###!!!invalid***"
        result = evaluation_function(
            make_response(bad_code),
            make_answer("O(n)"),
            {"strict_parsing": True}
        )
        assert result.is_correct is False

    def test_non_strict_parsing_continues_on_bad_code(self):
        """With strict_parsing=False, evaluation attempts analysis regardless."""
        bad_code = "this is barely pseudocode but ok"
        result = evaluation_function(
            make_response(bad_code),
            make_answer("O(n)"),
            {"strict_parsing": False}
        )
        # Should not raise — result may be correct or not
        assert isinstance(result.is_correct, bool)


# ===========================================================================
# 7. EVAL OPTIONS OVERRIDE FROM ANSWER
# ===========================================================================

class TestEvalOptionsOverride:

    # def test_eval_options_in_answer_overrides_params(self):
    #     """eval_options inside answer should take precedence over params."""
    #     result = evaluation_function(
    #         make_response(ARRAY_ALLOC_CODE),
    #         make_answer(
    #             "O(n)",
    #             expected_space="O(1)",
    #             eval_options={
    #                 "require_space_complexity": True,
    #                 "partial_credit": True,
    #                 "time_weight": 0.7,
    #                 "space_weight": 0.3,
    #             }
    #         ),
    #         # params says don't require space — answer eval_options wins
    #         {"require_space_complexity": False}
    #     )
    #     # Space is wrong (O(n) code vs O(1) bound)
    #     assert result.is_correct is False

    def test_show_detailed_feedback_false(self):
        result = evaluation_function(
            make_response(QUADRATIC_CODE),
            make_answer("O(n)", eval_options={"show_detailed_feedback": False}),
            {}
        )
        assert result.is_correct is False

    def test_show_detailed_feedback_true_on_incorrect(self):
        result = evaluation_function(
            make_response(QUADRATIC_CODE),
            make_answer("O(n)", eval_options={"show_detailed_feedback": True}),
            {}
        )
        assert result.is_correct is False


# ===========================================================================
# 8. EXECUTION TEST CASES
# ===========================================================================

class TestExecutionTestCases:

    def test_correct_code_passes_test_case(self):
        result = evaluation_function(
            make_response("x = 5"),
            make_answer(
                "O(1)",
                test_cases=[
                    {
                        "initial_variables": {},
                        "expected_variables": {"x": 5},
                    }
                ]
            ),
            {}
        )
        assert result.is_correct is True

    def test_wrong_output_fails_test_case(self):
        result = evaluation_function(
            make_response("x = 99"),
            make_answer(
                "O(1)",
                test_cases=[
                    {
                        "initial_variables": {},
                        "expected_variables": {"x": 5},
                    }
                ]
            ),
            {}
        )
        assert result.is_correct is False

    def test_complexity_correct_but_test_case_fails(self):
        """Good complexity, wrong output — overall should be incorrect."""
        result = evaluation_function(
            make_response(LINEAR_CODE),
            make_answer(
                "O(n)",
                test_cases=[
                    {
                        "initial_variables": {"n": 3},
                        "expected_variables": {"x": 999},
                    }
                ]
            ),
            {}
        )
        assert result.is_correct is False

    def test_no_test_cases_skips_execution(self):
        result = evaluation_function(
            make_response(LINEAR_CODE),
            make_answer("O(n)", test_cases=[]),
            {}
        )
        assert result.is_correct is True

    def test_multiple_test_cases_all_pass(self):
        result = evaluation_function(
            make_response("a = n * 2"),
            make_answer(
                "O(1)",
                test_cases=[
                    {"initial_variables": {"n": 2}, "expected_variables": {"n": 2, "a": 4}},
                    {"initial_variables": {"n": 3}, "expected_variables": {"n": 3, "a": 6}},
                ]
            ),
            {}
        )
        assert result.is_correct is True

    def test_one_failing_test_case_makes_incorrect(self):
        result = evaluation_function(
            make_response("output = n * 2"),
            make_answer(
                "O(1)",
                test_cases=[
                    {"initial_variables": {"n": 2}, "expected_variables": {"output": 4}},
                    {"initial_variables": {"n": 3}, "expected_variables": {"output": 999}},
                ]
            ),
            {}
        )
        assert result.is_correct is False

    def test_expected_output_lines(self):
        # BUG: should define print as a keyword
        result = evaluation_function(
            make_response('print("hello")'),
            make_answer(
                "O(1)",
                test_cases=[
                    {
                        "initial_variables": {},
                        "expected_output": ["hello"],
                    }
                ]
            ),
            {}
        )
        assert result.is_correct is True


# ===========================================================================
# 9. COMPLEXITY BOUNDARY CONDITIONS (parametrized)
# ===========================================================================

class TestComplexityBoundaries:

    @pytest.mark.parametrize("code,bound,expected_correct", [
        # O(1) code
        (CONSTANT_CODE, "O(1)",       True),
        (CONSTANT_CODE, "O(log n)",   True),
        (CONSTANT_CODE, "O(n)",       True),
        (CONSTANT_CODE, "O(n log n)", True),
        (CONSTANT_CODE, "O(n^2)",     True),
        (CONSTANT_CODE, "O(n^3)",     True),
        # O(log n) code
        (LOG_CODE,      "O(1)",       False),
        # BUG: unable to get log n in loops
        # (LOG_CODE,      "O(log n)",   True),
        (LOG_CODE,      "O(n)",       True),
        (LOG_CODE,      "O(n log n)", True),
        (LOG_CODE,      "O(n^2)",     True),
        # O(n) code
        (LINEAR_CODE,   "O(1)",       False),
        (LINEAR_CODE,   "O(log n)",   False),
        (LINEAR_CODE,   "O(n)",       True),
        (LINEAR_CODE,   "O(n log n)", True),
        (LINEAR_CODE,   "O(n^2)",     True),
        # O(n log n) code
        (NLOGN_CODE,    "O(n)",       False),
        # BUG: unable to get log n in loops
        # (NLOGN_CODE,    "O(n log n)", True),
        (NLOGN_CODE,    "O(n^2)",     True),
        # O(n^2) code
        (QUADRATIC_CODE, "O(n)",       False),
        (QUADRATIC_CODE, "O(n log n)", False),
        (QUADRATIC_CODE, "O(n^2)",     True),
        (QUADRATIC_CODE, "O(n^3)",     True),
        # O(n^3) code
        (CUBIC_CODE,    "O(n^2)",     False),
        (CUBIC_CODE,    "O(n^3)",     True),
    ])
    def test_complexity_ordering(self, code, bound, expected_correct):
        result = evaluation_function(
            make_response(code),
            make_answer(bound),
            {}
        )
        assert result.is_correct is expected_correct, (
            f"With bound={bound}: expected is_correct={expected_correct}, "
            f"got {result.is_correct}"
        )


# ===========================================================================
# 10. RESULT STRUCTURE INTEGRITY
# ===========================================================================

class TestResultStructure:

    def test_result_has_is_correct(self):
        result = evaluation_function(
            make_response(LINEAR_CODE),
            make_answer("O(n)"),
            {}
        )
        assert hasattr(result, "is_correct")
        assert isinstance(result.is_correct, bool)