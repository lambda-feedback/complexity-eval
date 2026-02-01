"""
Tests for the evaluation function.

Tests cover:
- Basic evaluation functionality
- Complexity bound checking (code should be <= bound)
- Different complexity classes
- Partial credit scoring
- Feedback generation
- Error handling
"""

import pytest


class MockParams:
    """Mock params object for testing."""
    def __init__(self, **kwargs):
        self._data = kwargs

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def to_dict(self):
        return self._data


@pytest.fixture
def params():
    """Default params fixture."""
    return MockParams()


class TestEvaluationBasic:
    """Basic evaluation function tests."""

    def test_evaluation_returns_result(self, params):
        """Test that evaluation returns a Result object."""
        from ..evaluation import evaluation_function

        response = "FOR i = 1 TO n DO\n    x = x + 1\nEND FOR"
        answer = "O(n)"
        result = evaluation_function(response, answer, params)

        assert hasattr(result, 'is_correct')
        assert hasattr(result, 'to_dict')

    def test_linear_meets_linear_bound(self, params):
        """Test linear code meets O(n) bound."""
        from ..evaluation import evaluation_function

        response = "FOR i = 1 TO n DO\n    x = x + 1\nEND FOR"
        answer = "O(n)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True

    def test_constant_meets_linear_bound(self, params):
        """Test constant code meets O(n) bound (better than required)."""
        from ..evaluation import evaluation_function

        response = "x = 1\ny = 2"
        answer = "O(n)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True

    def test_quadratic_exceeds_linear_bound(self, params):
        """Test quadratic code exceeds O(n) bound."""
        from ..evaluation import evaluation_function

        response = """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        x = x + 1
    END FOR
END FOR"""
        answer = "O(n)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is False

    def test_quadratic_meets_quadratic_bound(self, params):
        """Test quadratic code meets O(n^2) bound."""
        from ..evaluation import evaluation_function

        response = """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        x = x + 1
    END FOR
END FOR"""
        answer = "O(n^2)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True


class TestEvaluationComplexityBounds:
    """Test various complexity bounds."""

    def test_log_n_meets_log_n_bound(self, params):
        """Test O(log n) code meets O(log n) bound."""
        from ..evaluation import evaluation_function

        response = """FUNCTION binarySearch(A, target, low, high)
    IF low > high THEN
        RETURN -1
    END IF
    mid = (low + high) / 2
    IF A[mid] == target THEN
        RETURN mid
    ELSE IF A[mid] < target THEN
        RETURN binarySearch(A, target, mid + 1, high)
    ELSE
        RETURN binarySearch(A, target, low, mid - 1)
    END IF
END FUNCTION"""
        answer = "O(log n)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True

    def test_linear_exceeds_log_n_bound(self, params):
        """Test O(n) code exceeds O(log n) bound."""
        from ..evaluation import evaluation_function

        response = "FOR i = 1 TO n DO\n    x = x + 1\nEND FOR"
        answer = "O(log n)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is False

    def test_nlogn_meets_nlogn_bound(self, params):
        """Test O(n log n) code meets O(n log n) bound."""
        from ..evaluation import evaluation_function

        response = """FUNCTION mergeSort(A, low, high)
    IF low < high THEN
        mid = (low + high) / 2
        mergeSort(A, low, mid)
        mergeSort(A, mid + 1, high)
        merge(A, low, mid, high)
    END IF
END FUNCTION"""
        answer = "O(n log n)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True

    def test_linear_meets_nlogn_bound(self, params):
        """Test O(n) code meets O(n log n) bound (better than required)."""
        from ..evaluation import evaluation_function

        response = "FOR i = 1 TO n DO\n    x = x + 1\nEND FOR"
        answer = "O(n log n)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True

    def test_cubic_meets_cubic_bound(self, params):
        """Test O(n^3) code meets O(n^3) bound."""
        from ..evaluation import evaluation_function

        response = """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        FOR k = 1 TO n DO
            x = x + 1
        END FOR
    END FOR
END FOR"""
        answer = "O(n^3)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True


class TestEvaluationDictFormats:
    """Test evaluation with dict response/answer formats."""

    def test_dict_answer_time_complexity(self, params):
        """Test dict answer with expected_time_complexity."""
        from ..evaluation import evaluation_function

        response = "FOR i = 1 TO n DO\n    x = x + 1\nEND FOR"
        answer = {"expected_time_complexity": "O(n)"}
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True

    def test_dict_response_with_pseudocode(self, params):
        """Test dict response with pseudocode key."""
        from ..evaluation import evaluation_function

        response = {"pseudocode": "FOR i = 1 TO n DO\n    x = x + 1\nEND FOR"}
        answer = "O(n)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True


class TestEvaluationFeedback:
    """Test feedback generation."""

    def test_feedback_present_in_result(self, params):
        """Test feedback is present in result."""
        from ..evaluation import evaluation_function

        response = "FOR i = 1 TO n DO\n    x = x + 1\nEND FOR"
        answer = "O(n)"
        result = evaluation_function(response, answer, params)

        result_dict = result.to_dict()
        assert "feedback" in result_dict
        assert len(result_dict["feedback"]) > 0

    def test_feedback_shows_complexity(self, params):
        """Test feedback shows detected complexity."""
        from ..evaluation import evaluation_function

        response = "FOR i = 1 TO n DO\n    x = x + 1\nEND FOR"
        answer = "O(n)"
        result = evaluation_function(response, answer, params)

        assert "O(n)" in result.feedback

    def test_correct_feedback_positive(self, params):
        """Test correct answer gets positive feedback."""
        from ..evaluation import evaluation_function

        response = "FOR i = 1 TO n DO\n    x = x + 1\nEND FOR"
        answer = "O(n)"
        result = evaluation_function(response, answer, params)

        assert "Correct" in result.feedback or "meets" in result.feedback


class TestEvaluationErrorHandling:
    """Test error handling."""

    def test_empty_pseudocode(self, params):
        """Test handling of empty pseudocode."""
        from ..evaluation import evaluation_function

        response = ""
        answer = "O(n)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is False
        assert "No pseudocode" in result.feedback

    def test_none_response(self, params):
        """Test handling of None response."""
        from ..evaluation import evaluation_function

        response = None
        answer = "O(n)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is False


class TestEvaluationComplexityVariants:
    """Test different complexity notation variants."""

    def test_accepts_n_squared_notation(self, params):
        """Test accepts O(n^2) notation."""
        from ..evaluation import evaluation_function

        response = """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        x = x + 1
    END FOR
END FOR"""
        answer = "O(n^2)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True

    def test_accepts_unicode_squared(self, params):
        """Test accepts O(n²) unicode notation."""
        from ..evaluation import evaluation_function

        response = """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        x = x + 1
    END FOR
END FOR"""
        answer = "O(n²)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True

    def test_accepts_quadratic_word(self, params):
        """Test accepts 'quadratic' as answer."""
        from ..evaluation import evaluation_function

        response = """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        x = x + 1
    END FOR
END FOR"""
        answer = "quadratic"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True


class TestEvaluationCurlyBraceSyntax:
    """Test evaluation with curly brace syntax."""

    def test_curly_brace_loops(self, params):
        """Test curly brace loop syntax."""
        from ..evaluation import evaluation_function

        response = """FOR i = 1 TO n {
    FOR j = 1 TO n {
        x = x + 1
    }
}"""
        answer = "O(n^2)"
        result = evaluation_function(response, answer, params)

        assert result.is_correct is True
