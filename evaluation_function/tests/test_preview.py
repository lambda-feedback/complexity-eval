"""
Tests for the preview function.

Tests cover:
- Basic preview functionality
- Different pseudocode styles
- Error handling
- Complexity detection in preview
"""

import pytest


class MockParams:
    """Mock params object for testing."""
    pass


@pytest.fixture
def params():
    """Default params fixture."""
    return MockParams()


class TestPreviewBasic:
    """Basic preview function tests."""

    def test_preview_simple_loop(self, params):
        """Test preview of simple loop code."""
        from ..preview import preview_function

        code = """FOR i = 1 TO n DO
    x = x + 1
END FOR"""
        result = preview_function(code, params)

        assert "preview" in result
        preview = result["preview"]
        assert preview is not None
        assert "feedback" in preview
        assert "Parsing: Successful" in preview["feedback"]
        assert "Time Complexity" in preview["feedback"]

    def test_preview_nested_loops(self, params):
        """Test preview detects nested loops."""
        from ..preview import preview_function

        code = """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        sum = sum + A[i][j]
    END FOR
END FOR"""
        result = preview_function(code, params)

        preview = result["preview"]
        assert "Nested loops" in preview["feedback"]
        assert "O(n²)" in preview["feedback"]

    def test_preview_recursion(self, params):
        """Test preview detects recursion."""
        from ..preview import preview_function

        code = """FUNCTION factorial(n)
    IF n <= 1 THEN
        RETURN 1
    END IF
    RETURN n * factorial(n - 1)
END FUNCTION"""
        result = preview_function(code, params)

        preview = result["preview"]
        assert "Recursion" in preview["feedback"]


class TestPreviewEmptyInput:
    """Test handling of empty/invalid input."""

    def test_preview_empty_input(self, params):
        """Test preview handles empty input."""
        from ..preview import preview_function

        result = preview_function("", params)

        preview = result["preview"]
        assert "Please enter your pseudocode" in preview["feedback"]

    def test_preview_whitespace_only(self, params):
        """Test preview handles whitespace-only input."""
        from ..preview import preview_function

        result = preview_function("   \n\n   ", params)

        preview = result["preview"]
        assert "Please enter your pseudocode" in preview["feedback"]

    def test_preview_invalid_response_type(self, params):
        """Test preview handles invalid response type."""
        from ..preview import preview_function

        result = preview_function(12345, params)

        preview = result["preview"]
        assert "Invalid response format" in preview["feedback"]


class TestPreviewDictInput:
    """Test preview with dict response formats."""

    def test_preview_dict_response(self, params):
        """Test preview accepts dict response format."""
        from ..preview import preview_function

        response = {"pseudocode": "FOR i = 1 TO n DO\n    x = x + 1\nEND FOR"}
        result = preview_function(response, params)

        preview = result["preview"]
        assert "Parsing: Successful" in preview["feedback"]

    def test_preview_with_code_key(self, params):
        """Test preview accepts 'code' key in dict."""
        from ..preview import preview_function

        response = {"code": "FOR i = 1 TO n DO\n    x = x + 1\nEND FOR"}
        result = preview_function(response, params)

        preview = result["preview"]
        assert "Parsing: Successful" in preview["feedback"]


class TestPreviewLoopTypes:
    """Test preview with different loop types."""

    def test_preview_curly_brace_syntax(self, params):
        """Test preview handles curly brace block syntax."""
        from ..preview import preview_function

        code = """FOR i = 1 TO n {
    x = x + 1
}"""
        result = preview_function(code, params)

        preview = result["preview"]
        assert "Parsing: Successful" in preview["feedback"]
        assert "Loops" in preview["feedback"]

    def test_preview_while_loop(self, params):
        """Test preview detects while loop."""
        from ..preview import preview_function

        code = """WHILE x > 0 DO
    x = x - 1
END WHILE"""
        result = preview_function(code, params)

        preview = result["preview"]
        assert "Loops" in preview["feedback"]


class TestPreviewComplexityDetection:
    """Test complexity detection in preview."""

    def test_preview_binary_search(self, params):
        """Test preview analyzes binary search correctly."""
        from ..preview import preview_function

        code = """FUNCTION binarySearch(A, target, low, high)
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
        result = preview_function(code, params)

        preview = result["preview"]
        assert "Recursion" in preview["feedback"]
        assert "O(log n)" in preview["feedback"]

    def test_preview_constant_complexity(self, params):
        """Test preview detects constant complexity."""
        from ..preview import preview_function

        code = """x = 1
y = 2
z = x + y"""
        result = preview_function(code, params)

        preview = result["preview"]
        assert "O(1)" in preview["feedback"]

    def test_preview_merge_sort_pattern(self, params):
        """Test preview detects merge sort pattern."""
        from ..preview import preview_function

        code = """FUNCTION mergeSort(A, low, high)
    IF low < high THEN
        mid = (low + high) / 2
        mergeSort(A, low, mid)
        mergeSort(A, mid + 1, high)
        merge(A, low, mid, high)
    END IF
END FUNCTION"""
        result = preview_function(code, params)

        preview = result["preview"]
        assert "O(n log n)" in preview["feedback"]


class TestPreviewLatex:
    """Test LaTeX output in preview."""

    def test_preview_latex_output(self, params):
        """Test preview includes LaTeX formatted complexity."""
        from ..preview import preview_function

        code = "FOR i = 1 TO n DO\n    x = 1\nEND FOR"
        result = preview_function(code, params)

        preview = result["preview"]
        assert "latex" in preview
        assert "Time Complexity" in preview["latex"]


class TestPreviewEdgeCases:
    """Edge case tests for preview function."""

    def test_preview_very_long_code(self, params):
        """Test preview handles long code."""
        from ..preview import preview_function

        lines = ["FOR i = 1 TO n DO"]
        for j in range(50):
            lines.append(f"    x{j} = {j}")
        lines.append("END FOR")
        code = "\n".join(lines)

        result = preview_function(code, params)
        preview = result["preview"]
        assert "Parsing: Successful" in preview["feedback"]

    def test_preview_deeply_nested(self, params):
        """Test preview handles deeply nested loops."""
        from ..preview import preview_function

        code = """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        FOR k = 1 TO n DO
            x = x + 1
        END FOR
    END FOR
END FOR"""
        result = preview_function(code, params)

        preview = result["preview"]
        assert "depth" in preview["feedback"].lower()
