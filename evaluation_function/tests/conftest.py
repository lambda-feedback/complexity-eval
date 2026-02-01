"""
Pytest configuration and fixtures for the test suite.
"""

import pytest
from typing import List, Dict, Any

from ..parser.preprocessor import Preprocessor, PreprocessorConfig
from ..parser.parser import PseudocodeParser, ParserConfig
from ..schemas.complexity import ComplexityClass


# =============================================================================
# Parser Fixtures
# =============================================================================

@pytest.fixture
def preprocessor() -> Preprocessor:
    """Create a default preprocessor instance."""
    return Preprocessor()


@pytest.fixture
def preprocessor_strict() -> Preprocessor:
    """Create a preprocessor with strict settings."""
    config = PreprocessorConfig(
        normalize_case=True,
        normalize_operators=True,
        normalize_whitespace=True,
        fix_common_typos=False,  # Strict: don't fix typos
        preserve_strings=True,
    )
    return Preprocessor(config)


@pytest.fixture
def parser() -> PseudocodeParser:
    """Create a default parser instance."""
    return PseudocodeParser()


@pytest.fixture
def parser_strict() -> PseudocodeParser:
    """Create a parser with strict mode enabled."""
    config = ParserConfig(strict_mode=True)
    return PseudocodeParser(config)


# =============================================================================
# Sample Pseudocode Fixtures
# =============================================================================

@pytest.fixture
def simple_for_loop() -> str:
    """Simple FOR loop pseudocode."""
    return """FOR i = 1 TO n DO
    print(i)
END FOR"""


@pytest.fixture
def nested_for_loops() -> str:
    """Nested FOR loops pseudocode."""
    return """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        sum = sum + A[i][j]
    END FOR
END FOR"""


@pytest.fixture
def triple_nested_loops() -> str:
    """Triple nested loops pseudocode."""
    return """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        FOR k = 1 TO n DO
            result = result + A[i][j][k]
        END FOR
    END FOR
END FOR"""


@pytest.fixture
def while_loop() -> str:
    """Simple WHILE loop pseudocode."""
    return """WHILE i < n DO
    i = i + 1
    count = count + 1
END WHILE"""


@pytest.fixture
def binary_search() -> str:
    """Binary search algorithm (O(log n))."""
    return """FUNCTION binarySearch(A, target, low, high)
    WHILE low <= high DO
        mid = (low + high) / 2
        IF A[mid] == target THEN
            RETURN mid
        ELSE IF A[mid] < target THEN
            low = mid + 1
        ELSE
            high = mid - 1
        END IF
    END WHILE
    RETURN -1
END FUNCTION"""


@pytest.fixture
def bubble_sort() -> str:
    """Bubble sort algorithm (O(n²))."""
    return """FUNCTION bubbleSort(A, n)
    FOR i = 1 TO n-1 DO
        FOR j = 1 TO n-i DO
            IF A[j] > A[j+1] THEN
                swap(A[j], A[j+1])
            END IF
        END FOR
    END FOR
END FUNCTION"""


@pytest.fixture
def recursive_fibonacci() -> str:
    """Recursive Fibonacci (O(2^n))."""
    return """FUNCTION fib(n)
    IF n <= 1 THEN
        RETURN n
    END IF
    RETURN fib(n-1) + fib(n-2)
END FUNCTION"""


@pytest.fixture
def recursive_factorial() -> str:
    """Recursive factorial (O(n))."""
    return """FUNCTION factorial(n)
    IF n <= 1 THEN
        RETURN 1
    END IF
    RETURN n * factorial(n-1)
END FUNCTION"""


@pytest.fixture
def merge_sort() -> str:
    """Merge sort algorithm (O(n log n))."""
    return """FUNCTION mergeSort(A, left, right)
    IF left < right THEN
        mid = (left + right) / 2
        mergeSort(A, left, mid)
        mergeSort(A, mid+1, right)
        merge(A, left, mid, right)
    END IF
END FUNCTION

FUNCTION merge(A, left, mid, right)
    FOR i = left TO right DO
        temp[i] = A[i]
    END FOR
END FUNCTION"""


@pytest.fixture
def linear_search() -> str:
    """Linear search algorithm (O(n))."""
    return """FUNCTION linearSearch(A, n, target)
    FOR i = 1 TO n DO
        IF A[i] == target THEN
            RETURN i
        END IF
    END FOR
    RETURN -1
END FUNCTION"""


@pytest.fixture
def matrix_multiplication() -> str:
    """Matrix multiplication (O(n³))."""
    return """FUNCTION matrixMultiply(A, B, n)
    FOR i = 1 TO n DO
        FOR j = 1 TO n DO
            C[i][j] = 0
            FOR k = 1 TO n DO
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
            END FOR
        END FOR
    END FOR
    RETURN C
END FUNCTION"""


# =============================================================================
# Pseudocode Style Variations Fixtures
# =============================================================================

@pytest.fixture
def python_style_loop() -> str:
    """Python-style pseudocode."""
    return """def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j+1]:
                swap(arr[j], arr[j+1])"""


@pytest.fixture
def pascal_style_loop() -> str:
    """Pascal-style pseudocode."""
    return """PROCEDURE BubbleSort(A: ARRAY; n: INTEGER);
BEGIN
    FOR i := 1 TO n-1 DO
        FOR j := 1 TO n-i DO
            IF A[j] > A[j+1] THEN
                SWAP(A[j], A[j+1])
            END
        END
    END
END"""


@pytest.fixture
def c_style_loop() -> str:
    """C-style pseudocode."""
    return """function bubbleSort(A[], n) {
    for (i = 0; i < n-1; i++) {
        for (j = 0; j < n-i-1; j++) {
            if (A[j] > A[j+1]) {
                swap(A[j], A[j+1]);
            }
        }
    }
}"""


@pytest.fixture
def mixed_case_keywords() -> str:
    """Pseudocode with mixed case keywords."""
    return """FOR i = 1 To n DO
    While j < n Do
        IF condition Then
            j = j + 1
        ELSE
            j = j - 1
        End If
    End While
End For"""


@pytest.fixture
def unicode_operators() -> str:
    """Pseudocode with unicode operators."""
    return """FOR i ← 1 TO n DO
    IF A[i] ≤ max AND A[i] ≥ min THEN
        IF A[i] ≠ target THEN
            count ← count + 1
        END IF
    END IF
END FOR"""


@pytest.fixture
def typos_in_keywords() -> str:
    """Pseudocode with common typos."""
    return """FUCNTION test(n)
    WHLIE i < n DO
        i = i + 1
    END WHLIE
    RETRUN result
END FUCNTION"""


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def empty_function() -> str:
    """Empty function body."""
    return """FUNCTION emptyFunc()
END FUNCTION"""


@pytest.fixture
def deeply_nested() -> str:
    """Deeply nested structure."""
    return """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        FOR k = 1 TO n DO
            FOR l = 1 TO n DO
                FOR m = 1 TO n DO
                    x = x + 1
                END FOR
            END FOR
        END FOR
    END FOR
END FOR"""


@pytest.fixture
def multiple_functions() -> str:
    """Multiple function definitions."""
    return """FUNCTION helper(x)
    RETURN x * 2
END FUNCTION

FUNCTION main(n)
    FOR i = 1 TO n DO
        result = helper(i)
    END FOR
    RETURN result
END FUNCTION"""


@pytest.fixture
def foreach_loop() -> str:
    """For-each loop pseudocode."""
    return """FOR EACH item IN collection DO
    process(item)
END FOR"""


@pytest.fixture
def repeat_until_loop() -> str:
    """Repeat-until loop pseudocode."""
    return """REPEAT
    x = x + 1
UNTIL x >= n"""


# =============================================================================
# Expected Complexity Fixtures
# =============================================================================

@pytest.fixture
def complexity_test_cases() -> List[Dict[str, Any]]:
    """Test cases with expected complexities."""
    return [
        {
            "name": "constant",
            "code": "x = 1\ny = 2\nz = x + y",
            "expected_time": ComplexityClass.CONSTANT,
            "expected_space": ComplexityClass.CONSTANT,
        },
        {
            "name": "single_loop",
            "code": "FOR i = 1 TO n DO\n    print(i)\nEND FOR",
            "expected_time": ComplexityClass.LINEAR,
            "expected_space": ComplexityClass.CONSTANT,
        },
        {
            "name": "nested_loops",
            "code": "FOR i = 1 TO n DO\n    FOR j = 1 TO n DO\n        x = x + 1\n    END FOR\nEND FOR",
            "expected_time": ComplexityClass.QUADRATIC,
            "expected_space": ComplexityClass.CONSTANT,
        },
        {
            "name": "triple_nested",
            "code": "FOR i = 1 TO n DO\n    FOR j = 1 TO n DO\n        FOR k = 1 TO n DO\n            x = 1\n        END FOR\n    END FOR\nEND FOR",
            "expected_time": ComplexityClass.CUBIC,
            "expected_space": ComplexityClass.CONSTANT,
        },
        {
            "name": "logarithmic",
            "code": "WHILE n > 1 DO\n    n = n / 2\nEND WHILE",
            "expected_time": ComplexityClass.LOGARITHMIC,
            "expected_space": ComplexityClass.CONSTANT,
        },
    ]
