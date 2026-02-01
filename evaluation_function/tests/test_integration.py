"""
Integration tests for the complete evaluation pipeline.

Tests cover:
- End-to-end parsing and analysis
- Various algorithm complexities
- Different pseudocode styles
- Error handling scenarios
- Edge cases
"""

import pytest
from ..parser.parser import PseudocodeParser
from ..parser.preprocessor import Preprocessor
from ..schemas.complexity import ComplexityClass
from ..schemas.ast_nodes import NodeType, LoopType


class TestEndToEndParsing:
    """End-to-end tests for parsing pipeline."""
    
    def test_parse_simple_assignment(self, parser):
        """Test parsing and analyzing simple assignment."""
        code = "x = 1"
        result = parser.parse(code)
        
        assert result is not None
        assert result.normalized_code is not None
    
    def test_parse_simple_loop(self, parser):
        """Test parsing and analyzing simple loop."""
        code = """FOR i = 1 TO n DO
    print(i)
END FOR"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_loops']
        assert structure['loop_count'] >= 1
    
    def test_parse_nested_loops(self, parser):
        """Test parsing and analyzing nested loops."""
        code = """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        sum = sum + A[i][j]
    END FOR
END FOR"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_loops']
        assert structure['has_nested_loops']
    
    def test_parse_function_with_loop(self, parser):
        """Test parsing function containing loop."""
        code = """FUNCTION sum(A, n)
    total = 0
    FOR i = 1 TO n DO
        total = total + A[i]
    END FOR
    RETURN total
END FUNCTION"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_loops']


class TestAlgorithmComplexities:
    """Tests for various algorithm complexities."""
    
    def test_constant_complexity(self, parser):
        """Test O(1) constant complexity detection."""
        code = """x = 1
y = 2
z = x + y
RETURN z"""
        
        structure = parser.detect_structure(code)
        assert not structure['has_loops']
        assert not structure['has_recursion']
    
    def test_linear_complexity(self, parser, linear_search):
        """Test O(n) linear complexity detection."""
        structure = parser.detect_structure(linear_search)
        
        assert structure['has_loops']
        assert structure['loop_count'] == 1
        assert not structure['has_nested_loops']
    
    def test_quadratic_complexity(self, parser, bubble_sort):
        """Test O(n²) quadratic complexity detection."""
        structure = parser.detect_structure(bubble_sort)
        
        assert structure['has_loops']
        assert structure['has_nested_loops']
        assert structure['loop_count'] >= 2
    
    def test_cubic_complexity(self, parser, matrix_multiplication):
        """Test O(n³) cubic complexity detection."""
        structure = parser.detect_structure(matrix_multiplication)
        
        assert structure['has_loops']
        assert structure['has_nested_loops']
        assert structure['loop_count'] >= 3
    
    def test_logarithmic_complexity(self, parser, binary_search):
        """Test O(log n) logarithmic complexity detection."""
        structure = parser.detect_structure(binary_search)
        
        assert structure['has_loops']
        assert structure['has_conditionals']
    
    def test_linearithmic_complexity(self, parser, merge_sort):
        """Test O(n log n) linearithmic complexity detection."""
        structure = parser.detect_structure(merge_sort)
        
        assert structure['has_recursion']
    
    def test_exponential_complexity(self, parser, recursive_fibonacci):
        """Test O(2^n) exponential complexity detection."""
        structure = parser.detect_structure(recursive_fibonacci)
        
        assert structure['has_recursion']


class TestPseudocodeStyles:
    """Tests for different pseudocode style variations."""
    
    def test_pascal_style(self, parser, pascal_style_loop):
        """Test Pascal-style pseudocode."""
        result = parser.parse(pascal_style_loop)
        
        structure = parser.detect_structure(pascal_style_loop)
        assert structure['has_loops']
    
    def test_python_style(self, parser, python_style_loop):
        """Test Python-style pseudocode."""
        result = parser.parse(python_style_loop)
        
        structure = parser.detect_structure(python_style_loop)
        assert structure['has_loops']
    
    def test_mixed_case_keywords(self, parser, mixed_case_keywords):
        """Test mixed case keywords handling."""
        result = parser.parse(mixed_case_keywords)
        
        # Preprocessing should normalize case
        assert result.normalized_code is not None
        structure = parser.detect_structure(mixed_case_keywords)
        assert structure['has_loops']
    
    def test_unicode_operators(self, parser, unicode_operators):
        """Test unicode operator handling."""
        result = parser.parse(unicode_operators)
        
        # Preprocessing should normalize operators
        assert "←" not in result.normalized_code
        assert "≤" not in result.normalized_code


class TestLoopVariations:
    """Tests for various loop construct variations."""
    
    def test_for_loop_variations(self, parser):
        """Test different FOR loop syntaxes."""
        variations = [
            "FOR i = 1 TO n DO\n    x = x + 1\nEND FOR",
            "for i = 1 to n do\n    x = x + 1\nend for",
            "FOR i := 1 TO n DO\n    x = x + 1\nEND FOR",
            "FOR i ← 1 TO n DO\n    x = x + 1\nEND FOR",
        ]
        
        for code in variations:
            result = parser.parse(code)
            structure = parser.detect_structure(code)
            assert structure['has_loops'], f"Failed to detect loop in: {code[:30]}..."
    
    def test_while_loop_variations(self, parser):
        """Test different WHILE loop syntaxes."""
        variations = [
            "WHILE i < n DO\n    i = i + 1\nEND WHILE",
            "while i < n do\n    i = i + 1\nend while",
            "WHILE (i < n)\n    i = i + 1\nEND WHILE",
        ]
        
        for code in variations:
            result = parser.parse(code)
            structure = parser.detect_structure(code)
            assert structure['has_loops']
    
    def test_foreach_variations(self, parser):
        """Test different FOR-EACH loop syntaxes."""
        variations = [
            "FOR EACH item IN list DO\n    print(item)\nEND FOR",
            "FOR item IN list DO\n    print(item)\nEND FOR",
            "for each x in array do\n    process(x)\nend for",
        ]
        
        for code in variations:
            result = parser.parse(code)
            structure = parser.detect_structure(code)
            assert structure['has_loops']
    
    def test_repeat_until_variations(self, parser):
        """Test REPEAT-UNTIL loop syntaxes."""
        code = """REPEAT
    x = x + 1
UNTIL x >= n"""
        
        result = parser.parse(code)
        structure = parser.detect_structure(code)
        assert structure['has_loops']


class TestConditionalVariations:
    """Tests for various conditional construct variations."""
    
    def test_if_variations(self, parser):
        """Test different IF statement syntaxes."""
        variations = [
            "IF x > 0 THEN\n    y = 1\nEND IF",
            "if x > 0 then\n    y = 1\nend if",
            "IF x > 0:\n    y = 1\nEND IF",
            "IF (x > 0) THEN\n    y = 1\nENDIF",
        ]
        
        for code in variations:
            result = parser.parse(code)
            structure = parser.detect_structure(code)
            assert structure['has_conditionals']
    
    def test_if_else_variations(self, parser):
        """Test IF-ELSE variations."""
        code = """IF x > 0 THEN
    y = 1
ELSE
    y = -1
END IF"""
        
        result = parser.parse(code)
        structure = parser.detect_structure(code)
        assert structure['has_conditionals']
    
    def test_nested_conditionals(self, parser):
        """Test nested conditionals."""
        code = """IF x > 0 THEN
    IF y > 0 THEN
        z = 1
    ELSE
        z = 2
    END IF
ELSE
    z = 3
END IF"""
        
        result = parser.parse(code)
        structure = parser.detect_structure(code)
        assert structure['has_conditionals']


class TestFunctionVariations:
    """Tests for various function definition variations."""
    
    def test_function_keywords(self, parser):
        """Test different function definition keywords."""
        keywords = ["FUNCTION", "function", "ALGORITHM", "algorithm", 
                    "PROCEDURE", "procedure", "DEF", "def"]
        
        for kw in keywords:
            code = f"""{kw} test(x)
    RETURN x * 2
END {kw.upper() if kw.isupper() else 'FUNCTION'}"""
            
            result = parser.parse(code)
            assert result is not None
    
    def test_function_with_parameters(self, parser):
        """Test functions with various parameter styles."""
        variations = [
            "FUNCTION test(a, b)\n    RETURN a + b\nEND FUNCTION",
            "FUNCTION test(A[1..n])\n    RETURN A[1]\nEND FUNCTION",
            "FUNCTION test(x: INTEGER)\n    RETURN x\nEND FUNCTION",
        ]
        
        for code in variations:
            result = parser.parse(code)
            assert result is not None


class TestRecursionDetection:
    """Tests for recursion detection."""
    
    def test_simple_recursion(self, parser):
        """Test simple recursion detection."""
        code = """FUNCTION factorial(n)
    IF n <= 1 THEN
        RETURN 1
    END IF
    RETURN n * factorial(n-1)
END FUNCTION"""
        
        structure = parser.detect_structure(code)
        assert structure['has_recursion']
    
    def test_binary_recursion(self, parser):
        """Test binary recursion detection (Fibonacci)."""
        code = """FUNCTION fib(n)
    IF n <= 1 THEN
        RETURN n
    END IF
    RETURN fib(n-1) + fib(n-2)
END FUNCTION"""
        
        structure = parser.detect_structure(code)
        assert structure['has_recursion']
    
    def test_divide_and_conquer_recursion(self, parser):
        """Test divide-and-conquer recursion detection."""
        code = """FUNCTION mergeSort(A, left, right)
    IF left < right THEN
        mid = (left + right) / 2
        mergeSort(A, left, mid)
        mergeSort(A, mid+1, right)
        merge(A, left, mid, right)
    END IF
END FUNCTION"""
        
        structure = parser.detect_structure(code)
        assert structure['has_recursion']
    
    def test_no_recursion(self, parser):
        """Test that non-recursive code is not flagged."""
        code = """FUNCTION sum(A, n)
    total = 0
    FOR i = 1 TO n DO
        total = total + A[i]
    END FOR
    RETURN total
END FUNCTION"""
        
        structure = parser.detect_structure(code)
        assert not structure['has_recursion']


class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    def test_malformed_loop(self, parser):
        """Test handling of malformed loop."""
        code = """FOR i = 1 TO
    x = x + 1
END FOR"""
        
        result = parser.parse(code)
        # Should not crash, may have errors or use fallback
        assert result is not None
    
    def test_unclosed_block(self, parser):
        """Test handling of unclosed block."""
        code = """IF x > 0 THEN
    y = 1
    # Missing END IF"""
        
        result = parser.parse(code)
        assert result is not None
    
    def test_mismatched_keywords(self, parser):
        """Test handling of mismatched keywords."""
        code = """FOR i = 1 TO n DO
    x = x + 1
END WHILE"""  # Mismatched: FOR with END WHILE
        
        result = parser.parse(code)
        assert result is not None
    
    def test_unknown_constructs(self, parser):
        """Test handling of unknown constructs."""
        code = """MYSTERY_KEYWORD x = 1
ANOTHER_WEIRD_THING y = 2"""
        
        result = parser.parse(code)
        assert result is not None
    
    def test_empty_blocks(self, parser):
        """Test handling of empty blocks."""
        code = """IF x > 0 THEN
END IF

FOR i = 1 TO n DO
END FOR"""
        
        result = parser.parse(code)
        assert result is not None


class TestPreprocessorIntegration:
    """Tests for preprocessor integration."""
    
    def test_typo_correction_in_pipeline(self, parser):
        """Test that typos are corrected during parsing."""
        code = """WHLIE i < n DO
    i = i + 1
END WHLIE"""
        
        result = parser.parse(code)
        
        # Preprocessor should fix "WHLIE" to "WHILE"
        assert "while" in result.normalized_code.lower() or len(result.warnings) > 0
    
    def test_operator_normalization_in_pipeline(self, parser):
        """Test that operators are normalized during parsing."""
        code = """x ← 1
IF a ≤ b THEN
    y ← 2
END IF"""
        
        result = parser.parse(code)
        
        # Operators should be normalized
        assert "←" not in result.normalized_code
    
    def test_case_normalization_in_pipeline(self, parser):
        """Test that keywords are case-normalized."""
        code = """FOR i = 1 To n Do
    PRINT(i)
End FOR"""
        
        result = parser.parse(code)
        
        # Should detect loop regardless of case
        structure = parser.detect_structure(code)
        assert structure['has_loops']


class TestComplexAlgorithms:
    """Tests for complex algorithm parsing."""
    
    def test_quicksort(self, parser):
        """Test parsing quicksort algorithm."""
        code = """FUNCTION quickSort(A, low, high)
    IF low < high THEN
        pivot = partition(A, low, high)
        quickSort(A, low, pivot - 1)
        quickSort(A, pivot + 1, high)
    END IF
END FUNCTION

FUNCTION partition(A, low, high)
    pivot = A[high]
    i = low - 1
    FOR j = low TO high - 1 DO
        IF A[j] <= pivot THEN
            i = i + 1
            swap(A[i], A[j])
        END IF
    END FOR
    swap(A[i + 1], A[high])
    RETURN i + 1
END FUNCTION"""
        
        result = parser.parse(code)
        structure = parser.detect_structure(code)
        
        assert structure['has_loops']
        assert structure['has_recursion']
        assert structure['has_conditionals']
    
    def test_dijkstra(self, parser):
        """Test parsing Dijkstra's algorithm."""
        code = """FUNCTION dijkstra(G, source)
    dist[source] = 0
    FOR EACH vertex v IN G DO
        IF v != source THEN
            dist[v] = INFINITY
        END IF
        add v to Q
    END FOR
    
    WHILE Q is not empty DO
        u = vertex in Q with min dist[u]
        remove u from Q
        
        FOR EACH neighbor v of u DO
            alt = dist[u] + length(u, v)
            IF alt < dist[v] THEN
                dist[v] = alt
            END IF
        END FOR
    END WHILE
    
    RETURN dist
END FUNCTION"""
        
        result = parser.parse(code)
        structure = parser.detect_structure(code)
        
        assert structure['has_loops']
        assert structure['has_nested_loops']
        assert structure['has_conditionals']
    
    def test_dfs(self, parser):
        """Test parsing DFS algorithm."""
        code = """FUNCTION DFS(G, v, visited)
    visited[v] = true
    print(v)
    
    FOR EACH neighbor u of v DO
        IF NOT visited[u] THEN
            DFS(G, u, visited)
        END IF
    END FOR
END FUNCTION"""
        
        result = parser.parse(code)
        structure = parser.detect_structure(code)
        
        assert structure['has_loops']
        assert structure['has_recursion']
        assert structure['has_conditionals']


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_deeply_nested_structure(self, parser, deeply_nested):
        """Test deeply nested loops."""
        result = parser.parse(deeply_nested)
        structure = parser.detect_structure(deeply_nested)
        
        assert structure['has_loops']
        assert structure['has_nested_loops']
        assert structure['loop_count'] >= 5
    
    def test_very_long_code(self, parser):
        """Test parsing very long code."""
        # Generate 200 lines of code
        lines = []
        lines.append("FUNCTION longFunction(n)")
        lines.append("    x = 0")
        for i in range(100):
            lines.append(f"    x = x + {i}")
        lines.append("    FOR i = 1 TO n DO")
        lines.append("        y = y + 1")
        lines.append("    END FOR")
        lines.append("    RETURN x")
        lines.append("END FUNCTION")
        
        code = "\n".join(lines)
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_loops']
    
    def test_single_line_constructs(self, parser):
        """Test single-line constructs."""
        code = "IF x > 0 THEN y = 1"
        
        result = parser.parse(code)
        structure = parser.detect_structure(code)
        assert structure['has_conditionals']
    
    def test_multiple_functions(self, parser, multiple_functions):
        """Test multiple function definitions."""
        result = parser.parse(multiple_functions)
        
        assert result is not None
    
    def test_unicode_identifiers(self, parser):
        """Test code with unicode identifiers."""
        code = """σ = 0
FOR i = 1 TO n DO
    σ = σ + A[i]
END FOR"""
        
        result = parser.parse(code)
        # Should not crash
        assert result is not None
    
    def test_mixed_loops_and_recursion(self, parser):
        """Test code with both loops and recursion."""
        code = """FUNCTION process(A, n)
    FOR i = 1 TO n DO
        IF A[i] > 0 THEN
            result = process(A, A[i])
        END IF
    END FOR
    RETURN result
END FUNCTION"""
        
        result = parser.parse(code)
        structure = parser.detect_structure(code)
        
        assert structure['has_loops']
        assert structure['has_recursion']
        assert structure['has_conditionals']


class TestComplexityTestCases:
    """Tests using predefined complexity test cases."""
    
    def test_complexity_cases(self, parser, complexity_test_cases):
        """Test parsing of all complexity test cases."""
        for test_case in complexity_test_cases:
            code = test_case["code"]
            expected_time = test_case["expected_time"]
            
            result = parser.parse(code)
            
            # Should parse successfully
            assert result is not None, f"Failed to parse {test_case['name']}"
            
            # Structure detection should work
            structure = parser.detect_structure(code)
            
            # Basic sanity checks based on expected complexity
            if expected_time == ComplexityClass.CONSTANT:
                assert not structure['has_loops'], f"{test_case['name']} shouldn't have loops"
            elif expected_time in [ComplexityClass.LINEAR, ComplexityClass.QUADRATIC, ComplexityClass.CUBIC]:
                assert structure['has_loops'], f"{test_case['name']} should have loops"
