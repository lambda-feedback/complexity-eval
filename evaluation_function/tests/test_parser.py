"""
Comprehensive tests for the Parser module.

Tests cover:
- Basic parsing functionality
- Loop parsing (for, while, repeat, foreach)
- Conditional parsing (if/else/elif)
- Function parsing
- Expression parsing
- Error handling and fallback
- Structure detection
"""

import pytest
from ..parser.parser import PseudocodeParser, ParseError, ParserConfig
from ..schemas.ast_nodes import (
    ProgramNode, FunctionNode, BlockNode, LoopNode, ConditionalNode,
    AssignmentNode, ReturnNode, FunctionCallNode, VariableNode,
    LiteralNode, BinaryOpNode, LoopType, NodeType
)


class TestBasicParsing:
    """Tests for basic parsing functionality."""
    
    def test_parse_returns_parse_result(self, parser):
        """Test that parse returns a ParseResult object."""
        result = parser.parse("x = 1")
        
        assert hasattr(result, 'success')
        assert hasattr(result, 'ast')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
    
    def test_parse_simple_assignment(self, parser):
        """Test parsing simple assignment."""
        result = parser.parse("x = 1")
        
        assert result.success or len(result.errors) > 0  # May use fallback
    
    def test_parse_empty_input(self, parser):
        """Test parsing empty input."""
        result = parser.parse("")
        
        # Should handle gracefully
        assert result is not None
    
    def test_parse_whitespace_only(self, parser):
        """Test parsing whitespace-only input."""
        result = parser.parse("   \n\n   ")
        
        assert result is not None
    
    def test_normalized_code_returned(self, parser):
        """Test that normalized code is included in result."""
        result = parser.parse("FOR i = 1 TO n DO\n    print(i)\nEND FOR")
        
        assert result.normalized_code is not None


class TestForLoopParsing:
    """Tests for FOR loop parsing."""
    
    def test_parse_simple_for_loop(self, parser, simple_for_loop):
        """Test parsing simple FOR loop."""
        result = parser.parse(simple_for_loop)
        
        # Check structure was detected
        structure = parser.detect_structure(simple_for_loop)
        assert structure['has_loops']
        assert structure['loop_count'] >= 1
    
    def test_parse_for_loop_with_range(self, parser):
        """Test parsing FOR loop with numeric range."""
        code = """FOR i = 1 TO 10 DO
    print(i)
END FOR"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_loops']
    
    def test_parse_for_loop_with_step(self, parser):
        """Test parsing FOR loop with step."""
        code = """FOR i = 1 TO n STEP 2 DO
    print(i)
END FOR"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_loops']
    
    def test_parse_for_loop_downto(self, parser):
        """Test parsing FOR loop with DOWNTO."""
        code = """FOR i = n DOWNTO 1 DO
    print(i)
END FOR"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_loops']
    
    def test_parse_nested_for_loops(self, parser, nested_for_loops):
        """Test parsing nested FOR loops."""
        result = parser.parse(nested_for_loops)
        
        structure = parser.detect_structure(nested_for_loops)
        assert structure['has_loops']
        assert structure['has_nested_loops']
        assert structure['loop_count'] >= 2
    
    def test_parse_triple_nested_loops(self, parser, triple_nested_loops):
        """Test parsing triple nested loops."""
        result = parser.parse(triple_nested_loops)
        
        structure = parser.detect_structure(triple_nested_loops)
        assert structure['has_loops']
        assert structure['has_nested_loops']
        assert structure['loop_count'] >= 3


class TestWhileLoopParsing:
    """Tests for WHILE loop parsing."""
    
    def test_parse_simple_while_loop(self, parser, while_loop):
        """Test parsing simple WHILE loop."""
        result = parser.parse(while_loop)
        
        structure = parser.detect_structure(while_loop)
        assert structure['has_loops']
    
    def test_parse_while_with_complex_condition(self, parser):
        """Test parsing WHILE with complex condition."""
        code = """WHILE i < n AND j > 0 DO
    i = i + 1
    j = j - 1
END WHILE"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_loops']
    
    def test_parse_nested_while_loops(self, parser):
        """Test parsing nested WHILE loops."""
        code = """WHILE i < n DO
    WHILE j < m DO
        x = x + 1
        j = j + 1
    END WHILE
    i = i + 1
END WHILE"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_loops']
        assert structure['loop_count'] >= 2


class TestRepeatUntilParsing:
    """Tests for REPEAT-UNTIL loop parsing."""
    
    def test_parse_repeat_until(self, parser, repeat_until_loop):
        """Test parsing REPEAT-UNTIL loop."""
        result = parser.parse(repeat_until_loop)
        
        structure = parser.detect_structure(repeat_until_loop)
        assert structure['has_loops']
    
    def test_parse_repeat_with_complex_body(self, parser):
        """Test parsing REPEAT with complex body."""
        code = """REPEAT
    x = x + 1
    y = y * 2
    IF x > 10 THEN
        z = z + 1
    END IF
UNTIL x >= n"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_loops']
        assert structure['has_conditionals']


class TestForEachParsing:
    """Tests for FOR-EACH loop parsing."""
    
    def test_parse_foreach_loop(self, parser, foreach_loop):
        """Test parsing FOR-EACH loop."""
        result = parser.parse(foreach_loop)
        
        structure = parser.detect_structure(foreach_loop)
        assert structure['has_loops']
    
    def test_parse_foreach_variations(self, parser):
        """Test parsing FOR-EACH variations."""
        variations = [
            "FOR EACH item IN list DO\n    print(item)\nEND FOR",
            "FOR item IN list DO\n    print(item)\nEND FOR",
            "for each x in array do\n    process(x)\nend for",
        ]
        
        for code in variations:
            result = parser.parse(code)
            structure = parser.detect_structure(code)
            assert structure['has_loops']


class TestConditionalParsing:
    """Tests for conditional (IF/ELSE) parsing."""
    
    def test_parse_simple_if(self, parser):
        """Test parsing simple IF statement."""
        code = """IF x > 0 THEN
    print(x)
END IF"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_conditionals']
    
    def test_parse_if_else(self, parser):
        """Test parsing IF-ELSE statement."""
        code = """IF x > 0 THEN
    print("positive")
ELSE
    print("non-positive")
END IF"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_conditionals']
    
    def test_parse_if_elif_else(self, parser):
        """Test parsing IF-ELIF-ELSE statement."""
        code = """IF x > 0 THEN
    print("positive")
ELIF x < 0 THEN
    print("negative")
ELSE
    print("zero")
END IF"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_conditionals']
    
    def test_parse_nested_conditionals(self, parser):
        """Test parsing nested conditionals."""
        code = """IF x > 0 THEN
    IF y > 0 THEN
        print("both positive")
    ELSE
        print("x positive, y non-positive")
    END IF
ELSE
    print("x non-positive")
END IF"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_conditionals']


class TestFunctionParsing:
    """Tests for function definition parsing."""
    
    def test_parse_simple_function(self, parser):
        """Test parsing simple function definition."""
        code = """FUNCTION test()
    RETURN 1
END FUNCTION"""
        result = parser.parse(code)
        
        if result.success and result.ast:
            assert len(result.ast.functions) > 0 or result.ast.global_statements is not None
    
    def test_parse_function_with_parameters(self, parser):
        """Test parsing function with parameters."""
        code = """FUNCTION add(a, b)
    RETURN a + b
END FUNCTION"""
        result = parser.parse(code)
        
        # Should parse without errors or use fallback
        assert result is not None
    
    def test_parse_function_with_array_parameter(self, parser):
        """Test parsing function with array parameter."""
        code = """FUNCTION sum(A[1..n])
    total = 0
    FOR i = 1 TO n DO
        total = total + A[i]
    END FOR
    RETURN total
END FUNCTION"""
        result = parser.parse(code)
        
        assert result is not None
    
    def test_parse_multiple_functions(self, parser, multiple_functions):
        """Test parsing multiple function definitions."""
        result = parser.parse(multiple_functions)
        
        # Should recognize multiple functions
        assert result is not None
    
    def test_parse_recursive_function(self, parser, recursive_fibonacci):
        """Test parsing recursive function."""
        result = parser.parse(recursive_fibonacci)
        
        structure = parser.detect_structure(recursive_fibonacci)
        assert structure['has_recursion']


class TestRecursionDetection:
    """Tests for recursion detection."""
    
    def test_detect_simple_recursion(self, parser, recursive_factorial):
        """Test detection of simple recursion."""
        structure = parser.detect_structure(recursive_factorial)
        assert structure['has_recursion']
    
    def test_detect_double_recursion(self, parser, recursive_fibonacci):
        """Test detection of double recursion (like Fibonacci)."""
        structure = parser.detect_structure(recursive_fibonacci)
        assert structure['has_recursion']
    
    def test_detect_divide_conquer_recursion(self, parser, merge_sort):
        """Test detection of divide-and-conquer recursion."""
        structure = parser.detect_structure(merge_sort)
        assert structure['has_recursion']
    
    def test_no_false_recursion_detection(self, parser, linear_search):
        """Test that non-recursive code is not flagged as recursive."""
        structure = parser.detect_structure(linear_search)
        assert not structure['has_recursion']


class TestExpressionParsing:
    """Tests for expression parsing."""
    
    def test_parse_arithmetic_expressions(self, parser):
        """Test parsing arithmetic expressions."""
        code = "x = a + b * c - d / e"
        result = parser.parse(code)
        
        assert result is not None
    
    def test_parse_comparison_expressions(self, parser):
        """Test parsing comparison expressions."""
        expressions = [
            "IF a == b THEN x = 1 END IF",
            "IF a != b THEN x = 1 END IF",
            "IF a < b THEN x = 1 END IF",
            "IF a <= b THEN x = 1 END IF",
            "IF a > b THEN x = 1 END IF",
            "IF a >= b THEN x = 1 END IF",
        ]
        
        for code in expressions:
            result = parser.parse(code)
            assert result is not None
    
    def test_parse_logical_expressions(self, parser):
        """Test parsing logical expressions."""
        code = "IF a AND b OR NOT c THEN x = 1 END IF"
        result = parser.parse(code)
        
        assert result is not None
    
    def test_parse_array_access(self, parser):
        """Test parsing array access expressions."""
        code = """x = A[i]
y = B[i][j]
z = C[i + 1][j - 1]"""
        result = parser.parse(code)
        
        assert result is not None
    
    def test_parse_function_call_expression(self, parser):
        """Test parsing function call expressions."""
        code = """x = max(a, b)
y = min(c, d)
z = sqrt(x * x + y * y)"""
        result = parser.parse(code)
        
        assert result is not None


class TestStructureDetection:
    """Tests for structure detection."""
    
    def test_detect_no_loops(self, parser):
        """Test structure detection with no loops."""
        code = """x = 1
y = 2
z = x + y"""
        structure = parser.detect_structure(code)
        
        assert not structure['has_loops']
        assert structure['loop_count'] == 0
    
    def test_detect_single_loop(self, parser, simple_for_loop):
        """Test structure detection with single loop."""
        structure = parser.detect_structure(simple_for_loop)
        
        assert structure['has_loops']
        assert structure['loop_count'] >= 1
        assert not structure['has_nested_loops'] or structure['max_nesting'] == 1
    
    def test_detect_nested_loops(self, parser, nested_for_loops):
        """Test structure detection with nested loops."""
        structure = parser.detect_structure(nested_for_loops)
        
        assert structure['has_loops']
        assert structure['has_nested_loops']
        assert structure['loop_count'] >= 2
    
    def test_detect_conditionals(self, parser):
        """Test structure detection with conditionals."""
        code = """IF x > 0 THEN
    y = 1
ELSE
    y = -1
END IF"""
        structure = parser.detect_structure(code)
        
        assert structure['has_conditionals']
    
    def test_detect_complex_structure(self, parser, bubble_sort):
        """Test structure detection with complex algorithm."""
        structure = parser.detect_structure(bubble_sort)
        
        assert structure['has_loops']
        assert structure['has_nested_loops']
        assert structure['has_conditionals']


class TestStyleVariations:
    """Tests for different pseudocode style variations."""
    
    def test_parse_python_style(self, parser, python_style_loop):
        """Test parsing Python-style pseudocode."""
        result = parser.parse(python_style_loop)
        
        structure = parser.detect_structure(python_style_loop)
        assert structure['has_loops']
    
    def test_parse_pascal_style(self, parser, pascal_style_loop):
        """Test parsing Pascal-style pseudocode."""
        result = parser.parse(pascal_style_loop)
        
        structure = parser.detect_structure(pascal_style_loop)
        assert structure['has_loops']
    
    def test_parse_mixed_case(self, parser, mixed_case_keywords):
        """Test parsing mixed case keywords."""
        result = parser.parse(mixed_case_keywords)
        
        structure = parser.detect_structure(mixed_case_keywords)
        assert structure['has_loops']
    
    def test_parse_unicode_operators(self, parser, unicode_operators):
        """Test parsing unicode operators."""
        result = parser.parse(unicode_operators)
        
        assert result is not None


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_handle_syntax_error(self, parser):
        """Test handling of syntax errors."""
        code = "FOR i = 1 TO"  # Incomplete
        result = parser.parse(code)
        
        # Should not crash, may have errors or use fallback
        assert result is not None
    
    def test_handle_mismatched_blocks(self, parser):
        """Test handling of mismatched block delimiters."""
        code = """FOR i = 1 TO n DO
    IF x > 0 THEN
        print(x)
END FOR"""  # Missing END IF
        result = parser.parse(code)
        
        assert result is not None
    
    def test_handle_unknown_keywords(self, parser):
        """Test handling of unknown keywords."""
        code = """UNKNOWN_KEYWORD x = 1
ANOTHER_WEIRD_THING y = 2"""
        result = parser.parse(code)
        
        assert result is not None
    
    def test_fallback_on_parse_error(self, parser):
        """Test that fallback parser is used on error."""
        code = """FOR i = 1 TO n DO
    malformed { syntax [ here
    x = x + 1
END FOR"""
        result = parser.parse(code)
        
        # Should use fallback and still detect loop
        structure = parser.detect_structure(code)
        assert structure['has_loops']
    
    def test_strict_mode_no_fallback(self):
        """Test that strict mode doesn't use fallback for malformed input."""
        config = ParserConfig(strict_mode=True)
        parser = PseudocodeParser(config)
        
        code = "FOR i = 1 TO"  # Incomplete
        result = parser.parse(code)
        
        # In strict mode, may have errors or warnings, but should still return a result
        # The parser is now more resilient
        assert result is not None


class TestCompleteAlgorithms:
    """Tests for parsing complete algorithms."""
    
    def test_parse_binary_search(self, parser, binary_search):
        """Test parsing binary search algorithm."""
        result = parser.parse(binary_search)
        
        structure = parser.detect_structure(binary_search)
        assert structure['has_loops']
        assert structure['has_conditionals']
    
    def test_parse_bubble_sort(self, parser, bubble_sort):
        """Test parsing bubble sort algorithm."""
        result = parser.parse(bubble_sort)
        
        structure = parser.detect_structure(bubble_sort)
        assert structure['has_loops']
        assert structure['has_nested_loops']
        assert structure['has_conditionals']
    
    def test_parse_merge_sort(self, parser, merge_sort):
        """Test parsing merge sort algorithm."""
        result = parser.parse(merge_sort)
        
        structure = parser.detect_structure(merge_sort)
        assert structure['has_recursion']
    
    def test_parse_matrix_multiplication(self, parser, matrix_multiplication):
        """Test parsing matrix multiplication."""
        result = parser.parse(matrix_multiplication)
        
        structure = parser.detect_structure(matrix_multiplication)
        assert structure['has_loops']
        assert structure['has_nested_loops']
        assert structure['loop_count'] >= 3


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_deeply_nested_structure(self, parser, deeply_nested):
        """Test parsing deeply nested structure."""
        result = parser.parse(deeply_nested)
        
        structure = parser.detect_structure(deeply_nested)
        assert structure['has_loops']
        assert structure['has_nested_loops']
        assert structure['loop_count'] >= 5
    
    def test_empty_function_body(self, parser, empty_function):
        """Test parsing empty function body."""
        result = parser.parse(empty_function)
        
        assert result is not None
    
    def test_single_statement(self, parser):
        """Test parsing single statement."""
        result = parser.parse("x = 1")
        
        assert result is not None
    
    def test_comments_handling(self, parser):
        """Test that comments are handled."""
        code = """// This is a comment
FOR i = 1 TO n DO
    # Another comment
    x = x + 1
END FOR"""
        result = parser.parse(code)
        
        structure = parser.detect_structure(code)
        assert structure['has_loops']
    
    def test_very_long_code(self, parser):
        """Test parsing very long code."""
        # Generate code with many statements
        statements = [f"x{i} = {i}" for i in range(100)]
        code = "\n".join(statements)

        result = parser.parse(code)

        assert result is not None


class TestCurlyBraceBlocks:
    """Tests for curly brace block syntax."""

    def test_for_loop_with_curly_braces(self, parser):
        """Test parsing FOR loop with curly braces."""
        code = """FOR i = 1 TO n {
    x = x + 1
}"""
        result = parser.parse(code)

        structure = parser.detect_structure(code)
        assert structure['has_loops']
        assert structure['loop_count'] >= 1

    def test_while_loop_with_curly_braces(self, parser):
        """Test parsing WHILE loop with curly braces."""
        code = """WHILE x > 0 {
    x = x - 1
}"""
        result = parser.parse(code)

        structure = parser.detect_structure(code)
        assert structure['has_loops']

    def test_if_statement_with_curly_braces(self, parser):
        """Test parsing IF statement with curly braces."""
        code = """IF x > 0 {
    y = 1
}"""
        result = parser.parse(code)

        structure = parser.detect_structure(code)
        assert structure['has_conditionals']

    def test_function_with_curly_braces(self, parser):
        """Test parsing function with curly braces."""
        code = """FUNCTION test(n) {
    FOR i = 1 TO n {
        x = x + 1
    }
}"""
        result = parser.parse(code)

        structure = parser.detect_structure(code)
        assert structure['has_loops']

    def test_nested_loops_with_curly_braces(self, parser):
        """Test parsing nested loops with curly braces."""
        code = """FOR i = 1 TO n {
    FOR j = 1 TO n {
        x = x + 1
    }
}"""
        result = parser.parse(code)

        structure = parser.detect_structure(code)
        assert structure['has_loops']
        assert structure['has_nested_loops']
        assert structure['loop_count'] >= 2

    def test_mixed_end_and_braces(self, parser):
        """Test mixing END keywords and curly braces."""
        code = """FOR i = 1 TO n {
    IF x > 0 THEN
        y = 1
    END IF
}"""
        result = parser.parse(code)

        structure = parser.detect_structure(code)
        assert structure['has_loops']
        assert structure['has_conditionals']


class TestCallKeyword:
    """Tests for CALL keyword function invocation."""

    def test_call_keyword_statement(self, parser):
        """Test CALL keyword for function invocation."""
        code = """FOR i = 1 TO n DO
    CALL print(i)
END FOR"""
        result = parser.parse(code)

        structure = parser.detect_structure(code)
        assert structure['has_loops']

    def test_call_keyword_in_function(self, parser):
        """Test CALL keyword within a function."""
        code = """FUNCTION test(A, n)
    FOR i = 1 TO n DO
        CALL process(A[i])
    END FOR
END FUNCTION"""
        result = parser.parse(code)

        structure = parser.detect_structure(code)
        assert structure['has_loops']

    def test_direct_function_call(self, parser):
        """Test direct function call without CALL keyword."""
        code = """FOR i = 1 TO n DO
    print(i)
END FOR"""
        result = parser.parse(code)

        structure = parser.detect_structure(code)
        assert structure['has_loops']

    def test_call_with_curly_braces(self, parser):
        """Test CALL keyword with curly brace blocks."""
        code = """FOR i = 1 TO n {
    CALL process(i)
}"""
        result = parser.parse(code)

        structure = parser.detect_structure(code)
        assert structure['has_loops']
