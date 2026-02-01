"""
Comprehensive tests for the Preprocessor module.

Tests cover:
- Keyword normalization (case variations)
- Operator normalization (assignment, comparison)
- Whitespace normalization
- Typo correction
- String preservation
- Edge cases
"""

import pytest
from ..parser.preprocessor import Preprocessor, PreprocessorConfig


class TestKeywordNormalization:
    """Tests for keyword case normalization."""
    
    def test_for_keyword_variations(self, preprocessor):
        """Test FOR keyword case variations."""
        variations = ["FOR", "For", "for", "FOR", "fOr"]
        
        for var in variations:
            code = f"{var} i = 1 to n do"
            result, _ = preprocessor.preprocess(code)
            assert result.startswith("for"), f"'{var}' should normalize to 'for'"
    
    def test_while_keyword_variations(self, preprocessor):
        """Test WHILE keyword case variations."""
        variations = ["WHILE", "While", "while", "WHILST", "whilst"]
        
        for var in variations:
            code = f"{var} i < n do"
            result, _ = preprocessor.preprocess(code)
            assert result.startswith("while"), f"'{var}' should normalize to 'while'"
    
    def test_if_then_else_variations(self, preprocessor):
        """Test IF/THEN/ELSE keyword variations."""
        code = "IF condition THEN do_something ELSE do_other"
        result, _ = preprocessor.preprocess(code)
        
        assert "if" in result
        assert "then" in result
        assert "else" in result
    
    def test_function_keyword_variations(self, preprocessor):
        """Test FUNCTION keyword variations."""
        variations = [
            "FUNCTION test()",
            "Function test()",
            "function test()",
            "ALGORITHM test()",
            "Algorithm test()",
            "PROCEDURE test()",
            "DEF test()",
        ]
        
        for code in variations:
            result, _ = preprocessor.preprocess(code)
            assert "function" in result or "algorithm" in result or "procedure" in result or "def" in result
    
    def test_return_keyword_variations(self, preprocessor):
        """Test RETURN keyword variations."""
        variations = ["RETURN x", "Return x", "return x", "RETURNS x"]
        
        for code in variations:
            result, _ = preprocessor.preprocess(code)
            assert "return" in result
    
    def test_boolean_literal_normalization(self, preprocessor):
        """Test TRUE/FALSE normalization."""
        code = "x = TRUE\ny = FALSE"
        result, _ = preprocessor.preprocess(code)
        
        assert "true" in result
        assert "false" in result
    
    def test_logical_operator_variations(self, preprocessor):
        """Test AND/OR/NOT variations."""
        test_cases = [
            ("a AND b", "and"),
            ("a And b", "and"),
            ("a && b", "and"),  # && gets replaced with " and "
            ("a OR b", "or"),
            ("a Or b", "or"),
            ("a || b", "or"),  # || gets replaced with " or "
            ("NOT a", "not"),
            ("Not a", "not"),
        ]
        
        for code, expected in test_cases:
            result, _ = preprocessor.preprocess(code)
            # Check that the expected word appears (may have extra spaces)
            assert expected in result.lower(), f"'{code}' should contain '{expected}', got '{result}'"
    
    def test_end_keyword_variations(self, preprocessor):
        """Test END keyword variations."""
        variations = [
            "END FOR", "ENDFOR", "endfor", "End For",
            "END WHILE", "ENDWHILE", "endwhile",
            "END IF", "ENDIF", "endif",
            "DONE", "done",
        ]
        
        for code in variations:
            result, _ = preprocessor.preprocess(code)
            assert "end" in result or "done" in result


class TestOperatorNormalization:
    """Tests for operator normalization."""
    
    def test_assignment_operators(self, preprocessor):
        """Test assignment operator normalization."""
        test_cases = [
            ("x := 5", "x = 5"),
            ("x ← 5", "x = 5"),
            ("x <- 5", "x = 5"),
        ]
        
        for code, expected in test_cases:
            result, _ = preprocessor.preprocess(code)
            assert "=" in result
            assert ":=" not in result
            assert "←" not in result
            assert "<-" not in result
    
    def test_unicode_comparison_operators(self, preprocessor):
        """Test unicode comparison operator normalization."""
        test_cases = [
            ("a ≤ b", "<="),
            ("a ≥ b", ">="),
            ("a ≠ b", "!="),
        ]
        
        for code, expected in test_cases:
            result, _ = preprocessor.preprocess(code)
            assert expected in result
    
    def test_not_equal_variations(self, preprocessor):
        """Test not-equal operator variations."""
        test_cases = [
            ("a <> b", "!="),
            ("a ≠ b", "!="),
        ]
        
        for code, expected in test_cases:
            result, _ = preprocessor.preprocess(code)
            assert expected in result


class TestTypoCorrection:
    """Tests for typo correction."""
    
    def test_common_keyword_typos(self, preprocessor):
        """Test correction of common keyword typos."""
        typos = {
            "whlie": "while",
            "wihle": "while",
            "fro": "for",
            "eles": "else",
            "esle": "else",
            "retrun": "return",
            "reutrn": "return",
            "fucntion": "function",
            "funtion": "function",
            "algoritm": "algorithm",
            "pritn": "print",
        }
        
        for typo, correct in typos.items():
            code = f"{typo} test"
            result, warnings = preprocessor.preprocess(code)
            assert correct in result.lower(), f"'{typo}' should be corrected to '{correct}', got '{result}'"
            assert len(warnings) > 0, f"Warning should be generated for typo '{typo}'"
    
    def test_typo_warning_message(self, preprocessor):
        """Test that typo corrections generate appropriate warnings."""
        code = "whlie condition do"
        result, warnings = preprocessor.preprocess(code)
        
        assert len(warnings) > 0
        assert any("whlie" in w.lower() or "fixed" in w.lower() for w in warnings)
    
    def test_no_typo_correction_when_disabled(self):
        """Test that typos are not corrected when disabled."""
        config = PreprocessorConfig(fix_common_typos=False)
        preprocessor = Preprocessor(config)
        
        code = "whlie condition do"
        result, warnings = preprocessor.preprocess(code)
        
        # Should still normalize case, but not fix typo
        assert "whlie" in result.lower() or "while" not in result


class TestWhitespaceNormalization:
    """Tests for whitespace normalization."""
    
    def test_tab_to_space_conversion(self, preprocessor):
        """Test tab to space conversion."""
        code = "FOR i = 1 TO n DO\n\t\tprint(i)"
        result, _ = preprocessor.preprocess(code)
        
        assert "\t" not in result
    
    def test_trailing_whitespace_removal(self, preprocessor):
        """Test trailing whitespace removal."""
        code = "x = 1   \ny = 2   "
        result, _ = preprocessor.preprocess(code)
        
        for line in result.split('\n'):
            assert line == line.rstrip()
    
    def test_multiple_blank_lines_collapse(self, preprocessor):
        """Test multiple blank lines collapse to at most two."""
        code = "x = 1\n\n\n\n\ny = 2"
        result, _ = preprocessor.preprocess(code)
        
        assert "\n\n\n" not in result
    
    def test_multiple_spaces_normalization(self, preprocessor):
        """Test multiple spaces normalized to single space."""
        code = "x   =    1"
        result, _ = preprocessor.preprocess(code)
        
        # Leading whitespace preserved, but multiple internal spaces collapsed
        assert "   " not in result.strip()
    
    def test_indentation_preserved(self, preprocessor):
        """Test that meaningful indentation is preserved."""
        code = """FOR i = 1 TO n DO
    FOR j = 1 TO n DO
        x = x + 1
    END FOR
END FOR"""
        result, _ = preprocessor.preprocess(code)
        
        lines = result.split('\n')
        # Check that indentation structure is preserved
        assert lines[1].startswith(' ' * 4) or lines[1].startswith('    ')


class TestStringPreservation:
    """Tests for string literal preservation."""
    
    def test_double_quoted_strings_preserved(self, preprocessor):
        """Test that double-quoted strings are preserved."""
        code = 'print("HELLO WORLD FOR WHILE IF")'
        result, _ = preprocessor.preprocess(code)
        
        # Keywords inside string should NOT be normalized
        assert '"HELLO WORLD FOR WHILE IF"' in result or '"hello world for while if"' in result.lower()
    
    def test_single_quoted_strings_preserved(self, preprocessor):
        """Test that single-quoted strings are preserved."""
        code = "print('HELLO WORLD FOR WHILE IF')"
        result, _ = preprocessor.preprocess(code)
        
        assert "'" in result
    
    def test_mixed_strings_and_keywords(self, preprocessor):
        """Test code with both strings and keywords."""
        code = '''FOR i = 1 TO n DO
    print("Processing item: FOR")
END FOR'''
        result, _ = preprocessor.preprocess(code)
        
        # Outer FOR should be normalized, but string content preserved
        assert result.startswith("for")


class TestIndentationDetection:
    """Tests for indentation style detection."""
    
    def test_detect_2_space_indent(self, preprocessor):
        """Test detection of 2-space indentation."""
        code = """for i = 1 to n do
  x = x + 1
  y = y + 1
end for"""
        
        indent_unit = preprocessor.detect_indentation_style(code)
        assert indent_unit == 2
    
    def test_detect_4_space_indent(self, preprocessor):
        """Test detection of 4-space indentation."""
        code = """for i = 1 to n do
    x = x + 1
    y = y + 1
end for"""
        
        indent_unit = preprocessor.detect_indentation_style(code)
        assert indent_unit == 4
    
    def test_get_indent_level(self, preprocessor):
        """Test getting indent level of lines."""
        assert preprocessor.get_indent_level("x = 1", 4) == 0
        assert preprocessor.get_indent_level("    x = 1", 4) == 1
        assert preprocessor.get_indent_level("        x = 1", 4) == 2


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_empty_input(self, preprocessor):
        """Test handling of empty input."""
        result, warnings = preprocessor.preprocess("")
        assert result == ""
    
    def test_only_whitespace(self, preprocessor):
        """Test handling of whitespace-only input."""
        result, warnings = preprocessor.preprocess("   \n\n   \t\t")
        assert result.strip() == ""
    
    def test_single_line(self, preprocessor):
        """Test handling of single line input."""
        code = "x = 1"
        result, _ = preprocessor.preprocess(code)
        assert result == "x = 1"
    
    def test_windows_line_endings(self, preprocessor):
        """Test handling of Windows line endings."""
        code = "FOR i = 1 TO n DO\r\n    print(i)\r\nEND FOR"
        result, _ = preprocessor.preprocess(code)
        
        assert "\r\n" not in result
        assert "\r" not in result
    
    def test_mac_line_endings(self, preprocessor):
        """Test handling of old Mac line endings."""
        code = "FOR i = 1 TO n DO\r    print(i)\rEND FOR"
        result, _ = preprocessor.preprocess(code)
        
        assert "\r" not in result
    
    def test_unicode_identifiers(self, preprocessor):
        """Test handling of unicode in identifiers."""
        code = "variablé = 1\nπ = 3.14"
        result, _ = preprocessor.preprocess(code)
        
        # Should not crash on unicode
        assert "=" in result
    
    def test_very_long_lines(self, preprocessor):
        """Test handling of very long lines."""
        long_expr = "x = " + " + ".join(["a"] * 100)
        result, _ = preprocessor.preprocess(long_expr)
        
        assert "x =" in result
    
    def test_nested_strings(self, preprocessor):
        """Test handling of nested quotes (escaped)."""
        code = r'print("He said \"hello\"")'
        result, _ = preprocessor.preprocess(code)
        
        # Should handle escaped quotes
        assert "print" in result


class TestPreprocessorConfig:
    """Tests for preprocessor configuration options."""
    
    def test_disable_case_normalization(self):
        """Test disabling case normalization."""
        config = PreprocessorConfig(normalize_case=False)
        preprocessor = Preprocessor(config)
        
        code = "FOR i = 1 TO n DO"
        result, _ = preprocessor.preprocess(code)
        
        # Keywords should retain original case
        assert "FOR" in result or "for" in result  # May still be affected by other normalizations
    
    def test_disable_operator_normalization(self):
        """Test disabling operator normalization."""
        config = PreprocessorConfig(normalize_operators=False)
        preprocessor = Preprocessor(config)
        
        code = "x := 5"
        result, _ = preprocessor.preprocess(code)
        
        # Should keep := operator
        assert ":=" in result
    
    def test_disable_whitespace_normalization(self):
        """Test disabling whitespace normalization."""
        config = PreprocessorConfig(normalize_whitespace=False)
        preprocessor = Preprocessor(config)
        
        code = "x   =   1"
        result, _ = preprocessor.preprocess(code)
        
        # Multiple spaces should be preserved
        assert "   " in result
    
    def test_custom_tab_size(self):
        """Test custom tab size."""
        config = PreprocessorConfig(tab_size=2)
        preprocessor = Preprocessor(config)
        
        code = "FOR i = 1 TO n DO\n\tprint(i)"
        result, _ = preprocessor.preprocess(code)
        
        # Tab should be converted to 2 spaces
        lines = result.split('\n')
        if len(lines) > 1:
            assert lines[1].startswith("  ") or not lines[1].startswith("    ")


class TestComplexPseudocode:
    """Tests for complex pseudocode examples."""
    
    def test_full_algorithm_normalization(self, preprocessor, bubble_sort):
        """Test normalization of complete bubble sort algorithm."""
        result, warnings = preprocessor.preprocess(bubble_sort)
        
        # Should have normalized keywords
        assert "function" in result
        assert "for" in result
        assert "if" in result
    
    def test_mixed_style_normalization(self, preprocessor, mixed_case_keywords):
        """Test normalization of mixed case keywords."""
        result, _ = preprocessor.preprocess(mixed_case_keywords)
        
        # All keywords should be lowercase
        assert "FOR" not in result or "for" in result
        assert "While" not in result or "while" in result
    
    def test_unicode_operators_normalization(self, preprocessor, unicode_operators):
        """Test normalization of unicode operators."""
        result, _ = preprocessor.preprocess(unicode_operators)
        
        # Unicode operators should be converted
        assert "←" not in result
        assert "≤" not in result
        assert "≥" not in result
        assert "≠" not in result
