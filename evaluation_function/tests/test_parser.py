"""
Comprehensive test suite for the pseudocode parser.

Tests cover:
- Basic loop detection (FOR, WHILE, FOR EACH, REPEAT)
- Nested loops
- Different block styles (indentation, END keywords, curly braces)
- Conditionals with proper condition parsing
- Function definitions
- Edge cases and error handling
"""

import pytest
from ..parser.parser import PseudocodeParser, ParserConfig
from ..parser.preprocessor import PreprocessorConfig
from ..schemas.ast_nodes import (
    LoopType, NodeType, VariableNode, LiteralNode, 
    BinaryOpNode, OperatorType
)


class TestBasicForLoops:
    """Test basic FOR loop parsing."""
    
    def test_simple_for_loop_indented(self):
        """Test FOR loop with indentation-based block."""
        code = """
for i = 1 to n
  print(i)
  x = x + 1
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        assert result.ast is not None
        assert result.ast.global_statements is not None
        assert len(result.ast.global_statements.statements) == 1
        
        loop = result.ast.global_statements.statements[0]
        assert loop.node_type == NodeType.LOOP
        assert loop.loop_type == LoopType.FOR
        assert loop.iterator.name == "i"
        assert isinstance(loop.start, LiteralNode)
        assert loop.start.value == 1
        assert isinstance(loop.end, VariableNode)
        assert loop.end.name == "n"
    
    def test_for_loop_with_end_keyword(self):
        """Test FOR loop with END FOR terminator."""
        code = """
FOR i = 1 TO 10
  x = x + i
END FOR
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        assert result.ast is not None
        loop = result.ast.global_statements.statements[0]
        assert loop.loop_type == LoopType.FOR
        assert loop.iterator.name == "i"
    
    def test_for_loop_with_curly_braces(self):
        """Test FOR loop with curly brace block."""
        code = """
for i = 1 to n {
  print(i)
  sum = sum + i
}
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        loop = result.ast.global_statements.statements[0]
        assert loop.loop_type == LoopType.FOR
        assert loop.body is not None
    
    def test_for_loop_with_do_keyword(self):
        """Test FOR loop with DO keyword."""
        code = """
for i = 0 to n-1 do
  array[i] = 0
end for
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        loop = result.ast.global_statements.statements[0]
        assert loop.loop_type == LoopType.FOR


class TestWhileLoops:
    """Test WHILE loop parsing."""
    
    def test_simple_while_loop(self):
        """Test basic WHILE loop."""
        code = """
while x > 0
  x = x - 1
end while
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        loop = result.ast.global_statements.statements[0]
        assert loop.loop_type == LoopType.WHILE
        assert loop.condition is not None
    
    def test_while_with_do(self):
        """Test WHILE loop with DO keyword."""
        code = """
WHILE count < 100 DO
  count = count + 1
DONE
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        loop = result.ast.global_statements.statements[0]
        assert loop.loop_type == LoopType.WHILE
    
    def test_while_with_complex_condition(self):
        """Test WHILE with comparison operator."""
        code = """
while n >= 1 {
  n = n / 2
}
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        loop = result.ast.global_statements.statements[0]
        assert loop.loop_type == LoopType.WHILE
        # Check that condition was parsed
        assert loop.condition is not None
        assert isinstance(loop.condition, (VariableNode, BinaryOpNode))


class TestForEachLoops:
    """Test FOR EACH loop parsing."""
    
    def test_for_each_basic(self):
        """Test basic FOR EACH loop."""
        code = """
for each item in collection
  process(item)
end for
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        loop = result.ast.global_statements.statements[0]
        assert loop.loop_type == LoopType.FOR_EACH
        assert loop.iterator.name == "item"
        assert loop.collection.name == "collection"
    
    def test_for_in_without_each(self):
        """Test FOR...IN syntax without EACH."""
        code = """
for element in array {
  sum = sum + element
}
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        loop = result.ast.global_statements.statements[0]
        assert loop.loop_type == LoopType.FOR_EACH


class TestRepeatLoops:
    """Test REPEAT...UNTIL loops."""
    
    def test_repeat_until(self):
        """Test REPEAT...UNTIL loop."""
        code = """
repeat
  x = x * 2
until x > 100
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        loop = result.ast.global_statements.statements[0]
        assert loop.loop_type == LoopType.REPEAT_UNTIL


class TestNestedLoops:
    """Test nested loop parsing."""
    
    def test_nested_for_loops(self):
        """Test two nested FOR loops."""
        code = """
for i = 1 to n
  for j = 1 to m
    matrix[i][j] = 0
  end for
end for
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        outer_loop = result.ast.global_statements.statements[0]
        assert outer_loop.loop_type == LoopType.FOR
        assert outer_loop.nesting_level == 0
        
        # Check that inner loop exists in outer loop's body
        assert outer_loop.body is not None
        assert len(outer_loop.body.statements) > 0
        inner_loop = outer_loop.body.statements[0]
        assert inner_loop.loop_type == LoopType.FOR
        assert inner_loop.nesting_level > 0
    
    def test_triple_nested_loops_with_braces(self):
        """Test three nested loops with curly braces."""
        code = """
for i = 1 to n {
  for j = 1 to n {
    for k = 1 to n {
      sum = sum + 1
    }
  }
}
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        structure = parser.detect_structure(code)
        assert structure['has_nested_loops']
        assert structure['max_nesting'] == 3
        assert structure['loop_count'] == 3


class TestConditionals:
    """Test conditional (IF) statement parsing."""
    
    def test_simple_if(self):
        """Test basic IF statement."""
        code = """
if x > 0
  print("positive")
end if
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        cond = result.ast.global_statements.statements[0]
        assert cond.node_type == NodeType.CONDITIONAL
        # FIX: Check that condition was actually parsed
        assert cond.condition is not None
        assert cond.then_branch is not None
    
    def test_if_then_else(self):
        """Test IF-THEN-ELSE."""
        code = """
if n == 0 then
  return 1
else
  return n * factorial(n-1)
endif
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        cond = result.ast.global_statements.statements[0]
        assert cond.condition is not None
        assert cond.then_branch is not None
        assert cond.else_branch is not None
    
    def test_if_with_comparison_operators(self):
        """Test IF with various comparison operators."""
        test_cases = [
            ("if x == 5", OperatorType.EQUAL),
            ("if x != 5", OperatorType.NOT_EQUAL),
            ("if x < 5", OperatorType.LESS_THAN),
            ("if x <= 5", OperatorType.LESS_EQUAL),
            ("if x > 5", OperatorType.GREATER_THAN),
            ("if x >= 5", OperatorType.GREATER_EQUAL),
        ]
        
        parser = PseudocodeParser()
        
        for condition_str, expected_op in test_cases:
            code = f"""
{condition_str}
  x = 1
end if
"""
            result = parser.parse(code)
            assert result.success
            cond = result.ast.global_statements.statements[0]
            assert cond.condition is not None
            
            # If it's a binary op, check the operator
            if isinstance(cond.condition, BinaryOpNode):
                assert cond.condition.operator == expected_op


class TestFunctions:
    """Test function definition parsing."""
    
    def test_function_definition(self):
        """Test basic function definition."""
        code = """
function factorial(n)
  if n == 0
    return 1
  else
    return n * factorial(n-1)
  end if
end function
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        assert len(result.ast.functions) == 1
        func = result.ast.functions[0]
        assert func.name == "factorial"
    
    def test_algorithm_keyword(self):
        """Test ALGORITHM keyword for function definition."""
        code = """
algorithm bubbleSort(arr)
  for i = 1 to n
    for j = 1 to n-1
      if arr[j] > arr[j+1]
        swap(arr[j], arr[j+1])
      end if
    end for
  end for
end algorithm
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        assert len(result.ast.functions) == 1
        assert result.ast.functions[0].name == "bubbleSort"


class TestBlockStyles:
    """Test different block delimiting styles."""
    
    def test_mixed_indentation_and_end_keywords(self):
        """Test mixing indentation with END keywords."""
        code = """
for i = 1 to n
  if i % 2 == 0
    print(i)
  end if
end for
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
    
    def test_curly_braces_on_same_line(self):
        """Test opening brace on same line as statement."""
        code = """
for i = 1 to n {
  x = x + i
}
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
    
    def test_curly_braces_on_next_line(self):
        """Test opening brace on next line."""
        code = """
while x > 0
{
  x = x - 1
}
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success


class TestPreprocessing:
    """Test preprocessing and normalization."""
    
    def test_case_normalization(self):
        """Test that keywords are normalized to lowercase."""
        code = "FOR i = 1 TO n DO\n  PRINT(i)\nEND FOR"
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        assert "for" in result.normalized_code.lower()
    
    def test_operator_normalization(self):
        """Test that operators are normalized."""
        code = "x := 5\nif x ≠ 0 then\n  y = 1\nend if"
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        assert ":=" not in result.normalized_code
        assert "=" in result.normalized_code
    
    def test_typo_correction(self):
        """Test that common typos are fixed."""
        code = "whlie x > 0\n  x = x - 1\nend while"
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        assert len(result.warnings) > 0
        assert any("typo" in w.lower() for w in result.warnings)


class TestStructureDetection:
    """Test high-level structure detection."""
    
    def test_detect_simple_loop(self):
        """Test detecting a single loop."""
        code = "for i = 1 to n\n  print(i)\nend for"
        parser = PseudocodeParser()
        structure = parser.detect_structure(code)
        
        assert structure['has_loops']
        assert structure['loop_count'] == 1
        assert not structure['has_nested_loops']
        assert structure['max_nesting'] == 1
    
    def test_detect_nested_loops(self):
        """Test detecting nested loops."""
        code = """
for i = 1 to n
  for j = 1 to m
    print(i, j)
  end for
end for
"""
        parser = PseudocodeParser()
        structure = parser.detect_structure(code)
        
        assert structure['has_loops']
        assert structure['loop_count'] == 2
        assert structure['has_nested_loops']
        assert structure['max_nesting'] == 2
    
    def test_detect_recursion(self):
        """Test detecting recursive functions."""
        code = """
function factorial(n)
  if n == 0
    return 1
  else
    return n * factorial(n-1)
  end if
end function
"""
        parser = PseudocodeParser()
        structure = parser.detect_structure(code)
        
        assert structure['has_recursion']
    
    def test_detect_conditionals(self):
        """Test detecting conditionals."""
        code = "if x > 0\n  print(x)\nend if"
        parser = PseudocodeParser()
        structure = parser.detect_structure(code)
        
        assert structure['has_conditionals']


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_code(self):
        """Test parsing empty string."""
        parser = PseudocodeParser()
        result = parser.parse("")
        
        assert result.success
        assert result.ast is not None
    
    def test_only_comments(self):
        """Test code with only comments."""
        code = "// This is a comment\n# Another comment"
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
    
    def test_malformed_loop(self):
        """Test handling of malformed loop."""
        code = "for i =\n  print(i)"
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        # Should not crash, may or may not succeed
        assert result is not None
    
    def test_unmatched_braces(self):
        """Test handling of unmatched curly braces."""
        code = "for i = 1 to n {\n  print(i)"
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        # Should not crash
        assert result is not None


class TestIterationEstimation:
    """Test loop iteration estimation."""
    
    def test_estimate_n_iterations(self):
        """Test estimation for 1 to n loop."""
        code = "for i = 1 to n\n  print(i)\nend for"
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        loop = result.ast.global_statements.statements[0]
        assert loop.estimated_iterations == "n"
    
    def test_estimate_concrete_iterations(self):
        """Test estimation for concrete bounds."""
        code = "for i = 1 to 10\n  print(i)\nend for"
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        loop = result.ast.global_statements.statements[0]
        assert loop.estimated_iterations == "10"


class TestComplexPrograms:
    """Test parsing of complex, realistic programs."""
    
    def test_bubble_sort(self):
        """Test parsing bubble sort algorithm."""
        code = """
algorithm bubbleSort(arr, n)
  for i = 0 to n-1
    for j = 0 to n-i-1
      if arr[j] > arr[j+1]
        temp = arr[j]
        arr[j] = arr[j+1]
        arr[j+1] = temp
      end if
    end for
  end for
end algorithm
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        assert len(result.ast.functions) == 1
        
        structure = parser.detect_structure(code)
        assert structure['has_nested_loops']
        assert structure['has_conditionals']
    
    def test_binary_search(self):
        """Test parsing binary search with recursion."""
        code = """
function binarySearch(arr, target, low, high)
  if low > high then
    return -1
  end if
  
  mid = (low + high) / 2
  
  if arr[mid] == target then
    return mid
  elif arr[mid] > target then
    return binarySearch(arr, target, low, mid-1)
  else
    return binarySearch(arr, target, mid+1, high)
  end if
end function
"""
        parser = PseudocodeParser()
        result = parser.parse(code)
        
        assert result.success
        structure = parser.detect_structure(code)
        assert structure['has_recursion']
        assert structure['has_conditionals']


# Run tests with: pytest test_parser.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])