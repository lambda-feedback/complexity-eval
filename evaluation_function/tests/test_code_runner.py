"""
Comprehensive pytest test suite for CodeRunner.

All tests use pytest classes and functions with proper fixtures.
Tests integration of parser and interpreter.
"""

import pytest

from ..schemas.input_schema import ExecutionTestCase
from ..schemas.output_schema import CodeCorrectnessResult
from ..analyzer.code_runner import CodeRunner
from ..analyzer.interpreter import Interpreter
from ..parser.parser import PseudocodeParser


@pytest.fixture
def parser():
    """Create parser instance."""
    return PseudocodeParser()


@pytest.fixture
def interpreter():
    """Create interpreter instance."""
    return Interpreter()


@pytest.fixture
def runner(parser, interpreter):
    """Create CodeRunner with parser and interpreter."""
    return CodeRunner(parser, interpreter)


class TestParsingAndExecution:
    """Test basic parsing and execution."""
    
    def test_parsing_success(self, runner):
        """Test successful parsing."""
        code = "x = 42"
        result = runner.run(code, test_cases=[])
        
        assert result.parse_success
        assert result.is_correct  # No tests means success
        assert result.parse_errors == []
    
    def test_parsing_failure(self, runner):
        """Test parsing failure - FIX #7."""
        code = "this is not valid pseudocode !@#$%"
        result = runner.run(code, test_cases=[])
        
        assert not result.parse_success
        assert not result.is_correct
        # FIX #7: Just check that errors exist
        assert (len(result.parse_errors) > 0 or 
                "Parsing failed" in result.feedback)
    
    def test_simple_assignment(self, runner):
        """Test simple variable assignment."""
        code = "x = 42"
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"x": 42},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.parse_success
        assert result.is_correct
        assert len(result.execution_results) == 1
        assert result.execution_results[0].passed
        assert result.execution_results[0].actual_output["variables"]["x"] == 42
    
    def test_arithmetic_expression(self, runner):
        """Test arithmetic expression."""
        code = "result = 5 + 3 * 2"
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"result": 11},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct


class TestMultipleTestCases:
    """Test execution with multiple test cases."""
    
    def test_multiple_test_cases_all_pass(self, runner):
        """Test multiple test cases all passing."""
        code = """
for i = 1 to 5
  sum = sum + i
end for
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={"sum": 0},
                expected_variables={"sum": 15, "i": 5},
                expected_output=[]
            ),
            ExecutionTestCase(
                initial_variables={"sum": 10},
                expected_variables={"sum": 25, "i": 5},
                expected_output=[]
            ),
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.parse_success
        assert result.is_correct
        assert all(tc.passed for tc in result.execution_results)
    
    def test_multiple_test_cases_some_fail(self, runner):
        """Test multiple test cases with some failures."""
        code = "x = 10"
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"x": 10},
                expected_output=[]
            ),
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"x": 20},  # Wrong expectation
                expected_output=[]
            ),
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.parse_success
        assert not result.is_correct
        assert result.execution_results[0].passed
        assert not result.execution_results[1].passed
    
    def test_no_test_cases(self, runner):
        """Test with no test cases."""
        code = "x = 5"
        result = runner.run(code, test_cases=[])
        
        assert result.parse_success
        assert result.is_correct
        assert len(result.execution_results) == 0


class TestLoopExecution:
    """Test different loop types."""
    
    def test_for_loop(self, runner):
        """Test FOR loop execution."""
        code = """
sum = 0
for i = 1 to 4
  sum = sum + i
end for
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"sum": 10, "i": 4},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct
    
    def test_while_loop(self, runner):
        """Test WHILE loop execution."""
        code = """
x = 1
while x < 10
  x = x * 2
end while
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"x": 16},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct
    
    def test_for_each_loop(self, runner):
        """Test FOR EACH loop execution."""
        code = """
sum = 0
for each num in numbers
  sum = sum + num
end for
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={"numbers": [1, 2, 3, 4, 5]},
                expected_variables={
                    "numbers": [1, 2, 3, 4, 5], 
                    "sum": 15, 
                    "num": 5
                },
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct
    
    def test_repeat_until_loop(self, runner):
        """Test REPEAT UNTIL loop execution."""
        code = """
x = 1
repeat
  x = x * 2
until x >= 10
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"x": 16},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct
    
    def test_nested_loops(self, runner):
        """Test nested loops."""
        code = """
sum = 0
for i = 1 to 3
  for j = 1 to 2
    sum = sum + i * j
  end for
end for
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"sum": 18, "i": 3, "j": 2},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct


class TestConditionals:
    """Test conditional execution."""
    
    def test_if_then_else(self, runner):
        """Test IF-ELSE execution."""
        code = """
if x > 0
  result = 1
else
  result = -1
end if
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={"x": 5},
                expected_variables={"x": 5, "result": 1},
                expected_output=[]
            ),
            ExecutionTestCase(
                initial_variables={"x": -5},
                expected_variables={"x": -5, "result": -1},
                expected_output=[]
            ),
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct
    
    def test_if_without_else(self, runner):
        """Test IF without ELSE."""
        code = """
if x > 0
  positive = 1
end if
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={"x": 5},
                expected_variables={"x": 5, "positive": 1},
                expected_output=[]
            ),
            ExecutionTestCase(
                initial_variables={"x": -5},
                expected_variables={"x": -5},
                expected_output=[]
            ),
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct


class TestFunctions:
    """Test function definition and calls."""
    
    def test_function_with_return(self, runner):
        """Test function with return value."""
        code = """
function add(a, b)
  return a + b
end function

result = add(3, 4)
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"result": 7},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct
    
    def test_recursive_function(self, runner):
        """Test recursive function."""
        code = """
function factorial(n)
  if n <= 1
    return 1
  else
    return n * factorial(n - 1)
  end if
end function

result = factorial(5)
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"result": 120},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct
    
    def test_function_without_return(self, runner):
        """Test function without explicit return - modifies global variable."""
        code = """
function increment()
  x = x + 1
end function

x = 5
result = increment()
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"x": 6, "result": None},  # x is modified globally!
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct


class TestPrintOutput:
    """Test output capture."""
    
    def test_print_output(self, runner):
        """Test print function output."""
        code = """
print("Hello")
print("World")
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={},
                expected_output=["Hello", "World"]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct
    
    def test_print_with_variables(self, runner):
        """Test print with variable values."""
        code = """
x = 10
y = 20
print(x, y)
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"x": 10, "y": 20},
                expected_output=["10 20"]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct
    
    def test_print_in_loop(self, runner):
        """Test print inside loop."""
        code = """
for i = 1 to 3
  print(i)
end for
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"i": 3},
                expected_output=["1", "2", "3"]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct


class TestErrorHandling:
    """Test error handling in execution."""
    
    def test_undefined_variable_error(self, runner):
        """Test error when using undefined variable."""
        code = "y = x + 1"  # x is undefined
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"y": 1},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.parse_success
        assert not result.is_correct
        assert not result.execution_results[0].passed
        assert "not defined" in result.execution_results[0].error_message.lower()
    
    def test_division_by_zero_error(self, runner):
        """Test division by zero error."""
        code = "result = 10 / 0"
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"result": 0},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.parse_success
        assert not result.is_correct
        assert "division by zero" in result.execution_results[0].error_message.lower()
    
    def test_undefined_function_error(self, runner):
        """Test error when calling undefined function."""
        code = "result = missing_func()"
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.parse_success
        assert not result.is_correct
        assert "not defined" in result.execution_results[0].error_message.lower()
    
    def test_runtime_error_in_loop(self, runner):
        """Test runtime error inside loop."""
        code = """
for i = 0 to 3
  result = 10 / i
end for
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert not result.is_correct
        assert "division by zero" in result.execution_results[0].error_message.lower()


class TestVariableScoping:
    """Test variable scoping."""
    
    def test_function_parameters_isolated(self, runner):
        """Test that function parameters don't affect outer scope."""
        code = """
function set_local(x)
  x = 99
  return x
end function

x = 5
y = set_local(x)
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"x": 5, "y": 99},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct
    
    def test_function_can_access_globals(self, runner):
        """Test that functions can access and modify global variables."""
        code = """
function read_and_modify_global()
  y = x + 10
  return y
end function

x = 42
result = read_and_modify_global()
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"x": 42, "y": 52, "result": 52},  # y is now global!
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct


class TestInitialVariables:
    """Test initial variables handling."""
    
    def test_initial_variables_provided(self, runner):
        """Test execution with initial variables."""
        code = "result = x + y"
        test_cases = [
            ExecutionTestCase(
                initial_variables={"x": 10, "y": 20},
                expected_variables={"x": 10, "y": 20, "result": 30},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct
    
    def test_initial_variables_not_modified(self, runner):
        """Test that initial variables from one test don't affect another."""
        code = "x = x + 1"
        test_cases = [
            ExecutionTestCase(
                initial_variables={"x": 5},
                expected_variables={"x": 6},
                expected_output=[]
            ),
            ExecutionTestCase(
                initial_variables={"x": 10},
                expected_variables={"x": 11},
                expected_output=[]
            ),
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct


class TestComplexPrograms:
    """Test complex programs."""
    
    def test_fibonacci(self, runner):
        """Test Fibonacci function."""
        code = """
function fib(n)
  if n <= 1
    return n
  else
    return fib(n - 1) + fib(n - 2)
  end if
end function

result = fib(6)
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"result": 8},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct
    
    def test_sum_array(self, runner):
        """Test summing an array."""
        code = """
sum = 0
for each num in arr
  sum = sum + num
end for
"""
        test_cases = [
            ExecutionTestCase(
                initial_variables={"arr": [5, 10, 15, 20]},
                expected_variables={"arr": [5, 10, 15, 20], "sum": 50, "num": 20},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert result.is_correct


class TestFeedbackMessages:
    """Test feedback message generation."""
    
    def test_all_pass_feedback(self, runner):
        """Test feedback when all tests pass."""
        code = "x = 1"
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"x": 1},
                expected_output=[]
            )
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert "passed" in result.feedback.lower() or "✅" in result.feedback
    
    def test_some_fail_feedback(self, runner):
        """Test feedback when some tests fail."""
        code = "x = 1"
        test_cases = [
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"x": 1},
                expected_output=[]
            ),
            ExecutionTestCase(
                initial_variables={},
                expected_variables={"x": 2},
                expected_output=[]
            ),
        ]
        
        result = runner.run(code, test_cases=test_cases)
        
        assert "failed" in result.feedback.lower()
    
    def test_no_tests_feedback(self, runner):
        """Test feedback when no tests provided."""
        code = "x = 1"
        result = runner.run(code, test_cases=[])
        
        assert "no test" in result.feedback.lower() or result.is_correct


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])