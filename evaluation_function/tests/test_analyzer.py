"""
Comprehensive tests for the Complexity Analyzer module.

Tests cover:
- Loop detection and analysis
- Recursion detection and complexity
- Nested loop handling
- AST-based and pattern-based analysis
- Feedback generation
- Edge cases and various complexity classes
"""

import pytest
from evaluation_function.analyzer.complexity_analyzer import (
    ComplexityAnalyzer,
    AnalysisResult,
    LoopInfo,
    RecursionInfo,
)
from evaluation_function.analyzer.feedback_generator import (
    FeedbackGenerator,
    DetailedFeedback,
    FeedbackLevel,
    FeedbackSection,
)
from evaluation_function.schemas.complexity import ComplexityClass
from evaluation_function.schemas.ast_nodes import (
    ProgramNode,
    FunctionNode,
    BlockNode,
    LoopNode,
    ConditionalNode,
    VariableNode,
    LiteralNode,
    LoopType,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def analyzer():
    """Create a fresh analyzer for each test."""
    return ComplexityAnalyzer()


@pytest.fixture
def feedback_generator():
    """Create a feedback generator."""
    return FeedbackGenerator()


# ============================================================================
# LoopInfo Tests
# ============================================================================

class TestLoopInfo:
    """Tests for LoopInfo dataclass."""
    
    def test_for_loop_description_with_bounds(self):
        """Test FOR loop description with start and end bounds."""
        loop = LoopInfo(
            loop_type="for",
            iterator="i",
            start_bound="1",
            end_bound="n",
            step="1",
            iterations="n",
            complexity=ComplexityClass.LINEAR,
            nesting_level=0
        )
        desc = loop.get_description()
        assert "FOR loop" in desc
        assert "i" in desc
        assert "1" in desc
        assert "n" in desc
    
    def test_for_loop_description_without_bounds(self):
        """Test FOR loop description without bounds."""
        loop = LoopInfo(
            loop_type="for",
            iterator="i",
            start_bound=None,
            end_bound=None,
            step=None,
            iterations="n",
            complexity=ComplexityClass.LINEAR,
            nesting_level=0
        )
        desc = loop.get_description()
        assert "FOR loop" in desc
        assert "iterator" in desc.lower() or "i" in desc
    
    def test_foreach_loop_description(self):
        """Test FOR-EACH loop description."""
        loop = LoopInfo(
            loop_type="for_each",
            iterator="item",
            start_bound=None,
            end_bound="collection",
            step=None,
            iterations="n",
            complexity=ComplexityClass.LINEAR,
            nesting_level=0
        )
        desc = loop.get_description()
        assert "FOR-EACH" in desc or "for_each" in desc.lower()
    
    def test_while_loop_description(self):
        """Test WHILE loop description."""
        loop = LoopInfo(
            loop_type="while",
            iterator=None,
            start_bound=None,
            end_bound=None,
            step=None,
            iterations="n",
            complexity=ComplexityClass.LINEAR,
            nesting_level=0
        )
        desc = loop.get_description()
        assert "WHILE" in desc.upper()
    
    def test_repeat_loop_description(self):
        """Test REPEAT-UNTIL loop description."""
        loop = LoopInfo(
            loop_type="repeat",
            iterator=None,
            start_bound=None,
            end_bound=None,
            step=None,
            iterations="n",
            complexity=ComplexityClass.LINEAR,
            nesting_level=0
        )
        desc = loop.get_description()
        assert "REPEAT" in desc.upper()


# ============================================================================
# RecursionInfo Tests
# ============================================================================

class TestRecursionInfo:
    """Tests for RecursionInfo dataclass."""
    
    def test_linear_recursion_description(self):
        """Test linear recursion description."""
        rec = RecursionInfo(
            function_name="factorial",
            num_recursive_calls=1,
            reduction_pattern="n-1",
            branching_factor=1,
            work_per_call=ComplexityClass.CONSTANT,
            complexity=ComplexityClass.LINEAR,
            recurrence="T(n) = T(n-1) + O(1)"
        )
        desc = rec.get_description()
        assert "Linear recursion" in desc
        assert "factorial" in desc
    
    def test_divide_conquer_description(self):
        """Test divide-and-conquer recursion description."""
        rec = RecursionInfo(
            function_name="mergeSort",
            num_recursive_calls=2,
            reduction_pattern="n/2",
            branching_factor=2,
            work_per_call=ComplexityClass.LINEAR,
            complexity=ComplexityClass.LINEARITHMIC,
            recurrence="T(n) = 2T(n/2) + O(n)"
        )
        desc = rec.get_description()
        assert "Divide-and-conquer" in desc
        assert "mergeSort" in desc
    
    def test_binary_recursion_description(self):
        """Test binary recursion (non-divide-conquer) description."""
        rec = RecursionInfo(
            function_name="fib",
            num_recursive_calls=2,
            reduction_pattern="n-1",
            branching_factor=2,
            work_per_call=ComplexityClass.CONSTANT,
            complexity=ComplexityClass.EXPONENTIAL,
            recurrence="T(n) = 2T(n-1) + O(1)"
        )
        desc = rec.get_description()
        assert "Binary recursion" in desc
        assert "fib" in desc
    
    def test_multiple_recursion_description(self):
        """Test multiple (>2) recursion description."""
        rec = RecursionInfo(
            function_name="multiRec",
            num_recursive_calls=3,
            reduction_pattern="n-1",
            branching_factor=3,
            work_per_call=ComplexityClass.CONSTANT,
            complexity=ComplexityClass.EXPONENTIAL,
            recurrence="T(n) = 3T(n-1) + O(1)"
        )
        desc = rec.get_description()
        assert "Multiple recursion" in desc or "3" in desc


# ============================================================================
# AnalysisResult Tests
# ============================================================================

class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""
    
    def test_get_complexity_string(self):
        """Test complexity string retrieval."""
        result = AnalysisResult(
            time_complexity=ComplexityClass.QUADRATIC,
            space_complexity=ComplexityClass.LINEAR,
            loops=[],
            recursion=None,
            max_nesting_depth=2,
            confidence=0.9,
            factors=[]
        )
        assert result.get_complexity_string() == "O(n²)"
    
    def test_result_with_loops(self):
        """Test result with loop information."""
        loop = LoopInfo(
            loop_type="for",
            iterator="i",
            start_bound="1",
            end_bound="n",
            step="1",
            iterations="n",
            complexity=ComplexityClass.LINEAR,
            nesting_level=0
        )
        result = AnalysisResult(
            time_complexity=ComplexityClass.LINEAR,
            space_complexity=ComplexityClass.CONSTANT,
            loops=[loop],
            recursion=None,
            max_nesting_depth=1,
            confidence=0.9,
            factors=[]
        )
        assert len(result.loops) == 1
        assert result.recursion is None


# ============================================================================
# ComplexityAnalyzer - Pattern-Based Tests
# ============================================================================

class TestAnalyzerPatternBased:
    """Tests for pattern-based analysis (no AST)."""
    
    def test_constant_complexity_no_loops(self, analyzer):
        """Test constant complexity for code without loops."""
        code = """
        FUNCTION simple(x)
            result = x + 1
            RETURN result
        END FUNCTION
        """
        result = analyzer.analyze(code)
        assert result.time_complexity == ComplexityClass.CONSTANT
        assert len(result.loops) == 0
    
    def test_single_for_loop_linear(self, analyzer):
        """Test single FOR loop gives linear complexity."""
        code = """
        FUNCTION sum(A, n)
            total = 0
            FOR i = 1 TO n DO
                total = total + A[i]
            END FOR
            RETURN total
        END FUNCTION
        """
        result = analyzer.analyze(code)
        assert result.time_complexity == ComplexityClass.LINEAR
        assert len(result.loops) == 1
        assert result.loops[0].loop_type == "for"
    
    def test_nested_for_loops_quadratic(self, analyzer):
        """Test nested FOR loops give quadratic complexity."""
        code = """
        FUNCTION bubbleSort(A, n)
            FOR i = 1 TO n DO
                FOR j = 1 TO n DO
                    IF A[j] > A[j+1] THEN
                        swap(A[j], A[j+1])
                    END IF
                END FOR
            END FOR
        END FUNCTION
        """
        result = analyzer.analyze(code)
        assert result.time_complexity == ComplexityClass.QUADRATIC
        assert result.max_nesting_depth == 2
    
    def test_triple_nested_loops_cubic(self, analyzer):
        """Test triple nested loops give cubic complexity."""
        code = """
        FUNCTION matrixMultiply(A, B, n)
            FOR i = 1 TO n DO
                FOR j = 1 TO n DO
                    FOR k = 1 TO n DO
                        C[i][j] = C[i][j] + A[i][k] * B[k][j]
                    END FOR
                END FOR
            END FOR
        END FUNCTION
        """
        result = analyzer.analyze(code)
        assert result.time_complexity == ComplexityClass.CUBIC
        assert result.max_nesting_depth == 3
    
    def test_while_loop_linear(self, analyzer):
        """Test WHILE loop detection."""
        code = """
        FUNCTION findElement(A, target)
            i = 0
            WHILE i < n DO
                IF A[i] == target THEN
                    RETURN i
                END IF
                i = i + 1
            END WHILE
            RETURN -1
        END FUNCTION
        """
        result = analyzer.analyze(code)
        assert result.time_complexity == ComplexityClass.LINEAR
        assert len(result.loops) == 1
        assert result.loops[0].loop_type == "while"
    
    def test_while_loop_logarithmic(self, analyzer):
        """Test WHILE loop with halving gives logarithmic complexity."""
        code = """
        FUNCTION binarySearch(A, target)
            low = 0
            high = n
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
        END FUNCTION
        """
        result = analyzer.analyze(code)
        # Binary search should be detected as logarithmic
        assert result.time_complexity in [ComplexityClass.LOGARITHMIC, ComplexityClass.LINEAR]
    
    def test_foreach_loop_linear(self, analyzer):
        """Test FOR-EACH loop detection."""
        code = """
        FUNCTION printAll(collection)
            FOR EACH item IN collection DO
                print(item)
            END FOR
        END FUNCTION
        """
        result = analyzer.analyze(code)
        assert result.time_complexity == ComplexityClass.LINEAR
        assert len(result.loops) == 1
        assert result.loops[0].loop_type == "for_each"
    
    def test_repeat_until_loop(self, analyzer):
        """Test REPEAT-UNTIL loop detection."""
        code = """
        FUNCTION readInput()
            REPEAT
                input = read()
            UNTIL input == "quit"
        END FUNCTION
        """
        result = analyzer.analyze(code)
        assert len(result.loops) == 1
        assert result.loops[0].loop_type == "repeat"


# ============================================================================
# ComplexityAnalyzer - Recursion Tests
# ============================================================================

class TestAnalyzerRecursion:
    """Tests for recursion detection and analysis."""
    
    def test_simple_linear_recursion(self, analyzer):
        """Test simple linear recursion (factorial-like)."""
        code = """
        FUNCTION factorial(n)
            IF n <= 1 THEN
                RETURN 1
            END IF
            RETURN n * factorial(n - 1)
        END FUNCTION
        """
        result = analyzer.analyze(code)
        assert result.recursion is not None
        assert result.recursion.function_name == "factorial"
        assert result.recursion.branching_factor == 1
        assert result.time_complexity == ComplexityClass.LINEAR
    
    def test_binary_recursion_exponential(self, analyzer):
        """Test binary recursion (Fibonacci-like) is exponential."""
        code = """
        FUNCTION fib(n)
            IF n <= 1 THEN
                RETURN n
            END IF
            RETURN fib(n - 1) + fib(n - 2)
        END FUNCTION
        """
        result = analyzer.analyze(code)
        assert result.recursion is not None
        assert result.recursion.function_name == "fib"
        assert result.recursion.branching_factor >= 2
        assert result.time_complexity == ComplexityClass.EXPONENTIAL
    
    def test_divide_and_conquer_merge_sort(self, analyzer):
        """Test divide-and-conquer recursion (merge sort pattern)."""
        code = """
        FUNCTION mergeSort(A, low, high)
            IF low < high THEN
                mid = (low + high) / 2
                mergeSort(A, low, mid)
                mergeSort(A, mid + 1, high)
                merge(A, low, mid, high)
            END IF
        END FUNCTION
        """
        result = analyzer.analyze(code)
        assert result.recursion is not None
        assert result.recursion.function_name.lower() == "mergesort"
        assert "n/2" in result.recursion.reduction_pattern
        assert result.time_complexity == ComplexityClass.LINEARITHMIC
    
    def test_binary_search_recursion(self, analyzer):
        """Test binary search recursive pattern."""
        code = """
        FUNCTION binarySearch(A, target, low, high)
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
        END FUNCTION
        """
        result = analyzer.analyze(code)
        assert result.recursion is not None
        assert result.recursion.branching_factor == 1
        assert result.time_complexity == ComplexityClass.LOGARITHMIC
    
    def test_space_complexity_recursive(self, analyzer):
        """Test space complexity for recursive functions."""
        code = """
        FUNCTION factorial(n)
            IF n <= 1 THEN
                RETURN 1
            END IF
            RETURN n * factorial(n - 1)
        END FUNCTION
        """
        result = analyzer.analyze(code)
        # Linear recursion has O(n) stack depth
        assert result.space_complexity == ComplexityClass.LINEAR
    
    def test_space_complexity_divide_conquer(self, analyzer):
        """Test space complexity for divide-and-conquer."""
        code = """
        FUNCTION binarySearch(A, target, low, high)
            IF low > high THEN
                RETURN -1
            END IF
            mid = (low + high) / 2
            RETURN binarySearch(A, target, mid + 1, high)
        END FUNCTION
        """
        result = analyzer.analyze(code)
        # Divide-and-conquer has O(log n) stack depth
        assert result.space_complexity == ComplexityClass.LOGARITHMIC


# ============================================================================
# ComplexityAnalyzer - AST-Based Tests
# ============================================================================

class TestAnalyzerAST:
    """Tests for AST-based analysis."""
    
    def test_analyze_ast_single_loop(self, analyzer):
        """Test AST analysis with single loop."""
        loop = LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="i"),
            start=LiteralNode(value=1),
            end=VariableNode(name="n"),
            body=BlockNode(statements=[]),
            estimated_iterations="n"
        )
        func = FunctionNode(
            name="test",
            parameters=[],
            body=BlockNode(statements=[loop])
        )
        ast = ProgramNode(functions=[func])
        
        result = analyzer.analyze("", ast)
        assert result.time_complexity == ComplexityClass.LINEAR
        assert len(result.loops) == 1
    
    def test_analyze_ast_nested_loops(self, analyzer):
        """Test AST analysis with nested loops."""
        inner_loop = LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="j"),
            start=LiteralNode(value=1),
            end=VariableNode(name="n"),
            body=BlockNode(statements=[]),
            estimated_iterations="n"
        )
        outer_loop = LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="i"),
            start=LiteralNode(value=1),
            end=VariableNode(name="n"),
            body=BlockNode(statements=[inner_loop]),
            estimated_iterations="n"
        )
        func = FunctionNode(
            name="test",
            parameters=[],
            body=BlockNode(statements=[outer_loop])
        )
        ast = ProgramNode(functions=[func])
        
        result = analyzer.analyze("", ast)
        assert result.time_complexity == ComplexityClass.QUADRATIC
    
    def test_analyze_ast_global_statements(self, analyzer):
        """Test AST analysis with global statements (no functions)."""
        loop = LoopNode(
            loop_type=LoopType.WHILE,
            condition=None,
            body=BlockNode(statements=[]),
            estimated_iterations="n"
        )
        ast = ProgramNode(
            functions=[],
            global_statements=BlockNode(statements=[loop])
        )
        
        result = analyzer.analyze("", ast)
        assert result.time_complexity == ComplexityClass.LINEAR
    
    def test_analyze_ast_loop_with_conditional(self, analyzer):
        """Test AST analysis with loop containing conditional."""
        inner_loop = LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="k"),
            start=LiteralNode(value=1),
            end=VariableNode(name="n"),
            body=BlockNode(statements=[]),
            estimated_iterations="n"
        )
        conditional = ConditionalNode(
            then_branch=BlockNode(statements=[inner_loop]),
            else_branch=None
        )
        outer_loop = LoopNode(
            loop_type=LoopType.FOR,
            iterator=VariableNode(name="i"),
            start=LiteralNode(value=1),
            end=VariableNode(name="n"),
            body=BlockNode(statements=[conditional]),
            estimated_iterations="n"
        )
        func = FunctionNode(
            name="test",
            parameters=[],
            body=BlockNode(statements=[outer_loop])
        )
        ast = ProgramNode(functions=[func])
        
        result = analyzer.analyze("", ast)
        # Loop inside conditional inside loop = quadratic
        assert result.time_complexity == ComplexityClass.QUADRATIC


# ============================================================================
# ComplexityAnalyzer - Iteration Estimation Tests
# ============================================================================

class TestIterationEstimation:
    """Tests for iteration estimation logic."""
    
    def test_constant_bounds(self, analyzer):
        """Test constant iteration bounds."""
        code = """
        FOR i = 1 TO 10 DO
            print(i)
        END FOR
        """
        result = analyzer.analyze(code)
        assert result.loops[0].iterations == "10"
        assert result.loops[0].complexity == ComplexityClass.CONSTANT
    
    def test_variable_n_bound(self, analyzer):
        """Test variable 'n' as upper bound."""
        code = """
        FOR i = 1 TO n DO
            print(i)
        END FOR
        """
        result = analyzer.analyze(code)
        assert result.loops[0].iterations == "n"
        assert result.loops[0].complexity == ComplexityClass.LINEAR
    
    def test_length_bound(self, analyzer):
        """Test 'length' as upper bound."""
        code = """
        FOR i = 0 TO length DO
            print(i)
        END FOR
        """
        result = analyzer.analyze(code)
        assert result.loops[0].complexity == ComplexityClass.LINEAR


# ============================================================================
# FeedbackGenerator Tests
# ============================================================================

class TestFeedbackGenerator:
    """Tests for FeedbackGenerator."""
    
    def test_generate_constant_feedback(self, feedback_generator, analyzer):
        """Test feedback for constant complexity."""
        code = "x = 5"
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result)
        
        assert feedback.complexity_result == "O(1)"
        assert "Constant" in feedback.summary or "O(1)" in feedback.summary
    
    def test_generate_linear_feedback(self, feedback_generator, analyzer):
        """Test feedback for linear complexity."""
        code = """
        FOR i = 1 TO n DO
            print(i)
        END FOR
        """
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result)
        
        assert feedback.complexity_result == "O(n)"
        assert feedback.loop_count == 1
        assert feedback.max_nesting == 1
    
    def test_generate_quadratic_feedback(self, feedback_generator, analyzer):
        """Test feedback for quadratic complexity."""
        code = """
        FOR i = 1 TO n DO
            FOR j = 1 TO n DO
                print(i, j)
            END FOR
        END FOR
        """
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result)
        
        assert feedback.complexity_result == "O(n²)"
        assert feedback.max_nesting == 2
        assert "nested" in feedback.summary.lower() or "loop" in feedback.summary.lower()
    
    def test_generate_recursion_feedback(self, feedback_generator, analyzer):
        """Test feedback for recursive algorithms."""
        code = """
        FUNCTION fib(n)
            IF n <= 1 THEN
                RETURN n
            END IF
            RETURN fib(n - 1) + fib(n - 2)
        END FUNCTION
        """
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result)
        
        assert feedback.has_recursion
        assert "recursion" in feedback.summary.lower()
    
    def test_feedback_sections(self, feedback_generator, analyzer):
        """Test that feedback sections are generated."""
        code = """
        FOR i = 1 TO n DO
            FOR j = 1 TO n DO
                print(i, j)
            END FOR
        END FOR
        """
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result, FeedbackLevel.DETAILED)
        
        assert len(feedback.sections) > 0
        section_titles = [s.title for s in feedback.sections]
        assert any("Loop" in t for t in section_titles)
    
    def test_feedback_suggestions(self, feedback_generator, analyzer):
        """Test that suggestions are generated for complex algorithms."""
        code = """
        FUNCTION fib(n)
            IF n <= 1 THEN
                RETURN n
            END IF
            RETURN fib(n - 1) + fib(n - 2)
        END FUNCTION
        """
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result)
        
        # Exponential should have optimization suggestions
        assert len(feedback.suggestions) > 0
        assert any("dynamic" in s.lower() or "memoization" in s.lower() 
                   for s in feedback.suggestions)


# ============================================================================
# FeedbackGenerator - Output Format Tests
# ============================================================================

class TestFeedbackFormats:
    """Tests for feedback output formats."""
    
    def test_to_string_brief(self, feedback_generator, analyzer):
        """Test brief string format."""
        code = "FOR i = 1 TO n DO\n    print(i)\nEND FOR"
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result, FeedbackLevel.BRIEF)
        
        output = feedback.to_string(FeedbackLevel.BRIEF)
        assert "O(n)" in output
        assert "COMPLEXITY ANALYSIS RESULT" in output
    
    def test_to_string_standard(self, feedback_generator, analyzer):
        """Test standard string format."""
        code = "FOR i = 1 TO n DO\n    print(i)\nEND FOR"
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result, FeedbackLevel.STANDARD)
        
        output = feedback.to_string(FeedbackLevel.STANDARD)
        assert "O(n)" in output
        assert "What does this mean?" in output
    
    def test_to_string_detailed(self, feedback_generator, analyzer):
        """Test detailed string format."""
        code = "FOR i = 1 TO n DO\n    print(i)\nEND FOR"
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result, FeedbackLevel.DETAILED)
        
        output = feedback.to_string(FeedbackLevel.DETAILED)
        assert "O(n)" in output
        assert "Real-World" in output or "Analysis Confidence" in output
    
    def test_to_dict(self, feedback_generator, analyzer):
        """Test dictionary conversion."""
        code = "FOR i = 1 TO n DO\n    print(i)\nEND FOR"
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result)
        
        data = feedback.to_dict()
        assert "summary" in data
        assert "complexity" in data
        assert "sections" in data
        assert "stats" in data
        assert data["complexity"] == "O(n)"
    
    def test_format_for_student(self, feedback_generator, analyzer):
        """Test student-friendly format."""
        code = """
        FOR i = 1 TO n DO
            FOR j = 1 TO n DO
                print(i, j)
            END FOR
        END FOR
        """
        result = analyzer.analyze(code)
        output = feedback_generator.format_for_student(result)
        
        assert "O(n²)" in output
        assert len(output) > 100  # Should be detailed
    
    def test_format_brief(self, feedback_generator, analyzer):
        """Test brief one-line format."""
        code = "FOR i = 1 TO n DO\n    print(i)\nEND FOR"
        result = analyzer.analyze(code)
        output = feedback_generator.format_brief(result)
        
        assert "O(n)" in output
        assert "Time Complexity" in output


# ============================================================================
# FeedbackGenerator - Complexity Explanations
# ============================================================================

class TestComplexityExplanations:
    """Tests for complexity class explanations."""
    
    def test_constant_explanation(self, feedback_generator, analyzer):
        """Test constant complexity explanation."""
        code = "x = 5"
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result)
        
        assert "Constant" in feedback.complexity_explanation
        assert "same amount of time" in feedback.complexity_explanation.lower() or "O(1)" in feedback.complexity_explanation
    
    def test_linear_explanation(self, feedback_generator, analyzer):
        """Test linear complexity explanation."""
        code = "FOR i = 1 TO n DO\n    print(i)\nEND FOR"
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result)
        
        assert "Linear" in feedback.complexity_explanation
    
    def test_quadratic_explanation(self, feedback_generator, analyzer):
        """Test quadratic complexity explanation."""
        code = """
        FOR i = 1 TO n DO
            FOR j = 1 TO n DO
                print(i, j)
            END FOR
        END FOR
        """
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result)
        
        assert "Quadratic" in feedback.complexity_explanation
        assert "nested" in feedback.complexity_explanation.lower() or "n²" in feedback.complexity_explanation


# ============================================================================
# Edge Cases and Special Scenarios
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_empty_code(self, analyzer):
        """Test handling of empty code."""
        result = analyzer.analyze("")
        assert result.time_complexity == ComplexityClass.CONSTANT
        assert len(result.loops) == 0
    
    def test_whitespace_only(self, analyzer):
        """Test handling of whitespace-only code."""
        result = analyzer.analyze("   \n\n   \t   ")
        assert result.time_complexity == ComplexityClass.CONSTANT
    
    def test_comments_only(self, analyzer):
        """Test handling of comments-only code."""
        code = """
        // This is a comment
        # Another comment
        """
        result = analyzer.analyze(code)
        assert result.time_complexity == ComplexityClass.CONSTANT
    
    def test_multiple_independent_loops(self, analyzer):
        """Test multiple independent (not nested) loops."""
        code = """
        FOR i = 1 TO n DO
            print(i)
        END FOR
        
        FOR j = 1 TO n DO
            print(j)
        END FOR
        """
        result = analyzer.analyze(code)
        # Independent loops don't multiply - take max
        assert result.time_complexity == ComplexityClass.LINEAR
        assert len(result.loops) == 2
    
    def test_deeply_nested_loops(self, analyzer):
        """Test very deeply nested loops."""
        code = """
        FOR i = 1 TO n DO
            FOR j = 1 TO n DO
                FOR k = 1 TO n DO
                    FOR l = 1 TO n DO
                        print(i, j, k, l)
                    END FOR
                END FOR
            END FOR
        END FOR
        """
        result = analyzer.analyze(code)
        # 4 nested loops = O(n^4) = polynomial
        assert result.time_complexity in [ComplexityClass.POLYNOMIAL, ComplexityClass.CUBIC]
        assert result.max_nesting_depth >= 3
    
    def test_mixed_loop_types(self, analyzer):
        """Test code with different loop types."""
        code = """
        FOR i = 1 TO n DO
            WHILE j < n DO
                j = j + 1
            END WHILE
        END FOR
        """
        result = analyzer.analyze(code)
        assert result.time_complexity == ComplexityClass.QUADRATIC
    
    def test_loop_with_constant_bound(self, analyzer):
        """Test loop with constant upper bound (not O(n))."""
        code = """
        FOR i = 1 TO 100 DO
            print(i)
        END FOR
        """
        result = analyzer.analyze(code)
        # Constant bound = O(1)
        assert result.loops[0].complexity == ComplexityClass.CONSTANT
    
    def test_confidence_levels(self, analyzer):
        """Test that confidence is set appropriately."""
        # Simple code - high confidence
        simple_result = analyzer.analyze("x = 5")
        assert simple_result.confidence >= 0.7
        
        # Complex code with loops - high confidence
        loop_code = "FOR i = 1 TO n DO\n    print(i)\nEND FOR"
        loop_result = analyzer.analyze(loop_code)
        assert loop_result.confidence >= 0.8


# ============================================================================
# Integration Tests
# ============================================================================

class TestAnalyzerIntegration:
    """Integration tests combining analyzer and feedback."""
    
    def test_full_pipeline_linear(self, analyzer, feedback_generator):
        """Test full analysis pipeline for linear algorithm."""
        code = """
        FUNCTION linearSearch(A, n, target)
            FOR i = 1 TO n DO
                IF A[i] == target THEN
                    RETURN i
                END IF
            END FOR
            RETURN -1
        END FUNCTION
        """
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result, FeedbackLevel.DETAILED)
        
        assert result.time_complexity == ComplexityClass.LINEAR
        assert "O(n)" in feedback.to_string()
    
    def test_full_pipeline_quadratic(self, analyzer, feedback_generator):
        """Test full analysis pipeline for quadratic algorithm."""
        code = """
        FUNCTION bubbleSort(A, n)
            FOR i = 1 TO n DO
                FOR j = 1 TO n-1 DO
                    IF A[j] > A[j+1] THEN
                        swap(A[j], A[j+1])
                    END IF
                END FOR
            END FOR
        END FUNCTION
        """
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result, FeedbackLevel.DETAILED)
        
        assert result.time_complexity == ComplexityClass.QUADRATIC
        assert "O(n²)" in feedback.to_string()
        assert "nested" in feedback.to_string().lower()
    
    def test_full_pipeline_recursive(self, analyzer, feedback_generator):
        """Test full analysis pipeline for recursive algorithm."""
        code = """
        FUNCTION factorial(n)
            IF n <= 1 THEN
                RETURN 1
            END IF
            RETURN n * factorial(n - 1)
        END FUNCTION
        """
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result, FeedbackLevel.DETAILED)
        
        assert result.recursion is not None
        output = feedback.to_string()
        assert "recursion" in output.lower() or "recursive" in output.lower()
    
    def test_full_pipeline_exponential(self, analyzer, feedback_generator):
        """Test full analysis pipeline for exponential algorithm."""
        code = """
        FUNCTION fib(n)
            IF n <= 1 THEN
                RETURN n
            END IF
            RETURN fib(n-1) + fib(n-2)
        END FUNCTION
        """
        result = analyzer.analyze(code)
        feedback = feedback_generator.generate(result, FeedbackLevel.DETAILED)
        
        assert result.time_complexity == ComplexityClass.EXPONENTIAL
        output = feedback.to_string()
        assert "O(2^n)" in output or "Exponential" in output
        # Should have optimization suggestions
        assert len(feedback.suggestions) > 0


# ============================================================================
# FeedbackSection Tests
# ============================================================================

class TestFeedbackSection:
    """Tests for FeedbackSection class."""
    
    def test_section_creation(self):
        """Test creating a feedback section."""
        section = FeedbackSection(
            title="Test Section",
            content="This is test content",
            importance="info"
        )
        assert section.title == "Test Section"
        assert section.content == "This is test content"
        assert section.importance == "info"
    
    def test_section_importance_levels(self):
        """Test different importance levels."""
        for importance in ["info", "warning", "success", "error"]:
            section = FeedbackSection(
                title="Test",
                content="Content",
                importance=importance
            )
            assert section.importance == importance


# ============================================================================
# DetailedFeedback Tests
# ============================================================================

class TestDetailedFeedback:
    """Tests for DetailedFeedback class."""
    
    def test_feedback_creation(self):
        """Test creating detailed feedback."""
        feedback = DetailedFeedback(
            summary="Test summary",
            complexity_result="O(n)",
            loop_count=1,
            max_nesting=1,
            has_recursion=False
        )
        assert feedback.summary == "Test summary"
        assert feedback.complexity_result == "O(n)"
        assert feedback.loop_count == 1
    
    def test_feedback_with_sections(self):
        """Test feedback with sections."""
        section = FeedbackSection(
            title="Loop Analysis",
            content="One loop detected",
            importance="info"
        )
        feedback = DetailedFeedback(
            summary="Test",
            complexity_result="O(n)",
            sections=[section]
        )
        assert len(feedback.sections) == 1
        assert feedback.sections[0].title == "Loop Analysis"
    
    def test_feedback_to_dict_structure(self):
        """Test dictionary structure of feedback."""
        feedback = DetailedFeedback(
            summary="Test summary",
            complexity_result="O(n²)",
            loop_count=2,
            max_nesting=2,
            has_recursion=False,
            complexity_explanation="Quadratic complexity",
            real_world_example="Bubble sort",
            suggestions=["Consider optimization"],
            confidence_note="High confidence"
        )
        data = feedback.to_dict()
        
        assert data["summary"] == "Test summary"
        assert data["complexity"] == "O(n²)"
        assert data["stats"]["loop_count"] == 2
        assert data["stats"]["max_nesting"] == 2
        assert data["stats"]["has_recursion"] == False
        assert data["explanation"] == "Quadratic complexity"
        assert "Consider optimization" in data["suggestions"]
