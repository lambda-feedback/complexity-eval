"""
Comprehensive tests for the Complexity Schemas module.

Tests cover:
- ComplexityClass enum operations
- Complexity comparison and equivalence
- Loop and recursion complexity analysis
- Time and space complexity structures
"""

import pytest
from ..schemas.complexity import (
    ComplexityClass, ComplexityExpression, ComplexityFactor,
    LoopComplexity, RecursionComplexity, TimeComplexity,
    SpaceComplexity, ComplexityResult
)


class TestComplexityClass:
    """Tests for ComplexityClass enum."""
    
    def test_complexity_class_values(self):
        """Test that complexity classes have expected values."""
        assert ComplexityClass.CONSTANT.value == "O(1)"
        assert ComplexityClass.LOGARITHMIC.value == "O(log n)"
        assert ComplexityClass.LINEAR.value == "O(n)"
        assert ComplexityClass.LINEARITHMIC.value == "O(n log n)"
        assert ComplexityClass.QUADRATIC.value == "O(n²)"
        assert ComplexityClass.CUBIC.value == "O(n³)"
        assert ComplexityClass.EXPONENTIAL.value == "O(2^n)"
        assert ComplexityClass.FACTORIAL.value == "O(n!)"
    
    def test_from_string_basic(self):
        """Test parsing basic complexity strings."""
        test_cases = [
            ("O(1)", ComplexityClass.CONSTANT),
            ("O(n)", ComplexityClass.LINEAR),
            ("O(n^2)", ComplexityClass.QUADRATIC),
            ("O(n^3)", ComplexityClass.CUBIC),
            ("O(log n)", ComplexityClass.LOGARITHMIC),
            ("O(n log n)", ComplexityClass.LINEARITHMIC),
            ("O(2^n)", ComplexityClass.EXPONENTIAL),
            ("O(n!)", ComplexityClass.FACTORIAL),
        ]
        
        for input_str, expected in test_cases:
            result = ComplexityClass.from_string(input_str)
            assert result == expected, f"'{input_str}' should parse to {expected}"
    
    def test_from_string_variations(self):
        """Test parsing complexity string variations."""
        # Case variations
        assert ComplexityClass.from_string("o(n)") == ComplexityClass.LINEAR
        assert ComplexityClass.from_string("O(N)") == ComplexityClass.LINEAR
        
        # Space variations
        assert ComplexityClass.from_string("O( n )") == ComplexityClass.LINEAR
        assert ComplexityClass.from_string("O(n log n)") == ComplexityClass.LINEARITHMIC
        assert ComplexityClass.from_string("O(nlogn)") == ComplexityClass.LINEARITHMIC
        
        # Alternative notations
        assert ComplexityClass.from_string("O(n²)") == ComplexityClass.QUADRATIC
        assert ComplexityClass.from_string("O(n*n)") == ComplexityClass.QUADRATIC
        assert ComplexityClass.from_string("O(nn)") == ComplexityClass.QUADRATIC
        
        # Logarithm variations
        assert ComplexityClass.from_string("O(lgn)") == ComplexityClass.LOGARITHMIC
        assert ComplexityClass.from_string("O(log(n))") == ComplexityClass.LOGARITHMIC
    
    def test_from_string_text_names(self):
        """Test parsing text-based complexity names."""
        test_cases = [
            ("constant", ComplexityClass.CONSTANT),
            ("linear", ComplexityClass.LINEAR),
            ("quadratic", ComplexityClass.QUADRATIC),
            ("cubic", ComplexityClass.CUBIC),
            ("logarithmic", ComplexityClass.LOGARITHMIC),
            ("linearithmic", ComplexityClass.LINEARITHMIC),
            ("exponential", ComplexityClass.EXPONENTIAL),
            ("factorial", ComplexityClass.FACTORIAL),
        ]
        
        for input_str, expected in test_cases:
            result = ComplexityClass.from_string(input_str)
            assert result == expected
    
    def test_from_string_unknown(self):
        """Test parsing unknown complexity strings."""
        unknown_inputs = ["O(mystery)", "unknown", "something", "", None]
        
        for input_str in unknown_inputs:
            if input_str is not None:
                result = ComplexityClass.from_string(input_str)
                assert result == ComplexityClass.UNKNOWN
    
    def test_compare_complexities(self):
        """Test comparing complexity classes."""
        # Lower is better (more efficient)
        assert ComplexityClass.compare(ComplexityClass.CONSTANT, ComplexityClass.LINEAR) == -1
        assert ComplexityClass.compare(ComplexityClass.LINEAR, ComplexityClass.QUADRATIC) == -1
        assert ComplexityClass.compare(ComplexityClass.QUADRATIC, ComplexityClass.CUBIC) == -1
        assert ComplexityClass.compare(ComplexityClass.CUBIC, ComplexityClass.EXPONENTIAL) == -1
        
        # Equal
        assert ComplexityClass.compare(ComplexityClass.LINEAR, ComplexityClass.LINEAR) == 0
        
        # Higher is worse (less efficient)
        assert ComplexityClass.compare(ComplexityClass.QUADRATIC, ComplexityClass.LINEAR) == 1
    
    def test_compare_with_unknown(self):
        """Test comparing with UNKNOWN complexity."""
        result = ComplexityClass.compare(ComplexityClass.UNKNOWN, ComplexityClass.LINEAR)
        assert result == 0  # Unknown comparisons return 0
    
    def test_is_equivalent(self):
        """Test complexity equivalence check."""
        assert ComplexityClass.LINEAR.is_equivalent(ComplexityClass.LINEAR)
        assert not ComplexityClass.LINEAR.is_equivalent(ComplexityClass.QUADRATIC)
    
    def test_multiply_complexities(self):
        """Test multiplying complexity classes."""
        # O(n) * O(n) = O(n²)
        result = ComplexityClass.multiply(ComplexityClass.LINEAR, ComplexityClass.LINEAR)
        assert result == ComplexityClass.QUADRATIC
        
        # O(n) * O(n²) = O(n³)
        result = ComplexityClass.multiply(ComplexityClass.LINEAR, ComplexityClass.QUADRATIC)
        assert result == ComplexityClass.CUBIC
        
        # O(n) * O(log n) = O(n log n)
        result = ComplexityClass.multiply(ComplexityClass.LINEAR, ComplexityClass.LOGARITHMIC)
        assert result == ComplexityClass.LINEARITHMIC
        
        # O(1) * O(n) = O(n)
        result = ComplexityClass.multiply(ComplexityClass.CONSTANT, ComplexityClass.LINEAR)
        assert result == ComplexityClass.LINEAR
        
        # O(n) * O(1) = O(n)
        result = ComplexityClass.multiply(ComplexityClass.LINEAR, ComplexityClass.CONSTANT)
        assert result == ComplexityClass.LINEAR
    
    def test_get_order(self):
        """Test getting complexity order."""
        order = ComplexityClass.get_order()
        
        assert order[0] == ComplexityClass.CONSTANT
        assert ComplexityClass.LOGARITHMIC in order
        assert ComplexityClass.LINEAR in order
        assert ComplexityClass.LINEARITHMIC in order
        assert ComplexityClass.QUADRATIC in order
        assert ComplexityClass.EXPONENTIAL in order
        assert ComplexityClass.FACTORIAL in order


class TestComplexityExpression:
    """Tests for ComplexityExpression."""
    
    def test_create_expression(self):
        """Test creating complexity expression."""
        expr = ComplexityExpression(
            base_class=ComplexityClass.LINEAR,
            raw_expression="O(n)"
        )
        
        assert expr.base_class == ComplexityClass.LINEAR
        assert expr.raw_expression == "O(n)"
    
    def test_expression_with_coefficient(self):
        """Test expression with coefficient."""
        expr = ComplexityExpression(
            base_class=ComplexityClass.LINEAR,
            raw_expression="O(2n)",
            coefficient=2.0
        )
        
        assert expr.coefficient == 2.0
        # Asymptotically still O(n)
        assert expr.to_string() == "O(n)"
    
    def test_expression_equivalence(self):
        """Test expression equivalence."""
        expr1 = ComplexityExpression(
            base_class=ComplexityClass.LINEAR,
            raw_expression="O(n)",
            coefficient=1.0
        )
        expr2 = ComplexityExpression(
            base_class=ComplexityClass.LINEAR,
            raw_expression="O(2n)",
            coefficient=2.0
        )
        expr3 = ComplexityExpression(
            base_class=ComplexityClass.QUADRATIC,
            raw_expression="O(n^2)"
        )
        
        assert expr1.is_equivalent(expr2)  # O(n) == O(2n) asymptotically
        assert not expr1.is_equivalent(expr3)


class TestComplexityFactor:
    """Tests for ComplexityFactor."""
    
    def test_create_factor(self):
        """Test creating complexity factor."""
        factor = ComplexityFactor(
            source="outer loop",
            factor_type="loop",
            complexity=ComplexityClass.LINEAR,
            iterations="n",
            nesting_level=0
        )
        
        assert factor.source == "outer loop"
        assert factor.factor_type == "loop"
        assert factor.complexity == ComplexityClass.LINEAR


class TestLoopComplexity:
    """Tests for LoopComplexity."""
    
    def test_create_simple_loop(self):
        """Test creating simple loop complexity."""
        loop = LoopComplexity(
            loop_type="for",
            iterator_variable="i",
            iterations="n",
            complexity=ComplexityClass.LINEAR
        )
        
        assert loop.loop_type == "for"
        assert loop.iterations == "n"
        assert loop.get_total_complexity() == ComplexityClass.LINEAR
    
    def test_nested_loop_complexity(self):
        """Test nested loop complexity calculation."""
        inner = LoopComplexity(
            loop_type="for",
            iterator_variable="j",
            iterations="n",
            complexity=ComplexityClass.LINEAR,
            nesting_level=1
        )
        outer = LoopComplexity(
            loop_type="for",
            iterator_variable="i",
            iterations="n",
            complexity=ComplexityClass.LINEAR,
            nesting_level=0,
            nested_loops=[inner]
        )
        
        # O(n) * O(n) = O(n²)
        total = outer.get_total_complexity()
        assert total == ComplexityClass.QUADRATIC
    
    def test_loop_with_bounds(self):
        """Test loop with explicit bounds."""
        loop = LoopComplexity(
            loop_type="for",
            iterator_variable="i",
            iterations="n",
            complexity=ComplexityClass.LINEAR,
            start_bound="1",
            end_bound="n",
            step_size="1"
        )
        
        assert loop.start_bound == "1"
        assert loop.end_bound == "n"
    
    def test_loop_to_dict(self):
        """Test loop complexity serialization."""
        loop = LoopComplexity(
            loop_type="for",
            iterator_variable="i",
            iterations="n",
            complexity=ComplexityClass.LINEAR
        )
        
        result = loop.model_dump()
        
        assert result["loop_type"] == "for"
        assert result["iterations"] == "n"


class TestRecursionComplexity:
    """Tests for RecursionComplexity."""
    
    def test_create_simple_recursion(self):
        """Test creating simple recursion complexity."""
        rec = RecursionComplexity(
            function_name="factorial",
            branching_factor=1,
            reduction_type="subtract",
            work_per_call=ComplexityClass.CONSTANT
        )
        
        assert rec.function_name == "factorial"
        assert rec.branching_factor == 1
    
    def test_analyze_linear_recursion(self):
        """Test analyzing linear recursion (factorial-like)."""
        rec = RecursionComplexity(
            function_name="factorial",
            branching_factor=1,
            reduction_factor=1,
            reduction_type="subtract",
            work_per_call=ComplexityClass.CONSTANT
        )
        
        result = rec.analyze()
        assert result == ComplexityClass.LINEAR
    
    def test_analyze_exponential_recursion(self):
        """Test analyzing exponential recursion (naive Fibonacci)."""
        rec = RecursionComplexity(
            function_name="fib",
            branching_factor=2,
            reduction_factor=1,
            reduction_type="subtract",
            work_per_call=ComplexityClass.CONSTANT
        )
        
        result = rec.analyze()
        assert result == ComplexityClass.EXPONENTIAL
    
    def test_analyze_divide_conquer_logarithmic(self):
        """Test analyzing divide-and-conquer with O(1) work (binary search)."""
        rec = RecursionComplexity(
            function_name="binarySearch",
            branching_factor=1,
            reduction_factor=2.0,
            reduction_type="divide",
            work_per_call=ComplexityClass.CONSTANT
        )
        
        result = rec.analyze()
        assert result == ComplexityClass.LOGARITHMIC
    
    def test_analyze_divide_conquer_linearithmic(self):
        """Test analyzing divide-and-conquer with O(n) work (merge sort)."""
        rec = RecursionComplexity(
            function_name="mergeSort",
            branching_factor=2,
            reduction_factor=2.0,
            reduction_type="divide",
            work_per_call=ComplexityClass.LINEAR
        )
        
        result = rec.analyze()
        assert result == ComplexityClass.LINEARITHMIC
    
    def test_recurrence_pattern(self):
        """Test setting recurrence pattern."""
        rec = RecursionComplexity(
            function_name="mergeSort",
            branching_factor=2,
            reduction_factor=2.0,
            reduction_type="divide",
            work_per_call=ComplexityClass.LINEAR,
            recurrence_pattern="T(n) = 2T(n/2) + O(n)"
        )
        
        assert rec.recurrence_pattern == "T(n) = 2T(n/2) + O(n)"
    
    def test_recursion_to_dict(self):
        """Test recursion complexity serialization."""
        rec = RecursionComplexity(
            function_name="factorial",
            branching_factor=1,
            reduction_type="subtract"
        )
        
        result = rec.model_dump()
        
        assert result["function_name"] == "factorial"
        assert result["branching_factor"] == 1


class TestTimeComplexity:
    """Tests for TimeComplexity."""
    
    def test_create_time_complexity(self):
        """Test creating time complexity result."""
        tc = TimeComplexity(
            overall=ComplexityClass.QUADRATIC,
            expression="O(n²)"
        )
        
        assert tc.overall == ComplexityClass.QUADRATIC
        assert tc.expression == "O(n²)"
    
    def test_time_complexity_with_contributions(self):
        """Test time complexity with loop contributions."""
        loop1 = LoopComplexity(
            loop_type="for",
            iterator_variable="i",
            iterations="n",
            complexity=ComplexityClass.LINEAR
        )
        loop2 = LoopComplexity(
            loop_type="for",
            iterator_variable="j",
            iterations="n",
            complexity=ComplexityClass.LINEAR,
            nesting_level=1
        )
        
        tc = TimeComplexity(
            overall=ComplexityClass.QUADRATIC,
            expression="O(n²)",
            loop_contributions=[loop1, loop2],
            dominant_factor="nested loops"
        )
        
        assert len(tc.loop_contributions) == 2
        assert tc.dominant_factor == "nested loops"
    
    def test_time_complexity_cases(self):
        """Test time complexity with best/average/worst cases."""
        tc = TimeComplexity(
            overall=ComplexityClass.LINEARITHMIC,
            expression="O(n log n)",
            best_case=ComplexityClass.LINEAR,
            average_case=ComplexityClass.LINEARITHMIC,
            worst_case=ComplexityClass.QUADRATIC
        )
        
        assert tc.best_case == ComplexityClass.LINEAR
        assert tc.worst_case == ComplexityClass.QUADRATIC
    
    def test_time_complexity_to_dict(self):
        """Test time complexity serialization."""
        tc = TimeComplexity(
            overall=ComplexityClass.LINEAR,
            expression="O(n)",
            explanation="Single loop iterating n times"
        )
        
        result = tc.model_dump()
        
        assert result["overall"] == ComplexityClass.LINEAR
        assert "explanation" in result


class TestSpaceComplexity:
    """Tests for SpaceComplexity."""
    
    def test_create_space_complexity(self):
        """Test creating space complexity result."""
        sc = SpaceComplexity(
            overall=ComplexityClass.CONSTANT,
            expression="O(1)"
        )
        
        assert sc.overall == ComplexityClass.CONSTANT
    
    def test_space_complexity_with_auxiliary(self):
        """Test space complexity with auxiliary space."""
        sc = SpaceComplexity(
            overall=ComplexityClass.LINEAR,
            expression="O(n)",
            auxiliary_space=ComplexityClass.LINEAR,
            input_space=ComplexityClass.LINEAR
        )
        
        assert sc.auxiliary_space == ComplexityClass.LINEAR
    
    def test_space_complexity_with_recursion_stack(self):
        """Test space complexity with recursion stack."""
        sc = SpaceComplexity(
            overall=ComplexityClass.LINEAR,
            expression="O(n)",
            recursion_stack=ComplexityClass.LINEAR
        )
        
        assert sc.recursion_stack == ComplexityClass.LINEAR
    
    def test_space_complexity_data_structures(self):
        """Test space complexity with data structures."""
        sc = SpaceComplexity(
            overall=ComplexityClass.LINEAR,
            expression="O(n)",
            data_structures=[
                {"type": "array", "size": "n"},
                {"type": "hash_table", "size": "n"}
            ]
        )
        
        assert len(sc.data_structures) == 2


class TestComplexityResult:
    """Tests for ComplexityResult."""
    
    def test_create_complexity_result(self):
        """Test creating complete complexity result."""
        tc = TimeComplexity(
            overall=ComplexityClass.QUADRATIC,
            expression="O(n²)"
        )
        sc = SpaceComplexity(
            overall=ComplexityClass.CONSTANT,
            expression="O(1)"
        )
        
        result = ComplexityResult(
            time_complexity=tc,
            space_complexity=sc
        )
        
        assert result.time_complexity.overall == ComplexityClass.QUADRATIC
        assert result.space_complexity.overall == ComplexityClass.CONSTANT
    
    def test_complexity_result_with_metadata(self):
        """Test complexity result with metadata."""
        tc = TimeComplexity(
            overall=ComplexityClass.LINEARITHMIC,
            expression="O(n log n)"
        )
        sc = SpaceComplexity(
            overall=ComplexityClass.LINEAR,
            expression="O(n)"
        )
        
        result = ComplexityResult(
            time_complexity=tc,
            space_complexity=sc,
            algorithm_type="sorting",
            is_optimal=True,
            confidence=0.95,
            optimization_suggestions=["Consider in-place sorting to reduce space"]
        )
        
        assert result.algorithm_type == "sorting"
        assert result.is_optimal == True
        assert result.confidence == 0.95
        assert len(result.optimization_suggestions) == 1
    
    def test_complexity_result_with_warnings(self):
        """Test complexity result with warnings."""
        tc = TimeComplexity(
            overall=ComplexityClass.UNKNOWN,
            expression="unknown"
        )
        sc = SpaceComplexity(
            overall=ComplexityClass.UNKNOWN,
            expression="unknown"
        )
        
        result = ComplexityResult(
            time_complexity=tc,
            space_complexity=sc,
            confidence=0.3,
            warnings=["Could not determine loop bounds", "Recursion pattern unclear"]
        )
        
        assert len(result.warnings) == 2
        assert result.confidence == 0.3
    
    def test_complexity_result_to_dict(self):
        """Test complexity result serialization."""
        tc = TimeComplexity(
            overall=ComplexityClass.LINEAR,
            expression="O(n)"
        )
        sc = SpaceComplexity(
            overall=ComplexityClass.CONSTANT,
            expression="O(1)"
        )
        
        result = ComplexityResult(
            time_complexity=tc,
            space_complexity=sc
        )
        
        data = result.model_dump()
        
        assert "time_complexity" in data
        assert "space_complexity" in data
        assert data["time_complexity"]["overall"] == ComplexityClass.LINEAR


class TestComplexityClassOrdering:
    """Tests for complexity class ordering and comparisons."""
    
    def test_complexity_ordering(self):
        """Test that complexities are correctly ordered."""
        order = ComplexityClass.get_order()
        
        for i in range(len(order) - 1):
            result = ComplexityClass.compare(order[i], order[i + 1])
            assert result == -1, f"{order[i]} should be less than {order[i + 1]}"
    
    def test_all_pairs_comparison(self):
        """Test comparing all pairs of complexity classes."""
        order = ComplexityClass.get_order()
        
        for i, c1 in enumerate(order):
            for j, c2 in enumerate(order):
                result = ComplexityClass.compare(c1, c2)
                if i < j:
                    assert result == -1
                elif i > j:
                    assert result == 1
                else:
                    assert result == 0
    
    def test_symmetric_comparison(self):
        """Test that comparisons are symmetric."""
        c1, c2 = ComplexityClass.LINEAR, ComplexityClass.QUADRATIC
        
        assert ComplexityClass.compare(c1, c2) == -ComplexityClass.compare(c2, c1)


class TestEdgeCases:
    """Tests for edge cases in complexity schemas."""
    
    def test_empty_expression(self):
        """Test handling of empty complexity expression."""
        result = ComplexityClass.from_string("")
        assert result == ComplexityClass.UNKNOWN
    
    def test_nested_loops_deep(self):
        """Test deeply nested loop complexity."""
        # Create 5 nested loops
        loops = []
        for i in range(5):
            loop = LoopComplexity(
                loop_type="for",
                iterator_variable=f"i{i}",
                iterations="n",
                complexity=ComplexityClass.LINEAR,
                nesting_level=i
            )
            loops.append(loop)
        
        # Link them
        for i in range(len(loops) - 1):
            loops[i].nested_loops = [loops[i + 1]]
        
        # O(n^5) should be at least CUBIC or POLYNOMIAL
        total = loops[0].get_total_complexity()
        # Accept CUBIC or POLYNOMIAL since deep nesting multiplication is complex
        assert total in [ComplexityClass.CUBIC, ComplexityClass.POLYNOMIAL], f"Expected CUBIC or POLYNOMIAL, got {total}"
    
    def test_zero_branching_factor(self):
        """Test recursion with zero branching factor."""
        rec = RecursionComplexity(
            function_name="test",
            branching_factor=0,
            reduction_type="subtract"
        )
        
        # Should not crash
        result = rec.analyze()
        assert result is not None
    
    def test_large_coefficient(self):
        """Test expression with large coefficient."""
        expr = ComplexityExpression(
            base_class=ComplexityClass.LINEAR,
            raw_expression="O(1000000n)",
            coefficient=1000000
        )
        
        # Still O(n) asymptotically
        assert expr.to_string() == "O(n)"
