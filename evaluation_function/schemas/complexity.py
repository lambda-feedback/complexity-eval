"""
Complexity Analysis Schemas using Pydantic.

This module defines types and structures for representing algorithm complexity,
including time complexity, space complexity, and various complexity factors.

Complexity Classes (Big-O):
    O(1)        - Constant
    O(log n)    - Logarithmic
    O(n)        - Linear
    O(n log n)  - Linearithmic
    O(n²)       - Quadratic
    O(n³)       - Cubic
    O(2^n)      - Exponential
    O(n!)       - Factorial
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import math


class ComplexityClass(str, Enum):
    """
    Standard complexity classes in ascending order.
    """
    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    SQRT = "O(√n)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n²)"
    CUBIC = "O(n³)"
    POLYNOMIAL = "O(n^k)"  # For n^k where k > 3
    EXPONENTIAL = "O(2^n)"
    FACTORIAL = "O(n!)"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, s: str) -> "ComplexityClass":
        """
        Parse complexity from string representation.
        
        Handles various formats:
            - O(n^2), O(n²), O(n*n) -> QUADRATIC
            - O(nlogn), O(n log n) -> LINEARITHMIC
            - O(log n), O(logn), O(lg n) -> LOGARITHMIC
        """
        if not s:
            return cls.UNKNOWN
            
        # Normalize the string
        normalized = s.lower().replace(" ", "").replace("*", "")
        
        # Direct mappings
        mappings = {
            "o(1)": cls.CONSTANT,
            "1": cls.CONSTANT,
            "constant": cls.CONSTANT,
            "o(logn)": cls.LOGARITHMIC,
            "o(lgn)": cls.LOGARITHMIC,
            "o(log(n))": cls.LOGARITHMIC,
            "logn": cls.LOGARITHMIC,
            "logarithmic": cls.LOGARITHMIC,
            "o(√n)": cls.SQRT,
            "o(sqrtn)": cls.SQRT,
            "o(sqrt(n))": cls.SQRT,
            "o(n^0.5)": cls.SQRT,
            "o(n)": cls.LINEAR,
            "n": cls.LINEAR,
            "linear": cls.LINEAR,
            "o(nlogn)": cls.LINEARITHMIC,
            "o(nlgn)": cls.LINEARITHMIC,
            "o(nlog(n))": cls.LINEARITHMIC,
            "nlogn": cls.LINEARITHMIC,
            "linearithmic": cls.LINEARITHMIC,
            "o(n^2)": cls.QUADRATIC,
            "o(n²)": cls.QUADRATIC,
            "o(nn)": cls.QUADRATIC,
            "n^2": cls.QUADRATIC,
            "n²": cls.QUADRATIC,
            "quadratic": cls.QUADRATIC,
            "o(n^3)": cls.CUBIC,
            "o(n³)": cls.CUBIC,
            "n^3": cls.CUBIC,
            "cubic": cls.CUBIC,
            "o(2^n)": cls.EXPONENTIAL,
            "o(2ⁿ)": cls.EXPONENTIAL,
            "2^n": cls.EXPONENTIAL,
            "exponential": cls.EXPONENTIAL,
            "o(n!)": cls.FACTORIAL,
            "n!": cls.FACTORIAL,
            "factorial": cls.FACTORIAL,
        }
        
        return mappings.get(normalized, cls.UNKNOWN)
    
    @classmethod
    def get_order(cls) -> List["ComplexityClass"]:
        """Get complexity classes in ascending order."""
        return [
            cls.CONSTANT,
            cls.LOGARITHMIC,
            cls.SQRT,
            cls.LINEAR,
            cls.LINEARITHMIC,
            cls.QUADRATIC,
            cls.CUBIC,
            cls.POLYNOMIAL,
            cls.EXPONENTIAL,
            cls.FACTORIAL,
        ]
    
    @classmethod
    def compare(cls, a: "ComplexityClass", b: "ComplexityClass") -> int:
        """
        Compare two complexity classes.
        
        Returns:
            -1 if a < b (a is more efficient)
             0 if a == b
             1 if a > b (b is more efficient)
        """
        order = cls.get_order()
        
        if a == cls.UNKNOWN or b == cls.UNKNOWN:
            return 0
        
        try:
            idx_a = order.index(a)
            idx_b = order.index(b)
        except ValueError:
            return 0
        
        if idx_a < idx_b:
            return -1
        elif idx_a > idx_b:
            return 1
        return 0
    
    def is_equivalent(self, other: "ComplexityClass") -> bool:
        """Check if two complexity classes are equivalent."""
        return self == other
    
    @classmethod
    def multiply(cls, a: "ComplexityClass", b: "ComplexityClass") -> "ComplexityClass":
        """
        Multiply two complexity classes (for nested structures).
        
        Examples:
            O(n) * O(n) = O(n²)
            O(n) * O(log n) = O(n log n)
        """
        if a == cls.UNKNOWN or b == cls.UNKNOWN:
            return cls.UNKNOWN
        if a == cls.CONSTANT:
            return b
        if b == cls.CONSTANT:
            return a
        
        # Multiplication rules
        rules = {
            (cls.LINEAR, cls.LINEAR): cls.QUADRATIC,
            (cls.LINEAR, cls.QUADRATIC): cls.CUBIC,
            (cls.QUADRATIC, cls.LINEAR): cls.CUBIC,
            (cls.LINEAR, cls.CUBIC): cls.POLYNOMIAL,      # O(n⁴)
            (cls.CUBIC, cls.LINEAR): cls.POLYNOMIAL,      # O(n⁴)
            (cls.LINEAR, cls.LOGARITHMIC): cls.LINEARITHMIC,
            (cls.LOGARITHMIC, cls.LINEAR): cls.LINEARITHMIC,
            (cls.LINEAR, cls.LINEARITHMIC): cls.POLYNOMIAL,  # O(n² log n) ≈ polynomial
            (cls.LINEARITHMIC, cls.LINEAR): cls.POLYNOMIAL,
            (cls.QUADRATIC, cls.QUADRATIC): cls.POLYNOMIAL,  # O(n⁴)
            (cls.QUADRATIC, cls.CUBIC): cls.POLYNOMIAL,      # O(n⁵)
            (cls.CUBIC, cls.QUADRATIC): cls.POLYNOMIAL,      # O(n⁵)
            (cls.CUBIC, cls.CUBIC): cls.POLYNOMIAL,          # O(n⁶)
            (cls.LINEAR, cls.POLYNOMIAL): cls.POLYNOMIAL,    # Still polynomial
            (cls.POLYNOMIAL, cls.LINEAR): cls.POLYNOMIAL,    # Still polynomial
            (cls.QUADRATIC, cls.POLYNOMIAL): cls.POLYNOMIAL,
            (cls.POLYNOMIAL, cls.QUADRATIC): cls.POLYNOMIAL,
            (cls.POLYNOMIAL, cls.POLYNOMIAL): cls.POLYNOMIAL,
        }
        
        result = rules.get((a, b))
        if result:
            return result
        
        # Default: return the higher complexity
        order = cls.get_order()
        try:
            idx_a = order.index(a)
            idx_b = order.index(b)
            return order[max(idx_a, idx_b)]
        except ValueError:
            return cls.UNKNOWN


class ComplexityExpression(BaseModel):
    """
    Represents a complexity expression that can be more specific than standard classes.
    
    Examples:
        - O(n) with coefficient 2 -> O(2n) (still O(n) asymptotically)
        - O(n^2) with log factor -> O(n² log n)
    """
    base_class: ComplexityClass
    raw_expression: str  # Original expression as written
    coefficient: float = 1.0  # Constant factor (ignored in Big-O)
    log_factor: int = 0  # Number of log(n) factors
    exponent: float = 1.0  # For polynomial: n^exponent
    
    def to_string(self) -> str:
        """Convert to standard Big-O notation."""
        return self.base_class.value
    
    def is_equivalent(self, other: "ComplexityExpression") -> bool:
        """
        Check if two expressions are asymptotically equivalent.
        
        O(2n) == O(n), O(n^2) == O(n*n), etc.
        """
        return self.base_class == other.base_class


class ComplexityFactor(BaseModel):
    """
    Represents a single factor contributing to overall complexity.
    
    Used to track how different code constructs contribute to complexity.
    """
    source: str  # Description of what creates this factor
    factor_type: str  # "loop", "recursion", "operation", "data_structure"
    complexity: ComplexityClass
    iterations: Optional[str] = None  # e.g., "n", "n/2", "log(n)"
    nesting_level: int = 0
    location: Optional[str] = None  # Source location


class LoopComplexity(BaseModel):
    """
    Complexity analysis for a single loop construct.
    """
    loop_type: str  # "for", "while", "for_each"
    iterator_variable: Optional[str] = None
    iterations: str = "n"  # Expression for number of iterations
    complexity: ComplexityClass = ComplexityClass.LINEAR
    nesting_level: int = 0
    body_complexity: ComplexityClass = ComplexityClass.CONSTANT
    
    # Analysis details
    start_bound: Optional[str] = None
    end_bound: Optional[str] = None
    step_size: Optional[str] = None
    
    # Nested loop info
    nested_loops: List["LoopComplexity"] = Field(default_factory=list)
    
    def get_total_complexity(self) -> ComplexityClass:
        """Calculate total complexity including nested loops."""
        result = self.complexity
        
        for nested in self.nested_loops:
            nested_total = nested.get_total_complexity()
            result = ComplexityClass.multiply(result, nested_total)
        
        return result


class RecursionComplexity(BaseModel):
    """
    Complexity analysis for recursive functions.
    
    Uses recurrence relation analysis:
        T(n) = a*T(n/b) + f(n)
    
    Where:
        a = branching factor (number of recursive calls)
        b = reduction factor (how problem size decreases)
        f(n) = work done per call
    """
    function_name: str
    branching_factor: int = 1  # 'a' in recurrence
    reduction_factor: float = 2.0  # 'b' in recurrence
    reduction_type: str = "divide"  # "divide" (n/b), "subtract" (n-b)
    work_per_call: ComplexityClass = ComplexityClass.CONSTANT  # f(n)
    
    # Detected recurrence pattern
    recurrence_pattern: Optional[str] = None  # e.g., "T(n) = 2T(n/2) + O(n)"
    
    # Result
    time_complexity: ComplexityClass = ComplexityClass.UNKNOWN
    space_complexity: ComplexityClass = ComplexityClass.UNKNOWN  # Stack depth
    
    # Analysis details
    base_case: Optional[str] = None
    recursive_calls: List[str] = Field(default_factory=list)
    
    def analyze(self) -> ComplexityClass:
        """
        Analyze recursion complexity using Master Theorem or pattern matching.
        """
        if self.reduction_type == "divide":
            return self._apply_master_theorem()
        else:
            return self._analyze_linear_recursion()
    
    def _apply_master_theorem(self) -> ComplexityClass:
        """
        Apply Master Theorem for divide-and-conquer recurrences.
        
        T(n) = a*T(n/b) + O(n^d)
        
        Case 1: d < log_b(a) -> O(n^(log_b(a)))
        Case 2: d = log_b(a) -> O(n^d * log(n))
        Case 3: d > log_b(a) -> O(n^d)
        """
        a = self.branching_factor
        b = self.reduction_factor
        
        # Get d from work_per_call
        d = 0
        if self.work_per_call == ComplexityClass.CONSTANT:
            d = 0
        elif self.work_per_call == ComplexityClass.LINEAR:
            d = 1
        elif self.work_per_call == ComplexityClass.QUADRATIC:
            d = 2
        
        if b <= 1:
            return ComplexityClass.UNKNOWN
            
        log_b_a = math.log(a) / math.log(b)
        
        if d < log_b_a - 0.01:
            # Case 1
            if abs(log_b_a - 1) < 0.01:
                return ComplexityClass.LINEAR
            elif abs(log_b_a - 2) < 0.01:
                return ComplexityClass.QUADRATIC
            else:
                return ComplexityClass.POLYNOMIAL
        elif abs(d - log_b_a) < 0.01:
            # Case 2: d ≈ log_b(a)
            if d == 0:
                return ComplexityClass.LOGARITHMIC
            elif d == 1:
                return ComplexityClass.LINEARITHMIC
            else:
                return ComplexityClass.POLYNOMIAL
        else:
            # Case 3
            return self.work_per_call
    
    def _analyze_linear_recursion(self) -> ComplexityClass:
        """Analyze recursion with linear reduction (n-1, n-2, etc.)."""
        if self.branching_factor == 1:
            # Single recursive call: T(n) = T(n-1) + f(n)
            # Results in O(n * f(n))
            if self.work_per_call == ComplexityClass.CONSTANT:
                return ComplexityClass.LINEAR
            elif self.work_per_call == ComplexityClass.LINEAR:
                return ComplexityClass.QUADRATIC
        elif self.branching_factor == 2:
            # Two recursive calls: T(n) = 2T(n-1) + f(n) -> O(2^n)
            return ComplexityClass.EXPONENTIAL
        
        return ComplexityClass.UNKNOWN


class TimeComplexity(BaseModel):
    """
    Complete time complexity analysis result.
    """
    overall: ComplexityClass
    expression: str  # Raw expression, e.g., "O(n²)"
    
    # Breakdown by source
    loop_contributions: List[LoopComplexity] = Field(default_factory=list)
    recursion_contributions: List[RecursionComplexity] = Field(default_factory=list)
    other_factors: List[ComplexityFactor] = Field(default_factory=list)
    
    # Analysis
    dominant_factor: Optional[str] = None  # What dominates complexity
    explanation: str = ""
    
    # Best/Average/Worst case (if distinguishable)
    best_case: Optional[ComplexityClass] = None
    average_case: Optional[ComplexityClass] = None
    worst_case: Optional[ComplexityClass] = None


class SpaceComplexity(BaseModel):
    """
    Complete space complexity analysis result.
    """
    overall: ComplexityClass
    expression: str  # Raw expression, e.g., "O(n)"
    
    # Space breakdown
    auxiliary_space: ComplexityClass = ComplexityClass.CONSTANT  # Extra space used
    input_space: Optional[ComplexityClass] = None  # Space for input
    recursion_stack: Optional[ComplexityClass] = None  # Call stack depth
    
    # Data structures detected
    data_structures: List[Dict[str, Any]] = Field(default_factory=list)
    
    explanation: str = ""


class ComplexityResult(BaseModel):
    """
    Complete complexity analysis result combining time and space.
    """
    time_complexity: TimeComplexity
    space_complexity: SpaceComplexity
    
    # Overall analysis
    algorithm_type: Optional[str] = None  # e.g., "sorting", "searching", "graph traversal"
    is_optimal: Optional[bool] = None  # If we know optimal for this problem type
    optimization_suggestions: List[str] = Field(default_factory=list)
    
    # Confidence in analysis
    confidence: float = 1.0  # 0.0 to 1.0
    warnings: List[str] = Field(default_factory=list)


# Update forward references
LoopComplexity.model_rebuild()
