"""
Complexity Analyzer - Analyzes pseudocode AST to determine time complexity.

This module provides the core analysis logic for determining algorithm complexity
from parsed pseudocode, with detailed tracking of complexity factors.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import re
import math

from ..schemas.complexity import (
    ComplexityClass,
    LoopComplexity,
    RecursionComplexity,
    TimeComplexity,
    SpaceComplexity,
    ComplexityResult,
    ComplexityFactor,
)
from ..schemas.ast_nodes import (
    ProgramNode,
    FunctionNode,
    BlockNode,
    LoopNode,
    ConditionalNode,
    LoopType,
)


@dataclass
class LoopInfo:
    """Information about a detected loop."""
    loop_type: str
    iterator: Optional[str]
    start_bound: Optional[str]
    end_bound: Optional[str]
    step: Optional[str]
    iterations: str
    complexity: ComplexityClass
    nesting_level: int
    line_number: int = 0
    has_early_exit: bool = False
    nested_loops: List["LoopInfo"] = field(default_factory=list)
    
    def get_description(self) -> str:
        """Get human-readable description of this loop."""
        if self.loop_type == "for" and self.iterator:
            if self.start_bound and self.end_bound:
                return f"FOR loop ({self.iterator} from {self.start_bound} to {self.end_bound})"
            return f"FOR loop with iterator {self.iterator}"
        elif self.loop_type == "for_each":
            return f"FOR-EACH loop iterating over collection"
        elif self.loop_type == "while":
            return f"WHILE loop"
        elif self.loop_type == "repeat":
            return f"REPEAT-UNTIL loop"
        return f"{self.loop_type.upper()} loop"


@dataclass
class RecursionInfo:
    """Information about detected recursion."""
    function_name: str
    num_recursive_calls: int
    reduction_pattern: str  # "n-1", "n/2", etc.
    branching_factor: int
    work_per_call: ComplexityClass
    complexity: ComplexityClass
    recurrence: str  # e.g., "T(n) = 2T(n/2) + O(n)"
    
    def get_description(self) -> str:
        """Get human-readable description of this recursion."""
        if self.branching_factor == 1:
            return f"Linear recursion in {self.function_name}()"
        elif self.branching_factor == 2:
            if "n/2" in self.reduction_pattern:
                return f"Divide-and-conquer recursion in {self.function_name}()"
            else:
                return f"Binary recursion in {self.function_name}()"
        return f"Multiple recursion ({self.branching_factor} calls) in {self.function_name}()"


@dataclass
class AnalysisResult:
    """Complete result of complexity analysis."""
    time_complexity: ComplexityClass
    space_complexity: ComplexityClass
    loops: List[LoopInfo]
    recursion: Optional[RecursionInfo]
    max_nesting_depth: int
    confidence: float
    factors: List[ComplexityFactor]
    raw_code: str = ""
    
    def get_complexity_string(self) -> str:
        """Get the complexity as a string."""
        return self.time_complexity.value


class ComplexityAnalyzer:
    """
    Analyzes pseudocode to determine time and space complexity.
    
    This analyzer works with both parsed AST and raw code fallback,
    providing detailed analysis of loops, recursion, and other factors.
    """
    
    def __init__(self):
        self.loops: List[LoopInfo] = []
        self.recursion_info: Optional[RecursionInfo] = None
        self.factors: List[ComplexityFactor] = []
        self.current_function: Optional[str] = None
        self.function_calls: Dict[str, List[str]] = {}
    
    def analyze(self, code: str, ast: Optional[ProgramNode] = None) -> AnalysisResult:
        """
        Analyze code complexity.
        
        Args:
            code: The pseudocode string
            ast: Optional parsed AST (uses pattern matching if not provided)
            
        Returns:
            AnalysisResult with complexity information
        """
        self._reset()
        
        if ast:
            return self._analyze_ast(ast, code)
        else:
            return self._analyze_patterns(code)
    
    def _reset(self):
        """Reset analyzer state."""
        self.loops = []
        self.recursion_info = None
        self.factors = []
        self.current_function = None
        self.function_calls = {}
    
    def _analyze_ast(self, ast: ProgramNode, code: str) -> AnalysisResult:
        """Analyze from parsed AST."""
        # Analyze functions
        for func in ast.functions:
            self.current_function = func.name
            self.function_calls[func.name] = []

            if func.body:
                self._analyze_block(func.body, nesting_level=0)

            # Check for recursion using pattern matching on code
            # Look for recursive calls in the code
            if code:
                call_pattern = rf'\b{func.name}\s*\('
                calls = re.findall(call_pattern, code, re.IGNORECASE)
                if len(calls) > 1:  # More than just the definition
                    self._detect_recursion_in_function(func, code)

        # Analyze global statements
        if ast.global_statements:
            self._analyze_block(ast.global_statements, nesting_level=0)

        return self._compute_result(code)
    
    def _analyze_block(self, block: BlockNode, nesting_level: int):
        """Analyze a block of statements (top-level, adds to self.loops)."""
        for stmt in block.statements:
            if isinstance(stmt, LoopNode):
                loop_info = self._analyze_loop(stmt, nesting_level)
                self.loops.append(loop_info)
            elif isinstance(stmt, ConditionalNode):
                # For top-level conditionals, add any loops they contain to self.loops
                loops_from_cond = self._analyze_conditional(stmt, nesting_level)
                self.loops.extend(loops_from_cond)
    
    def _analyze_loop(self, loop: LoopNode, nesting_level: int) -> LoopInfo:
        """Analyze a loop node."""
        loop_type = loop.loop_type.value if loop.loop_type else "for"
        iterator = loop.iterator.name if loop.iterator else None
        
        # Determine iterations
        iterations, complexity = self._estimate_loop_iterations(loop)
        
        # Get bounds
        start_bound = self._expr_to_string(loop.start)
        end_bound = self._expr_to_string(loop.end)
        step = self._expr_to_string(loop.step) if loop.step else "1"
        
        loop_info = LoopInfo(
            loop_type=loop_type,
            iterator=iterator,
            start_bound=start_bound,
            end_bound=end_bound,
            step=step,
            iterations=iterations,
            complexity=complexity,
            nesting_level=nesting_level,
            line_number=loop.location.line if loop.location else 0
        )
        
        # Analyze nested content
        if loop.body:
            for stmt in loop.body.statements:
                if isinstance(stmt, LoopNode):
                    nested_info = self._analyze_loop(stmt, nesting_level + 1)
                    loop_info.nested_loops.append(nested_info)
                elif isinstance(stmt, ConditionalNode):
                    # Loops inside conditionals should be counted as nested
                    nested_from_cond = self._analyze_conditional(stmt, nesting_level + 1)
                    loop_info.nested_loops.extend(nested_from_cond)

        return loop_info
    
    def _analyze_conditional(self, cond: ConditionalNode, nesting_level: int) -> List[LoopInfo]:
        """Analyze conditional branches for loops. Returns loops found for nesting."""
        nested_loops = []
        if cond.then_branch:
            nested_loops.extend(self._extract_loops_from_block(cond.then_branch, nesting_level))
        if cond.else_branch:
            nested_loops.extend(self._extract_loops_from_block(cond.else_branch, nesting_level))
        return nested_loops

    def _extract_loops_from_block(self, block: BlockNode, nesting_level: int) -> List[LoopInfo]:
        """Extract loops from a block without adding to self.loops. Used for nested contexts."""
        loops = []
        for stmt in block.statements:
            if isinstance(stmt, LoopNode):
                loop_info = self._analyze_loop(stmt, nesting_level)
                loops.append(loop_info)
            elif isinstance(stmt, ConditionalNode):
                loops.extend(self._analyze_conditional(stmt, nesting_level))
        return loops
    
    def _estimate_loop_iterations(self, loop: LoopNode) -> Tuple[str, ComplexityClass]:
        """Estimate the number of iterations for a loop."""
        if loop.estimated_iterations:
            iterations = loop.estimated_iterations
        elif loop.loop_type == LoopType.FOR:
            iterations = self._estimate_for_iterations(loop)
        elif loop.loop_type == LoopType.FOR_EACH:
            iterations = "n"  # Collection size
        else:
            iterations = "n"  # Default for while/repeat
        
        # Convert iterations to complexity class
        complexity = self._iterations_to_complexity(iterations)
        
        return iterations, complexity
    
    def _estimate_for_iterations(self, loop: LoopNode) -> str:
        """Estimate FOR loop iterations from bounds."""
        start = self._expr_to_string(loop.start)
        end = self._expr_to_string(loop.end)
        
        if not start or not end:
            return "n"
        
        start_lower = start.lower()
        end_lower = end.lower()
        
        # Common patterns
        if start_lower in ("0", "1"):
            if end_lower in ("n", "len", "length", "size", "count"):
                return "n"
            if "n-" in end_lower or "n -" in end_lower:
                return "n"
            if "n/2" in end_lower or "n / 2" in end_lower:
                return "n/2"
            if "log" in end_lower:
                return "log(n)"
            if end_lower.startswith("sqrt") or "√" in end_lower:
                return "√n"
        
        # Check if both are constants
        try:
            s = int(start)
            e = int(end)
            return str(e - s + 1)
        except (ValueError, TypeError):
            pass
        
        # Default to n
        return "n"
    
    def _iterations_to_complexity(self, iterations: str) -> ComplexityClass:
        """Convert iteration count to complexity class."""
        iterations_lower = iterations.lower().replace(" ", "")
        
        if iterations_lower in ("1", "2", "3", "4", "5", "10", "100"):
            return ComplexityClass.CONSTANT
        if "log" in iterations_lower:
            return ComplexityClass.LOGARITHMIC
        if "sqrt" in iterations_lower or "√" in iterations_lower:
            return ComplexityClass.SQRT
        if iterations_lower in ("n", "n-1", "n+1", "len", "length", "size"):
            return ComplexityClass.LINEAR
        if "n/2" in iterations_lower:
            return ComplexityClass.LINEAR  # Still O(n)
        
        return ComplexityClass.LINEAR  # Default
    
    def _expr_to_string(self, expr) -> Optional[str]:
        """Convert expression node to string."""
        if expr is None:
            return None
        if hasattr(expr, 'name'):
            return expr.name
        if hasattr(expr, 'value'):
            return str(expr.value)
        return str(expr)
    
    def _detect_recursion_in_function(self, func: FunctionNode, code: str):
        """Detect recursion pattern in a function."""
        func_name = func.name
        
        # Count recursive calls in code
        call_pattern = rf'\b{func_name}\s*\('
        calls = re.findall(call_pattern, code, re.IGNORECASE)
        num_calls = len(calls) - 1  # Subtract definition
        
        if num_calls <= 0:
            return
        
        # Analyze reduction pattern
        reduction, branching = self._analyze_recursive_calls(func_name, code)
        
        # Determine complexity
        complexity = self._compute_recursion_complexity(branching, reduction)
        
        # Build recurrence relation
        recurrence = self._build_recurrence(branching, reduction)
        
        self.recursion_info = RecursionInfo(
            function_name=func_name,
            num_recursive_calls=num_calls,
            reduction_pattern=reduction,
            branching_factor=branching,
            work_per_call=ComplexityClass.CONSTANT,
            complexity=complexity,
            recurrence=recurrence
        )
    
    def _analyze_recursive_calls(self, func_name: str, code: str) -> Tuple[str, int]:
        """Analyze recursive call patterns."""
        code_lower = code.lower()

        # Count recursive calls, distinguishing between:
        # - Calls in mutually exclusive branches (if-else with RETURN) -> branching = 1
        # - Calls that all execute (sequential, or in same expression) -> branching = count
        lines = code.split('\n')
        total_recursive_calls = 0
        max_calls_per_line = 0
        calls_with_return = 0
        calls_without_return = 0

        for line in lines:
            line_lower = line.lower().strip()
            # Skip function definition line
            if re.search(rf'(function|def|procedure|algorithm)\s+{func_name}\s*\(', line, re.IGNORECASE):
                continue
            calls = len(re.findall(rf'\b{func_name}\s*\(', line, re.IGNORECASE))
            if calls > 0:
                total_recursive_calls += calls
                max_calls_per_line = max(max_calls_per_line, calls)
                # Check if this line has RETURN before the call
                if re.search(r'\breturn\b', line, re.IGNORECASE):
                    calls_with_return += calls
                else:
                    calls_without_return += calls

        # Determine branching factor:
        # - If all calls are in return statements (like binary search), likely mutually exclusive -> branching = 1
        # - If calls are in same expression (like fibonacci), use max per line
        # - If calls are sequential without return (like merge sort), all execute -> use total
        if max_calls_per_line >= 2:
            # Multiple calls on same line (e.g., fib(n-1) + fib(n-2))
            branching = max_calls_per_line
        elif calls_without_return >= 2:
            # Multiple sequential calls without return (merge sort pattern)
            branching = calls_without_return
        elif calls_with_return > 0 and calls_without_return == 0:
            # All calls are in return statements (binary search pattern - mutually exclusive)
            branching = 1
        else:
            branching = max(1, total_recursive_calls)

        # Detect reduction pattern
        # Use word boundaries to avoid matching parts of other words (e.g., "return -1")
        patterns = [
            (r'\bn\s*-\s*1\b', 'n-1'),
            (r'\bn\s*-\s*2\b', 'n-1'),  # Still linear reduction
            (r'\bn\s*/\s*2\b', 'n/2'),
            (r'\bn\s*//\s*2\b', 'n/2'),
            (r'\bmid\b', 'n/2'),
            (r'\blow\b.*\bhigh\b', 'n/2'),
        ]

        reduction = 'n-1'  # Default
        for pattern, result in patterns:
            if re.search(pattern, code_lower):
                reduction = result
                break

        return reduction, branching
    
    def _compute_recursion_complexity(self, branching: int, reduction: str) -> ComplexityClass:
        """Compute recursion complexity using Master Theorem."""
        if "n/2" in reduction:
            # Divide and conquer
            if branching == 1:
                return ComplexityClass.LOGARITHMIC  # Binary search
            elif branching == 2:
                return ComplexityClass.LINEARITHMIC  # Merge sort
            else:
                return ComplexityClass.POLYNOMIAL
        else:
            # Linear reduction (n-1)
            if branching == 1:
                return ComplexityClass.LINEAR  # Simple recursion
            elif branching == 2:
                return ComplexityClass.EXPONENTIAL  # Fibonacci-like
            else:
                return ComplexityClass.EXPONENTIAL
    
    def _build_recurrence(self, branching: int, reduction: str) -> str:
        """Build recurrence relation string."""
        if "n/2" in reduction:
            return f"T(n) = {branching}T(n/2) + O(1)"
        else:
            return f"T(n) = {branching}T(n-1) + O(1)"
    
    def _analyze_patterns(self, code: str) -> AnalysisResult:
        """Analyze code using pattern matching (fallback)."""
        lines = code.split('\n')
        
        # Detect function definitions
        func_match = re.search(r'(function|algorithm|def|procedure)\s+(\w+)', code, re.IGNORECASE)
        if func_match:
            self.current_function = func_match.group(2)
        
        # Analyze each line for loops
        self._detect_loops_from_code(lines)
        
        # Detect recursion
        if self.current_function:
            call_pattern = rf'\b{self.current_function}\s*\('
            calls = re.findall(call_pattern, code, re.IGNORECASE)
            if len(calls) > 1:
                reduction, branching = self._analyze_recursive_calls(self.current_function, code)
                complexity = self._compute_recursion_complexity(branching, reduction)
                self.recursion_info = RecursionInfo(
                    function_name=self.current_function,
                    num_recursive_calls=len(calls) - 1,
                    reduction_pattern=reduction,
                    branching_factor=branching,
                    work_per_call=ComplexityClass.CONSTANT,
                    complexity=complexity,
                    recurrence=self._build_recurrence(branching, reduction)
                )
        
        return self._compute_result(code)
    
    def _detect_loops_from_code(self, lines: List[str]):
        """Detect loops from raw code lines."""
        indent_stack: List[Tuple[int, LoopInfo]] = []
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip().lower()
            if not stripped:
                continue
            
            indent = len(line) - len(line.lstrip())
            
            # Pop loops that have ended
            while indent_stack and indent <= indent_stack[-1][0]:
                indent_stack.pop()
            
            nesting_level = len(indent_stack)
            
            # Check for FOR loop
            for_match = re.match(r'for\s+(\w+)\s*[=:]\s*(\w+)\s+to\s+(\w+)', stripped)
            if for_match:
                loop_info = self._create_loop_info_from_match(
                    "for", for_match, nesting_level, line_num
                )
                if indent_stack:
                    indent_stack[-1][1].nested_loops.append(loop_info)
                else:
                    self.loops.append(loop_info)
                indent_stack.append((indent, loop_info))
                continue
            
            # Check for FOR EACH loop
            foreach_match = re.match(r'for\s+(?:each\s+)?(\w+)\s+in\s+(\w+)', stripped)
            if foreach_match:
                loop_info = LoopInfo(
                    loop_type="for_each",
                    iterator=foreach_match.group(1),
                    start_bound=None,
                    end_bound=foreach_match.group(2),
                    step=None,
                    iterations="n",
                    complexity=ComplexityClass.LINEAR,
                    nesting_level=nesting_level,
                    line_number=line_num
                )
                if indent_stack:
                    indent_stack[-1][1].nested_loops.append(loop_info)
                else:
                    self.loops.append(loop_info)
                indent_stack.append((indent, loop_info))
                continue
            
            # Check for WHILE loop
            if stripped.startswith('while ') or stripped.startswith('while('):
                loop_info = LoopInfo(
                    loop_type="while",
                    iterator=None,
                    start_bound=None,
                    end_bound=None,
                    step=None,
                    iterations=self._estimate_while_iterations(stripped),
                    complexity=ComplexityClass.LINEAR,
                    nesting_level=nesting_level,
                    line_number=line_num
                )
                
                # Check for logarithmic pattern
                if any(p in stripped for p in ['/2', '//2', '* 2', '*2', 'log']):
                    loop_info.iterations = "log(n)"
                    loop_info.complexity = ComplexityClass.LOGARITHMIC
                
                if indent_stack:
                    indent_stack[-1][1].nested_loops.append(loop_info)
                else:
                    self.loops.append(loop_info)
                indent_stack.append((indent, loop_info))
                continue
            
            # Check for REPEAT loop
            if stripped.startswith('repeat'):
                loop_info = LoopInfo(
                    loop_type="repeat",
                    iterator=None,
                    start_bound=None,
                    end_bound=None,
                    step=None,
                    iterations="n",
                    complexity=ComplexityClass.LINEAR,
                    nesting_level=nesting_level,
                    line_number=line_num
                )
                if indent_stack:
                    indent_stack[-1][1].nested_loops.append(loop_info)
                else:
                    self.loops.append(loop_info)
                indent_stack.append((indent, loop_info))
    
    def _create_loop_info_from_match(self, loop_type: str, match, 
                                      nesting_level: int, line_num: int) -> LoopInfo:
        """Create LoopInfo from regex match."""
        iterator = match.group(1)
        start = match.group(2)
        end = match.group(3)
        
        # Estimate iterations
        iterations = self._estimate_iterations_from_bounds(start, end)
        complexity = self._iterations_to_complexity(iterations)
        
        return LoopInfo(
            loop_type=loop_type,
            iterator=iterator,
            start_bound=start,
            end_bound=end,
            step="1",
            iterations=iterations,
            complexity=complexity,
            nesting_level=nesting_level,
            line_number=line_num
        )
    
    def _estimate_iterations_from_bounds(self, start: str, end: str) -> str:
        """Estimate iterations from loop bounds."""
        start_lower = start.lower()
        end_lower = end.lower()
        
        if start_lower in ("0", "1"):
            if end_lower in ("n", "len", "length", "size", "count"):
                return "n"
            if "n-" in end_lower:
                return "n"
            if "n/2" in end_lower:
                return "n/2"
            if "log" in end_lower:
                return "log(n)"
        
        try:
            s = int(start)
            e = int(end)
            return str(e - s + 1)
        except (ValueError, TypeError):
            pass
        
        return "n"
    
    def _estimate_while_iterations(self, condition: str) -> str:
        """Estimate while loop iterations from condition."""
        if '/2' in condition or '//2' in condition or '* 2' in condition:
            return "log(n)"
        return "n"
    
    def _compute_result(self, code: str) -> AnalysisResult:
        """Compute final analysis result."""
        time_complexity = ComplexityClass.CONSTANT
        max_nesting = 0
        
        # Compute from loops
        for loop in self.loops:
            loop_complexity = self._compute_total_loop_complexity(loop)
            if ComplexityClass.compare(loop_complexity, time_complexity) > 0:
                time_complexity = loop_complexity
            max_nesting = max(max_nesting, self._get_max_nesting(loop))
        
        # Consider recursion
        if self.recursion_info:
            rec_complexity = self.recursion_info.complexity
            if ComplexityClass.compare(rec_complexity, time_complexity) > 0:
                time_complexity = rec_complexity
        
        # Compute space complexity
        space_complexity = self._compute_space_complexity()
        
        # Build factors list
        factors = self._build_factors()
        
        # Confidence based on analysis quality
        confidence = self._compute_confidence()
        
        return AnalysisResult(
            time_complexity=time_complexity,
            space_complexity=space_complexity,
            loops=self.loops,
            recursion=self.recursion_info,
            max_nesting_depth=max_nesting,
            confidence=confidence,
            factors=factors,
            raw_code=code
        )
    
    def _compute_total_loop_complexity(self, loop: LoopInfo) -> ComplexityClass:
        """Compute total complexity of a loop including nested loops."""
        result = loop.complexity
        
        for nested in loop.nested_loops:
            nested_complexity = self._compute_total_loop_complexity(nested)
            result = ComplexityClass.multiply(result, nested_complexity)
        
        return result
    
    def _get_max_nesting(self, loop: LoopInfo) -> int:
        """Get maximum nesting depth."""
        if not loop.nested_loops:
            return loop.nesting_level + 1
        return max(self._get_max_nesting(n) for n in loop.nested_loops)
    
    def _compute_space_complexity(self) -> ComplexityClass:
        """Compute space complexity."""
        if self.recursion_info:
            # Recursive = stack depth
            if "n/2" in self.recursion_info.reduction_pattern:
                return ComplexityClass.LOGARITHMIC
            else:
                return ComplexityClass.LINEAR
        
        return ComplexityClass.CONSTANT
    
    def _build_factors(self) -> List[ComplexityFactor]:
        """Build list of complexity factors."""
        factors = []
        
        for loop in self.loops:
            factors.append(ComplexityFactor(
                source=loop.get_description(),
                factor_type="loop",
                complexity=loop.complexity,
                iterations=loop.iterations,
                nesting_level=loop.nesting_level,
                location=f"line {loop.line_number}" if loop.line_number else None
            ))
        
        if self.recursion_info:
            factors.append(ComplexityFactor(
                source=self.recursion_info.get_description(),
                factor_type="recursion",
                complexity=self.recursion_info.complexity,
                iterations=self.recursion_info.recurrence
            ))
        
        return factors
    
    def _compute_confidence(self) -> float:
        """Compute confidence in the analysis."""
        confidence = 0.8  # Base confidence
        
        if self.loops or self.recursion_info:
            confidence = 0.9
        
        if self.recursion_info and self.loops:
            confidence = 0.85  # Mixed analysis is harder
        
        return confidence
