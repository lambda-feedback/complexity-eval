"""
Feedback Generator - Generates detailed human-readable feedback for complexity analysis.

This module provides clear, educational feedback explaining:
- What complexity was detected and why
- How loops and recursion contribute to complexity
- Step-by-step breakdown of the analysis
- Suggestions for improvement
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from ..schemas.complexity import ComplexityClass
from .complexity_analyzer import AnalysisResult, LoopInfo, RecursionInfo


class FeedbackLevel(str, Enum):
    """Level of detail for feedback."""
    BRIEF = "brief"      # Just the result
    STANDARD = "standard"  # Result with explanation
    DETAILED = "detailed"  # Full breakdown with examples


@dataclass
class FeedbackSection:
    """A section of feedback."""
    title: str
    content: str
    importance: str = "info"  # "info", "warning", "success", "error"


@dataclass
class DetailedFeedback:
    """Complete feedback for complexity analysis."""
    # Summary
    summary: str
    complexity_result: str
    
    # Breakdown sections
    sections: List[FeedbackSection] = field(default_factory=list)
    
    # Quick facts
    loop_count: int = 0
    max_nesting: int = 0
    has_recursion: bool = False
    
    # Educational content
    complexity_explanation: str = ""
    real_world_example: str = ""
    
    # Suggestions
    suggestions: List[str] = field(default_factory=list)
    
    # Confidence
    confidence_note: str = ""
    
    def to_string(self, level: FeedbackLevel = FeedbackLevel.STANDARD) -> str:
        """Convert feedback to formatted string."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("COMPLEXITY ANALYSIS RESULT")
        lines.append("=" * 60)
        lines.append("")
        
        # Summary
        lines.append(f"Time Complexity: {self.complexity_result}")
        lines.append("")
        lines.append(self.summary)
        lines.append("")
        
        if level == FeedbackLevel.BRIEF:
            return "\n".join(lines)
        
        # Sections
        for section in self.sections:
            lines.append("-" * 40)
            lines.append(f"[{section.importance.upper()}] {section.title}")
            lines.append("-" * 40)
            lines.append(section.content)
            lines.append("")
        
        # Complexity explanation
        if self.complexity_explanation:
            lines.append("-" * 40)
            lines.append("What does this mean?")
            lines.append("-" * 40)
            lines.append(self.complexity_explanation)
            lines.append("")
        
        if level == FeedbackLevel.STANDARD:
            # Add suggestions
            if self.suggestions:
                lines.append("-" * 40)
                lines.append("Suggestions")
                lines.append("-" * 40)
                for i, suggestion in enumerate(self.suggestions, 1):
                    lines.append(f"  {i}. {suggestion}")
                lines.append("")
            return "\n".join(lines)
        
        # Detailed level - add everything
        if self.real_world_example:
            lines.append("-" * 40)
            lines.append("Real-World Example")
            lines.append("-" * 40)
            lines.append(self.real_world_example)
            lines.append("")
        
        if self.suggestions:
            lines.append("-" * 40)
            lines.append("Optimization Suggestions")
            lines.append("-" * 40)
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {suggestion}")
            lines.append("")
        
        if self.confidence_note:
            lines.append("-" * 40)
            lines.append("Analysis Confidence")
            lines.append("-" * 40)
            lines.append(self.confidence_note)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": self.summary,
            "complexity": self.complexity_result,
            "sections": [
                {"title": s.title, "content": s.content, "importance": s.importance}
                for s in self.sections
            ],
            "stats": {
                "loop_count": self.loop_count,
                "max_nesting": self.max_nesting,
                "has_recursion": self.has_recursion
            },
            "explanation": self.complexity_explanation,
            "real_world_example": self.real_world_example,
            "suggestions": self.suggestions,
            "confidence": self.confidence_note
        }


class FeedbackGenerator:
    """
    Generates detailed, human-readable feedback for complexity analysis results.
    
    The feedback is designed to be educational and help students understand:
    - Why their code has a particular complexity
    - How different constructs contribute to complexity
    - How to improve their algorithms
    """
    
    # Complexity descriptions and examples
    COMPLEXITY_INFO = {
        ComplexityClass.CONSTANT: {
            "name": "Constant Time",
            "symbol": "O(1)",
            "description": "The algorithm takes the same amount of time regardless of input size.",
            "example": "Accessing an array element by index, checking if a number is even/odd.",
            "growth": "No growth - always the same time."
        },
        ComplexityClass.LOGARITHMIC: {
            "name": "Logarithmic Time",
            "symbol": "O(log n)",
            "description": "The algorithm reduces the problem size by half (or a constant factor) at each step.",
            "example": "Binary search - each comparison eliminates half the remaining elements.",
            "growth": "Very slow growth. Doubling n adds only one extra step."
        },
        ComplexityClass.SQRT: {
            "name": "Square Root Time",
            "symbol": "O(√n)",
            "description": "The algorithm's time grows with the square root of the input size.",
            "example": "Checking if a number is prime by testing divisors up to √n.",
            "growth": "Slow growth. For n=1,000,000, only ~1,000 operations."
        },
        ComplexityClass.LINEAR: {
            "name": "Linear Time",
            "symbol": "O(n)",
            "description": "The algorithm examines each element once. Time grows proportionally with input size.",
            "example": "Finding the maximum element in an unsorted array - must check every element.",
            "growth": "Direct growth. Doubling n doubles the time."
        },
        ComplexityClass.LINEARITHMIC: {
            "name": "Linearithmic Time",
            "symbol": "O(n log n)",
            "description": "Common in efficient sorting algorithms that use divide-and-conquer.",
            "example": "Merge sort, quicksort (average case), heapsort.",
            "growth": "Slightly faster than linear growth. Very efficient for sorting."
        },
        ComplexityClass.QUADRATIC: {
            "name": "Quadratic Time",
            "symbol": "O(n²)",
            "description": "Usually caused by nested loops where both iterate over the input.",
            "example": "Bubble sort, selection sort, comparing all pairs of elements.",
            "growth": "Doubling n quadruples the time. Becomes slow for large inputs."
        },
        ComplexityClass.CUBIC: {
            "name": "Cubic Time",
            "symbol": "O(n³)",
            "description": "Often seen with triple nested loops, common in matrix operations.",
            "example": "Standard matrix multiplication, Floyd-Warshall algorithm.",
            "growth": "Doubling n increases time by 8x. Impractical for large n."
        },
        ComplexityClass.POLYNOMIAL: {
            "name": "Polynomial Time",
            "symbol": "O(n^k)",
            "description": "Time grows as n raised to some power k > 3.",
            "example": "Some dynamic programming solutions, brute-force algorithms.",
            "growth": "Very fast growth. Only practical for small inputs."
        },
        ComplexityClass.EXPONENTIAL: {
            "name": "Exponential Time",
            "symbol": "O(2^n)",
            "description": "The algorithm doubles its work for each additional input element. Often from recursive solutions that branch multiple times.",
            "example": "Naive recursive Fibonacci, generating all subsets, brute-force combinatorial problems.",
            "growth": "Extremely fast growth! Adding one element doubles the time."
        },
        ComplexityClass.FACTORIAL: {
            "name": "Factorial Time",
            "symbol": "O(n!)",
            "description": "The worst common complexity class. Generates all permutations.",
            "example": "Brute-force traveling salesman, generating all permutations.",
            "growth": "Astronomically fast growth. n=20 has more operations than atoms in the universe!"
        }
    }
    
    def __init__(self):
        pass
    
    def generate(self, result: AnalysisResult, 
                 level: FeedbackLevel = FeedbackLevel.STANDARD) -> DetailedFeedback:
        """
        Generate detailed feedback from analysis result.
        
        Args:
            result: The complexity analysis result
            level: Level of detail for feedback
            
        Returns:
            DetailedFeedback object with complete feedback
        """
        feedback = DetailedFeedback(
            summary=self._generate_summary(result),
            complexity_result=result.time_complexity.value,
            loop_count=len(result.loops),
            max_nesting=result.max_nesting_depth,
            has_recursion=result.recursion is not None
        )
        
        # Add analysis sections
        feedback.sections = self._generate_sections(result)
        
        # Add complexity explanation
        feedback.complexity_explanation = self._explain_complexity(result.time_complexity)
        
        # Add real-world example
        feedback.real_world_example = self._get_real_world_example(result.time_complexity)
        
        # Generate suggestions
        feedback.suggestions = self._generate_suggestions(result)
        
        # Add confidence note
        feedback.confidence_note = self._generate_confidence_note(result)
        
        return feedback
    
    def _generate_summary(self, result: AnalysisResult) -> str:
        """Generate a summary of the analysis."""
        complexity = result.time_complexity
        info = self.COMPLEXITY_INFO.get(complexity, {})
        name = info.get("name", complexity.value)
        
        parts = []
        parts.append(f"Your algorithm has {name} complexity: {complexity.value}.")
        
        # Add context based on what was found
        if result.recursion:
            rec = result.recursion
            if rec.branching_factor == 1:
                parts.append(f"This is due to recursion in {rec.function_name}() with {rec.reduction_pattern} reduction.")
            else:
                parts.append(f"This is due to {rec.branching_factor}-way recursion in {rec.function_name}().")
        elif result.loops:
            if result.max_nesting_depth > 1:
                parts.append(f"This is caused by {result.max_nesting_depth} levels of nested loops.")
            elif len(result.loops) == 1:
                loop = result.loops[0]
                parts.append(f"This is due to a {loop.loop_type.upper()} loop that iterates {loop.iterations} times.")
            else:
                parts.append(f"This is due to {len(result.loops)} loops in your code.")
        else:
            parts.append("No loops or recursion detected - the algorithm runs in constant time.")
        
        return " ".join(parts)
    
    def _generate_sections(self, result: AnalysisResult) -> List[FeedbackSection]:
        """Generate detailed analysis sections."""
        sections = []
        
        # Loop analysis section
        if result.loops:
            sections.append(self._generate_loop_section(result))
        
        # Recursion analysis section
        if result.recursion:
            sections.append(self._generate_recursion_section(result.recursion))
        
        # Nesting analysis
        if result.max_nesting_depth > 1:
            sections.append(self._generate_nesting_section(result))
        
        # Complexity breakdown
        sections.append(self._generate_breakdown_section(result))
        
        return sections
    
    def _generate_loop_section(self, result: AnalysisResult) -> FeedbackSection:
        """Generate section explaining loop analysis."""
        lines = []
        
        for i, loop in enumerate(result.loops, 1):
            lines.append(f"Loop {i}: {loop.get_description()}")
            lines.append(f"  - Iterations: {loop.iterations}")
            lines.append(f"  - Complexity contribution: {loop.complexity.value}")
            
            if loop.nested_loops:
                lines.append(f"  - Contains {len(loop.nested_loops)} nested loop(s):")
                for j, nested in enumerate(loop.nested_loops, 1):
                    lines.append(f"    {j}. {nested.get_description()} - {nested.iterations} iterations")
            
            lines.append("")
        
        # Explain how loops combine
        if len(result.loops) > 1 or any(l.nested_loops for l in result.loops):
            lines.append("How loops combine:")
            lines.append("  - Nested loops: complexities MULTIPLY (O(n) × O(n) = O(n²))")
            lines.append("  - Sequential loops: take the MAXIMUM (O(n) + O(n) = O(n))")
        
        return FeedbackSection(
            title="Loop Analysis",
            content="\n".join(lines),
            importance="info"
        )
    
    def _generate_recursion_section(self, rec: RecursionInfo) -> FeedbackSection:
        """Generate section explaining recursion analysis."""
        lines = []
        
        lines.append(f"Recursive Function: {rec.function_name}()")
        lines.append(f"  - Number of recursive calls: {rec.branching_factor} per invocation")
        lines.append(f"  - Problem reduction: {rec.reduction_pattern}")
        lines.append(f"  - Recurrence relation: {rec.recurrence}")
        lines.append("")
        
        # Explain the recursion pattern
        if rec.branching_factor == 1:
            if "n/2" in rec.reduction_pattern:
                lines.append("Pattern: Single recursive call with halving (like binary search)")
                lines.append("Each call processes half the remaining problem.")
                lines.append("Total calls: log₂(n), giving O(log n) complexity.")
            else:
                lines.append("Pattern: Linear recursion (like factorial)")
                lines.append("Each call reduces problem by 1, requiring n calls total.")
                lines.append("Result: O(n) complexity.")
        elif rec.branching_factor == 2:
            if "n/2" in rec.reduction_pattern:
                lines.append("Pattern: Divide-and-conquer (like merge sort)")
                lines.append("Problem splits in half, but both halves are processed.")
                lines.append("Using Master Theorem: O(n log n) complexity.")
            else:
                lines.append("Pattern: Binary recursion with linear reduction (like naive Fibonacci)")
                lines.append("WARNING: This creates an exponential number of calls!")
                lines.append("Each call spawns 2 more, leading to O(2^n) complexity.")
        else:
            lines.append(f"Pattern: {rec.branching_factor}-way branching recursion")
            lines.append("Multiple recursive calls lead to exponential growth.")
        
        return FeedbackSection(
            title="Recursion Analysis",
            content="\n".join(lines),
            importance="warning" if rec.complexity in [ComplexityClass.EXPONENTIAL, ComplexityClass.FACTORIAL] else "info"
        )
    
    def _generate_nesting_section(self, result: AnalysisResult) -> FeedbackSection:
        """Generate section about loop nesting."""
        depth = result.max_nesting_depth
        
        lines = []
        lines.append(f"Maximum nesting depth: {depth} levels")
        lines.append("")
        
        # Explain impact
        if depth == 2:
            lines.append("Two nested loops typically result in O(n²) quadratic complexity.")
            lines.append("Example: for i in 1..n: for j in 1..n: → n × n = n² operations")
        elif depth == 3:
            lines.append("Three nested loops result in O(n³) cubic complexity.")
            lines.append("This grows VERY quickly - use with caution for large inputs!")
        elif depth > 3:
            lines.append(f"{depth} levels of nesting creates O(n^{depth}) complexity.")
            lines.append("This is extremely slow for large inputs!")
            lines.append("Consider if all these nested loops are necessary.")
        
        return FeedbackSection(
            title="Nesting Impact",
            content="\n".join(lines),
            importance="warning" if depth >= 3 else "info"
        )
    
    def _generate_breakdown_section(self, result: AnalysisResult) -> FeedbackSection:
        """Generate complexity breakdown section."""
        lines = []
        lines.append("Step-by-step complexity calculation:")
        lines.append("")
        
        if not result.loops and not result.recursion:
            lines.append("1. No loops or recursion detected")
            lines.append("2. Only simple operations (assignments, comparisons)")
            lines.append("3. Each operation takes constant time O(1)")
            lines.append("→ Final complexity: O(1)")
        elif result.recursion and not result.loops:
            rec = result.recursion
            lines.append(f"1. Found recursive function: {rec.function_name}()")
            lines.append(f"2. Recurrence: {rec.recurrence}")
            if "n/2" in rec.reduction_pattern:
                lines.append("3. Problem halves each call → logarithmic depth")
                if rec.branching_factor == 1:
                    lines.append("4. Single recursive call per level")
                    lines.append("→ Final complexity: O(log n)")
                else:
                    lines.append(f"4. {rec.branching_factor} calls per level")
                    lines.append("5. Apply Master Theorem")
                    lines.append(f"→ Final complexity: {rec.complexity.value}")
            else:
                lines.append(f"3. Linear reduction ({rec.reduction_pattern}) → n levels deep")
                if rec.branching_factor == 1:
                    lines.append("→ Final complexity: O(n)")
                else:
                    lines.append(f"4. {rec.branching_factor}^n calls total")
                    lines.append(f"→ Final complexity: {rec.complexity.value}")
        else:
            # Loop-based complexity
            step = 1
            current = ComplexityClass.CONSTANT
            
            for loop in result.loops:
                lines.append(f"{step}. {loop.get_description()}")
                lines.append(f"   Iterates {loop.iterations} times → {loop.complexity.value}")
                step += 1
                
                for nested in loop.nested_loops:
                    lines.append(f"{step}. Nested: {nested.get_description()}")
                    lines.append(f"   Iterates {nested.iterations} times → {nested.complexity.value}")
                    lines.append(f"   Nested inside previous loop → multiply complexities")
                    step += 1
            
            lines.append("")
            lines.append(f"→ Final time complexity: {result.time_complexity.value}")
        
        return FeedbackSection(
            title="Complexity Calculation",
            content="\n".join(lines),
            importance="success"
        )
    
    def _explain_complexity(self, complexity: ComplexityClass) -> str:
        """Get explanation for a complexity class."""
        info = self.COMPLEXITY_INFO.get(complexity, {})
        
        lines = []
        lines.append(f"{info.get('name', complexity.value)} - {info.get('symbol', '')}")
        lines.append("")
        lines.append(info.get('description', ''))
        lines.append("")
        lines.append(f"Growth rate: {info.get('growth', '')}")
        
        return "\n".join(lines)
    
    def _get_real_world_example(self, complexity: ComplexityClass) -> str:
        """Get real-world example for complexity class."""
        info = self.COMPLEXITY_INFO.get(complexity, {})
        example = info.get('example', '')
        
        if not example:
            return ""
        
        lines = []
        lines.append(f"Common algorithms with {complexity.value}:")
        lines.append(f"  {example}")
        
        # Add scale examples
        if complexity == ComplexityClass.LINEAR:
            lines.append("")
            lines.append("At different scales:")
            lines.append("  n=100: ~100 operations")
            lines.append("  n=1,000: ~1,000 operations")
            lines.append("  n=1,000,000: ~1,000,000 operations")
        elif complexity == ComplexityClass.QUADRATIC:
            lines.append("")
            lines.append("At different scales:")
            lines.append("  n=100: ~10,000 operations")
            lines.append("  n=1,000: ~1,000,000 operations")
            lines.append("  n=10,000: ~100,000,000 operations (slow!)")
        elif complexity == ComplexityClass.LOGARITHMIC:
            lines.append("")
            lines.append("At different scales:")
            lines.append("  n=100: ~7 operations")
            lines.append("  n=1,000: ~10 operations")
            lines.append("  n=1,000,000: ~20 operations")
        elif complexity == ComplexityClass.EXPONENTIAL:
            lines.append("")
            lines.append("At different scales:")
            lines.append("  n=10: ~1,024 operations")
            lines.append("  n=20: ~1,048,576 operations")
            lines.append("  n=30: ~1,073,741,824 operations (very slow!)")
            lines.append("  n=50: More than age of universe in nanoseconds!")
        
        return "\n".join(lines)
    
    def _generate_suggestions(self, result: AnalysisResult) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        complexity = result.time_complexity
        
        # Suggestions based on complexity
        if complexity == ComplexityClass.EXPONENTIAL:
            suggestions.append("Consider using dynamic programming to avoid redundant calculations.")
            suggestions.append("Memoization can often convert O(2^n) to O(n) or O(n²).")
            if result.recursion and result.recursion.branching_factor >= 2:
                suggestions.append(f"Your {result.recursion.function_name}() has overlapping subproblems - cache results!")
        
        if complexity == ComplexityClass.QUADRATIC and result.max_nesting_depth >= 2:
            suggestions.append("Look for ways to eliminate one of the nested loops.")
            suggestions.append("Consider using a hash table/dictionary to replace the inner loop with O(1) lookup.")
            suggestions.append("Sorting first might enable a more efficient algorithm.")
        
        if complexity == ComplexityClass.CUBIC:
            suggestions.append("Triple nested loops are usually avoidable with better algorithms.")
            suggestions.append("For matrix operations, consider Strassen's algorithm or library functions.")
        
        # Suggestions based on structure
        if result.max_nesting_depth >= 3:
            suggestions.append("Deep nesting makes code hard to read and slow. Consider refactoring.")
        
        if result.recursion and not result.loops:
            if result.recursion.branching_factor == 1:
                suggestions.append("This recursion could be converted to a simple loop for better performance.")
        
        # Generic good practices
        if not suggestions:
            if complexity in [ComplexityClass.CONSTANT, ComplexityClass.LOGARITHMIC]:
                suggestions.append("Your algorithm is already very efficient!")
            elif complexity == ComplexityClass.LINEAR:
                suggestions.append("Linear complexity is often optimal for problems requiring examination of all input.")
            elif complexity == ComplexityClass.LINEARITHMIC:
                suggestions.append("O(n log n) is optimal for comparison-based sorting. Good job!")
        
        return suggestions
    
    def _generate_confidence_note(self, result: AnalysisResult) -> str:
        """Generate note about analysis confidence."""
        confidence = result.confidence
        
        if confidence >= 0.9:
            return "High confidence: Clear loop/recursion patterns detected."
        elif confidence >= 0.7:
            return "Moderate confidence: Analysis based on detected patterns."
        else:
            return "Lower confidence: Some constructs may not have been fully analyzed."
    
    def format_for_student(self, result: AnalysisResult) -> str:
        """
        Format feedback specifically for student learning.
        
        Returns a clear, educational explanation suitable for students
        learning about algorithm complexity.
        """
        feedback = self.generate(result, FeedbackLevel.DETAILED)
        return feedback.to_string(FeedbackLevel.DETAILED)
    
    def format_brief(self, result: AnalysisResult) -> str:
        """Get brief one-line feedback."""
        feedback = self.generate(result, FeedbackLevel.BRIEF)
        return f"Time Complexity: {result.time_complexity.value} - {feedback.summary}"
