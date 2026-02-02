"""
Preview function for pseudocode complexity analysis.

This module provides a preview function that:
1. Validates that the pseudocode can be parsed
2. Analyzes the complexity of the code
3. Returns a preview showing the detected structure and complexity

The preview helps students verify their code before submission.
"""

from typing import Any, Dict, List, Optional
from lf_toolkit.preview import Result, Params, Preview

from .parser.parser import PseudocodeParser
from .analyzer.complexity_analyzer import ComplexityAnalyzer


def preview_function(response: Any, params: Params) -> Result:
    """
    Preview a student's pseudocode submission.

    This function parses and analyzes the pseudocode to provide immediate
    feedback on whether the code is valid and what complexity is detected.

    Args:
        response: The student's pseudocode (string)
        params: Additional parameters for configuration

    Returns:
        Result containing a Preview with:
        - latex: Formatted complexity result
        - feedback: Detailed feedback about parsing and analysis
    """
    try:
        # Extract pseudocode from response
        if isinstance(response, dict):
            pseudocode = response.get('pseudocode', response.get('code', ''))
        elif isinstance(response, str):
            pseudocode = response
        else:
            return Result(preview=Preview(
                feedback="Invalid response format. Please provide pseudocode as a string."
            ))

        if not pseudocode or not pseudocode.strip():
            return Result(preview=Preview(
                feedback="Please enter your pseudocode to see a preview."
            ))

        # Parse the pseudocode
        parser = PseudocodeParser()
        parse_result = parser.parse(pseudocode)

        if not parse_result.success:
            error_msg = "Failed to parse pseudocode."
            if parse_result.errors:
                error_msg += "\n\nErrors:\n" + "\n".join(f"- {e}" for e in parse_result.errors)
            if parse_result.warnings:
                error_msg += "\n\nWarnings:\n" + "\n".join(f"- {w}" for w in parse_result.warnings)
            return Result(preview=Preview(feedback=error_msg))

        # Analyze complexity
        analyzer = ComplexityAnalyzer()
        analysis = analyzer.analyze(pseudocode, parse_result.ast)

        # Detect structure
        structure = parser.detect_structure(pseudocode)

        # Generate preview content
        preview_content = _generate_preview_content(analysis, structure, parse_result)

        return Result(preview=Preview(
            latex=f"\\text{{Time Complexity: }} {_latex_complexity(analysis.time_complexity.value)}",
            feedback=preview_content
        ))

    except Exception as e:
        return Result(preview=Preview(
            feedback=f"An error occurred during preview: {str(e)}"
        ))


def _generate_preview_content(analysis, structure: Dict, parse_result) -> str:
    """Generate detailed preview content."""
    lines = []

    # Header
    lines.append("=" * 50)
    lines.append("PSEUDOCODE ANALYSIS PREVIEW")
    lines.append("=" * 50)
    lines.append("")

    # Parsing status
    lines.append("✓ Parsing: Successful")
    if parse_result.warnings:
        for w in parse_result.warnings[:3]:  # Show max 3 warnings
            lines.append(f"  ⚠ {w}")
    lines.append("")

    # Detected structure
    lines.append("-" * 50)
    lines.append("DETECTED STRUCTURE")
    lines.append("-" * 50)

    structure_items = []
    if structure.get('has_loops'):
        loop_count = structure.get('loop_count', 0)
        structure_items.append(f"• Loops: {loop_count}")
    if structure.get('has_nested_loops'):
        max_nesting = structure.get('max_nesting', 0)
        structure_items.append(f"• Nested loops: Yes (depth {max_nesting})")
    if structure.get('has_recursion'):
        structure_items.append("• Recursion: Yes")
    if structure.get('has_conditionals'):
        structure_items.append("• Conditionals: Yes")

    if structure_items:
        lines.extend(structure_items)
    else:
        lines.append("• No loops or recursion detected (constant time)")
    lines.append("")

    # Complexity result
    lines.append("-" * 50)
    lines.append("COMPLEXITY ANALYSIS")
    lines.append("-" * 50)
    lines.append(f"Time Complexity:  {analysis.time_complexity.value}")
    lines.append(f"Space Complexity: {analysis.space_complexity.value}")
    lines.append("")

    # Loop details
    if analysis.loops:
        lines.append("-" * 50)
        lines.append("LOOP DETAILS")
        lines.append("-" * 50)
        for i, loop in enumerate(analysis.loops, 1):
            lines.append(f"{i}. {loop.get_description()}")
            lines.append(f"   Iterations: {loop.iterations}")
            lines.append(f"   Contribution: {loop.complexity.value}")
            if loop.nested_loops:
                for j, nested in enumerate(loop.nested_loops, 1):
                    lines.append(f"   └─ Nested {j}: {nested.get_description()} ({nested.complexity.value})")
        lines.append("")

    # Recursion details
    if analysis.recursion:
        rec = analysis.recursion
        lines.append("-" * 50)
        lines.append("RECURSION DETAILS")
        lines.append("-" * 50)
        lines.append(f"Function: {rec.function_name}()")
        lines.append(f"Branching factor: {rec.branching_factor}")
        lines.append(f"Reduction pattern: {rec.reduction_pattern}")
        lines.append(f"Recurrence: {rec.recurrence}")
        lines.append("")

    # Confidence
    lines.append("-" * 50)
    confidence_pct = int(analysis.confidence * 100)
    lines.append(f"Analysis confidence: {confidence_pct}%")

    return "\n".join(lines)


def _latex_complexity(complexity: str) -> str:
    """Convert complexity to LaTeX format."""
    replacements = {
        "O(1)": "O(1)",
        "O(log n)": "O(\\log n)",
        "O(√n)": "O(\\sqrt{n})",
        "O(n)": "O(n)",
        "O(n log n)": "O(n \\log n)",
        "O(n²)": "O(n^2)",
        "O(n³)": "O(n^3)",
        "O(n^k)": "O(n^k)",
        "O(2^n)": "O(2^n)",
        "O(n!)": "O(n!)",
    }
    return replacements.get(complexity, complexity)
