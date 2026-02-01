"""
Main Parser module for pseudocode.

This module provides the PseudocodeParser class that combines:
- Preprocessing (syntax normalization)
- Lark parsing (grammar-based parsing) 
- Fallback parsing (pattern-based for when grammar fails)
"""

from typing import Tuple, List, Optional, Any
from dataclasses import dataclass
import re

from .preprocessor import Preprocessor, PreprocessorConfig
from .grammar import PSEUDOCODE_GRAMMAR, SIMPLIFIED_GRAMMAR

from ..schemas.ast_nodes import (
    ProgramNode,
    FunctionNode,
    BlockNode,
    LoopNode,
    ConditionalNode,
    VariableNode,
    LiteralNode,
    LoopType,
)
from ..schemas.output_schema import ParseResult


class ParseError(Exception):
    """Exception raised when parsing fails."""
    
    def __init__(self, message: str, line: int = 0, column: int = 0, 
                 context: Optional[str] = None):
        self.message = message
        self.line = line
        self.column = column
        self.context = context
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        msg = self.message
        if self.line:
            msg += f" at line {self.line}"
            if self.column:
                msg += f", column {self.column}"
        if self.context:
            msg += f"\n  Context: {self.context}"
        return msg


@dataclass
class ParserConfig:
    """Configuration for the parser."""
    use_indentation: bool = True
    strict_mode: bool = False
    max_errors: int = 10
    timeout: float = 5.0


class PseudocodeParser:
    """
    Main parser for pseudocode.
    
    Uses pattern-based fallback parsing for robustness since full grammar
    parsing of arbitrary pseudocode is challenging.
    
    Usage:
        parser = PseudocodeParser()
        result = parser.parse("FOR i = 1 TO n DO\\n  print(i)")
        if result.success:
            ast = result.ast
    """
    
    def __init__(self, config: Optional[ParserConfig] = None,
                 preprocessor_config: Optional[PreprocessorConfig] = None):
        self.config = config or ParserConfig()
        self.preprocessor = Preprocessor(preprocessor_config)
        self._lark_available = False
        self._parser = None
        
        # Try to initialize Lark parser, but don't fail if it doesn't work
        self._try_init_lark()
    
    def _try_init_lark(self):
        """Try to initialize Lark parser, gracefully handle failure."""
        try:
            from lark import Lark
            from lark.indenter import Indenter
            
            class PseudocodeIndenter(Indenter):
                NL_type = '_NL'
                OPEN_PAREN_types = []
                CLOSE_PAREN_types = []
                INDENT_type = '_INDENT'
                DEDENT_type = '_DEDENT'
                tab_len = 4
            
            self._parser = Lark(
                PSEUDOCODE_GRAMMAR,
                parser='lalr',
                postlex=PseudocodeIndenter() if self.config.use_indentation else None,
                propagate_positions=True,
                maybe_placeholders=True,
            )
            self._lark_available = True
        except Exception as e:
            # Lark parsing not available, will use fallback
            self._lark_available = False
            self._parser = None
    
    def parse(self, code: str) -> ParseResult:
        """
        Parse pseudocode and return a ParseResult.
        
        Args:
            code: The pseudocode to parse
            
        Returns:
            ParseResult with success status, AST (if successful),
            errors, and warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        # Step 1: Preprocess
        try:
            normalized_code, preprocess_warnings = self.preprocessor.preprocess(code)
            warnings.extend(preprocess_warnings)
        except Exception as e:
            errors.append(f"Preprocessing failed: {str(e)}")
            normalized_code = code
        
        # Step 2: Try Lark parsing if available and not in strict mode that requires it
        ast = None
        if self._lark_available and self._parser:
            try:
                tree = self._parser.parse(normalized_code)
                # For now, we don't fully transform - just indicate success
                ast = self._build_ast_from_tree(tree, normalized_code)
            except Exception as e:
                if self.config.strict_mode:
                    errors.append(f"Parse error: {str(e)}")
                else:
                    warnings.append(f"Full parsing failed, using fallback: {str(e)[:100]}")
        
        # Step 3: Use fallback pattern-based parsing
        if ast is None and not self.config.strict_mode:
            try:
                ast = self._parse_fallback(normalized_code)
            except Exception as e:
                errors.append(f"Fallback parsing failed: {str(e)}")
        
        return ParseResult(
            success=ast is not None,
            ast=ast,
            errors=errors,
            warnings=warnings,
            normalized_code=normalized_code
        )
    
    def _build_ast_from_tree(self, tree, code: str) -> Optional[ProgramNode]:
        """Build AST from Lark parse tree."""
        # Simplified AST building - mainly for structure detection
        try:
            functions = []
            statements = []
            
            for child in tree.children:
                if child is None:
                    continue
                if hasattr(child, 'data'):
                    if child.data == 'function_def':
                        func = self._extract_function(child)
                        if func:
                            functions.append(func)
                    elif child.data in ('for_stmt', 'while_stmt', 'if_stmt', 'repeat_stmt'):
                        stmt = self._extract_statement(child)
                        if stmt:
                            statements.append(stmt)
            
            global_block = BlockNode(statements=statements) if statements else None
            return ProgramNode(functions=functions, global_statements=global_block)
        except Exception:
            return None
    
    def _extract_function(self, node) -> Optional[FunctionNode]:
        """Extract function from parse tree node."""
        try:
            name = ""
            for child in node.children:
                if hasattr(child, 'type') and child.type == 'NAME':
                    name = str(child)
                    break
            return FunctionNode(name=name, parameters=[], body=None)
        except Exception:
            return None
    
    def _extract_statement(self, node) -> Optional[Any]:
        """Extract statement from parse tree node."""
        try:
            if node.data == 'for_stmt':
                return LoopNode(loop_type=LoopType.FOR, body=BlockNode(statements=[]))
            elif node.data == 'while_stmt':
                return LoopNode(loop_type=LoopType.WHILE, body=BlockNode(statements=[]))
            elif node.data == 'if_stmt':
                return ConditionalNode(then_branch=BlockNode(statements=[]))
        except Exception:
            pass
        return None
    
    def _parse_fallback(self, code: str) -> ProgramNode:
        """
        Fallback parsing using pattern detection.
        
        This method uses regex patterns to detect loops, conditionals,
        and functions when the full grammar fails.
        """
        functions: List[FunctionNode] = []
        statements = []
        
        lines = code.split('\n')
        indent_unit = self.preprocessor.detect_indentation_style(code)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            if not stripped:
                i += 1
                continue
            
            indent_level = self.preprocessor.get_indent_level(line, indent_unit)
            
            # Detect function definitions (with optional { at end)
            func_match = re.match(
                r'^(function|algorithm|procedure|def)\s+(\w+)\s*\([^)]*\)\s*\{?',
                stripped, re.IGNORECASE
            )
            if func_match:
                func_name = func_match.group(2)
                body_lines, end_idx = self._collect_block(lines, i + 1, indent_level, indent_unit)
                func = FunctionNode(
                    name=func_name,
                    parameters=[],
                    body=self._parse_block_fallback(body_lines, indent_unit)
                )
                functions.append(func)
                i = end_idx
                continue
            
            # Detect loops
            loop_node = self._detect_loop(stripped, lines, i, indent_level, indent_unit)
            if loop_node:
                statements.append(loop_node[0])
                i = loop_node[1]
                continue
            
            # Detect conditionals
            cond_node = self._detect_conditional(stripped, lines, i, indent_level, indent_unit)
            if cond_node:
                statements.append(cond_node[0])
                i = cond_node[1]
                continue
            
            # Other statement - skip
            i += 1
        
        global_block = BlockNode(statements=statements) if statements else None
        return ProgramNode(functions=functions, global_statements=global_block)
    
    def _collect_block(self, lines: List[str], start_idx: int,
                       base_indent: int, indent_unit: int) -> Tuple[List[str], int]:
        """Collect lines belonging to a block.

        Supports three block styles:
        1. Indentation-based (Python-like)
        2. END keyword-based (END IF, END FOR, etc.)
        3. Curly brace-based ({ ... })
        """
        block_lines = []
        i = start_idx
        brace_count = 0

        # Check if the block starts with an opening brace
        if i < len(lines):
            first_line = lines[i].strip()
            if first_line == '{' or first_line.endswith('{'):
                brace_count = 1
                # If just '{', skip it; if 'DO {', include content after
                if first_line == '{':
                    i += 1
                else:
                    # Remove the trailing brace from this line
                    block_lines.append(first_line[:-1].strip())
                    i += 1

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                block_lines.append(line)
                i += 1
                continue

            # Handle curly brace blocks
            if brace_count > 0:
                brace_count += stripped.count('{') - stripped.count('}')
                if brace_count <= 0:
                    # Remove trailing } if present
                    if stripped == '}':
                        i += 1
                    else:
                        block_lines.append(stripped.rstrip('}').strip())
                        i += 1
                    break
                block_lines.append(line)
                i += 1
                continue

            current_indent = self.preprocessor.get_indent_level(line, indent_unit)

            # End markers (keyword-based)
            if re.match(r'^(end\b|endif\b|endfor\b|endwhile\b|done\b|\})', stripped, re.IGNORECASE):
                i += 1
                break

            if current_indent <= base_indent and i > start_idx:
                break

            block_lines.append(line)
            i += 1

        return block_lines, i
    
    def _parse_block_fallback(self, lines: List[str], indent_unit: int) -> BlockNode:
        """Parse a block of lines into a BlockNode."""
        statements = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            if not stripped:
                i += 1
                continue
            
            indent_level = self.preprocessor.get_indent_level(line, indent_unit)
            
            loop_node = self._detect_loop(stripped, lines, i, indent_level, indent_unit)
            if loop_node:
                statements.append(loop_node[0])
                i = loop_node[1]
                continue
            
            cond_node = self._detect_conditional(stripped, lines, i, indent_level, indent_unit)
            if cond_node:
                statements.append(cond_node[0])
                i = cond_node[1]
                continue
            
            i += 1
        
        return BlockNode(statements=statements)
    
    def _detect_loop(self, line: str, lines: List[str], idx: int,
                     indent_level: int, indent_unit: int) -> Optional[Tuple[LoopNode, int]]:
        """Detect and parse a loop."""
        
        # FOR loop: for i = 1 to n (with optional DO or { at end)
        for_match = re.match(
            r'for\s+(\w+)\s*[=:]\s*(\w+)\s+to\s+(\w+)(?:\s+(?:do|step\s+\w+))?\s*\{?',
            line, re.IGNORECASE
        )
        if for_match:
            iterator = VariableNode(name=for_match.group(1))
            start = self._parse_simple_expr(for_match.group(2))
            end = self._parse_simple_expr(for_match.group(3))
            
            body_lines, next_idx = self._collect_block(lines, idx + 1, indent_level, indent_unit)
            body = self._parse_block_fallback(body_lines, indent_unit)
            estimated = self._estimate_iterations(for_match.group(2), for_match.group(3))
            
            return LoopNode(
                loop_type=LoopType.FOR,
                iterator=iterator,
                start=start,
                end=end,
                body=body,
                estimated_iterations=estimated,
                nesting_level=indent_level
            ), next_idx
        
        # WHILE loop (with optional DO or { at end)
        while_match = re.match(r'while\s+(.+?)(?:\s+do)?\s*\{?$', line, re.IGNORECASE)
        if while_match:
            body_lines, next_idx = self._collect_block(lines, idx + 1, indent_level, indent_unit)
            body = self._parse_block_fallback(body_lines, indent_unit)
            
            return LoopNode(
                loop_type=LoopType.WHILE,
                body=body,
                estimated_iterations="unknown",
                nesting_level=indent_level
            ), next_idx
        
        # FOR EACH loop (with optional { at end)
        foreach_match = re.match(
            r'for\s+(?:each\s+)?(\w+)\s+in\s+(\w+)(?:\s+do)?\s*\{?',
            line, re.IGNORECASE
        )
        if foreach_match:
            iterator = VariableNode(name=foreach_match.group(1))
            collection = VariableNode(name=foreach_match.group(2))
            
            body_lines, next_idx = self._collect_block(lines, idx + 1, indent_level, indent_unit)
            body = self._parse_block_fallback(body_lines, indent_unit)
            
            return LoopNode(
                loop_type=LoopType.FOR_EACH,
                iterator=iterator,
                collection=collection,
                body=body,
                estimated_iterations="n",
                nesting_level=indent_level
            ), next_idx
        
        # REPEAT loop  
        repeat_match = re.match(r'repeat\b', line, re.IGNORECASE)
        if repeat_match:
            body_lines, next_idx = self._collect_block(lines, idx + 1, indent_level, indent_unit)
            body = self._parse_block_fallback(body_lines, indent_unit)
            
            return LoopNode(
                loop_type=LoopType.REPEAT_UNTIL,
                body=body,
                estimated_iterations="unknown",
                nesting_level=indent_level
            ), next_idx
        
        return None
    
    def _detect_conditional(self, line: str, lines: List[str], idx: int,
                           indent_level: int, indent_unit: int) -> Optional[Tuple[ConditionalNode, int]]:
        """Detect and parse a conditional."""
        if_match = re.match(r'if\s+(.+?)(?:\s+then)?\s*\{?$', line, re.IGNORECASE)
        if if_match:
            body_lines, next_idx = self._collect_block(lines, idx + 1, indent_level, indent_unit)
            then_branch = self._parse_block_fallback(body_lines, indent_unit)
            
            else_branch = None
            if next_idx < len(lines):
                else_line = lines[next_idx].strip().lower()
                if else_line.startswith('else'):
                    else_lines, next_idx = self._collect_block(lines, next_idx + 1, indent_level, indent_unit)
                    else_branch = self._parse_block_fallback(else_lines, indent_unit)
            
            return ConditionalNode(
                then_branch=then_branch,
                else_branch=else_branch
            ), next_idx
        
        return None
    
    def _parse_simple_expr(self, expr_str: str) -> Any:
        """Parse a simple expression."""
        expr_str = expr_str.strip()
        try:
            if '.' in expr_str:
                return LiteralNode(value=float(expr_str), literal_type="float")
            return LiteralNode(value=int(expr_str), literal_type="int")
        except ValueError:
            return VariableNode(name=expr_str)
    
    def _estimate_iterations(self, start: str, end: str) -> str:
        """Estimate number of iterations."""
        start = start.strip().lower()
        end = end.strip().lower()
        
        if start in ('0', '1') and end in ('n', 'len', 'length', 'size'):
            return "n"
        if start in ('0', '1') and 'n-' in end:
            return "n"
        if start in ('0', '1') and end.startswith('n/'):
            return "n/2"
        if start in ('0', '1') and 'log' in end:
            return "log(n)"
        
        try:
            s = int(start)
            e = int(end)
            return str(e - s + 1)
        except ValueError:
            pass
        
        if end.isalpha():
            return end
        return "n"
    
    def detect_structure(self, code: str) -> dict:
        """
        Detect high-level structure without full parsing.
        """
        code_lower = code.lower()
        lines = code.split('\n')
        
        # Count loop keywords, excluding END keywords like "END FOR", "ENDFOR", etc.
        # First count raw matches, then subtract END matches
        loop_count = 0
        for line in lines:
            line_lower = line.strip().lower()
            # Skip end keywords
            if line_lower.startswith('end') or line_lower.startswith('done'):
                continue
            # Count loop starts
            if re.match(r'^for\b', line_lower):
                loop_count += 1
            elif re.match(r'^while\b', line_lower):
                loop_count += 1
            elif re.match(r'^repeat\b', line_lower):
                loop_count += 1
            elif re.match(r'^loop\b', line_lower):
                loop_count += 1
        
        max_nesting = 0
        indent_unit = self.preprocessor.detect_indentation_style(code)
        
        # Track loop nesting by counting active loops based on keywords
        current_nesting = 0
        for line in lines:
            stripped = line.strip().lower()
            if not stripped:
                continue
            
            # Check if this line starts a loop (also check for opening brace)
            if any(stripped.startswith(kw) for kw in ['for ', 'for(', 'while ', 'while(', 'repeat']):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
                # If the line ends with {, the brace is part of this loop start
                # (already counted above)
            
            # Check if this line ends a loop block (END keywords or closing brace)
            if (stripped.startswith('end for') or stripped.startswith('endfor') or
                stripped.startswith('end while') or stripped.startswith('endwhile') or
                stripped == 'done' or stripped.startswith('until ') or
                stripped == '}'):
                current_nesting = max(0, current_nesting - 1)
        
        has_recursion = False
        func_match = re.search(r'(function|algorithm|def)\s+(\w+)', code_lower)
        if func_match:
            func_name = func_match.group(2)
            call_pattern = rf'\b{func_name}\s*\('
            calls = re.findall(call_pattern, code_lower)
            has_recursion = len(calls) > 1
        
        has_conditionals = bool(re.search(r'\bif\b', code_lower))
        
        return {
            'has_loops': loop_count > 0,
            'has_nested_loops': max_nesting > 1,  # Only true if loops are actually nested
            'has_recursion': has_recursion,
            'loop_count': loop_count,
            'max_nesting': max_nesting,
            'has_conditionals': has_conditionals,
        }
