"""
Enhanced parser with better assignment and expression parsing.

This fixes the issue where simple assignments like "x = 42" weren't being parsed.
"""

import re
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass

from .preprocessor import Preprocessor, PreprocessorConfig
from .grammar import PSEUDOCODE_GRAMMAR
from ..schemas.ast_nodes import *
from ..schemas.output_schema import ParseResult


@dataclass
class ParserConfig:
    """Configuration for the parser."""
    use_indentation: bool = True
    strict_mode: bool = False
    max_errors: int = 10
    timeout: float = 5.0


class PseudocodeParser:
    """
    Enhanced parser with better assignment handling.
    """
    
    def __init__(self, config: Optional[ParserConfig] = None,
                 preprocessor_config: Optional[PreprocessorConfig] = None):
        self.config = config or ParserConfig()
        self.preprocessor = Preprocessor(preprocessor_config)
        self._lark_available = False
        self._parser = None
        self._try_init_lark()
    
    def _try_init_lark(self):
        """Try to initialize Lark parser."""
        try:
            from lark import Lark
            self._parser = Lark(
                PSEUDOCODE_GRAMMAR,
                parser='lalr',
                propagate_positions=True,
                maybe_placeholders=False,
            )
            self._lark_available = True
        except Exception:
            self._lark_available = False
            self._parser = None
    
    def parse(self, code: str) -> ParseResult:
        """Parse pseudocode and return ParseResult."""
        errors: List[str] = []
        warnings: List[str] = []
        
        # Preprocess
        try:
            normalized_code, preprocess_warnings = self.preprocessor.preprocess(code)
            warnings.extend(preprocess_warnings)
        except Exception as e:
            errors.append(f"Preprocessing failed: {str(e)}")
            normalized_code = code
        
        # Try Lark parsing
        ast = None
        if self._lark_available and self._parser:
            try:
                tree = self._parser.parse(normalized_code)
                ast = self._build_ast_from_tree(tree, normalized_code)
            except Exception as e:
                if self.config.strict_mode:
                    errors.append(f"Parse error: {str(e)}")
                else:
                    warnings.append(f"Full parsing failed, using fallback: {str(e)[:100]}")
        
        # Use fallback
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
                dummy_cond = VariableNode(name="condition")
                return ConditionalNode(condition=dummy_cond, then_branch=BlockNode(statements=[]))
        except Exception:
            pass
        return None
    
    def _parse_fallback(self, code: str) -> ProgramNode:
        """
        Enhanced fallback parsing with assignment support.
        
        Returns None if code appears to be completely invalid.
        """
        functions: List[FunctionNode] = []
        statements = []
        
        lines = code.split('\n')
        indent_unit = self.preprocessor.detect_indentation_style(code)
        
        # Check if code looks completely invalid (special characters, no keywords)
        non_empty_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
        if non_empty_lines:
            # Check if any line has recognizable patterns
            has_valid_pattern = False
            for line in non_empty_lines:
                # Check for valid patterns: assignment, keyword, function call, etc.
                if (re.match(r'\w+\s*=', line) or  # assignment
                    re.match(r'\b(for|while|if|function|return|print)\b', line, re.IGNORECASE) or  # keywords
                    re.match(r'\w+\(', line)):  # function call
                    has_valid_pattern = True
                    break
            
            # If no valid patterns and contains special chars, reject
            if not has_valid_pattern:
                # Check for invalid characters
                combined = ' '.join(non_empty_lines)
                invalid_chars = set('!@#$%^&*')
                if any(c in combined for c in invalid_chars) and len(combined) > 10:
                    return None  # Completely invalid code
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                i += 1
                continue
            
            indent_level = self.preprocessor.get_indent_level(line, indent_unit)
            
            # Detect function definitions (ENHANCED: parse parameters)
            func_match = re.match(
                r'^(function|algorithm|procedure|def)\s+(\w+)\s*\(([^)]*)\)\s*\{?',
                stripped, re.IGNORECASE
            )
            if func_match:
                func_name = func_match.group(2)
                params_str = func_match.group(3).strip()
                
                # Parse parameters
                params = []
                if params_str:
                    param_names = [p.strip() for p in params_str.split(',') if p.strip()]
                    params = [VariableNode(name=p) for p in param_names]
                
                body_lines, end_idx = self._collect_block(lines, i + 1, indent_level, indent_unit)
                func = FunctionNode(
                    name=func_name,
                    parameters=params,
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
            
            # Detect print statements
            print_match = re.match(r'^print\s*\((.+)\)\s*$', stripped, re.IGNORECASE)
            if print_match:
                # Parse print as function call wrapped in expression statement
                args_str = print_match.group(1)
                args = self._parse_argument_list(args_str)
                print_call = FunctionCallNode(
                    function_name="print",
                    arguments=args
                )
                statements.append(ExpressionStatementNode(expression=print_call))
                i += 1
                continue
            
            # Detect assignments (ENHANCED)
            assign_match = re.match(r'^(\w+(?:\[\w+\])?)\s*=\s*(.+)$', stripped)
            if assign_match:
                target_str = assign_match.group(1)
                value_str = assign_match.group(2)
                
                # Parse target
                if '[' in target_str:
                    # Array access
                    array_match = re.match(r'(\w+)\[(.+)\]', target_str)
                    if array_match:
                        array_name = array_match.group(1)
                        index_str = array_match.group(2)
                        target = ArrayAccessNode(
                            array=VariableNode(name=array_name),
                            index=self._parse_expression(index_str)
                        )
                else:
                    # Simple variable
                    target = VariableNode(name=target_str)
                
                # Parse value
                value = self._parse_expression(value_str)
                
                statements.append(AssignmentNode(target=target, value=value))
                i += 1
                continue
            
            # Detect return statements
            return_match = re.match(r'^return\s*(.*)$', stripped, re.IGNORECASE)
            if return_match:
                value_str = return_match.group(1).strip()
                if value_str:
                    value = self._parse_expression(value_str)
                else:
                    value = None
                statements.append(ReturnNode(value=value))
                i += 1
                continue
            
            # Skip unknown lines
            i += 1
        
        global_block = BlockNode(statements=statements) if statements else None
        return ProgramNode(functions=functions, global_statements=global_block)
    
    def _parse_expression(self, expr_str: str) -> ExpressionNode:
        """
        Parse an expression string into an ExpressionNode with proper precedence.
        
        Precedence (lowest to highest):
        1. Comparison: ==, !=, <, <=, >, >=
        2. Addition/Subtraction: +, -
        3. Multiplication/Division/Modulo: *, /, %
        4. Unary: -, not
        5. Atoms: literals, variables, function calls, array access
        """
        expr_str = expr_str.strip()
        
        # Level 1: Comparison operators (lowest precedence)
        for op_str, op_type in [
            ('==', OperatorType.EQUAL),
            ('!=', OperatorType.NOT_EQUAL),
            ('<=', OperatorType.LESS_EQUAL),
            ('>=', OperatorType.GREATER_EQUAL),
            ('<', OperatorType.LESS_THAN),
            ('>', OperatorType.GREATER_THAN),
        ]:
            parts = self._split_on_operator(expr_str, op_str)
            if len(parts) == 2 and parts[0] and parts[1]:
                left = self._parse_expression(parts[0])
                right = self._parse_expression(parts[1])
                return BinaryOpNode(operator=op_type, left=left, right=right)
        
        # Level 2: Addition and Subtraction
        for op_str, op_type in [
            ('+', OperatorType.ADD),
            ('-', OperatorType.SUBTRACT),
        ]:
            parts = self._split_on_operator(expr_str, op_str)
            if len(parts) == 2 and parts[0] and parts[1]:
                left = self._parse_expression(parts[0])
                right = self._parse_expression(parts[1])
                return BinaryOpNode(operator=op_type, left=left, right=right)
        
        # Level 3: Multiplication, Division, Modulo (higher precedence)
        for op_str, op_type in [
            ('*', OperatorType.MULTIPLY),
            ('/', OperatorType.DIVIDE),
            ('%', OperatorType.MODULO),
        ]:
            parts = self._split_on_operator(expr_str, op_str)
            if len(parts) == 2 and parts[0] and parts[1]:
                left = self._parse_expression(parts[0])
                right = self._parse_expression(parts[1])
                return BinaryOpNode(operator=op_type, left=left, right=right)
        
        # Level 4: Atoms (highest precedence)
        
        # Handle parenthesized expressions
        if expr_str.startswith('(') and expr_str.endswith(')'):
            # Check if these are matching parens
            paren_depth = 0
            all_enclosed = True
            for i, char in enumerate(expr_str):
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                    if paren_depth == 0 and i < len(expr_str) - 1:
                        # Closing paren before the end
                        all_enclosed = False
                        break
            
            if all_enclosed:
                # Remove outer parens and parse inner expression
                inner = expr_str[1:-1].strip()
                return self._parse_expression(inner)
        
        # Handle function calls (BEFORE variables to catch them first)
        func_match = re.match(r'^(\w+)\s*\((.+)\)\s*$', expr_str)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            args = self._parse_argument_list(args_str)
            return FunctionCallNode(function_name=func_name, arguments=args)
        
        # Handle function calls with no args
        func_match_no_args = re.match(r'^(\w+)\s*\(\s*\)\s*$', expr_str)
        if func_match_no_args:
            func_name = func_match_no_args.group(1)
            return FunctionCallNode(function_name=func_name, arguments=[])
        
        # Handle array access
        if '[' in expr_str and ']' in expr_str:
            array_match = re.match(r'^(\w+)\[(.+)\]$', expr_str)
            if array_match:
                array_name = array_match.group(1)
                index_str = array_match.group(2)
                return ArrayAccessNode(
                    array=VariableNode(name=array_name),
                    index=self._parse_expression(index_str)
                )
        
        # Handle literals
        # Try int
        try:
            return LiteralNode(value=int(expr_str), literal_type="int")
        except ValueError:
            pass
        
        # Try float
        try:
            return LiteralNode(value=float(expr_str), literal_type="float")
        except ValueError:
            pass
        
        # Try string
        if (expr_str.startswith('"') and expr_str.endswith('"')) or \
           (expr_str.startswith("'") and expr_str.endswith("'")):
            return LiteralNode(value=expr_str[1:-1], literal_type="string")
        
        # Try boolean
        if expr_str.lower() == 'true':
            return LiteralNode(value=True, literal_type="bool")
        if expr_str.lower() == 'false':
            return LiteralNode(value=False, literal_type="bool")
        
        # Try list literal
        if expr_str.startswith('[') and expr_str.endswith(']'):
            # Simple list parsing
            content = expr_str[1:-1].strip()
            if content:
                elements = [item.strip() for item in content.split(',')]
                values = []
                for elem in elements:
                    try:
                        values.append(int(elem))
                    except ValueError:
                        try:
                            values.append(float(elem))
                        except ValueError:
                            values.append(elem)
                return LiteralNode(value=values, literal_type="list")
            return LiteralNode(value=[], literal_type="list")
        
        # Default to variable
        return VariableNode(name=expr_str)
    
    def _split_on_operator(self, expr_str: str, op: str) -> List[str]:
        """Split expression on operator, respecting parentheses and function calls."""
        # Find last occurrence of operator not in parens
        paren_depth = 0
        best_pos = -1
        
        # Scan from right to left (for left-associativity)
        for i in range(len(expr_str) - len(op), -1, -1):
            if i < 0:
                break
                
            # Track parenthesis depth
            char = expr_str[i]
            if char == ')':
                paren_depth += 1
            elif char == '(':
                paren_depth -= 1
            
            # Check for operator at this position
            if paren_depth == 0 and expr_str[i:i+len(op)] == op:
                # Make sure it's not part of a number (e.g., -5)
                if op == '-' and i == 0:
                    continue  # This is unary minus
                if op == '-' and i > 0 and expr_str[i-1] in '(,':
                    continue  # This is unary minus
                best_pos = i
                break
        
        if best_pos >= 0:
            left = expr_str[:best_pos].strip()
            right = expr_str[best_pos+len(op):].strip()
            if left and right:  # Both sides must be non-empty
                return [left, right]
        
        return [expr_str]
    
    def _parse_argument_list(self, args_str: str) -> List[ExpressionNode]:
        """Parse comma-separated argument list, respecting nested parens."""
        if not args_str.strip():
            return []
        
        # Split on commas, but respect parentheses
        args = []
        current_arg = []
        paren_depth = 0
        
        for char in args_str:
            if char == '(':
                paren_depth += 1
                current_arg.append(char)
            elif char == ')':
                paren_depth -= 1
                current_arg.append(char)
            elif char == ',' and paren_depth == 0:
                # End of argument
                arg_str = ''.join(current_arg).strip()
                if arg_str:
                    args.append(self._parse_expression(arg_str))
                current_arg = []
            else:
                current_arg.append(char)
        
        # Don't forget the last argument
        arg_str = ''.join(current_arg).strip()
        if arg_str:
            args.append(self._parse_expression(arg_str))
        
        return args
    
    def _collect_block(self, lines: List[str], start_idx: int,
                       base_indent: int, indent_unit: int) -> Tuple[List[str], int]:
        """Collect lines belonging to a block."""
        block_lines = []
        i = start_idx
        brace_count = 0

        # Check if we're starting a brace block
        if start_idx > 0:
            prev_line = lines[start_idx - 1].strip()
            if prev_line.endswith('{'):
                brace_count = 1

        if i < len(lines) and lines[i].strip() == '{':
            brace_count = 1
            i += 1

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                if brace_count == 0:
                    block_lines.append(line)
                else:
                    block_lines.append(line)
                i += 1
                continue

            # Handle curly brace blocks
            if brace_count > 0:
                open_braces = stripped.count('{')
                close_braces = stripped.count('}')
                
                if stripped == '}':
                    brace_count -= 1
                    if brace_count <= 0:
                        i += 1
                        break
                    i += 1
                    continue
                
                if stripped.endswith('}'):
                    content = stripped[:-1].strip()
                    if content:
                        block_lines.append(content)
                    brace_count += open_braces - close_braces
                    if brace_count <= 0:
                        i += 1
                        break
                else:
                    block_lines.append(line)
                    brace_count += open_braces - close_braces
                
                i += 1
                continue

            current_indent = self.preprocessor.get_indent_level(line, indent_unit)

            # End markers (ENHANCED: include function/algorithm/procedure)
            if re.match(r'^(end|endif|endfor|endwhile|end\s+if|end\s+for|end\s+while|end\s+function|endfunction|end\s+algorithm|endalgorithm|end\s+procedure|endprocedure|done)\b', 
                       stripped, re.IGNORECASE):
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
            
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                i += 1
                continue
            
            indent_level = self.preprocessor.get_indent_level(line, indent_unit)
            
            # Check for loops
            loop_node = self._detect_loop(stripped, lines, i, indent_level, indent_unit)
            if loop_node:
                statements.append(loop_node[0])
                i = loop_node[1]
                continue
            
            # Check for conditionals
            cond_node = self._detect_conditional(stripped, lines, i, indent_level, indent_unit)
            if cond_node:
                statements.append(cond_node[0])
                i = cond_node[1]
                continue
            
            # Check for return statements
            return_match = re.match(r'^return\s*(.*)$', stripped, re.IGNORECASE)
            if return_match:
                value_str = return_match.group(1).strip()
                if value_str:
                    value = self._parse_expression(value_str)
                else:
                    value = None
                statements.append(ReturnNode(value=value))
                i += 1
                continue
            
            # Check for print statements
            print_match = re.match(r'^print\s*\((.+)\)\s*$', stripped, re.IGNORECASE)
            if print_match:
                args_str = print_match.group(1)
                args = self._parse_argument_list(args_str)
                print_call = FunctionCallNode(
                    function_name="print",
                    arguments=args
                )
                statements.append(ExpressionStatementNode(expression=print_call))
                i += 1
                continue
            
            # Check for assignments
            assign_match = re.match(r'^(\w+(?:\[\w+\])?)\s*=\s*(.+)$', stripped)
            if assign_match:
                target_str = assign_match.group(1)
                value_str = assign_match.group(2)
                
                # Parse target
                if '[' in target_str:
                    # Array access
                    array_match = re.match(r'(\w+)\[(.+)\]', target_str)
                    if array_match:
                        array_name = array_match.group(1)
                        index_str = array_match.group(2)
                        target = ArrayAccessNode(
                            array=VariableNode(name=array_name),
                            index=self._parse_expression(index_str)
                        )
                else:
                    # Simple variable
                    target = VariableNode(name=target_str)
                
                # Parse value
                value = self._parse_expression(value_str)
                
                statements.append(AssignmentNode(target=target, value=value))
                i += 1
                continue
            
            # Unknown statement - skip
            i += 1
        
        return BlockNode(statements=statements)
    
    def _detect_loop(self, line: str, lines: List[str], idx: int,
                     indent_level: int, indent_unit: int) -> Optional[Tuple[LoopNode, int]]:
        """Detect and parse a loop."""
        
        # FOR loop
        for_match = re.match(
            r'for\s+(\w+)\s*[=:]\s*(.+?)\s+to\s+(.+?)(?:\s+(?:do|step\s+\w+))?\s*\{?\s*$',
            line, re.IGNORECASE
        )
        if for_match:
            iterator = VariableNode(name=for_match.group(1))
            start = self._parse_expression(for_match.group(2))
            end = self._parse_expression(for_match.group(3))
            
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
        
        # WHILE loop
        while_match = re.match(r'while\s+(.+?)(?:\s+do)?\s*\{?\s*$', line, re.IGNORECASE)
        if while_match:
            condition = self._parse_expression(while_match.group(1))
            body_lines, next_idx = self._collect_block(lines, idx + 1, indent_level, indent_unit)
            body = self._parse_block_fallback(body_lines, indent_unit)
            
            return LoopNode(
                loop_type=LoopType.WHILE,
                condition=condition,
                body=body,
                estimated_iterations="unknown",
                nesting_level=indent_level
            ), next_idx
        
        # FOR EACH loop
        foreach_match = re.match(
            r'for\s+(?:each\s+)?(\w+)\s+in\s+(.+?)(?:\s+do)?\s*\{?\s*$',
            line, re.IGNORECASE
        )
        if foreach_match:
            iterator = VariableNode(name=foreach_match.group(1))
            collection = self._parse_expression(foreach_match.group(2))
            
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
        
        # REPEAT loop (ENHANCED: look for until condition)
        repeat_match = re.match(r'repeat\b', line, re.IGNORECASE)
        if repeat_match:
            body_lines, next_idx = self._collect_block(lines, idx + 1, indent_level, indent_unit)
            
            # Check if there's an UNTIL condition after the block
            condition = None
            if next_idx < len(lines):
                until_line = lines[next_idx].strip()
                until_match = re.match(r'until\s+(.+)$', until_line, re.IGNORECASE)
                if until_match:
                    condition = self._parse_expression(until_match.group(1))
                    next_idx += 1
            
            body = self._parse_block_fallback(body_lines, indent_unit)
            
            return LoopNode(
                loop_type=LoopType.REPEAT_UNTIL,
                condition=condition,
                body=body,
                estimated_iterations="unknown",
                nesting_level=indent_level
            ), next_idx
        
        return None
    
    def _detect_conditional(self, line: str, lines: List[str], idx: int,
                           indent_level: int, indent_unit: int) -> Optional[Tuple[ConditionalNode, int]]:
        """Detect and parse a conditional."""
        if_match = re.match(r'if\s+(.+?)(?:\s+then)?\s*\{?\s*$', line, re.IGNORECASE)
        if if_match:
            condition_str = if_match.group(1).strip()
            condition = self._parse_expression(condition_str)
            
            body_lines, next_idx = self._collect_block(lines, idx + 1, indent_level, indent_unit)
            then_branch = self._parse_block_fallback(body_lines, indent_unit)
            
            else_branch = None
            # Check for ELSE on the next line
            if next_idx < len(lines):
                else_line = lines[next_idx].strip()
                # Match ELSE (with optional opening brace)
                if re.match(r'^else\s*\{?\s*$', else_line, re.IGNORECASE):
                    else_lines, next_idx = self._collect_block(lines, next_idx + 1, indent_level, indent_unit)
                    else_branch = self._parse_block_fallback(else_lines, indent_unit)
            
            return ConditionalNode(
                condition=condition,
                then_branch=then_branch,
                else_branch=else_branch
            ), next_idx
        
        return None
    
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
        """Detect high-level structure without full parsing."""
        code_lower = code.lower()
        lines = code.split('\n')
        
        loop_count = 0
        for line in lines:
            line_lower = line.strip().lower()
            if re.match(r'^(end|done)', line_lower):
                continue
            if re.match(r'^for\b', line_lower):
                loop_count += 1
            elif re.match(r'^while\b', line_lower):
                loop_count += 1
            elif re.match(r'^repeat\b', line_lower):
                loop_count += 1
        
        max_nesting = 0
        current_nesting = 0
        
        for line in lines:
            stripped = line.strip().lower()
            if not stripped:
                continue
            
            if any(stripped.startswith(kw) for kw in ['for ', 'for(', 'while ', 'while(', 'repeat']):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            
            if re.match(r'^(end\s*for|endfor|end\s*while|endwhile|done|until\s+)', stripped):
                current_nesting = max(0, current_nesting - 1)
            elif stripped == '}':
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
            'has_nested_loops': max_nesting > 1,
            'has_recursion': has_recursion,
            'loop_count': loop_count,
            'max_nesting': max_nesting,
            'has_conditionals': has_conditionals,
        }