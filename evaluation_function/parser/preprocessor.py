"""
Preprocessor for normalizing pseudocode syntax variations.

This module handles the diverse ways students write pseudocode by normalizing:
- Keywords (FOR/for/For → for)
- Assignment operators (=, :=, ←, <- → =)
- Comparison operators
- Whitespace and indentation
- Common typos and variations
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreprocessorConfig:
    """Configuration for the preprocessor."""
    normalize_case: bool = True
    normalize_operators: bool = True
    normalize_whitespace: bool = True
    fix_common_typos: bool = True
    preserve_strings: bool = True
    tab_size: int = 4


class Preprocessor:
    """
    Normalizes pseudocode to a standard format for parsing.
    
    Handles variations in:
    - Loop keywords: FOR, for, For, LOOP, loop
    - Conditionals: IF, if, THEN, then, ELSE, else
    - Assignment: =, :=, ←, <-
    - Comparisons: ==, =, ≤, <=, ≥, >=, ≠, !=, <>
    - Keywords: AND, and, &&, OR, or, ||, NOT, not, !
    - Function definitions: FUNCTION, function, ALGORITHM, algorithm, PROCEDURE
    - Return: RETURN, return, RETURNS
    - Ranges: TO, to, DOWNTO, downto, ..
    """
    
    # Keyword mappings (normalized form → variations)
    # Note: algorithm is kept separate from function
    KEYWORD_MAPPINGS = {
        # Loop keywords
        "for": ["FOR", "For", "LOOP", "loop", "Loop"],
        "while": ["WHILE", "While", "WHILST", "whilst"],
        "do": ["DO", "Do"],
        "end": ["END", "End", "ENDFOR", "endfor", "ENDWHILE", "endwhile", 
                "ENDIF", "endif", "END IF", "end if", "END FOR", "end for",
                "END WHILE", "end while", "DONE", "done"],
        "repeat": ["REPEAT", "Repeat"],
        "until": ["UNTIL", "Until"],
        "to": ["TO", "To"],
        "downto": ["DOWNTO", "Downto", "DOWN TO", "down to"],
        "step": ["STEP", "Step", "BY", "by"],
        "in": ["IN", "In"],
        "each": ["EACH", "Each"],
        
        # Conditional keywords
        "if": ["IF", "If"],
        "then": ["THEN", "Then"],
        "else": ["ELSE", "Else"],
        "elif": ["ELIF", "Elif", "ELSEIF", "elseif", "ELSE IF", "else if", "elsif", "ELSIF"],
        
        # Logical operators (word forms only - && and || handled separately)
        "and": ["AND", "And"],
        "or": ["OR", "Or"],
        "not": ["NOT", "Not"],
        
        # Function keywords - algorithm and function are separate!
        "function": ["FUNCTION", "Function", "FUNC", "func"],
        "algorithm": ["ALGORITHM", "Algorithm"],
        "procedure": ["PROCEDURE", "Procedure"],
        "def": ["DEF"],
        "return": ["RETURN", "Return", "RETURNS", "returns"],
        
        # Boolean literals
        "true": ["TRUE", "True"],
        "false": ["FALSE", "False"],
        
        # Other keywords
        "null": ["NULL", "Null", "NIL", "nil", "NONE", "None", "none"],
        "print": ["PRINT", "Print", "OUTPUT", "output", "WRITE", "write", "DISPLAY", "display"],
        "input": ["INPUT", "Input", "READ", "read"],
        "swap": ["SWAP", "Swap"],
        "call": ["CALL", "Call"],
    }
    
    # Operator mappings (symbols that aren't word characters)
    OPERATOR_MAPPINGS = {
        "&&": "and",
        "||": "or",
        "&": "and",
        "|": "or",
    }
    
    # Assignment operator mappings
    ASSIGNMENT_OPERATORS = {
        ":=": "=",
        "←": "=",
        "<-": "=",
        "⟵": "=",
    }
    
    # Comparison operator mappings
    COMPARISON_OPERATORS = {
        "≤": "<=",
        "≥": ">=",
        "≠": "!=",
        "<>": "!=",
    }
    
    # Common typos
    TYPO_FIXES = {
        "whlie": "while",
        "wihle": "while",
        "fro": "for",
        "fo": "for",
        "eles": "else",
        "esle": "else",
        "retrun": "return",
        "reutrn": "return",
        "fucntion": "function",
        "funtion": "function",
        "funciton": "function",
        "algoritm": "algorithm",
        "algortihm": "algorithm",
        "pritn": "print",
        "pirnt": "print",
    }
    
    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns for normalization."""
        # Build keyword pattern (case-insensitive word boundaries)
        self._keyword_map = {}
        for normalized, variations in self.KEYWORD_MAPPINGS.items():
            for var in variations:
                self._keyword_map[var.lower()] = normalized
            self._keyword_map[normalized] = normalized
        
        # Build typo pattern
        if self.config.fix_common_typos:
            self._typo_map = {k.lower(): v for k, v in self.TYPO_FIXES.items()}
    
    def preprocess(self, code: str) -> Tuple[str, List[str]]:
        """
        Preprocess pseudocode and return normalized version.
        
        Args:
            code: Raw pseudocode string
            
        Returns:
            Tuple of (normalized_code, warnings)
        """
        warnings = []
        
        # Preserve string literals
        strings = []
        if self.config.preserve_strings:
            code, strings = self._extract_strings(code)
        
        # Normalize line endings
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Fix common typos
        if self.config.fix_common_typos:
            code, typo_warnings = self._fix_typos(code)
            warnings.extend(typo_warnings)
        
        # Normalize operators (including && and ||)
        if self.config.normalize_operators:
            code = self._normalize_operators(code)
        
        # Normalize keywords
        if self.config.normalize_case:
            code = self._normalize_keywords(code)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            code = self._normalize_whitespace(code)
        
        # Restore string literals
        if self.config.preserve_strings:
            code = self._restore_strings(code, strings)
        
        return code, warnings
    
    def _extract_strings(self, code: str) -> Tuple[str, List[str]]:
        """Extract string literals and replace with placeholders."""
        strings = []
        
        def replace_string(match):
            strings.append(match.group(0))
            return f"__STRING_{len(strings) - 1}__"
        
        # Match both single and double quoted strings
        pattern = r'"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\''
        code = re.sub(pattern, replace_string, code)
        
        return code, strings
    
    def _restore_strings(self, code: str, strings: List[str]) -> str:
        """Restore string literals from placeholders."""
        for i, s in enumerate(strings):
            code = code.replace(f"__STRING_{i}__", s)
        return code
    
    def _fix_typos(self, code: str) -> Tuple[str, List[str]]:
        """Fix common typos in keywords."""
        warnings = []
        
        def fix_word(match):
            word = match.group(0)
            lower = word.lower()
            if lower in self._typo_map:
                fixed = self._typo_map[lower]
                warnings.append(f"Fixed typo: '{word}' → '{fixed}'")
                return fixed
            return word
        
        # Match whole words only
        code = re.sub(r'\b[a-zA-Z]+\b', fix_word, code)
        
        return code, warnings
    
    def _normalize_operators(self, code: str) -> str:
        """Normalize assignment, comparison, and logical operators."""
        # First handle && and || (before other processing might interfere)
        # These need to be replaced with word equivalents
        for old, new in self.OPERATOR_MAPPINGS.items():
            # Use word boundaries to avoid partial replacements
            code = re.sub(re.escape(old), f' {new} ', code)
        
        # Assignment operators (must be done before comparison)
        for old, new in self.ASSIGNMENT_OPERATORS.items():
            code = code.replace(old, new)
        
        # Handle single = that should be == (in comparisons)
        # This is tricky - we need context. For now, handle obvious cases.
        # = after if/while/until and before then/do should be ==
        code = re.sub(
            r'(if|while|until)\s+([^=\n]+)\s+=\s+([^=\n]+)\s+(then|do)',
            r'\1 \2 == \3 \4',
            code,
            flags=re.IGNORECASE
        )
        
        # Unicode comparison operators
        for old, new in self.COMPARISON_OPERATORS.items():
            code = code.replace(old, new)
        
        return code
    
    def _normalize_keywords(self, code: str) -> str:
        """Normalize keyword case to lowercase standard form."""
        
        def normalize_word(match):
            word = match.group(0)
            lower = word.lower()
            return self._keyword_map.get(lower, word)
        
        # Match whole words only
        code = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', normalize_word, code)
        
        return code
    
    def _normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace and indentation."""
        lines = code.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Convert tabs to spaces
            line = line.replace('\t', ' ' * self.config.tab_size)
            
            # Remove trailing whitespace
            line = line.rstrip()
            
            # Normalize multiple spaces (except leading)
            if line:
                leading = len(line) - len(line.lstrip())
                content = ' '.join(line.split())
                line = ' ' * leading + content
            
            normalized_lines.append(line)
        
        # Remove multiple blank lines
        code = '\n'.join(normalized_lines)
        code = re.sub(r'\n{3,}', '\n\n', code)
        
        return code.strip()
    
    def detect_indentation_style(self, code: str) -> int:
        """
        Detect the indentation unit used in the code.
        
        Returns:
            Number of spaces per indent level (common values: 2, 4)
        """
        lines = code.split('\n')
        indents = []
        
        for line in lines:
            if line.strip():  # Non-empty line
                leading = len(line) - len(line.lstrip())
                if leading > 0:
                    indents.append(leading)
        
        if not indents:
            return 4  # Default
        
        # Find GCD of all indentations
        from math import gcd
        from functools import reduce
        
        indent_unit = reduce(gcd, indents)
        return max(indent_unit, 2)  # At least 2 spaces
    
    def get_indent_level(self, line: str, indent_unit: int = 4) -> int:
        """Get the indentation level of a line."""
        if not line.strip():
            return 0
        leading = len(line) - len(line.lstrip())
        return leading // indent_unit