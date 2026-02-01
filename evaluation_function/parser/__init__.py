"""
Parser module for pseudocode analysis.

This module provides:
- Preprocessor: Normalizes pseudocode syntax variations
- Grammar: Lark grammar for pseudocode parsing
- ASTBuilder: Transforms parse tree to AST nodes (when Lark available)
- PseudocodeParser: Main parser interface
"""

from .preprocessor import Preprocessor, PreprocessorConfig
from .grammar import PSEUDOCODE_GRAMMAR, SIMPLIFIED_GRAMMAR
from .parser import PseudocodeParser, ParseError, ParserConfig

# ASTBuilder is only available if Lark is installed
try:
    from .ast_builder import ASTBuilder
except ImportError:
    ASTBuilder = None

__all__ = [
    "Preprocessor",
    "PreprocessorConfig", 
    "PSEUDOCODE_GRAMMAR",
    "SIMPLIFIED_GRAMMAR",
    "ASTBuilder",
    "PseudocodeParser",
    "ParseError",
    "ParserConfig",
]
