from typing import Any, Dict, List, Set, Union
from z3 import Solver, Int, Bool, And, Or, Not, IntVal, BoolVal, simplify, Implies

from ..schemas.ast_nodes import *
# Assuming imports from your specific file structure
class FixedVerificationResult:
    def __init__(self):
        self.success: bool = False
        self.message: str = ""
        self.undefined_symbols: List[str] = []
        self.proof_steps: List[str] = []

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "message": self.message,
            "undefined_symbols": self.undefined_symbols,
            "proof_steps": self.proof_steps
        }
from typing import Any, Dict, List, Set, Optional, Tuple
from z3 import Solver, Int, Bool, And, Or, Not, IntVal, BoolVal, Implies, is_bool
from evaluation_function.schemas.ast_nodes import (
    ProgramNode, BlockNode, AssignmentNode, VariableNode, LiteralNode,
    BinaryOpNode, UnaryOpNode, ConditionalNode, LoopNode,
    OperatorType, ExpressionNode, NodeType, LoopType
)

class FixedALevelVerifier:
    def __init__(self):
        self.defined_variables: Set[str] = set()
        self.proof_steps: List[str] = []

    def collect_symbols(self, precondition: ExpressionNode, program: ProgramNode) -> None:
        self.defined_variables.clear()
        self._collect_from_expr(precondition)
        if program.global_statements:
            self._collect_from_block(program.global_statements)

    def _collect_from_expr(self, expr: Optional[ExpressionNode]):
        if not expr: return
        if isinstance(expr, VariableNode):
            self.defined_variables.add(expr.name)
        elif isinstance(expr, BinaryOpNode):
            self._collect_from_expr(expr.left)
            self._collect_from_expr(expr.right)
        elif isinstance(expr, UnaryOpNode):
            self._collect_from_expr(expr.operand)

    def _collect_from_block(self, block: BlockNode):
        for stmt in block.statements:
            if isinstance(stmt, AssignmentNode) and isinstance(stmt.target, VariableNode):
                self.defined_variables.add(stmt.target.name)
            elif isinstance(stmt, ConditionalNode):
                if stmt.then_branch: self._collect_from_block(stmt.then_branch)
                for elif_b in stmt.elif_branches:
                    if elif_b.then_branch: self._collect_from_block(elif_b.then_branch)
                if stmt.else_branch: self._collect_from_block(stmt.else_branch)
            elif isinstance(stmt, LoopNode) and stmt.body:
                self._collect_from_block(stmt.body)

    # ------------------------
    # WP Logic
    # ------------------------
    def wp(self, statements: List[Any], post: ExpressionNode) -> ExpressionNode:
        wp_expr = post
        for stmt in reversed(statements):
            wp_expr = self._wp_stmt(stmt, wp_expr)
            self.proof_steps.append(f"WP after {self._stmt_to_str(stmt)}: {self._expr_to_str(wp_expr)}")
        return wp_expr

    def _wp_stmt(self, stmt: Any, post: ExpressionNode) -> ExpressionNode:
        if isinstance(stmt, AssignmentNode) and isinstance(stmt.target, VariableNode):
            return self._substitute(post, stmt.target.name, stmt.value)

        elif isinstance(stmt, ConditionalNode):
            current_wp = self.wp(stmt.else_branch.statements if stmt.else_branch else [], post)
            for elif_b in reversed(stmt.elif_branches):
                wp_elif = self.wp(elif_b.then_branch.statements if elif_b.then_branch else [], post)
                current_wp = self._combine_conditional(elif_b.condition, wp_elif, current_wp)
            wp_then = self.wp(stmt.then_branch.statements if stmt.then_branch else [], post)
            return self._combine_conditional(stmt.condition, wp_then, current_wp)

        elif isinstance(stmt, LoopNode):
            # A-Level Simplification: Use provided invariant
            if "invariant" in stmt.metadata:
                return stmt.metadata["invariant"]
        return post

    def _combine_conditional(self, cond: ExpressionNode, wp_true: ExpressionNode, wp_false: ExpressionNode) -> ExpressionNode:
        left = BinaryOpNode(operator=OperatorType.AND, left=cond, right=wp_true)
        not_c = UnaryOpNode(operator=OperatorType.NOT, operand=cond)
        right = BinaryOpNode(operator=OperatorType.AND, left=not_c, right=wp_false)
        return BinaryOpNode(operator=OperatorType.OR, left=left, right=right)

    def _substitute(self, expr: ExpressionNode, var_name: str, replacement: ExpressionNode) -> ExpressionNode:
        if isinstance(expr, VariableNode) and expr.name == var_name:
            return replacement
        if isinstance(expr, BinaryOpNode):
            return BinaryOpNode(
                operator=expr.operator,
                left=self._substitute(expr.left, var_name, replacement),
                right=self._substitute(expr.right, var_name, replacement)
            )
        if isinstance(expr, UnaryOpNode):
            return UnaryOpNode(
                operator=expr.operator,
                operand=self._substitute(expr.operand, var_name, replacement)
            )
        return expr

    # ------------------------
    # Z3 Integration
    # ------------------------
    def verify(self, precondition: ExpressionNode, program: ProgramNode, postcondition: ExpressionNode):
        from evaluation_function.equivalence.equivalence import FixedVerificationResult
        result = FixedVerificationResult()
        try:
            self.collect_symbols(precondition, program)
            wp_final = self.wp(program.global_statements.statements, postcondition) if program.global_statements else postcondition
            
            success, msg = self._implies(precondition, wp_final)
            result.success, result.message = success, msg
            result.proof_steps, result.undefined_symbols = self.proof_steps, []
        except Exception as e:
            result.success, result.message = False, f"Internal Error: {str(e)}"
        return result

    def _implies(self, pre: ExpressionNode, post: ExpressionNode) -> tuple[bool, str]:
        s = Solver()
        env = {}
        try:
            pre_z3 = self._expr_to_z3(pre, env)
            post_z3 = self._expr_to_z3(post, env)
            
            # Ensure both are treated as Booleans in Z3 context
            # A-Level pseudocode usually treats 1 as True, 0 as False
            def ensure_bool(z3_expr):
                if not is_bool(z3_expr):
                    return z3_expr != 0
                return z3_expr

            s.add(And(ensure_bool(pre_z3), Not(ensure_bool(post_z3))))
            
            if s.check().r == -1: # UNSAT
                return True, "✓ Success"
            return False, f"✗ Counter-example: {s.model()}"
        except Exception as e:
            raise ValueError(f"Z3 Error: {e}")

    def _expr_to_z3(self, expr: ExpressionNode, env: Dict[str, Any]):
        if isinstance(expr, LiteralNode):
            if isinstance(expr.value, bool): return BoolVal(expr.value)
            # Check if it's a numeric string or actual int
            val = int(expr.value) if not isinstance(expr.value, bool) else expr.value
            return IntVal(val) if not isinstance(val, bool) else BoolVal(val)
        
        if isinstance(expr, VariableNode):
            if expr.name not in env:
                # Default to Int for A-Level variables unless specified
                env[expr.name] = Int(expr.name)
            return env[expr.name]

        if isinstance(expr, UnaryOpNode):
            operand = self._expr_to_z3(expr.operand, env)
            if expr.operator == OperatorType.NOT:
                if not is_bool(operand): operand = (operand != 0)
                return Not(operand)
            if expr.operator == OperatorType.SUBTRACT: return -operand
            
        if isinstance(expr, BinaryOpNode):
            l = self._expr_to_z3(expr.left, env)
            r = self._expr_to_z3(expr.right, env)
            op = expr.operator
            
            # Arithmetic
            if op == OperatorType.ADD: return l + r
            if op == OperatorType.SUBTRACT: return l - r
            if op == OperatorType.MULTIPLY: return l * r
            
            # Comparisons (return Bool)
            if op == OperatorType.EQUAL: return l == r
            if op == OperatorType.NOT_EQUAL: return l != r
            if op == OperatorType.GREATER_THAN: return l > r
            if op == OperatorType.GREATER_EQUAL: return l >= r
            if op == OperatorType.LESS_THAN: return l < r
            if op == OperatorType.LESS_EQUAL: return l <= r
            
            # Logical (ensure operands are Bool)
            if op in [OperatorType.AND, OperatorType.OR]:
                if not is_bool(l): l = (l != 0)
                if not is_bool(r): r = (r != 0)
                return And(l, r) if op == OperatorType.AND else Or(l, r)

        raise ValueError(f"Unsupported node or operator: {type(expr)} {getattr(expr, 'operator', '')}")

    def _expr_to_str(self, expr: ExpressionNode) -> str:
        if isinstance(expr, VariableNode): return expr.name
        if isinstance(expr, LiteralNode): return str(expr.value)
        if isinstance(expr, BinaryOpNode):
            return f"({self._expr_to_str(expr.left)} {expr.operator.value} {self._expr_to_str(expr.right)})"
        if isinstance(expr, UnaryOpNode):
            return f"{expr.operator.value}({self._expr_to_str(expr.operand)})"
        return "expr"

    def _stmt_to_str(self, stmt) -> str:
        if isinstance(stmt, AssignmentNode): return f"{stmt.target.name} = ..."
        return str(type(stmt).__name__)