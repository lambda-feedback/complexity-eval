"""
Fixed Interpreter for AST execution.

FIXES:
1. Added ExpressionStatementNode handling
2. Added None check in ReturnNode
3. Implemented FOR_EACH loops
4. Implemented REPEAT_UNTIL loops
5. Added None check in conditionals
6. Added None check in evaluate_expression
7. Added print() built-in function
8. Added None check for function body
9. Improved error messages
"""

from typing import Any, Dict, List, Optional, Union
from copy import deepcopy

from ..schemas.input_schema import RuntimeValue
from ..schemas.ast_nodes import *


class ReturnException(Exception):
    """Exception raised when a return statement is executed."""
    def __init__(self, value: RuntimeValue):
        self.value = value


class Interpreter:
    """
    Interprets and executes pseudocode AST.
    
    Maintains:
    - variables: Current variable scope
    - functions: Registered function definitions
    - output: Captured print output
    """
    
    def __init__(self):
        self.variables: Dict[str, RuntimeValue] = {}
        self.functions: Dict[str, FunctionNode] = {}
        self.output: List[str] = []

    # -----------------------------------
    # Public Entry Point
    # -----------------------------------

    def run(
        self,
        program: ProgramNode,
        initial_variables: Optional[Dict[str, RuntimeValue]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a program with optional initial variables.
        
        Returns:
            Dict with 'variables' and 'output' keys
        """
        self.variables = deepcopy(initial_variables) if initial_variables else {}
        self.functions = {}
        self.output = []

        # Register functions first
        for func in program.functions:
            self.functions[func.name] = func

        # Execute global statements
        if program.global_statements:
            self.execute_block(program.global_statements)

        return {
            "variables": deepcopy(self.variables),
            "output": list(self.output)
        }

    # -----------------------------------
    # Block Execution
    # -----------------------------------

    def execute_block(self, block: BlockNode):
        """Execute all statements in a block."""
        for stmt in block.statements:
            self.execute_statement(stmt)

    # -----------------------------------
    # Statement Execution
    # -----------------------------------

    def execute_statement(self, node: ASTNode):
        """Execute a single statement node."""

        if isinstance(node, AssignmentNode):
            value = self.evaluate_expression(node.value)
            if isinstance(node.target, VariableNode):
                self.variables[node.target.name] = value
            elif isinstance(node.target, ArrayAccessNode):
                arr = self.evaluate_expression(node.target.array)
                index = int(self.evaluate_expression(node.target.index))
                arr[index] = value
            else:
                raise RuntimeError(f"Invalid assignment target: {type(node.target)}")

        elif isinstance(node, ReturnNode):
            # FIX #2: Check if value is None
            value = self.evaluate_expression(node.value) if node.value else None
            raise ReturnException(value)

        elif isinstance(node, ExpressionStatementNode):
            # FIX #1: Handle expression statements
            self.evaluate_expression(node.expression)

        elif isinstance(node, FunctionCallNode):
            # Standalone function call (not as expression)
            self.call_function(node)

        elif isinstance(node, LoopNode):
            self.execute_loop(node)

        elif isinstance(node, ConditionalNode):
            self.execute_conditional(node)

        # elif isinstance(node, FunctionNode):
        #     # Already registered in run()
        #     pass

        elif isinstance(node, BlockNode):
            self.execute_block(node)

        else:
            raise NotImplementedError(f"Unsupported statement: {type(node).__name__}")

    # -----------------------------------
    # Loop Execution
    # -----------------------------------

    def execute_loop(self, node: LoopNode):
        """Execute different loop types."""

        if node.loop_type == LoopType.FOR:
            if not node.start or not node.end:
                raise RuntimeError("FOR loop requires start and end values")
            
            start = int(self.evaluate_expression(node.start))
            end = int(self.evaluate_expression(node.end))
            step = int(self.evaluate_expression(node.step)) if node.step else 1

            if not node.iterator:
                raise RuntimeError("FOR loop requires iterator variable")

            for i in range(start, end + 1, step):
                self.variables[node.iterator.name] = i
                if node.body:
                    self.execute_block(node.body)

        elif node.loop_type == LoopType.WHILE:
            # FIX #5: Check for None condition
            if not node.condition:
                raise RuntimeError("WHILE loop requires condition")
            
            while self.evaluate_expression(node.condition):
                if node.body:
                    self.execute_block(node.body)

        elif node.loop_type == LoopType.FOR_EACH:
            # FIX #3: Implement FOR_EACH
            if not node.collection:
                raise RuntimeError("FOR_EACH loop requires collection")
            if not node.iterator:
                raise RuntimeError("FOR_EACH loop requires iterator variable")
            
            collection = self.evaluate_expression(node.collection)
            if not isinstance(collection, (list, tuple, str)):
                raise RuntimeError(f"Cannot iterate over {type(collection).__name__}")
            
            for item in collection:
                self.variables[node.iterator.name] = item
                if node.body:
                    self.execute_block(node.body)

        elif node.loop_type == LoopType.REPEAT_UNTIL:
            # FIX #4: Implement REPEAT_UNTIL
            if not node.condition:
                raise RuntimeError("REPEAT_UNTIL loop requires condition")
            
            while True:
                if node.body:
                    self.execute_block(node.body)
                if self.evaluate_expression(node.condition):
                    break

        elif node.loop_type == LoopType.DO_WHILE:
            # Bonus: DO_WHILE implementation
            if not node.condition:
                raise RuntimeError("DO_WHILE loop requires condition")
            
            while True:
                if node.body:
                    self.execute_block(node.body)
                if not self.evaluate_expression(node.condition):
                    break

        else:
            raise NotImplementedError(f"Loop type {node.loop_type} not supported")

    # -----------------------------------
    # Conditional Execution
    # -----------------------------------

    def execute_conditional(self, node: ConditionalNode):
        """Execute IF/ELIF/ELSE conditional."""
        
        # FIX #5: Check for None condition
        if node.condition is None:
            # If no condition parsed, skip the conditional
            return

        if self.evaluate_expression(node.condition):
            self.execute_block(node.then_branch)
            return

        # Check elif branches
        for elif_branch in node.elif_branches:
            if elif_branch.condition and self.evaluate_expression(elif_branch.condition):
                self.execute_block(elif_branch.then_branch)
                return

        # Execute else branch if present
        if node.else_branch:
            self.execute_block(node.else_branch)

    # -----------------------------------
    # Expression Evaluation
    # -----------------------------------

    def evaluate_expression(self, node: Optional[ExpressionNode]) -> RuntimeValue:
        """Evaluate an expression and return its value."""
        
        # FIX #6: Handle None input
        if node is None:
            return None

        if isinstance(node, LiteralNode):
            return node.value

        if isinstance(node, VariableNode):
            if node.name not in self.variables:
                raise NameError(f"Variable '{node.name}' is not defined")
            return self.variables[node.name]

        if isinstance(node, UnaryOpNode):
            value = self.evaluate_expression(node.operand)
            if node.operator == OperatorType.SUBTRACT:
                return -value
            if node.operator == OperatorType.NOT:
                return not value
            raise NotImplementedError(f"Unary operator {node.operator} not implemented")

        if isinstance(node, BinaryOpNode):
            left = self.evaluate_expression(node.left)
            right = self.evaluate_expression(node.right)
            op = node.operator

            # Arithmetic
            if op == OperatorType.ADD:
                return left + right
            if op == OperatorType.SUBTRACT:
                return left - right
            if op == OperatorType.MULTIPLY:
                return left * right
            if op == OperatorType.DIVIDE:
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                return left / right
            if op == OperatorType.FLOOR_DIVIDE:
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                return left // right
            if op == OperatorType.MODULO:
                return left % right
            if op == OperatorType.POWER:
                return left ** right

            # Comparisons
            if op == OperatorType.EQUAL:
                return left == right
            if op == OperatorType.NOT_EQUAL:
                return left != right
            if op == OperatorType.LESS_THAN:
                return left < right
            if op == OperatorType.LESS_EQUAL:
                return left <= right
            if op == OperatorType.GREATER_THAN:
                return left > right
            if op == OperatorType.GREATER_EQUAL:
                return left >= right

            # Logical
            if op == OperatorType.AND:
                return bool(left) and bool(right)
            if op == OperatorType.OR:
                return bool(left) or bool(right)

            raise NotImplementedError(f"Binary operator {op} not implemented")

        if isinstance(node, ArrayAccessNode):
            arr = self.evaluate_expression(node.array)
            if not isinstance(arr, (list, tuple, str)):
                raise TypeError(f"Cannot index {type(arr).__name__}")
            index = int(self.evaluate_expression(node.index))
            if index < 0 or index >= len(arr):
                raise IndexError(f"Index {index} out of range for array of length {len(arr)}")
            return arr[index]

        if isinstance(node, RecursiveCallNode):
            return self.call_function(node)

        if isinstance(node, FunctionCallNode):
            return self.call_function(node)

        raise NotImplementedError(f"Unsupported expression: {type(node).__name__}")

    # -----------------------------------
    # Function Calls
    # -----------------------------------

    def call_function(self, node: FunctionCallNode) -> RuntimeValue:
        """Call a function with arguments."""
        
        # FIX #11: Handle built-in functions
        if node.function_name == "print":
            # Print function outputs to self.output
            values = [self.evaluate_expression(arg) for arg in node.arguments]
            output_str = " ".join(str(v) for v in values)
            self.output.append(output_str)
            return None

        # Check if function exists
        if node.function_name not in self.functions:
            raise NameError(f"Function '{node.function_name}' is not defined")

        func = self.functions[node.function_name]

        # Evaluate arguments in current scope
        arg_values = [self.evaluate_expression(arg) for arg in node.arguments]

        # Check argument count
        if len(arg_values) != len(func.parameters):
            raise TypeError(
                f"Function '{func.name}' expects {len(func.parameters)} arguments, "
                f"got {len(arg_values)}"
            )

        # FIXED SCOPING: Only save/restore parameter values, not all variables
        # This allows functions to modify global variables
        saved_param_values = {}
        for param in func.parameters:
            # Save current value of parameter name (if it exists globally)
            if param.name in self.variables:
                saved_param_values[param.name] = self.variables[param.name]

        # Set parameter values
        for param, value in zip(func.parameters, arg_values):
            self.variables[param.name] = value

        try:
            # FIX #12: Check if body exists
            if func.body:
                self.execute_block(func.body)
            
            # Restore only parameter values, leave other variables modified
            for param in func.parameters:
                if param.name in saved_param_values:
                    # Restore the saved value
                    self.variables[param.name] = saved_param_values[param.name]
                else:
                    # Parameter didn't exist before, remove it
                    if param.name in self.variables:
                        del self.variables[param.name]
            
            # If no return statement, return None (following Python logic)
            return None
        except ReturnException as e:
            # Restore only parameter values
            for param in func.parameters:
                if param.name in saved_param_values:
                    self.variables[param.name] = saved_param_values[param.name]
                else:
                    if param.name in self.variables:
                        del self.variables[param.name]
            
            return e.value