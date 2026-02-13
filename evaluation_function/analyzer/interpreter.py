from typing import Any, Dict, List, Union
from copy import deepcopy

from ..schemas.ast_nodes import *

RuntimeValue = Union[int, float, str, bool]


class ReturnException(Exception):
    def __init__(self, value: RuntimeValue):
        self.value = value


class Interpreter:
    def __init__(self):
        self.variables: Dict[str, RuntimeValue] = {}
        self.functions: Dict[str, FunctionNode] = {}
        self.output: List[str] = []

    # -----------------------------------
    # Public Entry Point
    # -----------------------------------

    def run(self, program: ProgramNode):
        self.variables = {}
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
        for stmt in block.statements:
            self.execute_statement(stmt)

    # -----------------------------------
    # Statement Execution
    # -----------------------------------

    def execute_statement(self, node: ASTNode):

        if isinstance(node, AssignmentNode):
            value = self.evaluate_expression(node.value)
            if isinstance(node.target, VariableNode):
                self.variables[node.target.name] = value
            elif isinstance(node.target, ArrayAccessNode):
                arr = self.evaluate_expression(node.target.array)
                index = int(self.evaluate_expression(node.target.index))
                arr[index] = value

        elif isinstance(node, ReturnNode):
            value = self.evaluate_expression(node.value)
            raise ReturnException(value)

        elif isinstance(node, FunctionCallNode):
            self.call_function(node)

        elif isinstance(node, LoopNode):
            self.execute_loop(node)

        elif isinstance(node, ConditionalNode):
            self.execute_conditional(node)

        elif isinstance(node, FunctionNode):
            # Already registered in run()
            pass

        elif isinstance(node, BlockNode):
            self.execute_block(node)

        else:
            raise NotImplementedError(f"Unsupported statement: {type(node)}")

    # -----------------------------------
    # Loop Execution
    # -----------------------------------

    def execute_loop(self, node: LoopNode):

        if node.loop_type == LoopType.FOR:
            start = int(self.evaluate_expression(node.start))
            end = int(self.evaluate_expression(node.end))
            step = int(self.evaluate_expression(node.step)) if node.step else 1

            for i in range(start, end + 1, step):
                self.variables[node.iterator.name] = i
                self.execute_block(node.body)

        elif node.loop_type == LoopType.WHILE:
            while self.evaluate_expression(node.condition):
                self.execute_block(node.body)

        else:
            raise NotImplementedError(f"Loop type {node.loop_type} not supported")

    # -----------------------------------
    # Conditional Execution
    # -----------------------------------

    def execute_conditional(self, node: ConditionalNode):

        if self.evaluate_expression(node.condition):
            self.execute_block(node.then_branch)
            return

        for elif_branch in node.elif_branches:
            if self.evaluate_expression(elif_branch.condition):
                self.execute_block(elif_branch.then_branch)
                return

        if node.else_branch:
            self.execute_block(node.else_branch)

    # -----------------------------------
    # Expression Evaluation
    # -----------------------------------

    def evaluate_expression(self, node: ExpressionNode) -> RuntimeValue:

        if isinstance(node, LiteralNode):
            return node.value

        if isinstance(node, VariableNode):
            return self.variables.get(node.name, 0)

        if isinstance(node, UnaryOpNode):
            value = self.evaluate_expression(node.operand)
            if node.operator == OperatorType.SUBTRACT:
                return -value
            if node.operator == OperatorType.NOT:
                return not value

        if isinstance(node, BinaryOpNode):
            left = self.evaluate_expression(node.left)
            right = self.evaluate_expression(node.right)
            op = node.operator

            if op == OperatorType.ADD:
                return left + right
            if op == OperatorType.SUBTRACT:
                return left - right
            if op == OperatorType.MULTIPLY:
                return left * right
            if op == OperatorType.DIVIDE:
                return left / right
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

        if isinstance(node, ArrayAccessNode):
            arr = self.evaluate_expression(node.array)
            index = int(self.evaluate_expression(node.index))
            return arr[index]

        if isinstance(node, RecursiveCallNode):
            return self.call_function(node)

        if isinstance(node, FunctionCallNode):
            return self.call_function(node)

        raise NotImplementedError(f"Unsupported expression: {type(node)}")

    # -----------------------------------
    # Function Calls
    # -----------------------------------

    def call_function(self, node: FunctionCallNode) -> RuntimeValue:

        if node.function_name not in self.functions:
            raise Exception(f"Function {node.function_name} not defined")

        func = self.functions[node.function_name]

        # Evaluate arguments in current scope
        arg_values = [self.evaluate_expression(arg) for arg in node.arguments]

        # Save previous scope
        previous_vars = deepcopy(self.variables)

        # Set parameters
        for param, value in zip(func.parameters, arg_values):
            self.variables[param.name] = value

        try:
            self.execute_block(func.body)
            self.variables = previous_vars
            return 0
        except ReturnException as e:
            self.variables = previous_vars
            return e.value
