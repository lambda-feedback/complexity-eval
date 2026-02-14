"""
Fixed Code Runner - orchestrates parsing and execution.

FIXES:
- Better error handling with specific exception types
- Improved error messages
- Handles ReturnException at global scope
"""

from typing import List
from ..schemas.input_schema import ExecutionTestCase
from .interpreter import Interpreter, ReturnException
from ..schemas.output_schema import CodeCorrectnessResult, TestCaseResult, ParseResult
from ..parser.parser import PseudocodeParser


class CodeRunner:
    """
    Orchestrates parsing and execution of pseudocode.
    
    Workflow:
    1. Parse source code to AST
    2. Execute AST with each test case
    3. Compare results with expected values
    4. Return aggregated results
    """

    def __init__(self, parser: PseudocodeParser, interpreter: Interpreter):
        self.parser = parser
        self.interpreter = interpreter

    def run(
        self,
        source_code: str,
        test_cases: List[ExecutionTestCase]
    ) -> CodeCorrectnessResult:
        """
        Parse and execute pseudocode with test cases.
        
        Args:
            source_code: Pseudocode to parse and execute
            test_cases: List of test cases with inputs and expected outputs
            
        Returns:
            CodeCorrectnessResult with parse status and execution results
        """

        # -----------------------------------
        # 1. Parse
        # -----------------------------------

        parse_result: ParseResult = self.parser.parse(source_code)

        if not parse_result.success or not parse_result.ast:
            return CodeCorrectnessResult(
                parse_success=False,
                parse_errors=parse_result.errors,
                parse_warnings=parse_result.warnings,
                normalized_code=parse_result.normalized_code,
                execution_results=[],
                is_correct=False,
                feedback="Parsing failed. Fix syntax errors before execution."
            )

        # -----------------------------------
        # 2. Execute Test Cases
        # -----------------------------------

        execution_results: List[TestCaseResult] = []

        for i, test_case in enumerate(test_cases):
            try:
                result = self.interpreter.run(
                    parse_result.ast,
                    initial_variables=test_case.initial_variables
                )

                # Check if results match expectations
                passed = True
                error_messages = []

                if test_case.expected_variables is not None:
                    if result["variables"] != test_case.expected_variables:
                        passed = False
                        error_messages.append(
                            f"Variables mismatch: expected {test_case.expected_variables}, "
                            f"got {result['variables']}"
                        )

                if test_case.expected_output is not None:
                    if result["output"] != test_case.expected_output:
                        passed = False
                        error_messages.append(
                            f"Output mismatch: expected {test_case.expected_output}, "
                            f"got {result['output']}"
                        )

                execution_results.append(
                    TestCaseResult(
                        input_data=test_case.initial_variables,
                        expected_output={
                            "variables": test_case.expected_variables,
                            "output": test_case.expected_output
                        },
                        actual_output=result,
                        passed=passed,
                        error_message="; ".join(error_messages) if error_messages else None
                    )
                )

            # FIX #13: More specific error handling
            except ReturnException as e:
                # Return at global scope - shouldn't happen
                execution_results.append(
                    TestCaseResult(
                        input_data=test_case.initial_variables,
                        expected_output={
                            "variables": test_case.expected_variables,
                            "output": test_case.expected_output
                        },
                        actual_output=None,
                        passed=False,
                        error_message=f"Unexpected return statement (returned {e.value})"
                    )
                )
            
            except NameError as e:
                execution_results.append(
                    TestCaseResult(
                        input_data=test_case.initial_variables,
                        expected_output={
                            "variables": test_case.expected_variables,
                            "output": test_case.expected_output
                        },
                        actual_output=None,
                        passed=False,
                        error_message=f"Variable or function not defined: {str(e)}"
                    )
                )
            
            except ZeroDivisionError as e:
                execution_results.append(
                    TestCaseResult(
                        input_data=test_case.initial_variables,
                        expected_output={
                            "variables": test_case.expected_variables,
                            "output": test_case.expected_output
                        },
                        actual_output=None,
                        passed=False,
                        error_message="Division by zero"
                    )
                )
            
            except IndexError as e:
                execution_results.append(
                    TestCaseResult(
                        input_data=test_case.initial_variables,
                        expected_output={
                            "variables": test_case.expected_variables,
                            "output": test_case.expected_output
                        },
                        actual_output=None,
                        passed=False,
                        error_message=f"Array index error: {str(e)}"
                    )
                )
            
            except TypeError as e:
                execution_results.append(
                    TestCaseResult(
                        input_data=test_case.initial_variables,
                        expected_output={
                            "variables": test_case.expected_variables,
                            "output": test_case.expected_output
                        },
                        actual_output=None,
                        passed=False,
                        error_message=f"Type error: {str(e)}"
                    )
                )
            
            except RuntimeError as e:
                execution_results.append(
                    TestCaseResult(
                        input_data=test_case.initial_variables,
                        expected_output={
                            "variables": test_case.expected_variables,
                            "output": test_case.expected_output
                        },
                        actual_output=None,
                        passed=False,
                        error_message=f"Runtime error: {str(e)}"
                    )
                )
            
            except Exception as e:
                # Catch-all for unexpected errors
                execution_results.append(
                    TestCaseResult(
                        input_data=test_case.initial_variables,
                        expected_output={
                            "variables": test_case.expected_variables,
                            "output": test_case.expected_output
                        },
                        actual_output=None,
                        passed=False,
                        error_message=f"{type(e).__name__}: {str(e)}"
                    )
                )

        # -----------------------------------
        # 3. Aggregate Results
        # -----------------------------------

        all_passed = all(r.passed for r in execution_results)

        # Generate feedback
        if not execution_results:
            feedback = "Parsing successful. No test cases provided."
            is_correct = True
        elif all_passed:
            feedback = f"All {len(execution_results)} test case(s) passed! ✅"
            is_correct = True
        else:
            failed_count = sum(1 for r in execution_results if not r.passed)
            feedback = (f"{failed_count} of {len(execution_results)} test case(s) failed. "
                       f"Check execution results for details.")
            is_correct = False

        return CodeCorrectnessResult(
            parse_success=True,
            parse_errors=parse_result.errors,
            parse_warnings=parse_result.warnings,
            normalized_code=parse_result.normalized_code,
            execution_results=execution_results,
            is_correct=is_correct,
            feedback=feedback
        )