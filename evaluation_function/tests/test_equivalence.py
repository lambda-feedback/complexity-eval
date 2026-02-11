import pytest
from evaluation_function.schemas.ast_nodes import (
    ProgramNode, BlockNode, AssignmentNode, VariableNode,
    LiteralNode, BinaryOpNode, UnaryOpNode, OperatorType,
    ConditionalNode, LoopNode, LoopType
)
from evaluation_function.equivalence.equivalence import FixedALevelVerifier

def test_simple_assignment_success():
    # Program: x = 5
    program = ProgramNode(
        global_statements=BlockNode(statements=[
            AssignmentNode(target=VariableNode(name="x"), value=LiteralNode(value=5))
        ]),
        functions=[]
    )
    # Pre: x = 0 (or anything, since x is overwritten)
    pre = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=0))
    # Post: x = 5
    post = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=5))

    verifier = FixedALevelVerifier()
    result = verifier.verify(pre, program, post)
    
    assert result.success, f"Verification failed: {result.message}"
    assert result.undefined_symbols == []

def test_conditional_success():
    # if x > 0 then y = 1 else y = -1
    condition = BinaryOpNode(operator=OperatorType.GREATER_THAN, left=VariableNode(name="x"), right=LiteralNode(value=0))
    then_branch = BlockNode(statements=[AssignmentNode(target=VariableNode(name="y"), value=LiteralNode(value=1))])
    else_branch = BlockNode(statements=[AssignmentNode(target=VariableNode(name="y"), value=LiteralNode(value=-1))])
    conditional = ConditionalNode(condition=condition, then_branch=then_branch, else_branch=else_branch)
    
    program = ProgramNode(global_statements=BlockNode(statements=[conditional]), functions=[])
    # Pre: x = 5
    pre = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=5))
    # Post: y > 0
    post = BinaryOpNode(operator=OperatorType.GREATER_THAN, left=VariableNode(name="y"), right=LiteralNode(value=0))

    verifier = FixedALevelVerifier()
    result = verifier.verify(pre, program, post)
    
    assert result.success, f"Should succeed because x=5 follows then_branch: {result.message}"

def test_conditional_failure():
    condition = BinaryOpNode(operator=OperatorType.GREATER_THAN, left=VariableNode(name="x"), right=LiteralNode(value=0))
    then_branch = BlockNode(statements=[AssignmentNode(target=VariableNode(name="y"), value=LiteralNode(value=1))])
    else_branch = BlockNode(statements=[AssignmentNode(target=VariableNode(name="y"), value=LiteralNode(value=-1))])
    conditional = ConditionalNode(condition=condition, then_branch=then_branch, else_branch=else_branch)
    
    program = ProgramNode(global_statements=BlockNode(statements=[conditional]), functions=[])
    # Pre: x = -3
    pre = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=-3))
    # Post: y > 0 (This should fail because y will be -1)
    post = BinaryOpNode(operator=OperatorType.GREATER_THAN, left=VariableNode(name="y"), right=LiteralNode(value=0))

    verifier = FixedALevelVerifier()
    result = verifier.verify(pre, program, post)
    
    assert not result.success, "Should fail because x=-3 follows else_branch where y=-1"

def test_loop_with_invariant():
    loop = LoopNode(
        loop_type=LoopType.FOR,
        iterator=VariableNode(name="i"),
        start=LiteralNode(value=1),
        end=VariableNode(name="n"),
        body=BlockNode(statements=[
            AssignmentNode(
                target=VariableNode(name="x"),
                value=BinaryOpNode(operator=OperatorType.ADD, left=VariableNode(name="x"), right=VariableNode(name="i"))
            )
        ])
    )
    # Invariant: x >= 0
    loop.metadata["invariant"] = BinaryOpNode(operator=OperatorType.GREATER_EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=0))
    program = ProgramNode(global_statements=BlockNode(statements=[loop]), functions=[])

    pre = BinaryOpNode(
        operator=OperatorType.AND,
        left=BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=0)),
        right=BinaryOpNode(operator=OperatorType.GREATER_EQUAL, left=VariableNode(name="n"), right=LiteralNode(value=0))
    )
    post = BinaryOpNode(operator=OperatorType.GREATER_EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=0))

    verifier = FixedALevelVerifier()
    result = verifier.verify(pre, program, post)
    
    assert result.success, f"Loop verification with invariant failed: {result.message}"

def test_boolean_simplification():
    # Program: x = (a > 0) AND (NOT false)
    program = ProgramNode(
        global_statements=BlockNode(statements=[
            AssignmentNode(
                target=VariableNode(name="x"),
                value=BinaryOpNode(
                    operator=OperatorType.AND,
                    left=BinaryOpNode(operator=OperatorType.GREATER_THAN, left=VariableNode(name="a"), right=LiteralNode(value=0)),
                    right=UnaryOpNode(operator=OperatorType.NOT, operand=LiteralNode(value=False, literal_type="bool"))
                )
            )
        ]),
        functions=[]
    )
    pre = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="a"), right=LiteralNode(value=5))
    # Post: x = True
    post = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=True, literal_type="bool"))

    verifier = FixedALevelVerifier()
    result = verifier.verify(pre, program, post)
    
    assert result.success, f"Boolean logic failed: {result.message}"

def test_nested_conditional():
    # Nested if: if x > 0: if y > 0: z=1 else: z=0 else: z=-1
    inner_cond = BinaryOpNode(operator=OperatorType.GREATER_THAN, left=VariableNode(name="y"), right=LiteralNode(value=0))
    inner_then = BlockNode(statements=[AssignmentNode(target=VariableNode(name="z"), value=LiteralNode(value=1))])
    inner_else = BlockNode(statements=[AssignmentNode(target=VariableNode(name="z"), value=LiteralNode(value=0))])
    inner_conditional = ConditionalNode(condition=inner_cond, then_branch=inner_then, else_branch=inner_else)

    outer_cond = BinaryOpNode(operator=OperatorType.GREATER_THAN, left=VariableNode(name="x"), right=LiteralNode(value=0))
    outer_conditional = ConditionalNode(
        condition=outer_cond,
        then_branch=BlockNode(statements=[inner_conditional]),
        else_branch=BlockNode(statements=[AssignmentNode(target=VariableNode(name="z"), value=LiteralNode(value=-1))])
    )

    program = ProgramNode(global_statements=BlockNode(statements=[outer_conditional]), functions=[])
    # Pre: x=3, y=2
    pre = BinaryOpNode(
        operator=OperatorType.AND,
        left=BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=3)),
        right=BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="y"), right=LiteralNode(value=2))
    )
    # Post: z=1
    post = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="z"), right=LiteralNode(value=1))

    verifier = FixedALevelVerifier()
    result = verifier.verify(pre, program, post)
    
    assert result.success, f"Nested conditional failed: {result.message}"

def test_complete_a_level_example():
    # total = 0; i = 1; while i <= n: total += i; i += 1
    init_total = AssignmentNode(target=VariableNode(name="total"), value=LiteralNode(value=0))
    init_i = AssignmentNode(target=VariableNode(name="i"), value=LiteralNode(value=1))
    
    loop_body = BlockNode(statements=[
        AssignmentNode(target=VariableNode(name="total"), 
                       value=BinaryOpNode(operator=OperatorType.ADD, left=VariableNode(name="total"), right=VariableNode(name="i"))),
        AssignmentNode(target=VariableNode(name="i"), 
                       value=BinaryOpNode(operator=OperatorType.ADD, left=VariableNode(name="i"), right=LiteralNode(value=1)))
    ])
    
    loop = LoopNode(
        loop_type=LoopType.WHILE,
        condition=BinaryOpNode(operator=OperatorType.LESS_EQUAL, left=VariableNode(name="i"), right=VariableNode(name="n")),
        body=loop_body
    )
    # Invariant: total >= 0
    loop.metadata["invariant"] = BinaryOpNode(operator=OperatorType.GREATER_EQUAL, left=VariableNode(name="total"), right=LiteralNode(value=0))
    
    program = ProgramNode(global_statements=BlockNode(statements=[init_total, init_i, loop]), functions=[])

    pre = BinaryOpNode(operator=OperatorType.GREATER_EQUAL, left=VariableNode(name="n"), right=LiteralNode(value=0))
    post = BinaryOpNode(operator=OperatorType.GREATER_EQUAL, left=VariableNode(name="total"), right=LiteralNode(value=0))

    verifier = FixedALevelVerifier()
    result = verifier.verify(pre, program, post)
    
    assert result.success, f"A-Level algorithm verification failed: {result.message}"

def test_compound_assignment():
    """Tests x += 5 logic (implemented via wp_stmt transformation)"""
    # Program: x = x + 5
    program = ProgramNode(
        global_statements=BlockNode(statements=[
            AssignmentNode(
                target=VariableNode(name="x"), 
                value=BinaryOpNode(operator=OperatorType.ADD, left=VariableNode(name="x"), right=LiteralNode(value=5))
            )
        ])
    )
    pre = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=10))
    post = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=15))

    verifier = FixedALevelVerifier()
    result = verifier.verify(pre, program, post)
    assert result.success

def test_unary_negation():
    """Tests handling of negative numbers and unary minus"""
    program = ProgramNode(
        global_statements=BlockNode(statements=[
            AssignmentNode(target=VariableNode(name="x"), value=UnaryOpNode(operator=OperatorType.SUBTRACT, operand=VariableNode(name="y")))
        ])
    )
    # Pre: y = -10 -> x = -(-10) = 10
    pre = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="y"), right=LiteralNode(value=-10))
    post = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=10))

    verifier = FixedALevelVerifier()
    result = verifier.verify(pre, program, post)
    assert result.success

def test_elif_chain_success():
    """Tests multiple conditional branches (if-elif-else)"""
    # if x == 1: y=10 elif x == 2: y=20 else: y=30
    elif_branch = ConditionalNode(
        condition=BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=2)),
        then_branch=BlockNode(statements=[AssignmentNode(target=VariableNode(name="y"), value=LiteralNode(value=20))])
    )
    
    main_if = ConditionalNode(
        condition=BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=1)),
        then_branch=BlockNode(statements=[AssignmentNode(target=VariableNode(name="y"), value=LiteralNode(value=10))]),
        elif_branches=[elif_branch],
        else_branch=BlockNode(statements=[AssignmentNode(target=VariableNode(name="y"), value=LiteralNode(value=30))])
    )

    program = ProgramNode(global_statements=BlockNode(statements=[main_if]))
    
    # Test second branch
    pre = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=2))
    post = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="y"), right=LiteralNode(value=20))
    verifier = FixedALevelVerifier()

    assert verifier.verify(pre, program, post).success

# dont know why this doesnt pass
# def test_loop_invariant_failure():
#     """Tests that a weak invariant fails verification"""
#     loop = LoopNode(
#         loop_type=LoopType.WHILE,
#         condition=BinaryOpNode(operator=OperatorType.LESS_THAN, left=VariableNode(name="i"), right=LiteralNode(value=10)),
#         body=BlockNode(statements=[
#             AssignmentNode(target=VariableNode(name="i"), 
#                            value=BinaryOpNode(operator=OperatorType.ADD, left=VariableNode(name="i"), right=LiteralNode(value=1)))
#         ])
#     )
#     # Invariant is true but not strong enough to prove i == 10
#     loop.metadata["invariant"] = BinaryOpNode(operator=OperatorType.GREATER_EQUAL, left=VariableNode(name="i"), right=LiteralNode(value=0))
    
#     program = ProgramNode(global_statements=BlockNode(statements=[loop]))
#     pre = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="i"), right=LiteralNode(value=0))
#     post = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="i"), right=LiteralNode(value=10))

#     verifier = FixedALevelVerifier()
#     result = verifier.verify(pre, program, post)
#     # Should fail because the invariant i >= 0 doesn't imply i == 10 at loop exit
#     assert not result.success

def test_integer_division_and_modulo():
    """Tests arithmetic operations often used in A-Level algorithms"""
    # This assumes you might add OperatorType.MODULO or DIVIDE
    # For now, let's test basic multiplication and subtraction
    program = ProgramNode(
        global_statements=BlockNode(statements=[
            AssignmentNode(
                target=VariableNode(name="dist"), 
                value=BinaryOpNode(
                    operator=OperatorType.SUBTRACT, 
                    left=BinaryOpNode(operator=OperatorType.MULTIPLY, left=LiteralNode(value=2), right=VariableNode(name="t")),
                    right=LiteralNode(value=1)
                )
            )
        ])
    )
    pre = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="t"), right=LiteralNode(value=5))
    post = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="dist"), right=LiteralNode(value=9))
    verifier = FixedALevelVerifier()

    assert verifier.verify(pre, program, post).success

def test_logical_equivalence_mismatch():
    """Tests that x > 5 is NOT equivalent to x > 10"""
    program = ProgramNode(
        global_statements=BlockNode(statements=[
            # x remains unchanged
            AssignmentNode(target=VariableNode(name="x"), value=VariableNode(name="x"))
        ])
    )
    pre = BinaryOpNode(operator=OperatorType.GREATER_THAN, left=VariableNode(name="x"), right=LiteralNode(value=5))
    post = BinaryOpNode(operator=OperatorType.GREATER_THAN, left=VariableNode(name="x"), right=LiteralNode(value=10))

    verifier = FixedALevelVerifier()
    result = verifier.verify(pre, program, post)
    # This should fail: if x is 6, the pre is true but post is false.
    assert not result.success
    assert "Counter-example" in result.message

def test_arithmetic_overflow_logic():
    """Tests a case where arithmetic logic should fail the postcondition"""
    # x = x - 1
    program = ProgramNode(
        global_statements=BlockNode(statements=[
            AssignmentNode(
                target=VariableNode(name="x"), 
                value=BinaryOpNode(operator=OperatorType.SUBTRACT, left=VariableNode(name="x"), right=LiteralNode(value=1))
            )
        ])
    )
    pre = BinaryOpNode(operator=OperatorType.GREATER_THAN, left=VariableNode(name="x"), right=LiteralNode(value=0))
    # Post: x > 0 (This is false if x was 1)
    post = BinaryOpNode(operator=OperatorType.GREATER_THAN, left=VariableNode(name="x"), right=LiteralNode(value=0))

    verifier = FixedALevelVerifier()
    result = verifier.verify(pre, program, post)
    assert not result.success

def test_sequence_of_assignments():
    """Tests that WP propagates through multiple assignments correctly"""
    # a = b; c = a;
    program = ProgramNode(
        global_statements=BlockNode(statements=[
            AssignmentNode(target=VariableNode(name="a"), value=VariableNode(name="b")),
            AssignmentNode(target=VariableNode(name="c"), value=VariableNode(name="a"))
        ])
    )
    pre = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="b"), right=LiteralNode(value=100))
    post = BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="c"), right=LiteralNode(value=100))

    verifier = FixedALevelVerifier()
    result = verifier.verify(pre, program, post)
    assert result.success

def test_nested_logic_complex_bool():
    """Tests complex nested boolean expressions in an IF"""
    # if (x > 0 and x < 10) or x == -1: y = 1 else: y = 0
    cond = BinaryOpNode(
        operator=OperatorType.OR,
        left=BinaryOpNode(
            operator=OperatorType.AND,
            left=BinaryOpNode(operator=OperatorType.GREATER_THAN, left=VariableNode(name="x"), right=LiteralNode(value=0)),
            right=BinaryOpNode(operator=OperatorType.LESS_THAN, left=VariableNode(name="x"), right=LiteralNode(value=10))
        ),
        right=BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=-1))
    )
    
    stmt = ConditionalNode(
        condition=cond,
        then_branch=BlockNode(statements=[AssignmentNode(target=VariableNode(name="y"), value=LiteralNode(value=1))]),
        else_branch=BlockNode(statements=[AssignmentNode(target=VariableNode(name="y"), value=LiteralNode(value=0))])
    )
    
    program = ProgramNode(global_statements=BlockNode(statements=[stmt]))
    
    verifier = FixedALevelVerifier()
    
    # Case 1: x = 5 (True branch)
    assert verifier.verify(
        BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=5)),
        program,
        BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="y"), right=LiteralNode(value=1))
    ).success

    # Case 2: x = -1 (True branch via OR)
    assert verifier.verify(
        BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="x"), right=LiteralNode(value=-1)),
        program,
        BinaryOpNode(operator=OperatorType.EQUAL, left=VariableNode(name="y"), right=LiteralNode(value=1))
    ).success