"""
Lark grammar definition for pseudocode parsing.

This grammar is designed to be flexible and handle various pseudocode styles.
It uses a simplified approach to avoid LALR conflicts.
"""

# Simplified Lark grammar for pseudocode
# Focuses on structure detection rather than full semantic parsing
PSEUDOCODE_GRAMMAR = r'''
start: (statement | function_def | _NEWLINE)*

// Function definition
function_def: FUNC_KW NAME "(" [params] ")" _NEWLINE? block
FUNC_KW: "function"i | "algorithm"i | "procedure"i | "def"i

params: NAME ("," NAME)*

// Block (indentation, end-delimited, or curly braces)
block: "{" _NEWLINE? (statement _NEWLINE?)* "}"
     | _NEWLINE (statement _NEWLINE?)* END_KW

END_KW: "end"i | "endif"i | "endfor"i | "endwhile"i | "done"i | "end" NAME

// Statements
statement: for_stmt
         | while_stmt
         | if_stmt
         | repeat_stmt
         | return_stmt
         | call_stmt
         | assignment
         | expr

// Call statement (standalone function call with CALL keyword)
call_stmt: "call"i NAME "(" [args] ")"

// For loop
for_stmt: "for"i NAME "=" expr "to"i expr ("step"i expr)? ("do"i)? _NEWLINE? block
        | "for"i NAME "=" expr "downto"i expr ("step"i expr)? ("do"i)? _NEWLINE? block
        | "for"i ("each"i)? NAME "in"i expr ("do"i)? _NEWLINE? block

// While loop
while_stmt: "while"i expr ("do"i)? _NEWLINE? block

// Repeat until
repeat_stmt: "repeat"i _NEWLINE? block "until"i expr

// If statement
if_stmt: "if"i expr ("then"i)? _NEWLINE? block (elif_branch)* (else_branch)?

elif_branch: "elif"i expr ("then"i)? _NEWLINE? block
           | "elseif"i expr ("then"i)? _NEWLINE? block
           | "else"i "if"i expr ("then"i)? _NEWLINE? block

else_branch: "else"i _NEWLINE? block

// Return statement
return_stmt: "return"i expr?

// Assignment
assignment: NAME "=" expr
          | NAME "[" expr "]" "=" expr

// Expressions
?expr: or_expr

?or_expr: and_expr ( ("or"i | "||") and_expr)*

?and_expr: not_expr ( ("and"i | "&&") not_expr)*

?not_expr: ("not"i | "!") not_expr -> not_op
         | comparison

?comparison: arith (COMP_OP arith)*
COMP_OP: "==" | "!=" | "<=" | ">=" | "<" | ">"

?arith: term (("+"|"-") term)*

?term: factor (("*"|"/"|"//"|"%") factor)*

?factor: power
       | "-" power -> neg
       | "+" power -> pos

?power: atom ("^" atom)*

?atom: NUMBER
     | STRING
     | "true"i -> true
     | "false"i -> false
     | "call"i NAME "(" [args] ")" -> func_call
     | NAME "(" [args] ")" -> func_call
     | NAME "[" expr "]" -> array_access
     | NAME -> var
     | "(" expr ")"

args: expr ("," expr)*

// Terminals
NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
NUMBER: /\d+(\.\d+)?/
STRING: /"[^"]*"/ | /'[^']*'/

// Whitespace
_NEWLINE: /\r?\n/

// Comments
COMMENT: "//" /[^\n]*/ | "#" /[^\n]*/

%ignore COMMENT
%ignore /[ \t\f]+/
'''


# Simplified grammar for fallback - focuses on structure detection only
SIMPLIFIED_GRAMMAR = r'''
start: line*

line: _NL
    | loop_line
    | conditional_line
    | function_line
    | return_line
    | other_line

loop_line: LOOP_KEYWORD /[^\n]*/
conditional_line: COND_KEYWORD /[^\n]*/
function_line: FUNC_KEYWORD /[^\n]*/
return_line: RETURN_KEYWORD /[^\n]*/
other_line: /[^\n]+/

LOOP_KEYWORD: /\b(for|while|repeat|do|loop)\b/i
COND_KEYWORD: /\b(if|else|elif|then)\b/i
FUNC_KEYWORD: /\b(function|algorithm|procedure|def)\b/i
RETURN_KEYWORD: /\b(return)\b/i

_NL: /\r?\n/

%ignore /[\t \f]+/
'''