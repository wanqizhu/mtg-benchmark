from __future__ import annotations

_SOLUTION_FORMAT = """When you are ready to answer, put your final winning line inside <solution> tags:

<solution>
1. First step...
2. Second step...
</solution>

Your line must be legal under the rules and must win against all possible opponent responses and blocking decisions.
"""


def solver_system_prompt_tools(clarifications: str, rules_path: str) -> str:
    return f"""You are solving a Magic: The Gathering puzzle.

You have access to local read-only tools, if you want to refer to the full rules:
- grep(path, pattern): search a file and return matching lines with line numbers
- read(path, line_number, num_lines): read a window of lines from a file

The comprehensive rules are available at this absolute path:
{rules_path}

Puzzle conventions:
{clarifications}

{_SOLUTION_FORMAT}
"""


def solver_system_prompt_inline(clarifications: str, rules_text: str) -> str:
    return f"""You are solving a Magic: The Gathering puzzle.

You have the full comprehensive rules below for reference.

<comprehensive_rules>
{rules_text.strip()}
</comprehensive_rules>

Puzzle conventions:
{clarifications}

You will be provided with puzzle details below.
{_SOLUTION_FORMAT}
"""


def solver_user_prompt(gold_md: str, *, inline: bool = False) -> str:
    return f"""Solve the following puzzle.

<puzzle>
{gold_md.strip()}
</puzzle>
Refer to the official magic rules for clarification. Return your solution in <solution> tags.
"""


JUDGE_SYSTEM = """You are grading a Magic: The Gathering puzzle.

You will be provided with the puzzle, official solution, and model solution, all in xml tags.

<puzzle>...</puzzle>
<official_solution>...</official_solution>
<model_solution>...</model_solution>

Use the official solution as guidance, determine if the model solution is correct. 
A solution passes if it is rules-legal and achieves the puzzle objective for all possible opponent responses and blocking decisions.
It must consider all reasonable opponent responses -- use your best judgement for what reasonable means here.
It does not need to match the official solution line-for-line, but usually the official solution is the only solution, so be extra thoughtful if the proposed solution is different; 
if it is actually correct and different, return "pass_with_unique_solution".

In rare cases, the problem may be misspecified. This is the case if the official solution does not make sense. You should be extra sure you have understood the rules correctly. If this is the case, return "misspecified_problem".

Think carefully, then return your verdict in your final message with

<verdict>
{"verdict": "pass" | "pass_with_unique_solution" | "misspecified_problem" | "fail", "reasoning": "..."}
</verdict>

Do not say anything after your verdict.
"""


def judge_user_prompt(
    *,
    problem_text: str,
    model_answer: str,
    official_solution: str,
) -> str:
    return f"""Grade the following puzzle.

<puzzle>
{problem_text}
</puzzle>
<official_solution>
{official_solution}
</official_solution>
<model_solution>
{model_answer}
</model_solution>

Return your verdict in your final message with
<verdict>
{{"verdict": "pass" | "pass_with_unique_solution" | "misspecified_problem" | "fail", "reasoning": "..."}}
</verdict>
"""
