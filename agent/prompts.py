from __future__ import annotations

from agent.state import ValidationResult


def planner_prompt(task_id: str, code: str) -> str:
    return f"""You are the Planner for a controlled code-repair experiment.
Task: {task_id}
Given only the buggy implementation below, identify the likely defective function and a concise bug hypothesis.
Do not use hidden tests or known QuixBugs solutions.

```python
{code}
```

Return:
TARGET_FUNCTION: <name>
HYPOTHESIS: <concise hypothesis>
"""


def executor_prompt(task_id: str, code: str, hypothesis: str, remaining_budget: int) -> str:
    return f"""You are the Executor for a controlled code-repair experiment.
Task: {task_id}
Remaining token budget: {remaining_budget}
Hypothesis: {hypothesis}

Produce one corrected minimal Python function body or full function replacing the buggy function.
Return only Python code, no markdown fences.

```python
{code}
```
"""


def retry_prompt(
    task_id: str,
    original_code: str,
    prior_patch: str,
    evidence: ValidationResult,
    hypothesis: str,
    remaining_budget: int,
) -> str:
    return f"""You are retrying a failed code repair after executable validation.
Task: {task_id}
Remaining token budget: {remaining_budget}
Previous hypothesis: {hypothesis}
Error category: {evidence.error_category}
Failing test information:
{evidence.failing_test_info}
stderr:
{evidence.stderr}

Original buggy code:
```python
{original_code}
```

Prior proposed patch:
```python
{prior_patch}
```

Produce a corrected minimal patch as Python code only.
"""


def critic_prompt(
    task_id: str,
    original_code: str,
    hypothesis: str,
    patch: str,
    evidence: ValidationResult | None,
) -> str:
    evidence_text = "No executable test evidence is available to you."
    if evidence is not None:
        evidence_text = f"""Executable evidence:
success={evidence.success}
return_code={evidence.return_code}
timed_out={evidence.timed_out}
runtime_seconds={evidence.runtime_seconds}
tests_passed={evidence.tests_passed}
tests_failed={evidence.tests_failed}
tests_total={evidence.tests_total}
stdout={evidence.stdout}
stderr={evidence.stderr}
failing_test_info={evidence.failing_test_info}
"""
    return f"""You are the Critic for a controlled code-repair experiment.
Task: {task_id}
Bug hypothesis: {hypothesis}

Original code:
```python
{original_code}
```

Proposed patch:
```python
{patch}
```

{evidence_text}

Return ACCEPT or REJECT on the first line, followed by a short rationale.
"""

