# Evidence-Gated Budget-Aware Code Repair

This repository implements a controlled experimental benchmark for studying:

**Evidence-Guided Budget-Aware Orchestration for LLM-Based Code Repair**

The central question is whether executable test evidence can allocate a fixed LLM token
budget more efficiently in an agentic code-repair workflow.

This repository is an experimental extension of
`emadhammami/langgraph-budget-aware-code-agent`. The original prototype is not modified by
this codebase.

## Variants

The benchmark implements exactly four methods under the same model configuration:

| Method | Workflow | Key rule |
|---|---|---|
| `single_shot` | Executor -> tests | One LLM repair attempt, no Planner, no Critic. |
| `pec` | Planner -> Executor -> Critic -> tests | Critic receives no executable test evidence. |
| `pevc` | Planner -> Executor -> validation -> Critic | Critic receives objective execution evidence for every patch. |
| `evidence_gated` | Planner -> Executor -> validation -> Evidence Gate | Failed tests skip the Critic and may trigger one evidence-guided retry. Passing patches may go to the Critic. |

V0, V1, and V2 each allow one Executor attempt. V3 allows an initial Executor attempt plus
at most one evidence-guided retry.

## Model And Budget

All variants use the centralized model configuration in `configs/experiments.yaml`:

- Model: `gemini-2.5-flash`
- Temperature: `0`
- API key: `GOOGLE_API_KEY`

The code performs pre-call budget admission control before every LLM invocation. A call is
not intentionally started unless the estimated prompt tokens plus allowed generation budget
fit inside the remaining per-task budget. Gemini calls also pass the admitted generation
budget as `max_output_tokens`, so provider-side output caps are enforced when supported by
the API.

Provider usage metadata is recorded when available; otherwise token counts are explicitly
marked as estimated. `budget_exhausted` means the workflow intentionally stopped because
the remaining budget could not admit another LLM call. `budget_violation` means actual
measured provider token usage exceeded the configured per-task budget. A refused call due
to insufficient remaining budget is not a budget violation.

Required budgets are `2000`, `4000`, and `8000`. The main comparison is all four variants
at `8000`; budget sensitivity is V3 at all three budgets.

## QuixBugs Validation

`python -m benchmark.setup` clones QuixBugs into `.benchmarks/QuixBugs/` and checks out the
exact SHA in `configs/benchmark.yaml`. The benchmark task list is an explicit 40-task
allowlist in that same config file. Task discovery never uses `*.py` glob inference and
fails clearly if a configured source or `python_testcases/test_<task>.py` file is missing.

Each individual run:

1. creates an isolated temporary copy of the QuixBugs checkout,
2. reads the buggy Python implementation,
3. asks the configured method for a candidate patch,
4. uses Python AST metadata to replace the target function where possible,
5. parses the patched file with `ast.parse`,
6. runs the relevant official QuixBugs Python test with `pytest`,
7. records JSON and CSV telemetry,
8. discards the temporary copy.

`candidate_correct` is true when the final candidate passes the relevant official
QuixBugs tests. `workflow_success` is the main repair-rate metric: for `single_shot` it is
equal to `candidate_correct`; for PEC, PEVC, and Evidence-Gated it requires both
`candidate_correct` and `critic_accepted`. `false_accept` means the Critic accepted a
candidate that failed the official tests. `false_reject` means the Critic rejected a
candidate that passed those tests. Single-shot has null Critic-related metrics.

Merely executing a function definition with return code `0` is not counted as a repair.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
python -m benchmark.setup
```

Set `GOOGLE_API_KEY` in the environment before real Gemini runs. Do not commit `.env`.

## Running

Run selected tasks:

```bash
python -m benchmark.run --method evidence_gated --budget 8000 --tasks gcd,quicksort
```

Run the 5-task infrastructure pilot:

```bash
python -m benchmark.matrix --pilot
```

Run the core matrix after the protocol is verified:

```bash
python -m benchmark.matrix --core
```

The core matrix is at most 240 individual runs: 40 tasks x 4 methods at 8000, plus
40 tasks x V3 at 4000 and 40 tasks x V3 at 2000. Pilot data is marked with
`is_pilot = true` and is excluded from aggregate statistics by default.

## Resume Semantics

After every individual task, the runner writes one raw JSON file under `results/raw/` and
appends one row to `results/runs.csv`. Before a run starts, the CLI checks for an existing
completed `(task_id, method, token_budget, repetition)` row and skips it by default.

Use `--force` to rerun an existing combination.

## Result Schema

`results/runs.csv` contains:

`experiment_id`, `timestamp_utc`, `git_commit`, `benchmark_commit`, `task_id`, `method`,
`token_budget`, `repetition`, `model`, `temperature`, `is_pilot`, `run_status`,
`candidate_correct`, `workflow_success`, `tests_passed`, `tests_failed`, `tests_total`,
`critic_invoked`, `critic_accepted`, `false_accept`, `false_reject`, `input_tokens`,
`output_tokens`, `total_tokens`, `token_count_estimated`, `llm_calls`, `planner_calls`,
`executor_attempts`, `critic_calls`, `validation_attempts`, `validation_failures`,
`retry_used`, `early_exit`, `budget_exhausted`, `budget_violation`, `budget_limit`,
`budget_used`, `budget_remaining`, `patch_applied`, `syntax_valid`, `final_error_category`,
`runtime_seconds`, `llm_runtime_seconds`, `validation_runtime_seconds`.

Raw JSON files preserve state, node-level telemetry, LLM call records, patch records, and
runtime fingerprints for auditability. Full LLM outputs are kept out of `runs.csv`.

Resume logic skips rows with `run_status = completed` unless `--force` is supplied. A
completed budget-exhausted run is still a completed experimental outcome; only
`infrastructure_error` rows are eligible for automatic retry.

## Analysis

Generate aggregate metrics:

```bash
python -m analysis.aggregate
```

Generate paper tables:

```bash
python -m analysis.tables
```

Generate figures:

```bash
python -m analysis.plots
```

The analysis reads only `results/runs.csv` and raw JSON outputs. It does not contain
hard-coded findings and should not be edited with manually entered performance numbers.

## Tests

```bash
pytest
ruff check .
```

Normal tests use deterministic mocked LLM responses and do not require a Gemini API key.
The GitHub Actions workflow runs `pytest` and `ruff` on Python 3.11 and 3.12.

## Scientific Integrity

This repository does not include real experimental results. Do not fabricate results, do
not cherry-pick successful QuixBugs tasks, and do not silently exclude failed runs. Any
incompatible task must be documented with an explicit exclusion reason.

Prompt or protocol changes after observing failures should be treated as a new experiment
version. V3 may receive failed-test feedback only after the first candidate has been
generated and executed.

## Limitations

The AST patcher is deliberately conservative: if it cannot identify a unique target
function, it records `patch_error` instead of attempting risky text replacement. QuixBugs
test path conventions may require updates if the upstream repository structure changes.
Provider token metadata availability depends on the Gemini client response shape; estimated
counts are marked explicitly.
