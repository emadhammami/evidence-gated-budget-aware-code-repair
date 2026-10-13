# Evidence-Gated Budget-Aware Code Repair

**Manuscript status (October 2026):** Submitted to AIPMV 2026; not yet published.

This repository is the reproducibility artifact for the paper:

**Evidence-Guided Budget Allocation for Multi-Stage Large Language Model Orchestration**

The paper studies evidence-guided budget allocation as a general
multi-stage LLM orchestration problem. This repository contains the
controlled code-repair instantiation used for the empirical evaluation.

In this instantiation, executable QuixBugs test outcomes act as
intermediate environment evidence for deciding whether the remaining
token budget should be allocated to downstream verification, recovery,
or termination.

The Digital Product Passport (DPP) case study discussed in the paper is
an illustrative application of the orchestration principle and is not
implemented or empirically evaluated in this repository.

This repository is an experimental extension of
`emadhammami/langgraph-budget-aware-code-agent`. The original prototype
is not modified by this codebase.

## Variants

The benchmark implements exactly four methods under the same model configuration:

| Method | Workflow | Key rule |
|---|---|---|
| `single_shot` | Executor -> tests | One LLM repair attempt, no Planner, no Critic. |
| `pec` | Planner -> Executor -> Critic -> tests | Critic receives no executable test evidence. |
| `pevc` | Planner -> Executor -> validation -> Critic | Critic receives objective execution evidence for every patch. |
| `evidence_gated` | Planner -> Executor -> validation -> Evidence Gate | Failed tests skip the Critic and may trigger one evidence-guided retry. Passing patches may go to the Critic. |

Single-shot, PEC, and PEVC each allow one Executor attempt. Evidence-Gated allows an
initial Executor attempt plus at most one evidence-guided retry.

## Model And Budget

All variants use the centralized model configuration in `configs/experiments.yaml`:

- Model: `gemini-2.5-flash`
- Temperature: `0`
- API key: `GOOGLE_API_KEY`

The code performs pre-call budget admission control before every LLM invocation. A call is
not intentionally started unless the prompt tokens plus allowed generation budget fit inside
the remaining per-task budget. For real Gemini calls, the client first calls the model's
`count_tokens(prompt)` capability. It then computes `available_output = remaining -
actual_prompt_tokens` and passes `max_output_tokens = min(configured_generation_budget,
available_output)` to Gemini. If the remaining output allowance is below the configured
minimum meaningful call size, the workflow stops with `budget_exhausted = true` before
calling the model.

Provider usage metadata is recorded when available; otherwise token counts are explicitly
marked as estimated. `budget_exhausted` means the workflow intentionally stopped because
the remaining budget could not admit another LLM call. `budget_violation` means actual
measured provider token usage exceeded the configured per-task budget. A refused call due
to insufficient remaining budget is not a budget violation.

The Gemini adapter uses the current `google-genai` SDK:

```python
types.GenerateContentConfig(
    temperature=0,
    max_output_tokens=max_output_tokens,
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)
```

The same thinking configuration is applied to Planner, Executor, retry, and Critic calls
for every method. Raw call telemetry records `thinking_budget = 0` and
`thinking_config_applied = true`.

`configs/experiments.yaml` also configures provider pacing. The default is
`requests_per_minute: 4`, intentionally below the observed 5-RPM free-tier limit. A
process-wide generate-content rate limiter is shared by every role. Transient provider
errors such as 429, 408, 500, 502, 503, and 504 are retried with bounded backoff. Retry and
pacing telemetry is recorded separately as provider operations, not as scientific LLM
calls, and deliberate wait time is excluded from `llm_runtime_seconds`.

Required budgets are `1000`, `2000`, and `4000`. The main comparison evaluates all four
methods at `4000`; budget sensitivity evaluates Evidence-Gated at all three budgets.

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
python -m benchmark.run --method evidence_gated --budget 4000 --tasks gcd,quicksort
```

Run the 5-task infrastructure pilot:

```bash
python -m benchmark.matrix --pilot
```

Run the core matrix after the protocol is verified:

```bash
python -m benchmark.matrix --core
```

The reported core evaluation contains 240 completed runs: 40 tasks x 4 methods at 4000,
plus 40 Evidence-Gated runs at 2000 and 40 Evidence-Gated runs at 1000. Pilot data is marked with
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
`output_tokens`, `total_tokens`, `token_count_estimated`, `llm_calls`,
`provider_attempts`, `transient_retries`, `rate_limit_retries`,
`rate_limit_wait_seconds`, `provider_wall_time_seconds`, `planner_calls`,
`executor_attempts`, `critic_calls`, `validation_attempts`, `validation_failures`,
`retry_used`, `early_exit`, `budget_exhausted`, `budget_violation`, `budget_limit`,
`budget_used`, `budget_remaining`, `patch_applied`, `syntax_valid`, `final_error_category`,
`runtime_seconds`, `llm_runtime_seconds`, `validation_runtime_seconds`,
`infrastructure_error`.

Raw JSON files preserve state, node-level telemetry, LLM call records, patch records, and
runtime fingerprints for auditability. Full LLM outputs are kept out of `runs.csv`.

Resume logic skips rows with `run_status = completed` unless `--force` is supplied. A
completed budget-exhausted run is still a completed experimental outcome. If transient
provider retries are exhausted, the run is saved as `run_status = infrastructure_error`;
analysis excludes that row from scientific aggregates by default, and resume will retry it
later.

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


## Publication Artifact

The complete experimental data used for the reported results are available
in [`final_core_results.zip`](final_core_results.zip).

The artifact contains exactly 240 completed non-pilot task-condition runs:

- Single-shot @ 4000 tokens: 40
- PEC @ 4000 tokens: 40
- PEVC @ 4000 tokens: 40
- Evidence-Gated @ 4000 tokens: 40
- Evidence-Gated @ 2000 tokens: 40
- Evidence-Gated @ 1000 tokens: 40

The archive includes:

- `core_runs.csv`, containing the 240 run-level records;
- `raw/`, containing one raw JSON telemetry file per reported run;
- `experiment_git_commit.txt`, identifying the implementation revision;
- `python_version.txt`;
- `requirements-final.txt`;
- an artifact-specific README.

All 240 reported runs use experiment commit
`93ce1a5901bf900033c8cc7c547b8e8926bb7103`, use provider-reported token
counts, and contain no budget violations. The five-task infrastructure
pilot and infrastructure-error attempts are excluded from the publication
artifact.

The SHA-256 checksum of the archive is provided in
[`final_core_results.sha256.txt`](final_core_results.sha256.txt).

## Limitations

The AST patcher is deliberately conservative: if it cannot identify a unique target
function, it records `patch_error` instead of attempting risky text replacement. QuixBugs
test path conventions may require updates if the upstream repository structure changes.
Provider token metadata availability depends on the Gemini client response shape; estimated
counts are marked explicitly.
