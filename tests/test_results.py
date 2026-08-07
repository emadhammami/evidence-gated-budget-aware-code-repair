from pathlib import Path

from agent.state import CriticOutput, PatchRecord, RepairState, ValidationResult
from benchmark.results import completed_keys, persist_result, state_to_row


def test_false_acceptance_calculation():
    state = RepairState(experiment_id="x", task_id="gcd", method="pec", token_budget=8000)
    state.critic = CriticOutput(accepted=True)
    state.validations.append(ValidationResult(success=False, tests_failed=1, tests_total=1))
    row = state_to_row(state, "git", "bench", "model", 0)
    assert row["false_accept"] is True
    assert row["candidate_correct"] is False
    assert row["workflow_success"] is False


def test_false_reject_calculation():
    state = RepairState(experiment_id="x", task_id="gcd", method="pevc", token_budget=8000)
    state.critic = CriticOutput(accepted=False)
    state.validations.append(ValidationResult(success=True, tests_passed=1, tests_total=1))
    row = state_to_row(state, "git", "bench", "model", 0)
    assert row["false_reject"] is True
    assert row["candidate_correct"] is True
    assert row["workflow_success"] is False


def test_result_serialization_and_resume_skip(tmp_path: Path):
    state = RepairState(experiment_id="x", task_id="gcd", method="single_shot", token_budget=8000)
    state.patch = PatchRecord(applied=True, syntax_valid=True, affected_function="gcd")
    state.validations.append(ValidationResult(success=True, tests_passed=1, tests_total=1))
    state.llm_calls = []
    state.ended_at_utc = "2026-08-07T00:00:00+00:00"
    persist_result(state, "git", "bench", "model", 0, results_root=tmp_path)
    assert (tmp_path / "raw" / "gcd__single_shot__8000__run1.json").exists()
    assert ("gcd", "single_shot", 8000, 1) in completed_keys(tmp_path / "runs.csv")


def test_budget_exhausted_completed_run_is_skipped(tmp_path: Path):
    state = RepairState(experiment_id="x", task_id="gcd", method="evidence_gated", token_budget=2000)
    state.early_exit = True
    state.budget_exhausted = True
    state.ended_at_utc = "2026-08-07T00:00:00+00:00"
    persist_result(state, "git", "bench", "model", 0, results_root=tmp_path)
    assert ("gcd", "evidence_gated", 2000, 1) in completed_keys(tmp_path / "runs.csv")


def test_infrastructure_error_is_not_skipped(tmp_path: Path):
    state = RepairState(experiment_id="x", task_id="gcd", method="evidence_gated", token_budget=2000)
    state.run_status = "infrastructure_error"
    state.ended_at_utc = "2026-08-07T00:00:00+00:00"
    persist_result(state, "git", "bench", "model", 0, results_root=tmp_path)
    assert ("gcd", "evidence_gated", 2000, 1) not in completed_keys(tmp_path / "runs.csv")
