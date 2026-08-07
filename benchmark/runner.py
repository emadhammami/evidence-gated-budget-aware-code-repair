from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from agent.models import GeminiClient, LLMClient, ModelConfig
from agent.state import MethodName, RepairState
from agent.variants import run_evidence_gated, run_pec, run_pevc, run_single_shot
from benchmark.quixbugs import QuixBugsBenchmark
from benchmark.results import persist_result

VARIANTS = {
    "single_shot": run_single_shot,
    "pec": run_pec,
    "pevc": run_pevc,
    "evidence_gated": run_evidence_gated,
}


def git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip() or "unknown"


def run_one(
    task_id: str,
    method: MethodName,
    budget: int,
    repetition: int = 1,
    is_pilot: bool = False,
    llm: LLMClient | None = None,
    benchmark: QuixBugsBenchmark | None = None,
    persist: bool = True,
) -> RepairState:
    bench = benchmark or QuixBugsBenchmark()
    original_code = bench.load_buggy_code(task_id)
    state = RepairState(
        experiment_id=str(uuid.uuid4()),
        task_id=task_id,
        method=method,
        token_budget=budget,
        repetition=repetition,
        is_pilot=is_pilot,
        original_code=original_code,
    )
    client = llm or GeminiClient(ModelConfig())
    final_state = VARIANTS[method](state, client, bench)
    if persist:
        persist_result(
            final_state,
            git_commit=git_commit(),
            benchmark_commit=bench.benchmark_commit(),
            model=ModelConfig().name,
            temperature=ModelConfig().temperature,
            results_root=Path("results"),
        )
    return final_state

