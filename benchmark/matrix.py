from __future__ import annotations

import argparse
from typing import cast

import yaml

from agent.state import MethodName
from benchmark.quixbugs import QuixBugsBenchmark
from benchmark.results import completed_keys
from benchmark.runner import run_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--core", action="store_true")
    group.add_argument("--pilot", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repetition", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = QuixBugsBenchmark()
    config = yaml.safe_load(open("configs/experiments.yaml", encoding="utf-8"))
    if args.pilot:
        plan = [(task, config["pilot"]["method"], config["pilot"]["budget"]) for task in config["pilot"]["tasks"]]
        is_pilot = True
    else:
        tasks = benchmark.discover_tasks()
        plan = [(task, method, 8000) for task in tasks for method in ["single_shot", "pec", "pevc", "evidence_gated"]]
        plan += [(task, "evidence_gated", budget) for task in tasks for budget in [4000, 2000]]
        is_pilot = False
    done = completed_keys()
    for task, method, budget in plan:
        key = (task, method, budget, args.repetition)
        if key in done and not args.force:
            print(f"skip completed {task} {method} {budget} run{args.repetition}")
            continue
        print(f"run {task} {method} {budget} run{args.repetition} pilot={is_pilot}")
        run_one(
            task_id=task,
            method=cast(MethodName, method),
            budget=int(budget),
            repetition=args.repetition,
            is_pilot=is_pilot,
            benchmark=benchmark,
        )


if __name__ == "__main__":
    main()

