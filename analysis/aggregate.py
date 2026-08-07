from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_runs(path: str | Path = "results/runs.csv", include_pilot: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "is_pilot" in df.columns and not include_pilot:
        df = df[df["is_pilot"].astype(str).str.lower() != "true"]
    return df


def false_acceptance_rate(group: pd.DataFrame) -> float:
    accepted = group[group["critic_accepted"].astype(str).str.lower() == "true"]
    if accepted.empty:
        return float("nan")
    return accepted["false_accept"].astype(str).str.lower().eq("true").mean() * 100


def aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(by, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        total_tokens = group["total_tokens"].sum()
        repairs = group["repaired"].astype(bool).sum()
        rows.append(
            {
                **dict(zip(by, key_tuple, strict=True)),
                "evaluated_tasks": len(group),
                "repair_rate_pct": group["repaired"].astype(bool).mean() * 100,
                "mean_tokens": group["total_tokens"].mean(),
                "median_tokens": group["total_tokens"].median(),
                "repairs_per_100k_tokens": (repairs / total_tokens * 100000) if total_tokens else 0,
                "false_acceptance_rate_pct": false_acceptance_rate(group),
                "mean_llm_calls": group["llm_calls"].mean(),
                "early_exit_rate_pct": group["early_exit"].astype(bool).mean() * 100,
                "budget_violation_rate_pct": group["budget_exceeded"].astype(bool).mean() * 100,
                "mean_runtime_seconds": group["runtime_seconds"].mean(),
                "median_runtime_seconds": group["runtime_seconds"].median(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--include-pilot", action="store_true")
    args = parser.parse_args()
    df = load_runs(args.runs, include_pilot=args.include_pilot)
    print(aggregate(df, ["method", "token_budget"]).to_string(index=False))


if __name__ == "__main__":
    main()

