import pandas as pd

from analysis.aggregate import aggregate


def test_aggregation_metrics():
    df = pd.DataFrame(
        [
            {"method": "evidence_gated", "token_budget": 8000, "repaired": True, "total_tokens": 1000, "critic_accepted": True, "false_accept": False, "llm_calls": 3, "early_exit": False, "budget_exceeded": False, "runtime_seconds": 1.0},
            {"method": "evidence_gated", "token_budget": 8000, "repaired": False, "total_tokens": 2000, "critic_accepted": True, "false_accept": True, "llm_calls": 2, "early_exit": False, "budget_exceeded": False, "runtime_seconds": 2.0},
        ]
    )
    result = aggregate(df, ["method", "token_budget"])
    row = result.iloc[0]
    assert row["repair_rate_pct"] == 50
    assert round(row["repairs_per_100k_tokens"], 2) == 33.33
    assert row["false_acceptance_rate_pct"] == 50

