from agent.budget import BudgetManager
from agent.models import ScriptedLLMClient


def test_pre_call_budget_admission_rejects_oversized_call():
    budget = BudgetManager(total_budget=10)
    decision = budget.admit("x" * 100, generation_budget=20)
    assert not decision.admitted
    assert decision.reason == "insufficient_token_budget"


def test_token_accounting_records_estimated_usage():
    budget = BudgetManager(total_budget=200)
    call = ScriptedLLMClient(["patched"]).generate("executor", "prompt", budget, 20)
    assert call.admitted
    assert budget.used.total_tokens == call.usage.total_tokens
    assert budget.used.token_count_estimated is True

