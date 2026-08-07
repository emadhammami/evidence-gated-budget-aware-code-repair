from agent.budget import BudgetManager
from agent.models import GeminiClient, ModelConfig, ScriptedLLMClient


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


def test_gemini_call_enforces_generation_budget_as_max_output_tokens():
    class Usage:
        prompt_token_count = 1
        candidates_token_count = 2
        total_token_count = 3

    class Response:
        text = "ok"
        usage_metadata = Usage()

    class Model:
        def __init__(self):
            self.generation_config = None

        def generate_content(self, prompt, generation_config):
            self.generation_config = generation_config
            return Response()

    client = GeminiClient.__new__(GeminiClient)
    client.config = ModelConfig(name="gemini-2.5-flash", temperature=0)
    client._model = Model()
    call = client.generate("executor", "prompt", BudgetManager(100), generation_budget=17)
    assert call.admitted
    assert client._model.generation_config["max_output_tokens"] == 17
