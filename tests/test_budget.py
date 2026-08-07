from agent.budget import BudgetManager
from agent.models import GeminiClient, ModelConfig, ScriptedLLMClient
from agent.state import TokenUsage


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
        prompt_token_count = 7
        candidates_token_count = 9
        total_token_count = 20

    class Response:
        text = "ok"
        usage_metadata = Usage()

    class Count:
        total_tokens = 7

    class Model:
        def __init__(self):
            self.generation_config = None
            self.count_called = False

        def count_tokens(self, prompt):
            self.count_called = True
            return Count()

        def generate_content(self, prompt, generation_config):
            self.generation_config = generation_config
            return Response()

    client = GeminiClient.__new__(GeminiClient)
    client.config = ModelConfig(name="gemini-2.5-flash", temperature=0)
    client._model = Model()
    call = client.generate("executor", "prompt", BudgetManager(100), generation_budget=17)
    assert call.admitted
    assert client._model.count_called
    assert client._model.generation_config["max_output_tokens"] == 17
    assert call.prompt_token_count_estimated is False
    assert call.usage.input_tokens == 7
    assert call.usage.output_tokens == 13
    assert call.usage.total_tokens == 20


def test_gemini_dynamic_max_output_tokens_uses_remaining_budget():
    class Count:
        total_tokens = 80

    class Usage:
        prompt_token_count = 80
        candidates_token_count = 10
        total_token_count = 90

    class Response:
        text = "ok"
        usage_metadata = Usage()

    class Model:
        def __init__(self):
            self.generation_config = None

        def count_tokens(self, prompt):
            return Count()

        def generate_content(self, prompt, generation_config):
            self.generation_config = generation_config
            return Response()

    budget = BudgetManager(100)
    budget.record(TokenUsage(total_tokens=10))
    client = GeminiClient.__new__(GeminiClient)
    client.config = ModelConfig(name="gemini-2.5-flash", temperature=0, min_output_tokens=5)
    client._model = Model()
    call = client.generate("executor", "prompt", budget, generation_budget=50)
    assert call.admitted
    assert call.max_output_tokens == 10
    assert client._model.generation_config["max_output_tokens"] == 10


def test_gemini_budget_exhaustion_when_counted_prompt_leaves_too_little_output():
    class Count:
        total_tokens = 90

    class Model:
        def __init__(self):
            self.generated = False

        def count_tokens(self, prompt):
            return Count()

        def generate_content(self, prompt, generation_config):
            self.generated = True
            raise AssertionError("Gemini should not be called when budget is exhausted")

    client = GeminiClient.__new__(GeminiClient)
    client.config = ModelConfig(name="gemini-2.5-flash", temperature=0, min_output_tokens=32)
    client._model = Model()
    call = client.generate("planner", "prompt", BudgetManager(100), generation_budget=50)
    assert not call.admitted
    assert call.skipped_reason == "insufficient_token_budget"
    assert call.max_output_tokens == 0
    assert client._model.generated is False


def test_gemini_records_fallback_when_provider_counting_fails():
    class Usage:
        prompt_token_count = 2
        candidates_token_count = 1
        total_token_count = 3

    class Response:
        text = "ok"
        usage_metadata = Usage()

    class Model:
        def count_tokens(self, prompt):
            raise RuntimeError("counting unavailable")

        def generate_content(self, prompt, generation_config):
            return Response()

    client = GeminiClient.__new__(GeminiClient)
    client.config = ModelConfig(name="gemini-2.5-flash", temperature=0)
    client._model = Model()
    call = client.generate("critic", "prompt", BudgetManager(100), generation_budget=10)
    assert call.admitted
    assert call.prompt_token_count_estimated is True
