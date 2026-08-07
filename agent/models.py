from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

from agent.budget import BudgetManager, estimate_tokens
from agent.state import LLMCallRecord, TokenUsage


@dataclass(frozen=True)
class ModelConfig:
    name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    provider: str = "google"


class LLMClient(Protocol):
    def generate(self, role: str, prompt: str, budget: BudgetManager, generation_budget: int) -> LLMCallRecord:
        ...


class GeminiClient:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for real Gemini runs.")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            self.config.name,
            generation_config={"temperature": self.config.temperature},
        )

    def generate(self, role: str, prompt: str, budget: BudgetManager, generation_budget: int) -> LLMCallRecord:
        decision = budget.admit(prompt, generation_budget)
        if not decision.admitted:
            return LLMCallRecord(
                role=role,
                admitted=False,
                prompt_tokens_estimate=decision.prompt_tokens_estimate,
                generation_budget=generation_budget,
                skipped_reason=decision.reason,
            )
        start = time.perf_counter()
        response = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": generation_budget,
            },
        )
        runtime = time.perf_counter() - start
        text = getattr(response, "text", "") or ""
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            usage = TokenUsage(
                input_tokens=int(getattr(usage_meta, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(usage_meta, "candidates_token_count", 0) or 0),
                total_tokens=int(getattr(usage_meta, "total_token_count", 0) or 0),
                token_count_estimated=False,
            )
        else:
            usage = TokenUsage(
                input_tokens=estimate_tokens(prompt),
                output_tokens=estimate_tokens(text),
                total_tokens=estimate_tokens(prompt) + estimate_tokens(text),
                token_count_estimated=True,
            )
        budget.record(usage)
        return LLMCallRecord(
            role=role,
            admitted=True,
            prompt_tokens_estimate=decision.prompt_tokens_estimate,
            generation_budget=generation_budget,
            usage=usage,
            runtime_seconds=runtime,
            raw_output=text,
        )


class ScriptedLLMClient:
    """Deterministic test double used for CI and mocked end-to-end tests."""

    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = list(outputs or [])

    def generate(self, role: str, prompt: str, budget: BudgetManager, generation_budget: int) -> LLMCallRecord:
        decision = budget.admit(prompt, generation_budget)
        if not decision.admitted:
            return LLMCallRecord(
                role=role,
                admitted=False,
                prompt_tokens_estimate=decision.prompt_tokens_estimate,
                generation_budget=generation_budget,
                skipped_reason=decision.reason,
            )
        text = self.outputs.pop(0) if self.outputs else "ACCEPT\n"
        usage = TokenUsage(
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(text),
            total_tokens=estimate_tokens(prompt) + estimate_tokens(text),
            token_count_estimated=True,
        )
        budget.record(usage)
        return LLMCallRecord(
            role=role,
            admitted=True,
            prompt_tokens_estimate=decision.prompt_tokens_estimate,
            generation_budget=generation_budget,
            usage=usage,
            raw_output=text,
        )
