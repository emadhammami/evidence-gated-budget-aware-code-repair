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
    thinking_budget: int | None = 0
    min_output_tokens: int = 32


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
        prompt_tokens, prompt_count_estimated = self._count_prompt_tokens(prompt)
        remaining = budget.remaining
        available_output = remaining - prompt_tokens
        if available_output < self.config.min_output_tokens:
            return LLMCallRecord(
                role=role,
                admitted=False,
                prompt_tokens_estimate=prompt_tokens,
                prompt_token_count_estimated=prompt_count_estimated,
                configured_generation_budget=generation_budget,
                generation_budget=0,
                max_output_tokens=0,
                thinking_budget=self.config.thinking_budget,
                thinking_config_note=self._thinking_config_note(),
                skipped_reason="insufficient_token_budget",
            )
        max_output_tokens = min(generation_budget, available_output)
        start = time.perf_counter()
        response = self._model.generate_content(
            prompt,
            generation_config=self._generation_config(max_output_tokens),
        )
        runtime = time.perf_counter() - start
        text = getattr(response, "text", "") or ""
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            input_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
            total_tokens = int(getattr(usage_meta, "total_token_count", 0) or 0)
            candidate_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
            output_tokens = max(0, total_tokens - input_tokens) if total_tokens else candidate_tokens
            usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens or input_tokens + output_tokens,
                token_count_estimated=False,
            )
        else:
            usage = TokenUsage(
                input_tokens=prompt_tokens,
                output_tokens=estimate_tokens(text),
                total_tokens=prompt_tokens + estimate_tokens(text),
                token_count_estimated=True,
            )
        budget.record(usage)
        return LLMCallRecord(
            role=role,
            admitted=True,
            prompt_tokens_estimate=prompt_tokens,
            prompt_token_count_estimated=prompt_count_estimated,
            configured_generation_budget=generation_budget,
            generation_budget=max_output_tokens,
            max_output_tokens=max_output_tokens,
            thinking_budget=self.config.thinking_budget,
            thinking_config_note=self._thinking_config_note(),
            usage=usage,
            runtime_seconds=runtime,
            raw_output=text,
        )

    def _count_prompt_tokens(self, prompt: str) -> tuple[int, bool]:
        try:
            result = self._model.count_tokens(prompt)
            return int(getattr(result, "total_tokens")), False
        except Exception:
            return estimate_tokens(prompt), True

    def _generation_config(self, max_output_tokens: int) -> dict[str, int | float]:
        return {
            "temperature": self.config.temperature,
            "max_output_tokens": max_output_tokens,
        }

    def _thinking_config_note(self) -> str | None:
        if self.config.thinking_budget is None:
            return "thinking budget not configured"
        return (
            "thinking_budget=0 requested in configs/experiments.yaml; "
            "google-generativeai generation_config does not expose a stable thinking_config field here, "
            "so no unsupported parameter is sent and the same provider default applies to every method"
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
                prompt_token_count_estimated=True,
                configured_generation_budget=generation_budget,
                generation_budget=generation_budget,
                max_output_tokens=generation_budget,
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
            prompt_token_count_estimated=True,
            configured_generation_budget=generation_budget,
            generation_budget=generation_budget,
            max_output_tokens=generation_budget,
            usage=usage,
            raw_output=text,
        )
