"""Thin OpenAI-compatible chat client with retry + a mock mode.

Defaults to a local Ollama server (http://localhost:11434/v1), which needs
no API key and no network access at all. Every value is still env-var
overridable, so this also works against plain OpenAI, a remote vLLM server,
or any other OpenAI-compatible endpoint if you'd rather not run models
locally.
"""

import os
import time


class LLMClient:
    def __init__(self, mock: bool = False):
        self.mock = mock
        if not mock:
            from openai import OpenAI  # imported lazily so --mock needs no package/server at all

            self.client = OpenAI(
                base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("LLM_API_KEY", "ollama"),  # Ollama ignores this; some clients require non-empty
            )

    def complete(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system: str | None = None,
        max_retries: int = 3,
    ) -> str:
        if self.mock:
            return self._mock_complete(prompt)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = resp.choices[0].message.content
                return (content or "").strip()
            except Exception as e:
                # Ollama can be slow/unresponsive on the first call to a model it
                # hasn't loaded yet (cold start = disk read into RAM) — worth retrying
                # with backoff rather than failing the whole question on one timeout.
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM call to model={model} failed after {max_retries} attempts: {last_err}")

    @staticmethod
    def _mock_complete(prompt: str) -> str:
        """Deterministic, zero-cost stand-in for offline pipeline sanity-checks only.

        This does NOT produce a meaningful answer or hypothesis — it exists so the
        rest of the pipeline (embedding, ranking, scoring, logging) can be verified
        to run end-to-end without an API key. Never report --mock numbers as results.
        """
        if "Evidence:" in prompt:
            return "mock-answer"
        return "This is a mock hypothesis passage generated without calling any LLM API."
