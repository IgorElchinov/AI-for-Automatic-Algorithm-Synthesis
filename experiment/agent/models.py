from __future__ import annotations

from pathlib import Path
import time
import typing as tp

import requests

from .types import ModelResponse


class OllamaClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 600,
        num_ctx: int = 4096,
        num_predict: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        keep_alive: str = "30m",
        think: bool = False,
        raw: bool = True,
        max_attempts: int = 10,
        debug_writer: tp.Callable[[str, str, str], Path] | None = None,
        logger: tp.Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.temperature = temperature
        self.top_p = top_p
        self.keep_alive = keep_alive
        self.think = think
        self.raw = raw
        self.debug_writer = debug_writer
        self.logger = logger
        self.max_attempts = max_attempts

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)

    def _dump(self, stem: str, text: str, suffix: str) -> None:
        if self.debug_writer is not None:
            path = self.debug_writer(stem, text, suffix)
            self._log(f"Saved debug artifact to {path}")

    def generate(self, prompt: str) -> ModelResponse:
        # max_attempts = 10
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            started_at = time.perf_counter()
            raw_text = ""

            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "raw": self.raw,
                        "keep_alive": self.keep_alive,
                        "think": self.think,
                        "options": {
                            "num_ctx": self.num_ctx,
                            "num_predict": self.num_predict,
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "stop": ["```", "```python", "~~~"],
                        },
                    },
                    timeout=self.timeout,
                )

                raw_text = response.text
                self._log(f"[ollama attempt {attempt}/{self.max_attempts}] status={response.status_code}")
                self._log(f"[ollama attempt {attempt}/{self.max_attempts}] raw head={raw_text[:300]!r}")

                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    self._dump(f"ollama_attempt_{attempt}_error", raw_text, ".json")
                    raise RuntimeError(f"Ollama error: {data['error']}")

                if data.get("done") is not True:
                    self._dump(f"ollama_attempt_{attempt}_non_final", raw_text, ".json")
                    raise RuntimeError(
                        f"Ollama returned non-final response with done={data.get('done')}"
                    )

                required_keys = [
                    "response",
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                ]
                missing = [key for key in required_keys if key not in data]
                if missing:
                    self._dump(f"ollama_attempt_{attempt}_missing_keys", raw_text, ".json")
                    raise RuntimeError(f"Missing keys in Ollama response: {missing}")

                elapsed = time.perf_counter() - started_at
                self._log(f"wall={elapsed:.2f}s")
                self._log(f"prompt_tokens={data['prompt_eval_count']}")
                self._log(f"gen_tokens={data['eval_count']}")

                return ModelResponse(
                    text=data["response"],
                    wall_time_sec=elapsed,
                    raw=data,
                )

            except Exception as exc:
                last_error = exc
                self._log(f"[ollama attempt {attempt}/{self.max_attempts}] failed: {exc}")
                if raw_text:
                    self._dump(f"ollama_attempt_{attempt}_raw", raw_text, ".txt")
                if attempt < self.max_attempts:
                    time.sleep(1.5 * attempt)

        raise RuntimeError(f"Ollama generate failed after {self.max_attempts} attempts: {last_error}")