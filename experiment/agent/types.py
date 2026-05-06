from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BAD_SCORE = -1e18


@dataclass
class Solution:
    path: Path
    description: str | None = None
    score: float | None = None
    generation_time_sec: float | None = None
    test_time_sec: float | None = None

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @staticmethod
    def extract_code(text: str) -> str:
        import re

        match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip() + "\n"
        return text.strip() + "\n"

    @classmethod
    def from_text(cls, text: str, path: Path, description: str | None = None) -> "Solution":
        path.parent.mkdir(parents=True, exist_ok=True)
        code = cls.extract_code(text)
        path.write_text(code, encoding="utf-8")
        return cls(path=path, description=description)


@dataclass
class TestCase:
    input_text: str
    expected: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    ok: bool
    stage: str
    message: str


@dataclass
class ExecutionResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    duration_sec: float


@dataclass
class ModelResponse:
    text: str
    wall_time_sec: float
    raw: dict[str, Any] | None = None