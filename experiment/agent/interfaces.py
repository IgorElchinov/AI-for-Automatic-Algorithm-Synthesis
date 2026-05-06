from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .types import ExecutionResult, ModelResponse, TestCase


class Runner(Protocol):
    def run(self, solution_path: Path, input_text: str, timeout: int) -> ExecutionResult:
        ...


class ModelClient(Protocol):
    def generate(self, prompt: str) -> ModelResponse:
        ...


class PromptStrategy(Protocol):
    def build_gen_prompt(self, problem_text: str) -> str:
        ...

    def build_combine_prompt(self, problem_text: str, candidate_codes: list[str]) -> str:
        ...

    def build_mutate_prompt(self, problem_text: str, candidate_code: str) -> str:
        ...

    def build_fix_prompt(self, problem_text: str, candidate_code: str, issue: str) -> str:
        ...


class ProblemAdapter(Protocol):
    @property
    def problem_text(self) -> str:
        ...

    def build_smoke_tests(self) -> list[TestCase]:
        ...

    def build_full_tests(self) -> list[TestCase]:
        ...

    def evaluate_output(self, output: str, test: TestCase) -> float:
        ...