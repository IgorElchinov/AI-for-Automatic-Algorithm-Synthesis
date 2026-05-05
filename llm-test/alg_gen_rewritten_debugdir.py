from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import math
import random
import re
import subprocess
import time
import typing as tp

import requests


BAD_SCORE = -1e18


@dataclass
class Solution:
    path: Path
    description: str | None = None
    score: float | None = None
    generation_time_sec: float | None = None
    test_time_sec: float | None = None

    def run(self, input_text: str, timeout: int = 60) -> str:
        result = subprocess.run(
            ["python3", str(self.path)],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return result.stdout

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @staticmethod
    def _extract_python_code(text: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip() + "\n"
        return text.strip() + "\n"

    @classmethod
    def from_text(cls, text: str, path: Path, description: str | None = None) -> "Solution":
        path.parent.mkdir(parents=True, exist_ok=True)
        code = cls._extract_python_code(text)
        path.write_text(code, encoding="utf-8")
        return cls(path=path, description=description)


@dataclass
class Test:
    text: str
    a: list[int]
    b: list[float]
    budget: int


@dataclass
class ValidationResult:
    ok: bool
    stage: str
    message: str


class Task:
    def __init__(
        self,
        text: str,
        tests: list[Test],
        objective: tp.Callable[[list[int], list[int], list[float], int], float],
    ) -> None:
        self.text = text
        self.tests = tests
        self.objective = objective

    def evaluate_output(self, output: str, test: Test) -> float:
        tokens = output.strip().split()
        if len(tokens) != len(test.a):
            raise ValueError(f"Expected {len(test.a)} integers, got {len(tokens)}")

        try:
            x = [int(token) for token in tokens]
        except ValueError as exc:
            raise ValueError("Solution output contains non-integer tokens") from exc

        if any(value < 0 for value in x):
            raise ValueError("Budget allocation must be non-negative")

        if sum(x) > test.budget:
            raise ValueError(f"Budget exceeded: sum(x)={sum(x)} > B={test.budget}")

        return self.objective(x, test.a, test.b, test.budget)

    def run(
        self,
        solution: Solution,
        timeout_per_test: int = 60,
        logger: tp.Callable[[str], None] | None = None,
    ) -> tuple[float, str]:
        total_score = 0.0

        for idx, test in enumerate(self.tests, start=1):
            try:
                output = solution.run(test.text, timeout=timeout_per_test)
                total_score += self.evaluate_output(output, test)
            except Exception as exc:
                msg = f"[test {idx}] failed for {solution.path.name}: {exc}"
                if logger is not None:
                    logger(msg)
                else:
                    print(msg)
                return BAD_SCORE, str(exc)

        return total_score, "OK"


class Agent:
    SOLUTIONS_PATH = Path("solutions")

    CODE_FORMAT_INSTRUCTION = """
Return only valid Python 3 code.
Do not include markdown fences.
The program must read from stdin and print to stdout fallowing format from the task.
A simple fast heuristic is acceptable.
""".strip()

    GEN_PROMPT = """
You are writing a Python program for an optimization problem.
Generate a complete solution from scratch.
""".strip()

    COMBINE_PROMPT = """
You are given several candidate solutions.
Combine their best ideas into one improved Python solution.
""".strip()

    MUTATE_PROMPT = """
You are given a problem and one candidate solution.
Improve the solution while keeping it correct and reasonably fast.
""".strip()

    FIX_PROMPT = """
You are given a problem and one candidate solution wich is incorrect.
Fix the solution while keeping it correct and reasonably fast.
""".strip()

    def __init__(
        self,
        task: Task,
        initial_solutions: list[Path | Solution],
        model: str,
        k: int = 3,
        timeout_per_test: int = 60,
        retry_initial_generation: int = 5,
        retry_fix_generation: int = 3,
        retry_runtime_fix_generation: int = 2,
        debug_dir: str | Path = "debug",
        verbouse: bool = False
    ) -> None:
        self.task = task
        self.model = model
        self.k = k
        self.timeout_per_test = timeout_per_test
        self.retry_initial_generation = retry_initial_generation
        self.retry_fix_generation = retry_fix_generation
        self.retry_runtime_fix_generation = retry_runtime_fix_generation
        self.verbouse = verbouse

        self.SOLUTIONS_PATH.mkdir(parents=True, exist_ok=True)

        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.debug_dir / "agent.log"
        self._artifact_counter = 0

        self.solutions: list[Solution] = []
        for item in initial_solutions:
            if isinstance(item, Solution):
                self.solutions.append(item)
            else:
                self.solutions.append(Solution(path=item))

        self._best_solution: Solution | None = None

        self._log(f"Agent initialized. model={self.model}, debug_dir={self.debug_dir}")

        for solution in self.solutions:
            if solution.score is None:
                solution.score = self.test_solution(solution)
            if self._best_solution is None or solution.score > self._best_solution.score:
                self._best_solution = solution

    @property
    def best_solution(self) -> Solution:
        if self._best_solution is None:
            raise RuntimeError("No best solution yet")
        return self._best_solution

    def _solution_path(self, idx: int) -> Path:
        return self.SOLUTIONS_PATH / f"solution_{idx:04d}.py"

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def _log(self, message: str) -> None:
        line = f"[{self._timestamp()}] {message}"
        if self.verbouse:
            print(line)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _next_artifact_path(self, stem: str, suffix: str) -> Path:
        self._artifact_counter += 1
        return self.debug_dir / f"{self._artifact_counter:04d}_{stem}{suffix}"

    def _write_debug_text(self, stem: str, text: str, suffix: str = ".txt") -> Path:
        path = self._next_artifact_path(stem, suffix)
        path.write_text(text, encoding="utf-8")
        return path

    def _build_prompt(
        self,
        instruction: str,
        extra_sections: list[tuple[str, str]] | None = None,
    ) -> str:
        parts = [
            "### Instruction",
            instruction,
            "",
            "### Output requirements",
            self.CODE_FORMAT_INSTRUCTION,
            "",
            "### Task",
            self.task.text.strip(),
        ]

        if extra_sections:
            for title, body in extra_sections:
                parts.extend(["", f"### {title}", body.strip()])

        return "\n".join(parts) + "\n"

    def ask_model(self, prompt: str) -> tuple[str, float]:
        full_prompt = (
            "/no_think\n"
            "Return only valid Python 3 code.\n"
            "No explanations.\n"
            "No markdown fences.\n\n"
            + prompt
        )
        prompt_path = self._write_debug_text("prompt", full_prompt, ".txt")
        self._log(f"Prompt saved to {prompt_path}")

        max_attempts = 10
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            raw_text = ""

            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "raw": True,
                        "keep_alive": "30m",
                        "think": False,
                        "options": {
                            "num_ctx": 4096,
                            "num_predict": 512,
                            "temperature": 0.1,
                            "top_p": 0.9,
                        },
                    },
                    timeout=600,
                )

                raw_text = response.text
                content_type = response.headers.get("content-type", "")

                self._log(f"[ollama attempt {attempt}/{max_attempts}] status={response.status_code}")
                self._log(f"[ollama attempt {attempt}/{max_attempts}] content-type={content_type}")
                self._log(f"[ollama attempt {attempt}/{max_attempts}] raw head={raw_text[:300]!r}")

                response.raise_for_status()

                try:
                    data = response.json()
                except Exception as exc:
                    bad_path = self._write_debug_text(
                        f"ollama_attempt_{attempt}_non_json",
                        raw_text,
                        ".txt",
                    )
                    self._log(f"Saved non-JSON Ollama response to {bad_path}")
                    raise RuntimeError(
                        f"Ollama returned non-JSON response on attempt {attempt}"
                    ) from exc

                self._log(f"[ollama attempt {attempt}/{max_attempts}] keys={list(data.keys())}")
                self._log(f"[ollama attempt {attempt}/{max_attempts}] done={data.get('done')}")

                if "error" in data:
                    bad_path = self._write_debug_text(
                        f"ollama_attempt_{attempt}_error",
                        raw_text,
                        ".json",
                    )
                    self._log(f"Saved Ollama error response to {bad_path}")
                    raise RuntimeError(f"Ollama error on attempt {attempt}: {data['error']}")

                if data.get("done") is not True:
                    bad_path = self._write_debug_text(
                        f"ollama_attempt_{attempt}_non_final",
                        raw_text,
                        ".json",
                    )
                    self._log(f"Saved non-final Ollama response to {bad_path}")
                    raise RuntimeError(
                        "Ollama returned a non-final response even though stream=False. "
                        f"done={data.get('done')}, keys={list(data.keys())}"
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
                    bad_path = self._write_debug_text(
                        f"ollama_attempt_{attempt}_missing_keys",
                        raw_text,
                        ".json",
                    )
                    self._log(f"Saved malformed Ollama response to {bad_path}")
                    raise RuntimeError(
                        f"Unexpected Ollama response format on attempt {attempt}. "
                        f"Missing keys: {missing}. Keys present: {list(data.keys())}"
                    )

                t1 = time.perf_counter()

                self._log(f"wall={t1 - t0:.2f}s")
                self._log(f"total={data['total_duration'] / 1e9:.2f}s")
                self._log(f"load={data['load_duration'] / 1e9:.2f}s")
                self._log(f"prompt_tokens={data['prompt_eval_count']}")
                self._log(f"gen_tokens={data['eval_count']}")

                prompt_eval_sec = max(data["prompt_eval_duration"] / 1e9, 1e-9)
                eval_sec = max(data["eval_duration"] / 1e9, 1e-9)
                self._log(f"prompt_tps={data['prompt_eval_count'] / prompt_eval_sec:.2f}")
                self._log(f"gen_tps={data['eval_count'] / eval_sec:.2f}")

                return data["response"], t1 - t0

            except Exception as exc:
                last_error = exc
                self._log(f"[ollama attempt {attempt}/{max_attempts}] failed: {exc}")

                if raw_text:
                    raw_path = self._write_debug_text(
                        f"ollama_attempt_{attempt}_raw",
                        raw_text,
                        ".txt",
                    )
                    self._log(f"Saved raw Ollama response to {raw_path}")

                if attempt < max_attempts:
                    sleep_sec = 1.5 * attempt
                    self._log(f"Sleeping {sleep_sec:.1f}s before retry")
                    time.sleep(sleep_sec)

        raise RuntimeError(f"ask_model failed after {max_attempts} attempts: {last_error}")

    def validate_solution_file(self, solution: Solution) -> ValidationResult:
        try:
            text = solution.text()
        except Exception as exc:
            return ValidationResult(False, "read", f"Cannot read file: {exc}")

        try:
            ast.parse(text)
        except SyntaxError as exc:
            return ValidationResult(False, "syntax", str(exc))

        res = subprocess.run(
            ["python3", "-m", "py_compile", str(solution.path)],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            msg = res.stderr.strip() or res.stdout.strip() or "Compilation failed"
            return ValidationResult(False, "compile", msg)

        return ValidationResult(True, "ok", "OK")

    def generate_candidate(self, prompt: str, path: Path, description: str):
        text, elapsed = self.ask_model(prompt)
        solution = Solution.from_text(text, path, description=description)
        solution.generation_time_sec = elapsed

        validation = self.validate_solution_file(solution)
        self._log(
            f"[generate_candidate] {solution.path.name} "
            f"validation={validation.stage} ok={validation.ok}"
        )
        return solution, validation

    def fix_solution(self, solution: Solution, issue: str, path: Path) -> tuple[Solution, ValidationResult]:
        extra_sections = [
            ("Candidate solution", solution.text()),
            ("Issue", issue),
        ]
        prompt = self._build_prompt(self.FIX_PROMPT, extra_sections)

        text, elapsed = self.ask_model(prompt)
        fixed = Solution.from_text(text, path, description="fixed")
        fixed.generation_time_sec = elapsed

        validation = self.validate_solution_file(fixed)
        self._log(
            f"[fix_solution] {fixed.path.name} validation={validation.stage} ok={validation.ok}"
        )
        return fixed, validation

    def generate_valid_solution(self, prompt: str, path: Path, description: str) -> Solution:
        last_issue = "no error"

        for gen_attempt in range(self.retry_initial_generation):
            candidate, validation = self.generate_candidate(prompt, path, description)

            if validation.ok:
                return candidate

            last_issue = f"{validation.stage}: {validation.message}"
            self._log(f"[generate attempt {gen_attempt + 1}] {last_issue}")

            current = candidate
            for fix_attempt in range(self.retry_fix_generation):
                fixed_path = self._solution_path(len(self.solutions) + fix_attempt + 1000)
                fixed, fixed_validation = self.fix_solution(current, last_issue, fixed_path)

                if fixed_validation.ok:
                    return fixed

                last_issue = f"{fixed_validation.stage}: {fixed_validation.message}"
                self._log(f"[fix attempt {fix_attempt + 1}] {last_issue}")
                current = fixed

        raise RuntimeError(f"Could not produce valid solution. Last issue: {last_issue}")

    def gen_solution(self, path: Path) -> Solution:
        prompt = self._build_prompt(self.GEN_PROMPT)
        return self.generate_valid_solution(prompt, path, "generated")

    def choose_ancestors(self, num: int) -> list[Solution]:
        k = min(num, len(self.solutions))
        if k == 0:
            return []
        return random.sample(self.solutions, k=k)

    def combine_solutions(self, ancestors: list[Solution], path: Path) -> Solution:
        extra_sections = [
            (f"Candidate solution {i + 1}", solution.text())
            for i, solution in enumerate(ancestors)
        ]
        prompt = self._build_prompt(self.COMBINE_PROMPT, extra_sections)
        return self.generate_valid_solution(prompt, path, "combined")

    def mutate_solution(self, solution: Solution, path: Path) -> Solution:
        extra_sections = [("Candidate solution", solution.text())]
        prompt = self._build_prompt(self.MUTATE_PROMPT, extra_sections)
        return self.generate_valid_solution(prompt, path, "mutated")

    def repair_runtime_failures(self, solution: Solution, issue: str, path: Path) -> Solution:
        current = solution
        last_issue = issue

        for fix_attempt in range(self.retry_runtime_fix_generation):
            self._log(f"[runtime fix attempt {fix_attempt + 1}] {last_issue}")

            fixed_path = self._solution_path(len(self.solutions) + 2000 + fix_attempt)
            fixed, validation = self.fix_solution(current, last_issue, fixed_path)

            if not validation.ok:
                last_issue = f"{validation.stage}: {validation.message}"
                self._log(f"[runtime fix attempt {fix_attempt + 1}] invalid fix: {last_issue}")
                current = fixed
                continue

            score, test_issue = self.test_solution(fixed)
            fixed.score = score

            if score != BAD_SCORE:
                return fixed

            last_issue = test_issue
            current = fixed

        raise RuntimeError(f"Could not repair runtime failure. Last issue: {last_issue}")

    def get_tested_solution(self, prompt: str, path: Path, description: str) -> Solution:
        candidate = self.generate_valid_solution(prompt, path, description)

        score, issue = self.test_solution(candidate)
        candidate.score = score

        if score != BAD_SCORE:
            return candidate

        return self.repair_runtime_failures(candidate, issue, path)

    def test_solution(self, solution: Solution) -> float:
        started_at = time.perf_counter()
        score, issue = self.task.run(
            solution,
            timeout_per_test=self.timeout_per_test,
            logger=self._log,
        )
        elapsed = time.perf_counter() - started_at
        solution.test_time_sec = elapsed
        self._log(
            f"[tests] {solution.path.name} score={score:.6f} "
            f"time={elapsed:.2f}s issue={issue}"
        )
        return score, issue

    def _register_solution(self, solution: Solution) -> None:
        self.solutions.append(solution)
        if self._best_solution is None or solution.score > self._best_solution.score:
            self._best_solution = solution

    def run(self, iterations: int) -> Solution:
        if not self.solutions:
            initial_path = self._solution_path(len(self.solutions))
            prompt = self._build_prompt(self.GEN_PROMPT)
            solution = self.get_tested_solution(prompt, initial_path, "generated")
            self._register_solution(solution)

        for iteration in range(iterations):
            self._log(f"=== Iteration {iteration + 1}/{iterations} ===")
            ancestors = self.choose_ancestors(self.k)
            combined_path = self._solution_path(len(self.solutions))
            combined = self.combine_solutions(ancestors, combined_path)

            mutated_path = self._solution_path(len(self.solutions) + 1)
            extra_sections = [("Candidate solution", combined.text())]
            prompt = self._build_prompt(self.MUTATE_PROMPT, extra_sections)

            candidate = self.get_tested_solution(prompt, mutated_path, "mutated")
            self._register_solution(candidate)

            self._log(
                f"[best] {self.best_solution.path.name} "
                f"score={self.best_solution.score:.6f}"
            )

        return self.best_solution


def objective(x: list[int], a: list[int], b: list[float], budget: int) -> float:
    _ = budget
    return sum(ai * (1 - math.exp(-bi * xi)) for ai, bi, xi in zip(a, b, x))


def gen_test(n: int) -> Test:
    budget = random.randrange(max(1, n), 10 * max(1, n))
    a = [random.randint(1, 100) for _ in range(n)]
    b = [max(1e-4, round(random.random(), 7)) for _ in range(n)]
    text = (
        f"{n}\n"
        + " ".join(map(str, a))
        + "\n"
        + " ".join(map(str, b))
        + "\n"
        + f"{budget}\n"
    )
    return Test(text=text, a=a, b=b, budget=budget)


def build_default_tests() -> list[Test]:
    sizes = [2, 5, 10, 50, 100]
    repeats = 3
    tests: list[Test] = []
    for _ in range(repeats):
        for n in sizes:
            tests.append(gen_test(n))
    return tests


if __name__ == "__main__":
    random.seed(42)

    task_path = Path("task.txt")
    if not task_path.exists():
        raise FileNotFoundError("task.txt not found")

    statement = task_path.read_text(encoding="utf-8")

    task = Task(
        text=statement,
        tests=build_default_tests(),
        objective=objective,
    )

    agent = Agent(
        task=task,
        initial_solutions=[],
        model="qwen2.5-coder:3b",
        k=3,
        timeout_per_test=20,
        debug_dir="debug",
    )

    result = agent.run(iterations=6)
    print(
        f"\nResult: {result.path} "
        f"score={result.score} "
        f"gen_time={result.generation_time_sec} "
        f"test_time={result.test_time_sec}"
    )
