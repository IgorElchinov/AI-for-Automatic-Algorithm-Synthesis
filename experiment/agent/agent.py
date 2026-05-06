from __future__ import annotations

from pathlib import Path
import random
import re
import time

from .interfaces import ModelClient, ProblemAdapter, PromptStrategy, Runner
from .prompts import DefaultPromptStrategy
from .runner import PythonRunner
from .types import BAD_SCORE, Solution, ValidationResult
from .validation import validate_python_file


class Agent:
    SOLUTIONS_PATH = Path("solutions")

    def __init__(
        self,
        problem: ProblemAdapter,
        model_client: ModelClient,
        runner: Runner | None = None,
        prompt_strategy: PromptStrategy | None = None,
        k: int = 3,
        timeout_per_test: int = 60,
        retry_initial_generation: int = 5,
        retry_fix_generation: int = 3,
        retry_runtime_fix_generation: int = 2,
        debug_dir: str | Path = "debug",
        verbose: bool = False,
        smoke_test_count: int = 1,
        ancestor_pool_size: int = 5,
        initial_solutions: list[Path | Solution] | None = None,
    ) -> None:
        self.problem = problem
        self.model_client = model_client
        self.runner = runner or PythonRunner()
        self.prompt_strategy = prompt_strategy or DefaultPromptStrategy()

        self.k = k
        self.timeout_per_test = timeout_per_test
        self.retry_initial_generation = retry_initial_generation
        self.retry_fix_generation = retry_fix_generation
        self.retry_runtime_fix_generation = retry_runtime_fix_generation
        self.verbose = verbose
        self.smoke_test_count = smoke_test_count
        self.ancestor_pool_size = ancestor_pool_size

        self.SOLUTIONS_PATH.mkdir(parents=True, exist_ok=True)

        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.debug_dir / "agent.log"
        self._artifact_counter = 0
        self._solution_counter = self._initialize_solution_counter()

        self.solutions: list[Solution] = []
        initial_solutions = initial_solutions or []
        for item in initial_solutions:
            if isinstance(item, Solution):
                self.solutions.append(item)
            else:
                self.solutions.append(Solution(path=item))

        self._best_solution: Solution | None = None
        self._log("Agent initialized")

        for solution in self.solutions:
            if solution.score is None:
                score, _ = self.test_solution(solution)
                solution.score = score
            if self._best_solution is None or solution.score > self._best_solution.score:
                self._best_solution = solution

    @property
    def best_solution(self) -> Solution:
        if self._best_solution is None:
            raise RuntimeError("No best solution yet")
        return self._best_solution

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def _log(self, message: str) -> None:
        line = f"[{self._timestamp()}] {message}"
        if self.verbose:
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

    def _initialize_solution_counter(self) -> int:
        max_idx = -1
        pattern = re.compile(r"solution_(\d+)\.py$")
        for path in self.SOLUTIONS_PATH.glob("solution_*.py"):
            match = pattern.search(path.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        return max_idx + 1

    def _new_solution_path(self) -> Path:
        path = self.SOLUTIONS_PATH / f"solution_{self._solution_counter:04d}.py"
        self._solution_counter += 1
        return path

    def validate_solution_file(self, solution: Solution) -> ValidationResult:
        return validate_python_file(solution)

    def generate_candidate(
        self,
        prompt: str,
        path: Path,
        description: str,
    ) -> tuple[Solution, ValidationResult]:
        prompt_path = self._write_debug_text("prompt", prompt, ".txt")
        self._log(f"Prompt saved to {prompt_path}")

        response = self.model_client.generate(prompt)
        solution = Solution.from_text(response.text, path, description=description)
        solution.generation_time_sec = response.wall_time_sec

        validation = self.validate_solution_file(solution)
        self._log(
            f"[generate_candidate] {solution.path.name} "
            f"validation={validation.stage} ok={validation.ok}"
        )
        return solution, validation

    def fix_solution(self, solution: Solution, issue: str, path: Path) -> tuple[Solution, ValidationResult]:
        prompt = self.prompt_strategy.build_fix_prompt(
            self.problem.problem_text,
            solution.text(),
            issue,
        )
        return self.generate_candidate(prompt, path, "fixed")

    def generate_valid_solution(self, prompt: str, path: Path, description: str) -> Solution:
        last_issue = "no error"

        for gen_attempt in range(self.retry_initial_generation):
            candidate, validation = self.generate_candidate(prompt, path, description)

            if validation.ok:
                return candidate

            last_issue = f"{validation.stage}: {validation.message}"
            self._log(f"[generate attempt {gen_attempt + 1}] {last_issue}")

            if validation.stage in {"degenerate", "empty"}:
                self._log(
                    f"[generate attempt {gen_attempt + 1}] "
                    "degenerate candidate -> fresh generation"
                )
                continue

            current = candidate
            for fix_attempt in range(self.retry_fix_generation):
                fixed_path = self._new_solution_path()
                fixed, fixed_validation = self.fix_solution(current, last_issue, fixed_path)

                if fixed_validation.ok:
                    return fixed

                last_issue = f"{fixed_validation.stage}: {fixed_validation.message}"
                self._log(f"[fix attempt {fix_attempt + 1}] {last_issue}")
                current = fixed

        raise RuntimeError(f"Could not produce valid solution. Last issue: {last_issue}")

    def choose_ancestors(self, num: int) -> list[Solution]:
        valid = [
            solution for solution in self.solutions
            if solution.score is not None and solution.score != BAD_SCORE
        ]
        if not valid:
            return []

        valid.sort(key=lambda s: s.score if s.score is not None else BAD_SCORE, reverse=True)
        pool = valid[: max(1, self.ancestor_pool_size)]
        k = min(num, len(pool))
        return random.sample(pool, k=k)

    def test_solution(self, solution: Solution, tests=None) -> tuple[float, str]:
        active_tests = self.problem.build_full_tests() if tests is None else tests
        total_score = 0.0
        started_at = time.perf_counter()

        for idx, test in enumerate(active_tests, start=1):
            execution = self.runner.run(solution.path, test.input_text, self.timeout_per_test)
            if not execution.ok:
                issue = (
                    f"[test {idx}] runtime failure for {solution.path.name}\n"
                    f"returncode={execution.returncode}\n"
                    f"stdout:\n{execution.stdout}\n"
                    f"stderr:\n{execution.stderr}"
                )
                self._log(issue)
                dump_path = self._write_debug_text(f"test_failure_{solution.path.stem}", issue, ".txt")
                self._log(f"Saved test failure details to {dump_path}")
                solution.test_time_sec = time.perf_counter() - started_at
                return BAD_SCORE, issue

            try:
                total_score += self.problem.evaluate_output(execution.stdout, test)
            except Exception as exc:
                issue = f"[test {idx}] invalid output for {solution.path.name}: {exc}"
                self._log(issue)
                dump_path = self._write_debug_text(f"test_failure_{solution.path.stem}", issue, ".txt")
                self._log(f"Saved test failure details to {dump_path}")
                solution.test_time_sec = time.perf_counter() - started_at
                return BAD_SCORE, issue

        elapsed = time.perf_counter() - started_at
        solution.test_time_sec = elapsed
        self._log(f"[tests] {solution.path.name} score={total_score:.6f} time={elapsed:.2f}s issue=OK")
        return total_score, "OK"

    def _run_smoke_test(self, solution: Solution) -> tuple[float, str]:
        if self.smoke_test_count <= 0:
            return 0.0, "SKIPPED"
        smoke_tests = self.problem.build_smoke_tests()[: self.smoke_test_count]
        return self.test_solution(solution, tests=smoke_tests)

    def repair_runtime_failures(self, solution: Solution, issue: str) -> Solution:
        current = solution
        last_issue = issue

        for fix_attempt in range(self.retry_runtime_fix_generation):
            self._log(f"[runtime fix attempt {fix_attempt + 1}] {last_issue}")

            fixed_path = self._new_solution_path()
            fixed, validation = self.fix_solution(current, last_issue, fixed_path)

            if not validation.ok:
                last_issue = f"{validation.stage}: {validation.message}"
                self._log(f"[runtime fix attempt {fix_attempt + 1}] invalid fix: {last_issue}")
                current = fixed
                continue

            smoke_score, smoke_issue = self._run_smoke_test(fixed)
            if smoke_score == BAD_SCORE:
                last_issue = smoke_issue
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

        smoke_score, smoke_issue = self._run_smoke_test(candidate)
        if smoke_score == BAD_SCORE:
            return self.repair_runtime_failures(candidate, smoke_issue)

        score, issue = self.test_solution(candidate)
        candidate.score = score

        if score != BAD_SCORE:
            return candidate

        return self.repair_runtime_failures(candidate, issue)

    def _register_solution(self, solution: Solution) -> None:
        self.solutions.append(solution)
        if self._best_solution is None or solution.score > self._best_solution.score:
            self._best_solution = solution

    def run(self, iterations: int) -> Solution:
        if not self.solutions:
            initial_path = self._new_solution_path()
            prompt = self.prompt_strategy.build_gen_prompt(self.problem.problem_text)
            solution = self.get_tested_solution(prompt, initial_path, "generated")
            self._register_solution(solution)

        for iteration in range(iterations):
            self._log(f"=== Iteration {iteration + 1}/{iterations} ===")
            ancestors = self.choose_ancestors(self.k)

            if not ancestors:
                gen_path = self._new_solution_path()
                prompt = self.prompt_strategy.build_gen_prompt(self.problem.problem_text)
                candidate = self.get_tested_solution(prompt, gen_path, "generated")
                self._register_solution(candidate)
                continue

            combined_path = self._new_solution_path()
            combine_prompt = self.prompt_strategy.build_combine_prompt(
                self.problem.problem_text,
                [ancestor.text() for ancestor in ancestors],
            )
            combined = self.get_tested_solution(combine_prompt, combined_path, "combined")

            mutated_path = self._new_solution_path()
            mutate_prompt = self.prompt_strategy.build_mutate_prompt(
                self.problem.problem_text,
                combined.text(),
            )
            candidate = self.get_tested_solution(mutate_prompt, mutated_path, "mutated")
            self._register_solution(candidate)

            self._log(
                f"[best] {self.best_solution.path.name} "
                f"score={self.best_solution.score:.6f}"
            )

        return self.best_solution