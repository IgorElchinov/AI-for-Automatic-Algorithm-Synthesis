from __future__ import annotations

from pathlib import Path
import subprocess
import time
import sys
import importlib.util
import math

from .types import ExecutionResult
import json


class PythonRunner:
    def __init__(self, python_executable: str | None = None) -> None:
        # Use the current Python interpreter by default so subprocesses inherit
        # the active virtual environment when the agent is launched from it.
        self.python_executable = python_executable or sys.executable

    def run(self, solution_path: Path, input_text: str, timeout: int) -> ExecutionResult:
        started_at = time.perf_counter()
        result = subprocess.run(
            [self.python_executable, str(solution_path)],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - started_at

        return ExecutionResult(
            ok=(result.returncode == 0),
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            duration_sec=elapsed,
        )


class CommandRunner:
    """
    Generic runner 
    Can execute any command, i.e.:
        CommandRunner(["pypy3"])
        CommandRunner(["./run_solution.sh"])
    """
    def __init__(self, command_prefix: list[str]) -> None:
        self.command_prefix = command_prefix

    def run(self, solution_path: Path, input_text: str, timeout: int) -> ExecutionResult:
        started_at = time.perf_counter()
        result = subprocess.run(
            self.command_prefix + [str(solution_path)],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - started_at

        return ExecutionResult(
            ok=(result.returncode == 0),
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            duration_sec=elapsed,
        )


class BudgetExceededError(RuntimeError):
    pass


class BudgetedObjective:
    def __init__(self, problem, budget: int) -> None:
        self.problem = problem
        self.budget = int(budget)
        self.calls = 0

    def __call__(self, x) -> float:
        if self.calls >= self.budget:
            raise BudgetExceededError(
                f"Evaluation budget exceeded: attempted call "
                f"{self.calls + 1}, budget={self.budget}"
            )

        self.calls += 1
        return float(self.problem(x))


class FunctionRunner:
    def __init__(self, python_executable: str | None = None) -> None:
        self.python_executable = python_executable or sys.executable

    def run(
        self,
        solution_path: Path,
        input_text: str,
        timeout: int,
        candidate_seed_offset: int = 0,
    ) -> ExecutionResult:
        started_at = time.perf_counter()

        try:
            lines = [
                line.strip()
                for line in input_text.strip().splitlines()
                if line.strip()
            ]

            if len(lines) != 6:
                raise ValueError(
                    f"Expected 6 BBOB input lines, got {len(lines)}"
                )

            suite_name = lines[0]
            function_index = int(lines[1])
            dimension = int(lines[2])
            instance = int(lines[3])
            budget = int(lines[4])
            seed = int(lines[5])

            payload = {
                "solution_path": str(solution_path.resolve()),
                "suite_name": suite_name,
                "function_index": function_index,
                "dimension": dimension,
                "instance": instance,
                "budget": budget,
                "seed": seed,
                "candidate_seed_offset": candidate_seed_offset,
            }

            completed = subprocess.run(
                [self.python_executable, "-m", "agent.function_worker"],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                ok=False,
                stdout="",
                stderr=f"TimeoutExpired: exceeded {timeout} seconds",
                returncode=-1,
                duration_sec=time.perf_counter() - started_at,
            )

        except Exception as exc:
            return ExecutionResult(
                ok=False,
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
                returncode=1,
                duration_sec=time.perf_counter() - started_at,
            )

        duration_sec = time.perf_counter() - started_at

        if completed.returncode != 0:
            return ExecutionResult(
                ok=False,
                stdout="",
                stderr=(completed.stderr or completed.stdout).strip(),
                returncode=completed.returncode,
                duration_sec=duration_sec,
            )

        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            return ExecutionResult(
                ok=False,
                stdout="",
                stderr=(
                    f"Invalid worker JSON: {exc}; "
                    f"worker output: {completed.stdout[:500]}"
                ),
                returncode=completed.returncode,
                duration_sec=duration_sec,
            )

        if not result.get("ok"):
            return ExecutionResult(
                ok=False,
                stdout="",
                stderr=str(result.get("error", "Unknown worker error")),
                returncode=completed.returncode,
                duration_sec=completed.returncode,
            )

        x = result["x"]

        return ExecutionResult(
            ok=True,
            stdout=" ".join(map(str, x)),
            stderr=(
                f"objective_calls={result['objective_calls']}/"
                f"{result['budget']}"
            ),
            returncode=0,
            duration_sec=duration_sec,
        )
