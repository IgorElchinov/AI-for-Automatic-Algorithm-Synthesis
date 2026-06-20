from __future__ import annotations

from pathlib import Path
import subprocess
import time
import sys
import importlib.util
import math

from .types import ExecutionResult


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


class FunctionRunner:
    def run_function(
        self,
        solution_path: Path,
        objective,
        lower_bounds,
        upper_bounds,
        dimension: int,
        budget: int,
        seed: int,
        timeout: int,
    ) -> ExecutionResult:
        started_at = time.perf_counter()

        try:
            module_name = f"candidate_{abs(hash(solution_path))}"
            spec = importlib.util.spec_from_file_location(module_name, solution_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Cannot load candidate module")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "optimize"):
                raise RuntimeError("Candidate must define optimize(...)")

            result = module.optimize(
                objective,
                lower_bounds,
                upper_bounds,
                dimension,
                budget,
                seed,
            )

            if not isinstance(result, (list, tuple)):
                raise RuntimeError("optimize(...) must return a list or tuple")

            if len(result) != dimension:
                raise RuntimeError(f"Expected {dimension} values, got {len(result)}")

            x = [float(v) for v in result]

            if any(not math.isfinite(v) for v in x):
                raise RuntimeError("Returned vector contains non-finite values")

            stdout = " ".join(map(str, x))

            return ExecutionResult(
                ok=True,
                stdout=stdout,
                stderr="",
                returncode=0,
                duration_sec=time.perf_counter() - started_at,
            )

        except Exception as exc:
            return ExecutionResult(
                ok=False,
                stdout="",
                stderr=str(exc),
                returncode=1,
                duration_sec=time.perf_counter() - started_at,
            )

    def run(self, solution_path: Path, input_text: str, timeout: int) -> ExecutionResult:
        try:
            from tasks.coco_bbob import BudgetedObjective, make_bbob_problem

            lines = [line.strip() for line in input_text.strip().splitlines() if line.strip()]
            if len(lines) != 6:
                raise ValueError(f"Expected 6 lines of BBOB parameters, got {len(lines)}")

            suite_name = lines[0]
            function_index = int(lines[1])
            dimension = int(lines[2])
            instance = int(lines[3])
            budget = int(lines[4])
            seed = int(lines[5])

            problem = make_bbob_problem(
                function_index=function_index,
                dimension=dimension,
                instance=instance,
                suite_name=suite_name,
            )
            objective = BudgetedObjective(problem, budget)

            return self.run_function(
                solution_path=solution_path,
                objective=objective,
                lower_bounds=problem.lower_bounds,
                upper_bounds=problem.upper_bounds,
                dimension=dimension,
                budget=budget,
                seed=seed,
                timeout=timeout,
            )
        except Exception as exc:
            return ExecutionResult(
                ok=False,
                stdout="",
                stderr=str(exc),
                returncode=1,
                duration_sec=0.0,
            )
