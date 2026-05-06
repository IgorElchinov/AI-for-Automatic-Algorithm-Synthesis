from __future__ import annotations

from pathlib import Path
import subprocess
import time

from .types import ExecutionResult


class PythonRunner:
    def __init__(self, python_executable: str = "python3") -> None:
        self.python_executable = python_executable

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