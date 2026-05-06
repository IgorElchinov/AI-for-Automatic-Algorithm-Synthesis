from __future__ import annotations

from pathlib import Path
import ast
import subprocess

from .types import Solution, ValidationResult


def is_degenerate_code(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return True

    if stripped.replace("`", "").strip() == "":
        return True

    # bad_markers = [
    #     "### Candidate solution",
    #     "### Instruction",
    #     "### Output requirements",
    #     "### Task",
    # ]
    # if any(marker in stripped for marker in bad_markers):
    #     return True

    # lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    # if lines:
    #     if any(line.startswith("###") or line.startswith("```") for line in lines[:3]):
    #         return True

    extracted = Solution.extract_code(text).strip()
    if not extracted:
        return True

    tokens = stripped.split()
    if tokens and all(token.replace("`", "") == "" for token in tokens):
        return True

    return False


def validate_python_file(solution: Solution) -> ValidationResult:
    try:
        text = solution.text()
    except Exception as exc:
        return ValidationResult(False, "read", f"Cannot read file: {exc}")

    if is_degenerate_code(text):
        return ValidationResult(
            False,
            "degenerate",
            "Generated file is empty or contains only markdown/backticks",
        )

    try:
        ast.parse(text)
    except SyntaxError as exc:
        return ValidationResult(False, "syntax", str(exc))

    result = subprocess.run(
        ["python3", "-m", "py_compile", str(solution.path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "Compilation failed"
        return ValidationResult(False, "compile", msg)

    return ValidationResult(True, "ok", "OK")