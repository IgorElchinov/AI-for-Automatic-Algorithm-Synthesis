from __future__ import annotations

from pathlib import Path
import ast
import subprocess

from .types import Solution, ValidationResult


BAD_MARKERS = [
    "### Candidate solution",
    "### Instruction",
    "### Output requirements",
    "### Task",
    "```",
]

def is_degenerate_code(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return True

    if stripped.replace("`", "").strip() == "":
        return True

    extracted = Solution.extract_code(text).strip()
    if not extracted:
        return True

    if any(marker in stripped for marker in BAD_MARKERS):
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        first_lines = lines[:8]
        if any(
            line.startswith("###")
            or line.startswith("```")
            or "Candidate solution" in line
            or line.startswith("Instruction")
            for line in first_lines
        ):
            return True

    tokens = stripped.split()
    if tokens and all(token.replace("`", "") == "" for token in tokens):
        return True

    return False


ALLOWED_TOP_LEVEL_IMPORTS = {
    "math",
    "random",
    "statistics",
    "itertools",
    "functools",
    "collections",
    "heapq",
    "bisect",
    "dataclasses",
    "typing",
    "pathlib",
    "time",
    "json",
    "re",
    "numpy",
    "cocoex",
    "opytimizer",
}

FORBIDDEN_PREFIXES = (
    "scipy",
    "nevergrad",
    "pymoo",
    "sklearn",
    "torch",
    "tensorflow",
)

BAD_IMPORT_PREFIXES = (
    "opytimizer.optimizers.swarm",
    "opytimizer.optimizers.population",
    "opytimizer.optimizers.evolutionary",
    "opytimizer.optimizers.science",
)

def validate_imports(text: str) -> ValidationResult | None:                 # TODO: Use this
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None  # syntax stage already handles this

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                top = name.split(".")[0]

                if name.startswith(BAD_IMPORT_PREFIXES):
                    return ValidationResult(
                        False,
                        "bad_import",
                        f"Outdated Opytimizer import path: {name}",
                    )

                if name.startswith(FORBIDDEN_PREFIXES):
                    return ValidationResult(
                        False,
                        "forbidden_import",
                        f"Forbidden library import: {name}",
                    )

                if top not in ALLOWED_TOP_LEVEL_IMPORTS:
                    return ValidationResult(
                        False,
                        "unknown_import",
                        f"Unexpected import: {name}",
                    )

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0] if module else ""

            if module.startswith(BAD_IMPORT_PREFIXES):
                return ValidationResult(
                    False,
                    "bad_import",
                    f"Outdated Opytimizer import path: {module}",
                )

            if module.startswith(FORBIDDEN_PREFIXES):
                return ValidationResult(
                    False,
                    "forbidden_import",
                    f"Forbidden library import: {module}",
                )

            if module and top not in ALLOWED_TOP_LEVEL_IMPORTS:
                return ValidationResult(
                    False,
                    "unknown_import",
                    f"Unexpected from-import: {module}",
                )

    return None


REQUIRED_OPY_SYMBOLS = {
    "Opytimizer",
    "Function",
    "SearchSpace",
}

def validate_opytimizer_usage(text: str) -> ValidationResult | None:                 # TODO: Use this
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    imported_names = set()
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_modules.add(module)
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)

    if "opytimizer" not in {m.split(".")[0] for m in imported_modules}:
        return ValidationResult(False, "missing_opytimizer", "No Opytimizer import found")

    missing = REQUIRED_OPY_SYMBOLS - imported_names
    if missing:
        return ValidationResult(
            False,
            "missing_opytimizer_symbols",
            f"Missing required Opytimizer symbols: {sorted(missing)}",
        )

    return None


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

    validation_result = validate_imports(text)
    if validation_result is not None:
        return validation_result

    validation_result = validate_opytimizer_usage(text)
    if validation_result is not None:
        return validation_result

    result = subprocess.run(
        ["python3", "-m", "py_compile", str(solution.path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "Compilation failed"
        return ValidationResult(False, "compile", msg)

    return ValidationResult(True, "ok", "OK")