from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

from tasks.coco_bbob import BudgetedObjective, make_bbob_problem

# import logging
# logging.getLogger("opytimizer").setLevel(logging.ERROR)


def fail(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}))
    raise SystemExit(1)


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())

        solution_path = Path(payload["solution_path"]).resolve()
        function_index = int(payload["function_index"])
        dimension = int(payload["dimension"])
        instance = int(payload["instance"])
        budget = int(payload["budget"])
        seed = int(payload["seed"])
        suite_name = payload["suite_name"]
        # budget_multiplier = int(payload["budget_multiplier"])

        # adapter = CocoBbobAdapter()
        problem = make_bbob_problem(
            function_index=function_index,
            dimension=dimension,
            instance=instance,
            suite_name=suite_name,
        )

        objective = BudgetedObjective(problem, budget)

        module_name = f"candidate_{solution_path.stem}_{abs(hash(str(solution_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, solution_path)

        if spec is None or spec.loader is None:
            fail(f"Cannot import candidate from {solution_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        optimize = getattr(module, "optimize", None)
        if not callable(optimize):
            fail("Candidate must define callable optimize(...)")

        result = optimize(
            objective,
            problem.lower_bounds,
            problem.upper_bounds,
            dimension,
            budget,
            seed,
        )

        if not isinstance(result, (list, tuple)):
            fail("optimize(...) must return list or tuple")

        if len(result) != dimension:
            fail(f"Expected {dimension} coordinates, got {len(result)}")

        x = [float(value) for value in result]

        if any(not math.isfinite(value) for value in x):
            fail("Candidate returned non-finite coordinates")

        for i, value in enumerate(x):
            lower = float(problem.lower_bounds[i])
            upper = float(problem.upper_bounds[i])

            if value < lower or value > upper:
                fail(
                    f"Coordinate {i}={value} outside bounds [{lower}, {upper}]"
                )

        print(
            json.dumps(
                {
                    "ok": True,
                    "x": x,
                    "objective_calls": objective.calls,
                    "budget": budget,
                }
            )
        )

    except Exception as exc:
        fail(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()