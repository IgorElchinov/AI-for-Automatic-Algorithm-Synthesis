from __future__ import annotations

import argparse
from pathlib import Path

from agent.runner import BudgetedObjective, FunctionRunner
from tasks.coco_bbob import CocoBbobAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manually written optimize(...) candidate on COCO BBOB."
    )

    parser.add_argument(
        "--solution",
        type=Path,
        default=Path(
            "manual_solutions/random_search_function.py"
        ),
    )
    parser.add_argument(
        "--test-set",
        choices=["smoke", "full", "final"],
        default="smoke",
    )
    parser.add_argument(
        "--budget-multiplier",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Maximum wall-clock seconds per candidate/test case",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    adapter = CocoBbobAdapter(
        budget_multiplier=args.budget_multiplier,
    )

    tests_by_name = {
        "smoke": adapter.build_smoke_tests(),
        "full": adapter.build_full_tests(),
        "final": adapter.build_final_tests(),
    }
    tests = tests_by_name[args.test_set]

    runner = FunctionRunner()

    total_score = 0.0
    successful_tests = 0

    for index, test in enumerate(tests, start=1):
        meta = test.meta

        function_index = int(meta["function_index"])
        dimension = int(meta["dimension"])
        instance = int(meta["instance"])
        budget = int(meta["budget"])
        seed = int(meta["seed"])

        result = runner.run(
            solution_path=args.solution,
            function_index=function_index,
            dimension=dimension,
            instance=instance,
            budget=budget,
            seed=seed,
            budget_multiplier=args.budget_multiplier,
            timeout=args.timeout,
        )

        if not result.ok:
            print(
                f"[FAIL] test={index} "
                f"f={function_index} d={dimension} instance={instance}\n"
                f"{result.stderr}"
            )
            continue

        try:
            score = adapter.evaluate_output(result.stdout, test)
        except Exception as exc:
            print(
                f"[INVALID OUTPUT] test={index} "
                f"f={function_index} d={dimension} instance={instance}\n"
                f"{type(exc).__name__}: {exc}"
            )
            continue

        successful_tests += 1
        total_score += score

        print(
            f"[OK] test={index}/{len(tests)} "
            f"f={function_index} d={dimension} instance={instance} "
            f"score={score:.6f} "
            f"{result.stderr}"
        )

    mean_score = (
        total_score / successful_tests
        if successful_tests > 0
        else None
    )

    print("\n=== Summary ===")
    print(f"Solution: {args.solution}")
    print(f"Test set: {args.test_set}")
    print(f"Successful tests: {successful_tests}/{len(tests)}")
    print(f"Total score: {total_score}")
    print(f"Mean score: {mean_score}")


if __name__ == "__main__":
    main()