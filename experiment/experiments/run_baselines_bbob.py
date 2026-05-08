from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from baselines.opytimizer_bbob import (
    aggregate_results,
    available_default_specs,
    run_opytimizer_suite,
    save_results_json,
    save_summary_json,
    unavailable_default_specs,
)
from tasks.coco_bbob import CocoBbobAdapter, DEFAULT_BBOB_PROBLEM_TEXT


@dataclass
class BaselineRunConfig:
    run_name: str
    model_group: str
    seed: int
    budget_multiplier: int
    test_set: str
    selected_algorithms: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Opytimizer baselines on COCO BBOB.")
    parser.add_argument("--run-name", type=str, default=None, help="Output run name.")
    parser.add_argument("--base-dir", type=str, default="results/baselines_bbob")
    parser.add_argument(
        "--test-set",
        type=str,
        choices=["full", "final"],
        default="final",
        help="Which frozen test set from CocoBbobAdapter to evaluate on.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-multiplier", type=int, default=20)
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of baseline algorithm names to run, e.g. --algorithms pso de",
    )
    return parser.parse_args()


def timestamp_run_name() -> str:
    return time.strftime("baseline_run_%Y%m%d_%H%M%S")


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def choose_tests(adapter: CocoBbobAdapter, test_set: str):
    if test_set == "full":
        return adapter.build_full_tests()
    if test_set == "final":
        return adapter.build_final_tests()
    raise ValueError(f"Unknown test_set={test_set}")


def main() -> None:
    args = parse_args()
    run_name = args.run_name or timestamp_run_name()

    base_dir = Path(args.base_dir)
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = BaselineRunConfig(
        run_name=run_name,
        model_group="opytimizer_baselines",
        seed=args.seed,
        budget_multiplier=args.budget_multiplier,
        test_set=args.test_set,
        selected_algorithms=list(args.algorithms) if args.algorithms else [],
    )

    adapter = CocoBbobAdapter(
        problem_text=DEFAULT_BBOB_PROBLEM_TEXT,
        budget_multiplier=args.budget_multiplier,
        random_seed=args.seed,
    )
    tests = choose_tests(adapter, args.test_set)

    available_specs = available_default_specs()
    missing_specs = unavailable_default_specs()

    if args.algorithms:
        requested = set(args.algorithms)
        available_specs = [spec for spec in available_specs if spec.name in requested]

    if not available_specs:
        raise RuntimeError(
            "No available Opytimizer baseline specs to run. "
            "Check installation or selected algorithm names."
        )

    config_path = run_dir / "config.json"
    save_config = asdict(config)
    save_config["n_tests"] = len(tests)
    save_config["available_algorithms"] = [spec.name for spec in available_specs]
    save_config["missing_algorithms"] = [
        {"name": spec.name, "module_path": spec.module_path, "class_name": spec.class_name, "error": error}
        for spec, error in missing_specs
    ]
    atomic_write_json(config_path, save_config)

    (run_dir / "problem_text.txt").write_text(DEFAULT_BBOB_PROBLEM_TEXT, encoding="utf-8")

    print(f"Run dir: {run_dir}")
    print(f"Test set: {args.test_set}")
    print(f"Number of tests: {len(tests)}")
    print(f"Algorithms: {[spec.name for spec in available_specs]}")

    results = run_opytimizer_suite(
        adapter=adapter,
        specs=available_specs,
        tests=tests,
        seed=args.seed,
    )
    summary = aggregate_results(results)

    raw_results_path = run_dir / "raw_results.json"
    summary_path = run_dir / "summary.json"
    unavailable_path = run_dir / "unavailable_specs.json"

    save_results_json(raw_results_path, results)
    save_summary_json(summary_path, summary)
    atomic_write_json(
        unavailable_path,
        {
            "missing_specs": [
                {
                    "name": spec.name,
                    "module_path": spec.module_path,
                    "class_name": spec.class_name,
                    "error": error,
                }
                for spec, error in missing_specs
            ]
        },
    )

    print(f"Saved raw results to: {raw_results_path}")
    print(f"Saved summary to: {summary_path}")

    for algorithm, stats in summary["algorithms"].items():
        print(
            f"{algorithm}: "
            f"mean_score={stats['mean_score']:.6f} "
            f"median_score={stats['median_score']:.6f} "
            f"std_score={stats['std_score']:.6f} "
            f"success_rate={stats['success_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
