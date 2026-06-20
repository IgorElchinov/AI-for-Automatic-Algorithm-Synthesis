from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median, pstdev

from agent.runner import FunctionRunner
from tasks.coco_bbob import CocoBbobAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an optimize(...) candidate on COCO BBOB."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument(
        "--test-set",
        choices=["full", "final"],
        default="final",
    )
    parser.add_argument("--timeout-per-test", type=int, default=15)
    parser.add_argument(
        "--candidate-seed-offset",
        type=int,
        default=0,
        help="Integer added to each benchmark seed before calling optimize(...).",
    )
    return parser.parse_args()


def trimmed_mean(values: list[float], trim_fraction: float = 0.05) -> float:
    if not values:
        return float("nan")

    ordered = sorted(values)
    k = int(len(ordered) * trim_fraction)

    if 2 * k >= len(ordered):
        return mean(ordered)

    return mean(ordered[k: len(ordered) - k])


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)

    if not ordered:
        return float("nan")

    index = round((len(ordered) - 1) * q)
    return ordered[index]


def main() -> None:
    args = parse_args()

    output_dir = Path("results/function_candidates") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = CocoBbobAdapter()
    runner = FunctionRunner()

    if args.test_set == "full":
        tests = adapter.build_full_tests()
    else:
        tests = adapter.build_final_tests()

    rows: list[dict] = []

    for test_index, test in enumerate(tests, start=1):
        execution = runner.run(
            args.solution,
            test.input_text,
            args.timeout_per_test,
            candidate_seed_offset=args.candidate_seed_offset,
        )

        row = {
            "test_index": test_index,
            "meta": test.meta,
            "ok": execution.ok,
            "returncode": execution.returncode,
            "duration_sec": execution.duration_sec,
            "stderr": execution.stderr,
            "score": None,
        }

        if execution.ok:
            try:
                row["score"] = adapter.evaluate_output(
                    execution.stdout,
                    test,
                )
            except Exception as exc:
                row["ok"] = False
                row["stderr"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)

        score_str = (
            f"{row['score']:.6f}"
            if row["score"] is not None
            else "NA"
        )

        print(
            f"[{test_index}/{len(tests)}] "
            f"ok={row['ok']} score={score_str} "
            f"time={row['duration_sec']:.2f}s"
        )

    valid_scores = [
        float(row["score"])
        for row in rows
        if row["ok"] and row["score"] is not None
    ]

    summary = {
        "run_name": args.run_name,
        "solution": str(args.solution),
        "test_set": args.test_set,
        "n_tests": len(rows),
        "n_success": len(valid_scores),
        "success_rate": len(valid_scores) / len(rows) if rows else 0.0,
        "mean_score": mean(valid_scores) if valid_scores else None,
        "median_score": median(valid_scores) if valid_scores else None,
        "std_score": pstdev(valid_scores) if len(valid_scores) > 1 else 0.0,
        "trimmed_mean_5pct": (
            trimmed_mean(valid_scores) if valid_scores else None
        ),
        "q25": quantile(valid_scores, 0.25) if valid_scores else None,
        "q75": quantile(valid_scores, 0.75) if valid_scores else None,
        "positive_score_rate": (
            sum(score > 0 for score in valid_scores) / len(valid_scores)
            if valid_scores
            else 0.0
        ),
    }

    (output_dir / "raw_results.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()