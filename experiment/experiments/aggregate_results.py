from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate agent and baseline BBOB experiment results.")
    parser.add_argument("--agent-dir", type=str, default="results/agent_bbob")
    parser.add_argument("--baseline-dir", type=str, default="results/baselines_bbob")
    parser.add_argument("--out-dir", type=str, default="results/aggregate_bbob")
    parser.add_argument("--tag", type=str, default="latest", help="Label to include in output filenames.")
    return parser.parse_args()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        value = float(x)
        if math.isfinite(value):
            return value
        return None
    except Exception:
        return None


def compute_numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_agent_runs(agent_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not agent_dir.exists():
        return runs

    for run_dir in sorted([p for p in agent_dir.iterdir() if p.is_dir()]):
        final_result = read_json(run_dir / "final_result.json", {})
        state = read_json(run_dir / "state.json", {})
        config = read_json(run_dir / "config.json", {})
        if not final_result and not state:
            continue

        record = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "status": final_result.get("status", state.get("status")),
            "completed_iterations": final_result.get("completed_iterations", state.get("completed_iterations")),
            "best_solution_path": final_result.get("best_solution_path", state.get("best_solution_path")),
            "best_score_total_inner_loop": safe_float(final_result.get("best_score_total_inner_loop", state.get("best_score_total_inner_loop"))),
            "best_score_mean_inner_loop": safe_float(final_result.get("best_score_mean_inner_loop", state.get("best_score_mean_inner_loop"))),
            "n_inner_loop_tests": final_result.get("n_inner_loop_tests"),
            "final_holdout_total_score": safe_float(final_result.get("final_holdout_total_score", state.get("final_holdout_total_score"))),
            "final_holdout_mean_score": safe_float(final_result.get("final_holdout_mean_score", state.get("final_holdout_mean_score"))),
            "n_final_holdout_tests": final_result.get("n_final_holdout_tests"),
            "best_solution_test_time_sec": safe_float(final_result.get("best_solution_test_time_sec")),
            "model": config.get("model", {}).get("model") if isinstance(config.get("model"), dict) else config.get("model"),
            "temperature": (config.get("model", {}).get("temperature") if isinstance(config.get("model"), dict) else config.get("temperature")),
            "top_p": (config.get("model", {}).get("top_p") if isinstance(config.get("model"), dict) else config.get("top_p")),
            "num_ctx": (config.get("model", {}).get("num_ctx") if isinstance(config.get("model"), dict) else config.get("num_ctx")),
            "num_predict": (config.get("model", {}).get("num_predict") if isinstance(config.get("model"), dict) else config.get("num_predict")),
            "budget_multiplier": (config.get("bbob", {}).get("budget_multiplier") if isinstance(config.get("bbob"), dict) else config.get("budget_multiplier")),
        }
        runs.append(record)
    return runs


def collect_baseline_runs(baseline_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    algorithm_rows: list[dict[str, Any]] = []
    if not baseline_dir.exists():
        return run_rows, algorithm_rows

    for run_dir in sorted([p for p in baseline_dir.iterdir() if p.is_dir()]):
        config = read_json(run_dir / "config.json", {})
        summary = read_json(run_dir / "summary.json", {})
        raw_results = read_json(run_dir / "raw_results.json", [])
        if not summary and not raw_results:
            continue

        run_rows.append({
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "test_set": config.get("test_set"),
            "seed": config.get("seed"),
            "budget_multiplier": config.get("budget_multiplier"),
            "n_tests": config.get("n_tests"),
            "available_algorithms": ",".join(config.get("available_algorithms", [])),
        })

        algorithms = summary.get("algorithms", {})
        for algorithm, stats in algorithms.items():
            algorithm_rows.append({
                "run_name": run_dir.name,
                "algorithm": algorithm,
                "n_runs": stats.get("n_runs"),
                "mean_score": safe_float(stats.get("mean_score")),
                "median_score": safe_float(stats.get("median_score")),
                "std_score": safe_float(stats.get("std_score")),
                "success_rate": safe_float(stats.get("success_rate")),
                "mean_used_evaluations": safe_float(stats.get("mean_used_evaluations")),
                "test_set": config.get("test_set"),
                "seed": config.get("seed"),
                "budget_multiplier": config.get("budget_multiplier"),
            })
    return run_rows, algorithm_rows


def summarize_agent_runs(agent_runs: list[dict[str, Any]]) -> dict[str, Any]:
    inner_total_scores = [x for x in (safe_float(r.get("best_score_total_inner_loop")) for r in agent_runs) if x is not None]
    inner_mean_scores = [x for x in (safe_float(r.get("best_score_mean_inner_loop")) for r in agent_runs) if x is not None]
    holdout_total_scores = [x for x in (safe_float(r.get("final_holdout_total_score")) for r in agent_runs) if x is not None]
    holdout_mean_scores = [x for x in (safe_float(r.get("final_holdout_mean_score")) for r in agent_runs) if x is not None]

    best_run = None
    if agent_runs:
        ranked = sorted(agent_runs, key=lambda r: (safe_float(r.get("final_holdout_mean_score")) if safe_float(r.get("final_holdout_mean_score")) is not None else -10**18), reverse=True)
        best_run = ranked[0]

    return {
        "n_agent_runs": len(agent_runs),
        "inner_loop_total_score_summary": compute_numeric_summary(inner_total_scores),
        "inner_loop_mean_score_summary": compute_numeric_summary(inner_mean_scores),
        "final_holdout_total_score_summary": compute_numeric_summary(holdout_total_scores),
        "final_holdout_mean_score_summary": compute_numeric_summary(holdout_mean_scores),
        "best_run_by_holdout_mean_score": best_run,
    }


def summarize_baseline_algorithms(algorithm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in algorithm_rows:
        grouped.setdefault(str(row["algorithm"]), []).append(row)

    by_algorithm: dict[str, Any] = {}
    for algorithm, rows in grouped.items():
        mean_scores = [x for x in (safe_float(r.get("mean_score")) for r in rows) if x is not None]
        median_scores = [x for x in (safe_float(r.get("median_score")) for r in rows) if x is not None]
        success_rates = [x for x in (safe_float(r.get("success_rate")) for r in rows) if x is not None]
        by_algorithm[algorithm] = {
            "n_runs": len(rows),
            "mean_score_summary": compute_numeric_summary(mean_scores),
            "median_score_summary": compute_numeric_summary(median_scores),
            "success_rate_summary": compute_numeric_summary(success_rates),
        }

    return {"n_algorithm_entries": len(algorithm_rows), "algorithms": by_algorithm}


def compare_agent_vs_baselines(agent_runs: list[dict[str, Any]], algorithm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    agent_holdout_mean_scores = [x for x in (safe_float(r.get("final_holdout_mean_score")) for r in agent_runs) if x is not None]

    grouped: dict[str, list[float]] = {}
    for row in algorithm_rows:
        val = safe_float(row.get("mean_score"))
        if val is not None:
            grouped.setdefault(str(row["algorithm"]), []).append(val)

    comparison_rows: list[dict[str, Any]] = []
    agent_mean = mean(agent_holdout_mean_scores) if agent_holdout_mean_scores else None
    agent_best = max(agent_holdout_mean_scores) if agent_holdout_mean_scores else None

    for algorithm, scores in sorted(grouped.items()):
        baseline_mean = mean(scores) if scores else None
        baseline_best = max(scores) if scores else None
        comparison_rows.append({
            "algorithm": algorithm,
            "baseline_mean_score": baseline_mean,
            "baseline_best_score": baseline_best,
            "agent_mean_holdout_score": agent_mean,
            "agent_best_holdout_score": agent_best,
            "mean_delta_agent_minus_baseline": (agent_mean - baseline_mean if agent_mean is not None and baseline_mean is not None else None),
            "best_delta_agent_minus_baseline": (agent_best - baseline_best if agent_best is not None and baseline_best is not None else None),
        })

    return {"agent_vs_baselines": comparison_rows}


def main() -> None:
    args = parse_args()
    agent_dir = Path(args.agent_dir)
    baseline_dir = Path(args.baseline_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    agent_runs = collect_agent_runs(agent_dir)
    baseline_runs, baseline_algorithms = collect_baseline_runs(baseline_dir)

    agent_summary = summarize_agent_runs(agent_runs)
    baseline_summary = summarize_baseline_algorithms(baseline_algorithms)
    comparison = compare_agent_vs_baselines(agent_runs, baseline_algorithms)

    write_json(out_dir / f"agent_runs_{args.tag}.json", agent_runs)
    write_json(out_dir / f"agent_summary_{args.tag}.json", agent_summary)
    write_json(out_dir / f"baseline_runs_{args.tag}.json", baseline_runs)
    write_json(out_dir / f"baseline_algorithms_{args.tag}.json", baseline_algorithms)
    write_json(out_dir / f"baseline_summary_{args.tag}.json", baseline_summary)
    write_json(out_dir / f"comparison_{args.tag}.json", comparison)

    write_csv(out_dir / f"agent_runs_{args.tag}.csv", agent_runs)
    write_csv(out_dir / f"baseline_runs_{args.tag}.csv", baseline_runs)
    write_csv(out_dir / f"baseline_algorithms_{args.tag}.csv", baseline_algorithms)
    write_csv(out_dir / f"comparison_{args.tag}.csv", comparison["agent_vs_baselines"])

    print(f"Saved aggregate outputs to: {out_dir}")
    print(f"Agent runs found: {len(agent_runs)}")
    print(f"Baseline run groups found: {len(baseline_runs)}")
    print(f"Baseline algorithm entries found: {len(baseline_algorithms)}")

    holdout_summary = agent_summary["final_holdout_mean_score_summary"]
    print(
        "Agent final holdout mean summary: "
        f"count={holdout_summary['count']} "
        f"mean={holdout_summary['mean']} "
        f"median={holdout_summary['median']} "
        f"std={holdout_summary['std']}"
    )

    for algorithm, stats in baseline_summary["algorithms"].items():
        mean_stats = stats["mean_score_summary"]
        print(
            f"{algorithm}: "
            f"n_runs={stats['n_runs']} "
            f"mean_score_mean={mean_stats['mean']} "
            f"mean_score_median={mean_stats['median']}"
        )


if __name__ == "__main__":
    main()
