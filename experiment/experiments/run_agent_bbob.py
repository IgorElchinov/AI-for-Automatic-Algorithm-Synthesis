from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import time
import traceback

from agent.agent import Agent
from agent.models import OllamaClient
from agent.types import BAD_SCORE
from tasks.coco_bbob import CocoBbobAdapter, DEFAULT_BBOB_PROBLEM_TEXT


@dataclass
class RunConfig:
    run_name: str
    iterations: int
    max_consecutive_failures: int
    sleep_on_failure_sec: float
    pause_after_max_failures_sec: float
    model: str
    timeout_per_test: int
    k: int
    retry_initial_generation: int
    retry_fix_generation: int
    retry_runtime_fix_generation: int
    smoke_test_count: int
    ancestor_pool_size: int
    budget_multiplier: int
    temperature: float
    top_p: float
    num_ctx: int
    num_predict: int
    think: bool
    raw: bool
    verbose: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checkpointable BBOB experiment with Agent.")
    parser.add_argument("--run-name", type=str, default=None, help="Run name. Reuse to resume.")
    parser.add_argument("--base-dir", type=str, default="results/agent_bbob")
    parser.add_argument("--iterations", type=int, default=10, help="Evolution iterations after initialization.")
    parser.add_argument("--max-consecutive-failures", type=int, default=20)
    parser.add_argument("--sleep-on-failure-sec", type=float, default=5.0)
    parser.add_argument("--pause-after-max-failures-sec", type=float, default=300.0)
    parser.add_argument("--model", type=str, default="qwen2.5-coder:14b")
    parser.add_argument("--timeout-per-test", type=int, default=120)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--retry-initial-generation", type=int, default=5)
    parser.add_argument("--retry-fix-generation", type=int, default=3)
    parser.add_argument("--retry-runtime-fix-generation", type=int, default=2)
    parser.add_argument("--smoke-test-count", type=int, default=2)
    parser.add_argument("--ancestor-pool-size", type=int, default=5)
    parser.add_argument("--budget-multiplier", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=768)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def timestamp_run_name() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def list_solution_files(solutions_dir: Path) -> list[Path]:
    return sorted(solutions_dir.glob("solution_*.py"))


def build_run_config(args: argparse.Namespace, run_name: str) -> RunConfig:
    return RunConfig(
        run_name=run_name,
        iterations=args.iterations,
        max_consecutive_failures=args.max_consecutive_failures,
        sleep_on_failure_sec=args.sleep_on_failure_sec,
        pause_after_max_failures_sec=args.pause_after_max_failures_sec,
        model=args.model,
        timeout_per_test=args.timeout_per_test,
        k=args.k,
        retry_initial_generation=args.retry_initial_generation,
        retry_fix_generation=args.retry_fix_generation,
        retry_runtime_fix_generation=args.retry_runtime_fix_generation,
        smoke_test_count=args.smoke_test_count,
        ancestor_pool_size=args.ancestor_pool_size,
        budget_multiplier=args.budget_multiplier,
        temperature=args.temperature,
        top_p=args.top_p,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        think=args.think,
        raw=args.raw,
        verbose=args.verbose,
    )


def make_adapter(config: RunConfig) -> CocoBbobAdapter:
    return CocoBbobAdapter(problem_text=DEFAULT_BBOB_PROBLEM_TEXT, budget_multiplier=config.budget_multiplier)


def make_client(config: RunConfig) -> OllamaClient:
    return OllamaClient(
        model=config.model,
        think=config.think,
        raw=config.raw,
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
        temperature=config.temperature,
        top_p=config.top_p,
    )


def build_agent(config: RunConfig, adapter: CocoBbobAdapter, run_dir: Path, solutions_dir: Path) -> Agent:
    Agent.SOLUTIONS_PATH = solutions_dir
    client = make_client(config)
    agent = Agent(
        problem=adapter,
        model_client=client,
        k=config.k,
        timeout_per_test=config.timeout_per_test,
        retry_initial_generation=config.retry_initial_generation,
        retry_fix_generation=config.retry_fix_generation,
        retry_runtime_fix_generation=config.retry_runtime_fix_generation,
        debug_dir=run_dir / "debug",
        verbose=config.verbose,
        smoke_test_count=config.smoke_test_count,
        ancestor_pool_size=config.ancestor_pool_size,
        initial_solutions=list_solution_files(solutions_dir),
    )
    client.logger = agent._log
    client.debug_writer = agent._write_debug_text
    return agent


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_outer_error(outer_error_log: Path, tb: str) -> None:
    with outer_error_log.open("a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"ts={time.time()}\n")
        f.write(tb)
        f.write("\n")


def normalized_score(total_score: float | None, n_tests: int) -> float | None:
    if total_score is None or total_score == BAD_SCORE or n_tests <= 0:
        return None
    return total_score / n_tests


def main() -> None:
    args = parse_args()
    run_name = args.run_name or timestamp_run_name()
    config = build_run_config(args, run_name)

    base_dir = Path(args.base_dir)
    run_dir = base_dir / run_name
    solutions_dir = run_dir / "solutions"
    debug_dir = run_dir / "debug"
    state_path = run_dir / "state.json"
    events_path = run_dir / "events.jsonl"
    final_result_path = run_dir / "final_result.json"
    config_path = run_dir / "config.json"
    outer_error_log = debug_dir / "outer_errors.log"
    best_solution_copy = run_dir / "best_solution.py"

    run_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_json(config_path, asdict(config))
    write_text(run_dir / "problem_text.txt", DEFAULT_BBOB_PROBLEM_TEXT)

    state = load_json(
        state_path,
        {
            "run_name": run_name,
            "status": "running",
            "initialized": False,
            "completed_iterations": 0,
            "consecutive_failures": 0,
            "total_failures": 0,
            "pause_cycles": 0,
            "best_solution_path": None,
            "best_score_total_inner_loop": None,
            "best_score_mean_inner_loop": None,
            "last_error": None,
            "final_holdout_total_score": None,
            "final_holdout_mean_score": None,
            "final_holdout_issue": None,
            "last_update_ts": time.time(),
        },
    )
    state["status"] = "running"
    atomic_write_json(state_path, state)

    append_jsonl(events_path, {
        "ts": time.time(),
        "event": "resume_or_start",
        "run_name": run_name,
        "initialized": state["initialized"],
        "completed_iterations": state["completed_iterations"],
        "existing_solution_files": len(list_solution_files(solutions_dir)),
    })

    random.seed(42)
    adapter = make_adapter(config)
    n_inner_tests = len(adapter.build_full_tests())

    while True:
        completed_iterations = int(state["completed_iterations"])
        initialized = bool(state["initialized"])
        consecutive_failures = int(state["consecutive_failures"])

        if initialized and completed_iterations >= config.iterations:
            break

        if consecutive_failures >= config.max_consecutive_failures:
            state["status"] = "paused_retrying"
            state["pause_cycles"] = int(state.get("pause_cycles", 0)) + 1
            state["last_update_ts"] = time.time()
            atomic_write_json(state_path, state)
            append_jsonl(events_path, {
                "ts": time.time(),
                "event": "pause_after_max_consecutive_failures",
                "consecutive_failures": consecutive_failures,
                "pause_cycles": state["pause_cycles"],
                "sleep_sec": config.pause_after_max_failures_sec,
            })
            print(
                f"[warning] Reached max_consecutive_failures={config.max_consecutive_failures}. "
                f"Pausing for {config.pause_after_max_failures_sec:.1f}s and then retrying.",
                file=sys.stderr,
            )
            time.sleep(config.pause_after_max_failures_sec)
            state["consecutive_failures"] = 0
            state["status"] = "running"
            state["last_update_ts"] = time.time()
            atomic_write_json(state_path, state)
            continue

        try:
            agent = build_agent(config, adapter, run_dir, solutions_dir)

            if not initialized:
                agent.run(0)
                state["initialized"] = True
                state["consecutive_failures"] = 0
                state["last_error"] = None
                append_jsonl(events_path, {
                    "ts": time.time(),
                    "event": "initialized",
                    "best_solution_path": str(agent.best_solution.path),
                    "best_score_total_inner_loop": agent.best_solution.score,
                    "best_score_mean_inner_loop": normalized_score(agent.best_solution.score, n_inner_tests),
                })
            else:
                agent.run(1)
                state["completed_iterations"] = completed_iterations + 1
                state["consecutive_failures"] = 0
                state["last_error"] = None
                append_jsonl(events_path, {
                    "ts": time.time(),
                    "event": "iteration_complete",
                    "completed_iterations": state["completed_iterations"],
                    "best_solution_path": str(agent.best_solution.path),
                    "best_score_total_inner_loop": agent.best_solution.score,
                    "best_score_mean_inner_loop": normalized_score(agent.best_solution.score, n_inner_tests),
                })

            best = agent.best_solution
            state["best_solution_path"] = str(best.path)
            state["best_score_total_inner_loop"] = best.score
            state["best_score_mean_inner_loop"] = normalized_score(best.score, n_inner_tests)
            state["last_update_ts"] = time.time()
            atomic_write_json(state_path, state)

        except Exception as exc:
            tb = traceback.format_exc()
            state["consecutive_failures"] = consecutive_failures + 1
            state["total_failures"] = int(state["total_failures"]) + 1
            state["last_error"] = str(exc)
            state["last_update_ts"] = time.time()
            atomic_write_json(state_path, state)

            append_outer_error(outer_error_log, tb)

            append_jsonl(events_path, {
                "ts": time.time(),
                "event": "agent_exception",
                "completed_iterations": state["completed_iterations"],
                "consecutive_failures": state["consecutive_failures"],
                "error": str(exc),
            })

            print(
                f"[warning] Agent failed but run is checkpointed. "
                f"consecutive_failures={state['consecutive_failures']} error={exc}",
                file=sys.stderr,
            )
            time.sleep(config.sleep_on_failure_sec)
            continue

    try:
        agent = build_agent(config, adapter, run_dir, solutions_dir)
        best = agent.best_solution
        best_solution_copy.write_text(best.text(), encoding="utf-8")

        final_tests = adapter.build_final_tests()
        n_final_tests = len(final_tests)
        final_score_total, final_issue = agent.test_solution(best, tests=final_tests)
        final_score_mean = normalized_score(final_score_total, n_final_tests)

        final_result = {
            "run_name": run_name,
            "status": "completed",
            "completed_iterations": state["completed_iterations"],
            "best_solution_path": str(best.path),
            "best_score_total_inner_loop": best.score,
            "best_score_mean_inner_loop": normalized_score(best.score, n_inner_tests),
            "n_inner_loop_tests": n_inner_tests,
            "best_solution_test_time_sec": best.test_time_sec,
            "final_holdout_total_score": final_score_total,
            "final_holdout_mean_score": final_score_mean,
            "n_final_holdout_tests": n_final_tests,
            "final_holdout_issue": final_issue,
            "copied_best_solution_path": str(best_solution_copy),
            "solution_files_count": len(list_solution_files(solutions_dir)),
            "config": asdict(config),
        }
        atomic_write_json(final_result_path, final_result)

        state["status"] = "completed"
        state["best_solution_path"] = str(best.path)
        state["best_score_total_inner_loop"] = best.score
        state["best_score_mean_inner_loop"] = normalized_score(best.score, n_inner_tests)
        state["final_holdout_total_score"] = final_score_total
        state["final_holdout_mean_score"] = final_score_mean
        state["final_holdout_issue"] = final_issue
        state["last_update_ts"] = time.time()
        atomic_write_json(state_path, state)

        append_jsonl(events_path, {
            "ts": time.time(),
            "event": "completed",
            "best_solution_path": str(best.path),
            "best_score_total_inner_loop": best.score,
            "best_score_mean_inner_loop": normalized_score(best.score, n_inner_tests),
            "final_holdout_total_score": final_score_total,
            "final_holdout_mean_score": final_score_mean,
        })

        print(
            f"Run completed.\n"
            f"Run dir: {run_dir}\n"
            f"Best solution: {best.path}\n"
            f"Inner-loop total score: {best.score}\n"
            f"Inner-loop mean score: {normalized_score(best.score, n_inner_tests)}\n"
            f"Final holdout total score: {final_score_total}\n"
            f"Final holdout mean score: {final_score_mean}\n"
            f"Final issue: {final_issue}"
        )

    except Exception as exc:
        tb = traceback.format_exc()
        append_outer_error(outer_error_log, tb)

        state["status"] = "completed_inner_loop_final_eval_failed"
        state["last_error"] = str(exc)
        state["last_update_ts"] = time.time()
        atomic_write_json(state_path, state)

        append_jsonl(events_path, {
            "ts": time.time(),
            "event": "final_holdout_exception",
            "error": str(exc),
        })

        print(
            "[warning] Inner-loop run is complete, but final holdout evaluation failed. "
            f"Run dir: {run_dir} error={exc}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
