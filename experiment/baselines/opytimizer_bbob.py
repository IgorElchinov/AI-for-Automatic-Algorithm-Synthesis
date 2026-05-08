from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import json
import math
import random
from typing import Any, Callable
from pathlib import Path

import numpy as np

from tasks.coco_bbob import CocoBbobAdapter, make_bbob_problem
from agent.types import TestCase


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    module_path: str
    class_name: str
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    n_agents: int = 20


# PSO path is verified in Opytimizer README.
# The others are best-effort defaults and are filtered at runtime if import fails.
DEFAULT_BASELINE_SPECS: tuple[OptimizerSpec, ...] = (
    OptimizerSpec(
        name="pso",
        module_path="opytimizer.optimizers.single_objective.swarm",
        class_name="PSO",
        optimizer_kwargs={},
        n_agents=20,
    ),
    OptimizerSpec(
        name="de",
        module_path="opytimizer.optimizers.single_objective.evolutionary",
        class_name="DE",
        optimizer_kwargs={},
        n_agents=20,
    ),
    OptimizerSpec(
        name="gwo",
        module_path="opytimizer.optimizers.single_objective.population",
        class_name="GWO",
        optimizer_kwargs={},
        n_agents=20,
    ),
    OptimizerSpec(
        name="woa",
        module_path="opytimizer.optimizers.single_objective.population",
        class_name="WOA",
        optimizer_kwargs={},
        n_agents=20,
    ),
    OptimizerSpec(
        name="sa",
        module_path="opytimizer.optimizers.single_objective.science",
        class_name="SA",
        optimizer_kwargs={},
        n_agents=1,
    ),
)


@dataclass
class SingleRunResult:
    algorithm: str
    function_index: int
    dimension: int
    instance: int
    budget: int
    seed: int
    used_evaluations: int
    best_fx: float
    baseline_fx: float
    score: float
    x: list[float]
    ok: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BudgetExhausted(RuntimeError):
    """Raised internally when the BBOB evaluation budget is exhausted."""


class EvaluationCounter:
    def __init__(self, budget: int) -> None:
        self.budget = int(budget)
        self.count = 0
        self.best_x: np.ndarray | None = None
        self.best_fx = float("inf")

    def evaluate(self, x: np.ndarray, fn: Callable[[np.ndarray], float]) -> float:
        if self.count >= self.budget:
            raise BudgetExhausted(f"Evaluation budget exhausted: {self.count}/{self.budget}")

        fx = float(fn(x))
        self.count += 1

        if fx < self.best_fx:
            self.best_fx = fx
            self.best_x = np.asarray(x, dtype=float).copy()

        return fx


def _try_import(spec: OptimizerSpec):
    module = importlib.import_module(spec.module_path)
    return getattr(module, spec.class_name)


def available_default_specs() -> list[OptimizerSpec]:
    available: list[OptimizerSpec] = []
    for spec in DEFAULT_BASELINE_SPECS:
        try:
            _try_import(spec)
            available.append(spec)
        except Exception:
            continue
    return available


def unavailable_default_specs() -> list[tuple[OptimizerSpec, str]]:
    missing: list[tuple[OptimizerSpec, str]] = []
    for spec in DEFAULT_BASELINE_SPECS:
        try:
            _try_import(spec)
        except Exception as exc:
            missing.append((spec, str(exc)))
    return missing


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _extract_best_from_opytimizer(opt) -> tuple[np.ndarray | None, float | None]:
    """
    Tries several plausible places where Opytimizer may store the best solution.
    This is intentionally defensive to survive API drift across versions.
    """
    candidates: list[tuple[Any, str, str]] = [
        (getattr(opt, "space", None), "best_agent", "fit"),
        (getattr(opt, "space", None), "best_agent", "position"),
        (getattr(opt, "optimizer", None), "best_agent", "fit"),
        (getattr(opt, "optimizer", None), "best_agent", "position"),
        (getattr(opt, "history", None), "best_agent", "fit"),
        (getattr(opt, "history", None), "best_agent", "position"),
    ]

    # First try the canonical "space.best_agent".
    space = getattr(opt, "space", None)
    best_agent = getattr(space, "best_agent", None) if space is not None else None
    if best_agent is not None:
        position = getattr(best_agent, "position", None)
        fitness = getattr(best_agent, "fit", None)
        if position is not None:
            return np.asarray(position, dtype=float).reshape(-1), float(fitness) if fitness is not None else None

    # Try history.best_agent, if present.
    history = getattr(opt, "history", None)
    best_agent = getattr(history, "best_agent", None) if history is not None else None
    if best_agent is not None:
        position = getattr(best_agent, "position", None)
        fitness = getattr(best_agent, "fit", None)
        if position is not None:
            return np.asarray(position, dtype=float).reshape(-1), float(fitness) if fitness is not None else None

    return None, None


def _score_from_values(baseline_fx: float, best_fx: float) -> float:
    denom = max(1.0, abs(baseline_fx))
    return (baseline_fx - best_fx) / denom


def _build_opytimizer_objects(
    spec: OptimizerSpec,
    lower_bound: list[float],
    upper_bound: list[float],
    dimension: int,
    objective_fn: Callable[[np.ndarray], float],
):
    from opytimizer import Opytimizer
    from opytimizer.core import Function
    from opytimizer.spaces import SearchSpace

    optimizer_cls = _try_import(spec)

    space = SearchSpace(
        n_agents=spec.n_agents,
        n_variables=dimension,
        n_objectives=1,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    optimizer = optimizer_cls(**spec.optimizer_kwargs)
    function = Function(objective_fn)
    opt = Opytimizer(space, optimizer, function)
    return opt


def run_opytimizer_on_testcase(
    adapter: CocoBbobAdapter,
    test: TestCase,
    spec: OptimizerSpec,
    seed: int | None = None,
) -> SingleRunResult:
    meta = test.meta
    function_index = int(meta["function_index"])
    dimension = int(meta["dimension"])
    instance = int(meta["instance"])
    budget = int(meta["budget"])
    baseline_fx = float(meta["baseline_value"])
    seed = int(meta.get("seed", 0) if seed is None else seed)

    _set_seed(seed)

    problem = make_bbob_problem(
        function_index=function_index,
        dimension=dimension,
        instance=instance,
        suite_name=str(meta.get("suite_name", adapter.suite_name)),
        suite_instance=str(meta.get("suite_instance", adapter.suite_instance)),
    )

    lower_bound = np.asarray(problem.lower_bounds, dtype=float).reshape(-1).tolist()
    upper_bound = np.asarray(problem.upper_bounds, dtype=float).reshape(-1).tolist()

    counter = EvaluationCounter(budget=budget)

    def counted_objective(x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=float).reshape(-1)
        return counter.evaluate(arr, problem)

    ok = True
    error: str | None = None

    try:
        opt = _build_opytimizer_objects(
            spec=spec,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dimension=dimension,
            objective_fn=counted_objective,
        )

        # Approximate iteration budget. We also hard-stop exactly inside counted_objective.
        # The +1 gives the optimizer at least one update round after initialization.
        n_iterations = max(1, math.ceil(budget / max(1, spec.n_agents)))
        try:
            opt.start(n_iterations=n_iterations)
        except BudgetExhausted:
            pass

        best_x, extracted_fx = _extract_best_from_opytimizer(opt)
        if counter.best_x is not None:
            best_x = counter.best_x
        if counter.best_fx < float("inf"):
            best_fx = counter.best_fx
        elif extracted_fx is not None:
            best_fx = float(extracted_fx)
        else:
            raise RuntimeError("Could not extract a best solution from Opytimizer run")

    except Exception as exc:
        ok = False
        error = str(exc)
        if counter.best_x is not None and counter.best_fx < float("inf"):
            best_x = counter.best_x
            best_fx = counter.best_fx
        else:
            best_x = np.zeros(dimension, dtype=float)
            best_fx = float(problem(best_x))

    x_list = np.asarray(best_x, dtype=float).reshape(-1).tolist()
    score = _score_from_values(baseline_fx=baseline_fx, best_fx=float(best_fx))

    return SingleRunResult(
        algorithm=spec.name,
        function_index=function_index,
        dimension=dimension,
        instance=instance,
        budget=budget,
        seed=seed,
        used_evaluations=counter.count,
        best_fx=float(best_fx),
        baseline_fx=baseline_fx,
        score=score,
        x=x_list,
        ok=ok,
        error=error,
    )


def run_opytimizer_suite(
    adapter: CocoBbobAdapter,
    specs: list[OptimizerSpec] | None = None,
    tests: list[TestCase] | None = None,
    seed: int = 42,
) -> list[SingleRunResult]:
    specs = available_default_specs() if specs is None else specs
    tests = adapter.build_full_tests() if tests is None else tests

    results: list[SingleRunResult] = []
    for spec in specs:
        for test in tests:
            result = run_opytimizer_on_testcase(
                adapter=adapter,
                test=test,
                spec=spec,
                seed=seed,
            )
            results.append(result)
    return results


def aggregate_results(results: list[SingleRunResult]) -> dict[str, Any]:
    by_algorithm: dict[str, list[SingleRunResult]] = {}
    for result in results:
        by_algorithm.setdefault(result.algorithm, []).append(result)

    summary: dict[str, Any] = {
        "algorithms": {},
        "n_results": len(results),
    }

    for algorithm, items in by_algorithm.items():
        scores = [item.score for item in items]
        oks = [1.0 if item.ok else 0.0 for item in items]
        evals = [item.used_evaluations for item in items]

        summary["algorithms"][algorithm] = {
            "n_runs": len(items),
            "mean_score": float(np.mean(scores)) if scores else None,
            "median_score": float(np.median(scores)) if scores else None,
            "std_score": float(np.std(scores)) if scores else None,
            "success_rate": float(np.mean(oks)) if oks else None,
            "mean_used_evaluations": float(np.mean(evals)) if evals else None,
        }

    return summary


def save_results_json(path: str | Path, results: list[SingleRunResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.to_dict() for result in results]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_summary_json(path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "OptimizerSpec",
    "DEFAULT_BASELINE_SPECS",
    "SingleRunResult",
    "available_default_specs",
    "unavailable_default_specs",
    "run_opytimizer_on_testcase",
    "run_opytimizer_suite",
    "aggregate_results",
    "save_results_json",
    "save_summary_json",
]
