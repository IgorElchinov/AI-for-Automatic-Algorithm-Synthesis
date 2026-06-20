from __future__ import annotations

from dataclasses import dataclass, field
import math

from agent.types import TestCase


LIBRARY_RESTRICTIONS = '''
Library restrictions:
- you may use Python standard library
- you may use cocoex to access the benchmark problem
- You can use arbitrary optimization libraries such as opytimizer, scipy, nevergrad, pymoo, etc.
'''

# LIBRARY_RESTRICTIONS = '''
# Library restrictions:
# - you may use Python standard library
# - you may use cocoex to access the benchmark problem
# - for the optimization algorithm itself, use Opytimizer
# - do NOT rely on arbitrary third-party optimization libraries such as scipy, nevergrad, pymoo, etc.
# - the point of the task is to build the optimizer using Opytimizer components

# Opytimizer API memo

# Use only the current Opytimizer API.

# Verified core imports:
# from opytimizer import Opytimizer
# from opytimizer.core import Function
# from opytimizer.spaces import SearchSpace

# Verified optimizer import example:
# from opytimizer.optimizers.single_objective.swarm import PSO

# Current package structure:
# - opytimizer.optimizers.single_objective.evolutionary
# - opytimizer.optimizers.single_objective.misc
# - opytimizer.optimizers.single_objective.population
# - opytimizer.optimizers.single_objective.science
# - opytimizer.optimizers.single_objective.social
# - opytimizer.optimizers.single_objective.swarm

# Do not use old import paths such as:
# - from opytimizer.optimizers.swarm import PSO

# Typical usage pattern:
# 1. Define objective function and wrap it with Function(...)
# 2. Build SearchSpace(...)
# 3. Instantiate optimizer, e.g. PSO()
# 4. Build Opytimizer(space, optimizer, function)
# 5. Run opt.start(n_iterations=...)

# Reference pattern:
# space = SearchSpace(
#     n_agents=n_agents,
#     n_variables=dimension,
#     n_objectives=1,
#     lower_bound=lower_bound,
#     upper_bound=upper_bound,
# )
# optimizer = PSO()
# function = Function(objective_fn)
# opt = Opytimizer(space, optimizer, function)
# opt.start(n_iterations=n_iterations)
# '''


DEFAULT_BBOB_PROBLEM_TEXT = """
Write Python 3 code that defines exactly one function:

def optimize(objective, lower_bounds, upper_bounds, dimension, budget, seed):
    ...

The function receives:
- objective: callable objective(x) -> float
- lower_bounds: sequence of floats of length dimension
- upper_bounds: sequence of floats of length dimension
- dimension: integer
- budget: maximum allowed number of objective evaluations
- seed: integer random seed

The function must:
- return a list or tuple of exactly `dimension` finite floats
- keep all coordinates inside [lower_bounds[i], upper_bounds[i]]
- not read from stdin
- not print anything
- not import or use cocoex
- not reconstruct the COCO problem
- not call objective more than `budget` times

Library restrictions:
- you may use Python standard library
- you may use numpy
- for optimization primitives, use only Opytimizer
- do NOT use scipy, nevergrad, pymoo, sklearn, torch, tensorflow

Current Opytimizer API:
- from opytimizer import Opytimizer
- from opytimizer.core import Function
- from opytimizer.spaces import SearchSpace
- from opytimizer.optimizers.single_objective.swarm import PSO

Do not use deprecated import paths such as:
- from opytimizer.optimizers.swarm import PSO
- from opytimizer.optimizers.population import GWO
- from opytimizer.optimizers.evolutionary import DE

Important:
- return only Python code
- do not include markdown
- do not include explanations
- a simple robust optimizer is better than a fancy broken one
""".strip()


def default_smoke_specs() -> tuple[tuple[int, int, int], ...]:
    return (
        (1, 2, 1),
        (8, 2, 1),
        (15, 2, 1),
    )


def default_full_specs() -> tuple[tuple[int, int, int], ...]:
    specs: list[tuple[int, int, int]] = []
    for function_index in range(1, 25):
        for dimension in (2, 5):
            for instance in (1,):
                specs.append((function_index, dimension, instance))
    return tuple(specs)


def default_final_specs() -> tuple[tuple[int, int, int], ...]:
    specs: list[tuple[int, int, int]] = []
    for function_index in range(1, 25):
        for dimension in (2, 3, 5, 10):
            for instance in range(1, 6):
                specs.append((function_index, dimension, instance))
    return tuple(specs)


class BudgetExceededError(RuntimeError):
    pass


class BudgetedObjective:
    def __init__(self, problem, budget: int):
        self.problem = problem
        self.budget = int(budget)
        self.calls = 0

    def __call__(self, x):
        if self.calls >= self.budget:
            raise BudgetExceededError(
                f"Evaluation budget exceeded: {self.calls + 1} > {self.budget}"
            )
        self.calls += 1
        return float(self.problem(x))


def make_bbob_problem(
    function_index: int,
    dimension: int,
    instance: int,
    suite_name: str = "bbob",
    suite_instance: str = "year: 2009",
):
    import cocoex

    suite = cocoex.Suite(
        suite_name=suite_name,
        suite_instance=suite_instance,
        suite_options=(
            f"function_indices: {function_index} "
            f"instance_indices: {instance} "
            f"dimensions: {dimension}"
        ),
    )
    for problem in suite:
        return problem
    raise RuntimeError(
        f"Could not create BBOB problem for "
        f"f={function_index}, d={dimension}, instance={instance}"
    )


@dataclass
class CocoBbobAdapter:
    problem_text: str = DEFAULT_BBOB_PROBLEM_TEXT
    suite_name: str = "bbob"
    suite_instance: str = "year: 2009"
    budget_multiplier: int = 20
    random_seed: int = 42

    smoke_specs: tuple[tuple[int, int, int], ...] = field(default_factory=default_smoke_specs)
    full_specs: tuple[tuple[int, int, int], ...] = field(default_factory=default_full_specs)
    final_specs: tuple[tuple[int, int, int], ...] = field(default_factory=default_final_specs)

    _problem_cache: dict[tuple[int, int, int], object] = field(init=False, default_factory=dict)
    _smoke_tests: list[TestCase] = field(init=False, default_factory=list)
    _full_tests: list[TestCase] = field(init=False, default_factory=list)
    _final_tests: list[TestCase] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._smoke_tests = [self._make_test_case(*spec) for spec in self.smoke_specs]
        self._full_tests = [self._make_test_case(*spec) for spec in self.full_specs]
        self._final_tests = [self._make_test_case(*spec) for spec in self.final_specs]

    def _problem_key(self, function_index: int, dimension: int, instance: int) -> tuple[int, int, int]:
        return (function_index, dimension, instance)

    def _get_problem(self, function_index: int, dimension: int, instance: int):
        key = self._problem_key(function_index, dimension, instance)
        if key not in self._problem_cache:
            self._problem_cache[key] = make_bbob_problem(
                function_index=function_index,
                dimension=dimension,
                instance=instance,
                suite_name=self.suite_name,
                suite_instance=self.suite_instance,
            )
        return self._problem_cache[key]

    def _baseline_point(self, dimension: int) -> list[float]:
        return [0.0] * dimension

    def _make_test_case(self, function_index: int, dimension: int, instance: int) -> TestCase:
        budget = self.budget_multiplier * dimension
        problem = self._get_problem(function_index, dimension, instance)

        baseline_x = self._baseline_point(dimension)
        baseline_value = float(problem(baseline_x))

        input_text = "\n".join(
            [
                self.suite_name,
                str(function_index),
                str(dimension),
                str(instance),
                str(budget),
                str(self.random_seed),
            ]
        ) + "\n"

        return TestCase(
            input_text=input_text,
            meta={
                "suite_name": self.suite_name,
                "suite_instance": self.suite_instance,
                "function_index": function_index,
                "dimension": dimension,
                "instance": instance,
                "budget": budget,
                "seed": self.random_seed,
                "baseline_value": baseline_value,
            },
        )

    def build_smoke_tests(self) -> list[TestCase]:
        return list(self._smoke_tests)

    def build_full_tests(self) -> list[TestCase]:
        return list(self._full_tests)

    def build_final_tests(self) -> list[TestCase]:
        return list(self._final_tests)

    def evaluate_output(self, output: str, test: TestCase) -> float:
        meta = test.meta
        dimension = int(meta["dimension"])
        function_index = int(meta["function_index"])
        instance = int(meta["instance"])
        baseline_value = float(meta["baseline_value"])

        tokens = output.strip().split()
        if len(tokens) != dimension:
            raise ValueError(f"Expected {dimension} floats, got {len(tokens)}")

        try:
            x = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError("Output contains non-float tokens") from exc

        if any(not math.isfinite(v) for v in x):
            raise ValueError("Output contains non-finite values")

        problem = self._get_problem(function_index, dimension, instance)
        lower_bounds = problem.lower_bounds
        upper_bounds = problem.upper_bounds
        for i, value in enumerate(x):
            if value < lower_bounds[i] or value > upper_bounds[i]:
                raise ValueError(
                    f"Coordinate {i}={value} is outside bounds "
                    f"[{lower_bounds[i]}, {upper_bounds[i]}]"
                )

        fx = float(problem(x))

        denom = max(1.0, abs(baseline_value))
        score = (baseline_value - fx) / denom
        return score


__all__ = [
    "DEFAULT_BBOB_PROBLEM_TEXT",
    "CocoBbobAdapter",
    "default_smoke_specs",
    "default_full_specs",
    "default_final_specs",
    "make_bbob_problem",
]
