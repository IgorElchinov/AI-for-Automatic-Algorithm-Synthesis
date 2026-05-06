from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import random
from typing import Iterable

import cocoex

from agent.types import TestCase
from agent.models import OllamaClient
from agent.agent import Agent


def make_bbob_problem(
    function_index: int,
    dimension: int,
    instance: int,
    suite_name: str = "bbob",
):
    """
    Reconstruct a single COCO BBOB problem.
    """
    suite = cocoex.Suite(
        suite_name,
        f"instances: {instance}",
        f"dimensions: {dimension} function_indices: {function_index}",
    )
    for problem in suite:
        return problem
    raise RuntimeError(
        f"Could not create problem for f={function_index}, d={dimension}, instance={instance}"
    )


@dataclass
class CocoTestsuiteAdapter:
    problem_text: str
    suite_name: str = "bbob"
    budget_multiplier: int = 20

    # smoke set: tiny and cheap
    smoke_function_indices: tuple[int, ...] = (1, 8, 15)
    smoke_dimensions: tuple[int, ...] = (2,)
    smoke_instances: tuple[int, ...] = (1,)

    # full set: still manageable for the agent inner loop
    full_function_indices: tuple[int, ...] = tuple(range(1, 25))
    full_dimensions: tuple[int, ...] = (2, 5)
    full_instances: tuple[int, ...] = (1,)

    random_seed: int = 42

    def __post_init__(self) -> None:
        random.seed(self.random_seed)

    def _make_test_case(
        self,
        function_index: int,
        dimension: int,
        instance: int,
    ) -> TestCase:
        budget = self.budget_multiplier * dimension

        problem = make_bbob_problem(
            function_index=function_index,
            dimension=dimension,
            instance=instance,
            suite_name=self.suite_name,
        )

        # COCO example evaluates x=0 for comparability; we use that as baseline.
        zero = [0.0] * dimension
        baseline_value = float(problem(zero))

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
                "function_index": function_index,
                "dimension": dimension,
                "instance": instance,
                "budget": budget,
                "seed": self.random_seed,
                "baseline_value": baseline_value,
            },
        )

    def build_smoke_tests(self) -> list[TestCase]:
        tests: list[TestCase] = []
        for f_idx in self.smoke_function_indices:
            for dim in self.smoke_dimensions:
                for inst in self.smoke_instances:
                    tests.append(self._make_test_case(f_idx, dim, inst))
        return tests

    def build_full_tests(self) -> list[TestCase]:
        tests: list[TestCase] = []
        for f_idx in self.full_function_indices:
            for dim in self.full_dimensions:
                for inst in self.full_instances:
                    tests.append(self._make_test_case(f_idx, dim, inst))
        return tests

    def evaluate_output(self, output: str, test: TestCase) -> float:
        dim = int(test.meta["dimension"])
        function_index = int(test.meta["function_index"])
        instance = int(test.meta["instance"])
        suite_name = str(test.meta["suite_name"])
        baseline_value = float(test.meta["baseline_value"])

        tokens = output.strip().split()
        if len(tokens) != dim:
            raise ValueError(f"Expected {dim} floats, got {len(tokens)}")

        try:
            x = [float(tok) for tok in tokens]
        except ValueError as exc:
            raise ValueError("Output contains non-float tokens") from exc

        if any(not math.isfinite(v) for v in x):
            raise ValueError("Output contains non-finite values")

        problem = make_bbob_problem(
            function_index=function_index,
            dimension=dim,
            instance=instance,
            suite_name=suite_name,
        )
        fx = float(problem(x))

        # Higher score is better.
        # We normalize improvement relative to the baseline x=0.
        denom = max(1.0, abs(baseline_value))
        score = (baseline_value - fx) / denom
        return score


problem_text = """
Write a complete Python 3 program.

The program reads from stdin exactly 6 lines:
1) suite name (always "bbob")
2) function index (integer)
3) dimension (integer)
4) instance (integer)
5) evaluation budget (integer)
6) random seed (integer)

The program must:
- import cocoex
- reconstruct exactly the requested COCO BBOB problem
- optimize the black-box function under the given evaluation budget
- print exactly `dimension` floating-point numbers separated by spaces:
  these numbers are the final candidate solution vector x

Important requirements:
- output only the numbers, nothing else
- do not print explanations
- do not print markdown
- the code must be executable Python 3
- you may use any Python approach and any installed libraries
- you are NOT required to use Opytimizer, but you MAY use it if you want
- you may also use numpy, scipy, random search, evolution strategies, hill climbing, etc.

Helpful hints:
- use:
    suite = cocoex.Suite(
        suite_name,
        f"instances: {instance}",
        f"dimensions: {dimension} function_indices: {function_index}",
      )
  and then take the first problem from the suite
- the problem object is callable: value = problem(x)
- use problem.lower_bounds and problem.upper_bounds
- respect the evaluation budget
- a simple but robust optimizer is better than a fancy broken one

Your answer must be only Python code.
""".strip()


def main() -> None:
    client = OllamaClient(
        model="qwen2.5-coder:14b",
        think=False,
        raw=False,
        num_ctx=8192,
        num_predict=768,
        temperature=0.15,
        top_p=0.9,
        keep_alive="30m",
    )

    adapter = CocoTestsuiteAdapter(
        problem_text=problem_text,
        suite_name="bbob",
        budget_multiplier=20,
        smoke_function_indices=(1, 8, 15),
        smoke_dimensions=(2,),
        smoke_instances=(1,),
        full_function_indices=tuple(range(1, 25)),
        full_dimensions=(2, 5),
        full_instances=(1,),
        random_seed=42,
    )

    agent = Agent(
        problem=adapter,
        model_client=client,
        k=3,
        timeout_per_test=120,
        retry_initial_generation=5,
        retry_fix_generation=5,
        retry_runtime_fix_generation=5,
        smoke_test_count=2,
        ancestor_pool_size=5,
        debug_dir="debug_bbob",
        verbose=False,
    )

    best_solution = agent.run(20)
    print(
        f"Best solution: {best_solution.path} "
        f"with score {best_solution.score} "
        f"and time {best_solution.test_time_sec}"
    )


if __name__ == "__main__":
    main()