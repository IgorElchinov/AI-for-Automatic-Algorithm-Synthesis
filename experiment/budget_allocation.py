from dataclasses import dataclass
import math
import random

from agent.types import TestCase
from agent.models import OllamaClient
from agent.agent import Agent


@dataclass
class BudgetAllocationAdapter:
    problem_text: str
    smoke_sizes: tuple[int, ...] = (2, 5)
    full_sizes: tuple[int, ...] = (2, 5, 10, 50, 100)
    repeats: int = 3

    def _gen_test(self, n: int) -> TestCase:
        budget = random.randrange(max(1, n), 10 * max(1, n))
        a = [random.randint(1, 100) for _ in range(n)]
        b = [max(1e-4, round(random.random(), 7)) for _ in range(n)]
        input_text = (
            f"{n}\n"
            + " ".join(map(str, a)) + "\n"
            + " ".join(map(str, b)) + "\n"
            + f"{budget}\n"
        )
        return TestCase(
            input_text=input_text,
            meta={"a": a, "b": b, "budget": budget},
        )

    def build_smoke_tests(self) -> list[TestCase]:
        return [self._gen_test(n) for n in self.smoke_sizes]

    def build_full_tests(self) -> list[TestCase]:
        tests = []
        for _ in range(self.repeats):
            for n in self.full_sizes:
                tests.append(self._gen_test(n))
        return tests

    def evaluate_output(self, output: str, test: TestCase) -> float:
        tokens = output.strip().split()
        a = test.meta["a"]
        b = test.meta["b"]
        budget = test.meta["budget"]

        if len(tokens) != len(a):
            raise ValueError(f"Expected {len(a)} integers, got {len(tokens)}")

        x = [int(t) for t in tokens]
        if any(v < 0 for v in x):
            raise ValueError("Budget allocation must be non-negative")
        if sum(x) > budget:
            raise ValueError(f"Budget exceeded: sum(x)={sum(x)} > B={budget}")

        return sum(ai * (1 - math.exp(-bi * xi)) for ai, bi, xi in zip(a, b, x))

problem_text = """
/no_think
Return only executable Python 3 code.
No markdown. No explanations.

Write a complete Python 3 program.

Input format:
- line 1: integer n
- line 2: n integers a[0], ..., a[n-1]
- line 3: n real numbers b[0], ..., b[n-1]
- line 4: integer B

Output format:
- print exactly n non-negative integers x[0], ..., x[n-1]
- integers separated by spaces
- must satisfy sum(x) <= B

Goal:
maximize sum(a[i] * (1 - exp(-b[i] * x[i])))

Constraints:
- 1 <= n <= 10
- 1 <= B <= 10000
- a[i] > 0
- b[i] > 0

Use this baseline algorithm:
Start with x = [0] * n.
Repeat B times:
choose i with maximum marginal gain
a[i] * (exp(-b[i] * x[i]) - exp(-b[i] * (x[i] + 1)))
then increment x[i].

Requirements:
- read from stdin
- write to stdout
- use import math
- no comments
- no helper text
"""


def main():
    client = OllamaClient(
        model = "qwen2.5-coder:14b",
        think=False,
        num_ctx = 8192,
        num_predict = 768,
        temperature = 0.15,
    )
    agent = Agent(
        problem=BudgetAllocationAdapter(problem_text=problem_text),
        model_client=client,
    )
    best_solution = agent.run(10)
    print(f'Best solution: {best_solution.path} with score {best_solution.score} and time {best_solution.test_time_sec}')

if __name__ == '__main__':
    main()
