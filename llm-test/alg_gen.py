from pathlib import Path
import subprocess
from ollama import chat
import random
import typing as tp

class Solution:
    def __init__(self, path: Path, description: str | None = None) -> None:
        self.path = path
        self.score: float | None = None
        self.description: str | None = description
    
    def run(self, input: Path) -> str:
        print('Running tests...')
        res = subprocess.run(
                [self.path.absolute(), '<', input.absolute()],
                timeout=60,
                check=True,
            )
        return str(res.stdout)

    def text(self) -> str:
        with open(self.path, 'r') as f:
            return '\n'.join(f.readlines())

    @staticmethod
    def fromtext(text: str, path: Path, description: str | None = None):
        with open(path, '+w') as f:
            f.writelines(text)
        return Solution(path, description)

class Test:
    def __init__(self, text: str, *args) -> None:
        self.text = text
        self.args = args

class Task:
    def __init__(self, text: str, tests: list[Test], objective: tp.Callable) -> None:
        self.text = text
        self.tests = tests
        self.objective = objective
    
    def run(self, solution: Solution) -> float:
        score = 0.0
        for test in self.tests:
            inp = Path('input.txt')
            with open(inp, '+w') as f:
                f.writelines(test.text)
            x = solution.run(inp).strip().split()
            score += self.objective(x, *test.args)
        return score
        

class Agent:
    SOLUTIONS_PATH = Path('solutions/')
    COMBINE_PROMPT = 'combine these solutions:\n'
    GEN_PROMPT = 'generate a solution to solve the problem.\n'
    MUTATE_PROMPT = 'You are given a problem and a solution. Mutate a solution to increase its score'

    def __init__(
            self,
            task: Task,
            initial_solutions: list[Path] | list[Solution],
            model: str,
            k: int = 3
            ) -> None:
        self.task = task
        self.model = model
        self.k = k
        self.solutions = [Solution(self.SOLUTIONS_PATH/solution) if isinstance(solution, Path) else solution for solution in initial_solutions]
        self._best_solution: Solution | None = None
        for solution in self.solutions:
            if solution.score is None:
                solution.score = self.test_solution(solution)
            if self._best_solution is None or self._best_solution.score < solution.score:
                self._best_solution = solution

    @property
    def best_solution(self) -> Solution:
        if self.best_solution is None:
            raise
        return self.best_solution

    def ask_model(self, prompt: str) -> str:
        response = chat(
        model= self.model,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
        )
        return response.message.content

    def choose_ancestors(self, num: int) -> list[Solution]:
        return random.choices(self.solutions, k=num)

    def combine_solutions(self, ancestors: list[Solution], path: Path) -> Solution:
        prompt = '### Instruction:\n' + self.MUTATE_PROMPT + '### Task:\n' + self.task.text
        for i, solution in enumerate(ancestors):
            prompt +=  f'### Solution {i + 1}:' + solution.text() + '\n'
        return Solution.fromtext(self.ask_model(prompt), path)

    def mutate_solution(self, solution: Solution, path: Path) -> Solution:
        prompt = '### Instruction:\n' + self.MUTATE_PROMPT + '### Task:\n' + self.task.text + '### Solution:' + solution.text()
        return Solution.fromtext(self.ask_model(prompt), path)

    def gen_solution(self, prompt: str, path: Path) -> Solution:
        prompt = '### Instruction:\n' + self.GEN_PROMPT + '### Task:\n' + self.task.text
        return Solution.fromtext(self.ask_model(prompt), path)

    def test_solution(self, solution: Solution) -> float:
        return self.task.run(solution)

    def run(self, iterations: int) -> Solution:
        if not self.solutions:
            path = self.SOLUTIONS_PATH/Path(f'solution{len(self.solutions)}.txt')
            solution = self.gen_solution(self.task.text, path)
            self.solutions.append(solution)
            solution.score = self.test_solution(solution)
        for i in range(iterations):
            print(f'Iteration {i + 1}...')
            path = self.SOLUTIONS_PATH/Path(f'solution{len(self.solutions)}.txt')
            ancestors = self.choose_ancestors(self.k)
            solution = self.combine_solutions(ancestors, path)
            solution = self.mutate_solution(solution, path)
            solution.score = self.test_solution(solution)
            self.solutions.append(solution)
            if self._best_solution is None or self._best_solution.score < solution.score:
                self._best_solution = solution
            print('Done')

        return self.best_solution
    
if __name__ == '__main__':
    import math
    def objective(answer: str, a: list[int], b: list[float]):
        x: list[int] = list(map(int, answer.strip().split()))
        return sum(ai * (1 - math.exp(-bi * xi)) for ai, bi, xi in zip(a, b, x))
    def gen_test(n: int) -> Test:
        B = random.randrange(n, 10 * n)
        a = [random.randint(0, 100) for _ in range(n)]
        b = [round(random.random(), 7) for _ in range(n)]
        text = f'{n}\n' + ' '.join(map(str, a)) + '\n' + ' '.join(map(str, b)) + f'\n{B}\n'
        return Test(text, a, b)
    with open('task.txt', 'r') as f:
        statement = '\n'.join(f.readlines())
    task = Task(statement, [gen_test(i) for i in [2, 5, 10, 100, 1000, 10_000] * 20], objective)
    agent = Agent(task, [], 'qwen3:1.7b')
    result = agent.run(3)
    print(f'Result: {result.path} with score {result.score}')
