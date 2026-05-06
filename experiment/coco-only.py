from dataclasses import dataclass
import math
import random

from agent.types import TestCase
from agent.models import OllamaClient
from agent.agent import Agent

import cocoex

@dataclass
class CocoTestsuitAdapter:
    pass

problem_text = """
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
        problem=CocoTestsuitAdapter(problem_text=problem_text),
        model_client=client,
    )
    best_solution = agent.run(10)
    print(f'Best solution: {best_solution.path} with score {best_solution.score} and time {best_solution.test_time_sec}')

if __name__ == '__main__':
    main()