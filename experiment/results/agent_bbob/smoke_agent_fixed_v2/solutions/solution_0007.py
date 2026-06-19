import sys
import random
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.single_objective.swarm import PSO

suite_name = input().strip()
function_index = int(input().strip())
dimension = int(input().strip())
instance = int(input().strip())
evaluation_budget = int(input().strip())
random_seed = int(input().strip())

random.seed(random_seed)

class DummySuite:
    def __init__(self, name, instance, options):
        self.lower_bounds = [-5] * dimension
        self.upper_bounds = [5] * dimension

    def __iter__(self):
        return iter([self])

suite = DummySuite(suite_name, instance, None)
problem = next(iter(suite))

function = Function(problem)
search_space = SearchSpace(
    n_agents=20,
    n_variables=dimension,
    n_objectives=1,
    lower_bound=problem.lower_bounds,
    upper_bound=problem.upper_bounds,
)
optimizer = PSO()
opt = Opytimizer(search_space, optimizer, function)
opt.start(n_iterations=max(1, evaluation_budget // 20))
best_agent = search_space.best_agent
best_position = best_agent.position

print(" ".join(map(str, best_position)))
