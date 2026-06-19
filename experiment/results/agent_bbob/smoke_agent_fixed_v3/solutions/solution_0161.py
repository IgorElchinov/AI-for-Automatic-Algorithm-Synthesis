import cocoex
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.single_objective.swarm import PSO
import sys

suite_name = input().strip()
function_index = int(input().strip())
dimension = int(input().strip())
instance = int(input().strip())
evaluation_budget = int(input().strip())
random_seed = int(input().strip())

suite = cocoex.Suite(
    suite_name=suite_name,
    suite_instance="year: 2009",
    suite_options=(
        f"function_indices: {function_index} "
        f"instance_indices: {instance} "
        f"dimensions: {dimension}"
    ),
)
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
