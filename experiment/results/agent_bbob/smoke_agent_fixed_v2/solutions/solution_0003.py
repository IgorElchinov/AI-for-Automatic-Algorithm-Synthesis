import sys
import random
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.single_objective.swarm import PSO

# Read input
suite_name = sys.stdin.readline().strip()
function_index = int(sys.stdin.readline().strip())
dimension = int(sys.stdin.readline().strip())
instance = int(sys.stdin.readline().strip())
evaluation_budget = int(sys.stdin.readline().strip())
random_seed = int(sys.stdin.readline().strip())

# Set random seed
random.seed(random_seed)

# Create COCO BBOB problem
import cocoex
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

# Set up the optimizer
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

# Run optimization
opt.start(n_iterations=max(1, evaluation_budget // 20))
best_agent = search_space.best_agent
best_position = best_agent.position

# Print the best position
print(" ".join(map(str, best_position)))
