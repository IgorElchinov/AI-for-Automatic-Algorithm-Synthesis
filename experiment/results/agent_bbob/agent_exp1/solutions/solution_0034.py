import sys
import cocoex
from opytimizer import Opytimizer
from opytimizer.spaces.search import SearchSpace
from opytimizer.optimizers.swarm import PSO

# Read input from stdin
suite_name = sys.stdin.readline().strip()
function_index = int(sys.stdin.readline().strip())
dimension = int(sys.stdin.readline().strip())
instance = int(sys.stdin.readline().strip())
evaluation_budget = int(sys.stdin.readline().strip())
random_seed = int(sys.stdin.readline().strip())

# Set the random seed for reproducibility
import numpy as np
np.random.seed(random_seed)

# Reconstruct the COCO BBOB problem
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

# Define the search space
search_space = SearchSpace(
    n_variables=dimension,
    lower_bound=[-5.0] * dimension,
    upper_bound=[5.0] * dimension
)

# Define the optimizer
optimizer = PSO(search_space, algorithm_params={"w": 0.7, "c1": 1.5, "c2": 1.5})

# Run the optimization
history = optimizer.run(problem, n_iterations=evaluation_budget // dimension)

# Get the best solution
best_solution = history.best_agent.position

# Print the final candidate solution vector x
print(" ".join(map(str, best_solution)))
