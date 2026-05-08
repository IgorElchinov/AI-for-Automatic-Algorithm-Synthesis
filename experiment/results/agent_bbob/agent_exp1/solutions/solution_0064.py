import sys
import cocoex
from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces.search import SearchSpace

# Read input from stdin
suite_name = sys.stdin.readline().strip()
function_index = int(sys.stdin.readline().strip())
dimension = int(sys.stdin.readline().strip())
instance = int(sys.stdin.readline().strip())
evaluation_budget = int(sys.stdin.readline().strip())
random_seed = int(sys.stdin.readline().strip())

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
    n_iterations=evaluation_budget,
    n_particles=10,
    dimensions=(dimension, -5.0, 5.0),
)

# Define the optimizer
optimizer = PSO()

# Initialize the Opytimizer
opt = Opytimizer(search_space, optimizer)

# Execute the optimization
history = opt.start()

# Get the best solution
best_solution = history.best_agent.position

# Print the final candidate solution vector x
print(" ".join(map(str, best_solution)))
