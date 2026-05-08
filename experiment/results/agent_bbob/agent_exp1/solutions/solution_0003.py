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
search_space = SearchSpace(n_agents=1, n_iterations=int(evaluation_budget / dimension), n_variables=dimension,
                           lower_bound=[-5.0] * dimension, upper_bound=[5.0] * dimension)

# Define the optimizer
optimizer = PSO()

# Initialize the Opytimizer
opytimizer = Opytimizer(search_space, optimizer, problem)

# Run the optimization
opytimizer.start()

# Get the best solution
best_solution = opytimizer.space.best_agent.position

# Print the final candidate solution vector x
print(" ".join(map(str, best_solution)))
