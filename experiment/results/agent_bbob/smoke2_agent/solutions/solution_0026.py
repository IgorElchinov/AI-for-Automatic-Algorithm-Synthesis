import sys
import cocoex
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.single_objective.swarm import PSO

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

# Define the objective function
def objective_function(x):
    return problem(x)

# Create a search space
search_space = SearchSpace(
    n_variables=dimension,
    n_agents=30,
    lower_bound=[-5.0] * dimension,
    upper_bound=[5.0] * dimension
)

# Initialize the optimizer
optimizer = PSO(search_space, Function(objective_function))

# Run the optimization
optimizer.run(evaluation_budget)

# Get the best solution
best_solution = optimizer.best_agent.position

# Print the final candidate solution vector x
print(" ".join(map(str, best_solution)))
