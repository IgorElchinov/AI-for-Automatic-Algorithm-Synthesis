import sys
import cocoex
from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces.search import SearchSpace

# Read input from stdin
input_lines = [line.strip() for line in sys.stdin.readlines()]

suite_name = input_lines[0]
function_index = int(input_lines[1])
dimension = int(input_lines[2])
instance = int(input_lines[3])
evaluation_budget = int(input_lines[4])
random_seed = int(input_lines[5])

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
optimizer = PSO()

# Initialize the Opytimizer
opt = Opytimizer(search_space, optimizer, iterations=1, save_history=False)

# Define the objective function
def objective_function(x):
    return problem.function(x.reshape((dimension,)))

# Set the objective function
opt.set_function(objective_function)

# Run the optimization
opt.start()

# Get the best solution
best_solution = opt.best_agent.position

# Print the final candidate solution vector x
print(" ".join(map(str, best_solution)))
