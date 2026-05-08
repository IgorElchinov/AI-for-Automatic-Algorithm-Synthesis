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
def objective_fn(x):
    return problem(x)

# Build the search space
space = SearchSpace(
    n_agents=10,
    n_variables=dimension,
    n_objectives=1,
    lower_bound=[-5.0] * dimension,
    upper_bound=[5.0] * dimension,
)

# Instantiate the optimizer
optimizer = PSO()

# Wrap the objective function with Opytimizer's Function class
function = Function(objective_fn)

# Build the Opytimizer instance
opt = Opytimizer(space, optimizer, function)

# Run the optimization
opt.start(n_iterations=evaluation_budget // 10)

# Get the best solution
best_solution = opt.best_agent.position

# Print the final candidate solution vector x
print(" ".join(map(str, best_solution)))
