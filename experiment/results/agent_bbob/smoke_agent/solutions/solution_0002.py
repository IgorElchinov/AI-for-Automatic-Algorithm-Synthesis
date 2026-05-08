import sys
import random
import numpy as np
import cocoex

# Read input from stdin
suite_name = sys.stdin.readline().strip()
function_index = int(sys.stdin.readline().strip())
dimension = int(sys.stdin.readline().strip())
instance = int(sys.stdin.readline().strip())
evaluation_budget = int(sys.stdin.readline().strip())
random_seed = int(sys.stdin.readline().strip())

# Set the random seed for reproducibility
random.seed(random_seed)
np.random.seed(random_seed)

# Create the COCO BBOB problem
suite = cocoex.Suite(
    suite_name=suite_name,
    suite_instance=instance,
    suite_options=f"function_indices: {function_index} instance_indices: {instance} dimensions: {dimension}"
)
problem = suite.get_problem(0)  # Add the id argument

# Initialize the candidate solution vector x within the bounds
x = np.random.uniform(problem.lower_bounds, problem.upper_bounds, size=dimension)

# Simple random search optimizer
for _ in range(evaluation_budget):
    # Generate a new candidate solution vector y within the bounds
    y = np.random.uniform(problem.lower_bounds, problem.upper_bounds, size=dimension)
    
    # Evaluate both x and y
    fx = problem(x)
    fy = problem(y)
    
    # Update x if y is better
    if fy < fx:
        x = y

# Print the final candidate solution vector x
print(" ".join(map(str, x)))
