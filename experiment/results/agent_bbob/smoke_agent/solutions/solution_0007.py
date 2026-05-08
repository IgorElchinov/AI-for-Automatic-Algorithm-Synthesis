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
problem = next(iter(suite))

# Initialize the candidate solution vector x
x = np.random.uniform(problem.lower_bounds, problem.upper_bounds)

# Simple random search optimizer with local search
for _ in range(evaluation_budget - 1):
    # Generate a new candidate solution vector x_new
    x_new = np.random.uniform(problem.lower_bounds, problem.upper_bounds)
    
    # Evaluate the current and new solutions
    f_x = problem(x)
    f_x_new = problem(x_new)
    
    # Update the candidate solution if the new one is better
    if f_x_new < f_x:
        x = x_new

# Local search around the best found solution
for _ in range(10):  # Perform a few local searches
    for i in range(dimension):
        # Perturb each dimension slightly
        perturbation = np.random.uniform(-0.1, 0.1)
        x_perturbed = x.copy()
        x_perturbed[i] += perturbation
        
        # Ensure the perturbed solution is within bounds
        x_perturbed = np.clip(x_perturbed, problem.lower_bounds, problem.upper_bounds)
        
        # Evaluate the perturbed solution
        f_x_perturbed = problem(x_perturbed)
        
        # Update the candidate solution if the new one is better
        if f_x_perturbed < problem(x):
            x = x_perturbed

# Print the final candidate solution vector x
print(" ".join(map(str, x)))
