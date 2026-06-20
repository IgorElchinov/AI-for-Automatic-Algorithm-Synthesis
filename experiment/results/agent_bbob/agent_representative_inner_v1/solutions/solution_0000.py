import random
import numpy as np
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.single_objective.swarm import PSO


def optimize(objective, lower_bounds, upper_bounds, dimension, budget, seed):
    random.seed(seed)
    np.random.seed(seed)

    # Ensure bounds are numpy arrays
    lb = np.array(lower_bounds, dtype=float)
    ub = np.array(upper_bounds, dtype=float)

    # Simple handling for zero budget
    if budget <= 0:
        return list((lb + ub) / 2.0)

    # Define the function wrapper for Opytimizer
    def _wrapped(x):
        return float(objective(x.tolist()))

    func = Function(_wrapped)

    # Choose a swarm size that fits the budget
    swarm_size = min(20, max(1, budget))
    max_iter = max(1, budget // swarm_size)

    # Create search space
    space = SearchSpace(n_agents=swarm_size, n_variables=dimension, lower_bound=lb, upper_bound=ub)

    # Initialize PSO optimizer
    optimizer = PSO()

    # Run optimization
    opt = Opytimizer(search_space=space, function=func, optimizer=optimizer)
    opt.run(n_iterations=max_iter)

    # Retrieve best solution
    best_position = opt.best_agent.position

    # Clip to bounds and ensure finite
    best_position = np.clip(best_position, lb, ub)
    if not np.all(np.isfinite(best_position)):
        best_position = np.where(np.isfinite(best_position), best_position, (lb + ub) / 2.0)

    return best_position.tolist()
