import numpy as np
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.single_objective.swarm import PSO

def optimize(objective, lower_bounds, upper_bounds, dimension, budget, seed):
    rng = np.random.default_rng(seed)
    # Ensure bounds are arrays
    lb = np.asarray(lower_bounds, dtype=float)
    ub = np.asarray(upper_bounds, dtype=float)

    # Simple handling for zero budget
    if budget <= 0:
        return list((lb + ub) / 2.0)

    # Determine population size (at least 2, not exceeding budget)
    pop_size = min(max(2, dimension * 2), budget)
    # Determine number of iterations (each iteration evaluates pop_size individuals)
    max_iter = max(1, budget // pop_size)

    # Wrap objective for Opytimizer
    def _wrapped(x):
        return float(objective(x.tolist()))

    func = Function(_wrapped)

    # Create search space
    space = SearchSpace(n_agents=pop_size, n_variables=dimension, lower_bound=lb, upper_bound=ub)

    # Initialize optimizer
    optimizer = PSO()
    optimizer.n_iterations = max_iter

    # Run optimization
    opt = Opytimizer(space, func, optimizer, seed=seed)
    opt.start()

    # Retrieve best solution found
    best_position = opt.space.best_agent.position
    # Clip to bounds and ensure finite
    best_position = np.clip(best_position, lb, ub)
    if not np.all(np.isfinite(best_position)):
        best_position = np.where(np.isfinite(best_position), best_position, (lb + ub) / 2.0)

    return best_position.tolist()
