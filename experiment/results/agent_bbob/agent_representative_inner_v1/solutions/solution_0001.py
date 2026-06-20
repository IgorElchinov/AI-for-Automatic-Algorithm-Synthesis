import random
import numpy as np

def optimize(objective, lower_bounds, upper_bounds, dimension, budget, seed):
    random.seed(seed)
    np.random.seed(seed)

    lb = np.array(lower_bounds, dtype=float)
    ub = np.array(upper_bounds, dtype=float)

    if budget <= 0:
        return ((lb + ub) / 2.0).tolist()

    best_x = None
    best_f = float('inf')
    evals = 0

    # include the midpoint as first candidate
    x0 = (lb + ub) / 2.0
    f0 = float(objective(x0.tolist()))
    evals += 1
    if np.isfinite(f0):
        best_x, best_f = x0, f0

    while evals < budget:
        # sample uniformly
        x = np.random.uniform(lb, ub)
        f = float(objective(x.tolist()))
        evals += 1
        if np.isfinite(f) and f < best_f:
            best_x, best_f = x, f

    if best_x is None:
        best_x = (lb + ub) / 2.0

    return np.clip(best_x, lb, ub).tolist()
