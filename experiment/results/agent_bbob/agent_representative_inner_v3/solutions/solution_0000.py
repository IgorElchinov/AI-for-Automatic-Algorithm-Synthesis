import random
import math

def optimize(objective, lower_bounds, upper_bounds, dimension, budget, seed):
    random.seed(seed)
    best_x = None
    best_f = math.inf

    for _ in range(budget):
        x = [random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(dimension)]
        try:
            f = objective(x)
        except Exception:
            continue
        if not (isinstance(f, (int, float)) and math.isfinite(f)):
            continue
        if f < best_f:
            best_f = f
            best_x = x

    if best_x is None:
        # fallback to middle of bounds
        best_x = [(lb + ub) / 2.0 for lb, ub in zip(lower_bounds, upper_bounds)]

    return best_x
