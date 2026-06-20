import numpy as np

def optimize(objective, lower_bounds, upper_bounds, dimension, budget, seed):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lower_bounds, dtype=float)
    ub = np.asarray(upper_bounds, dtype=float)

    if budget <= 0:
        return ((lb + ub) / 2.0).tolist()

    best_x = None
    best_f = np.inf

    # Ensure at least one evaluation
    for _ in range(budget):
        x = rng.uniform(lb, ub)
        try:
            f = float(objective(x.tolist()))
        except Exception:
            f = np.inf
        if np.isfinite(f) and f < best_f:
            best_f = f
            best_x = x

    if best_x is None:
        best_x = (lb + ub) / 2.0

    # Clip to bounds and ensure finite
    best_x = np.clip(best_x, lb, ub)
    if not np.all(np.isfinite(best_x)):
        best_x = np.where(np.isfinite(best_x), best_x, (lb + ub) / 2.0)

    return best_x.tolist()
