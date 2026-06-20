# manual_solutions/over_budget.py

import random


def optimize(
    objective,
    lower_bounds,
    upper_bounds,
    dimension,
    budget,
    seed,
):
    random.seed(seed)

    x = [
        random.uniform(float(lower_bounds[i]), float(upper_bounds[i]))
        for i in range(dimension)
    ]

    for _ in range(budget + 1):
        objective(x)

    return x