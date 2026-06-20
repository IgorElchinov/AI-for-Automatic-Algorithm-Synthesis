import math
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

    best_x = [
        random.uniform(
            float(lower_bounds[i]),
            float(upper_bounds[i]),
        )
        for i in range(dimension)
    ]
    best_y = objective(best_x)

    for _ in range(max(0, budget - 1)):
        x = [
            random.uniform(
                float(lower_bounds[i]),
                float(upper_bounds[i]),
            )
            for i in range(dimension)
        ]

        y = objective(x)

        if math.isfinite(y) and y < best_y:
            best_x = x
            best_y = y

    return best_x