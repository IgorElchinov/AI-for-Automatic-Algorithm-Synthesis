import sys, random, math
import cocoex

def main():
    data = [line.strip() for line in sys.stdin.readlines()]
    if len(data) < 6:
        return
    suite_name = data[0]
    function_index = int(data[1])
    dimension = int(data[2])
    instance = int(data[3])
    budget = int(data[4])
    seed = int(data[5])
    random.seed(seed)

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

    lb = problem.lower_bounds
    ub = problem.upper_bounds

    best_x = [0.0] * dimension
    best_f = float('inf')
    evals = 0

    while evals < budget:
        x = [random.uniform(lb[i], ub[i]) for i in range(dimension)]
        f = problem(x)
        evals += 1
        if f < best_f:
            best_f = f
            best_x = x

    # ensure we output exactly dimension numbers
    print(' '.join(str(v) for v in best_x))

if __name__ == "__main__":
    main()
