import sys, random
import numpy as np
import cocoex
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.single_objective.swarm import PSO

def main():
    lines = [l.strip() for l in sys.stdin.readlines()]
    if len(lines) < 6:
        return
    suite_name = lines[0]
    function_index = int(lines[1])
    dimension = int(lines[2])
    instance = int(lines[3])
    budget = int(lines[4])
    seed = int(lines[5])

    random.seed(seed)
    np.random.seed(seed)

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

    lower = np.full(dimension, problem.lower_bounds, dtype=float)
    upper = np.full(dimension, problem.upper_bounds, dtype=float)

    n_agents = 20
    n_iterations = max(1, budget // n_agents)

    space = SearchSpace(
        n_agents=n_agents,
        n_variables=dimension,
        n_objectives=1,
        lower_bound=lower,
        upper_bound=upper,
    )

    def objective(x):
        arr = np.asarray(x).ravel()
        return problem(arr.tolist())

    func = Function(objective)

    optimizer = PSO()
    opt = Opytimizer(space, optimizer, func)
    opt.start(n_iterations=n_iterations)

    best = opt.best_agent.position
    print(" ".join(map(str, best.tolist())))

if __name__ == "__main__":
    main()
