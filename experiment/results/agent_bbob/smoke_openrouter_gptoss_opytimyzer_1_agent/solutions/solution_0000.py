import sys
import cocoex
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.single_objective.swarm import PSO

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

    # set random seed for reproducibility
    import random, numpy as np
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

    lower = np.full(dimension, problem.lower_bounds)
    upper = np.full(dimension, problem.upper_bounds)

    # simple PSO settings
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
        return problem(x.tolist())[0]

    func = Function(objective)

    optimizer = PSO()
    opt = Opytimizer(space, optimizer, func)
    opt.start(n_iterations=n_iterations)

    best = opt.best_agent.position
    print(" ".join(map(str, best)))

if __name__ == "__main__":
    main()
