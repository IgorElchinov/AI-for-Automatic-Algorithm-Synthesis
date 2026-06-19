import cocoex
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.single_objective.swarm import PSO
import sys

def main():
    input_data = sys.stdin.read().strip().splitlines()
    suite_name = input_data[0]
    function_index = int(input_data[1])
    dimension = int(input_data[2])
    instance = int(input_data[3])
    evaluation_budget = int(input_data[4])
    random_seed = int(input_data[5])

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

    function = Function(problem)
    search_space = SearchSpace(
        n_agents=20,
        n_variables=dimension,
        n_objectives=1,
        lower_bound=problem.lower_bounds,
        upper_bound=problem.upper_bounds,
    )
    optimizer = PSO()
    opt = Opytimizer(search_space, optimizer, function)
    opt.start(n_iterations=max(1, evaluation_budget // 20))
    
    best_agent = search_space.best_agent
    best_position = best_agent.position
    
    print(" ".join(map(str, best_position)))

if __name__ == "__main__":
    main()
