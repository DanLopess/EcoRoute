import pants
from lib import utils
import matplotlib.pyplot as plt
import time

points, distances = utils.read_points_and_distances()
nodes = utils.get_nodes(points, distances)

print(nodes)


def get_distance(index_a: int, index_b: int):
    return distances.loc[index_a, index_b]


def run_iteration():
    world = pants.World(nodes, get_distance)
    solver = pants.Solver()
    solution = solver.solve(world)
    return solution


def run_for_duration(func, seconds, *args):
    threshold = 20
    total_time = 0
    result_1st, duration_1st = utils.run_with_timer(func, *args)
    total_time += duration_1st
    func_results = [result_1st]
    while (total_time + duration_1st) < (seconds - threshold):
        result, duration = utils.run_with_timer(func, *args)
        func_results.append(result)
        total_time += duration
    return func_results


def plot_aco_iterations(results):
    x = range(1, len(results) + 1)
    y = [p.distance for p in results]
    plt.scatter(x, y, color="green")
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.title("Ant Colony Optimization x Iterations")
    plt.show()


if __name__ == "__main__":
    st = time.time()
    # run many times to obtain the best possible result in a given time
    time_to_run = 20  # 20 seconds (example)
    results = run_for_duration(run_iteration, time_to_run)
    best_aco = None
    for res in results:
        if (best_aco is None or res.distance < best_aco.distance):
            best_aco = res

    path = best_aco.tour
    path = utils.normalize_final_path(path)
    print("==== Best Result ====")
    print(f'Path: {path}')
    print(f'Distance: {best_aco.distance:.2f}')
    # plot_aco_iterations(results)
    print(f"Took {time.time() - st:.2f} seconds to execute.")
