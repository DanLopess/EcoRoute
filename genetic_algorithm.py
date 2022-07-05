from deap import tools
from deap import creator
from deap import base
from lib import utils
import matplotlib.pyplot as plt
import random
import time

points, distances = utils.read_points_and_distances()
nodes = utils.get_nodes(points, distances)


def get_distance(index_a: int, index_b: int):
    return distances.loc[index_a, index_b]


possible_genes = nodes


def fitness_func(individual):
    """Calculate the fitness of an individual 
    by summing the distances between each gene (eco)"""
    total_dist = 0
    if len(individual) >= 2:
        for eco1, eco2 in zip(individual[::1], individual[1::1]):
            dist = get_distance(eco1, eco2)
            total_dist += dist
        return_home_dist = get_distance(individual[-1], individual[0])
        total_dist += return_home_dist
    return [round(total_dist, 2)]


def mutate(individual, indpb):
    """Interchange single gene position for another"""
    prob = random.random()
    if prob < indpb and len(individual) >= 2:
        pos_i = random.randint(0, len(individual)-1)
        pos_f = random.randint(0, len(individual)-1)
        temp = individual[pos_f]
        individual[pos_f] = individual[pos_i]
        individual[pos_i] = temp
    return individual


def generate_individual():
    """Generate random sample of genes from possible genes"""
    return random.sample(possible_genes, len(possible_genes))


def cxOnePointOrdered(ind1, ind2):
    """Custom single-point ordered crossover"""
    size = min(len(ind1), len(ind2))
    k = random.randint(0, size-1)
    c1, c2 = ind1[:k],  ind2[:k]
    for val1, val2 in zip(ind1, ind2):
        if val2 not in c1:
            c1.append(val2)
        if val1 not in c2:
            c2.append(val1)
    return c1, c2


creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("gen_ind", generate_individual)
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.gen_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_func)
toolbox.register("mate", cxOnePointOrdered)
toolbox.register("mutate", mutate, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


def run_generation(cxpb, mutpb, hof, pop, g, logbook):
    print("-- Generation %i --" % g)

    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring
    hof.update(pop)
    best_ind = hof[0]
    logbook.record(gen=g, best_individual_distance=best_ind.fitness)
    print(logbook.stream)


def run_ga(cxpb=0.6, mutpb=0.3, gen=1000, pop_size=500, max_time=120, threshold=20):
    total_time = 0
    hof = tools.HallOfFame(1)
    random.seed(64)
    pop = toolbox.population(n=pop_size)

    print("Start of evolution")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    #print("  Evaluated %i individuals" % len(pop))
    g = 0
    logbook = tools.Logbook()
    logbook.header = ["gen", "best_individual_distance"]
    while g < gen and total_time < max_time - threshold:
        g = g + 1
        _, time = utils.run_with_timer(
            run_generation, cxpb, mutpb, hof, pop, g, logbook)
        total_time += time

    print(f"-- End of (successful) evolution (Generations: {g}) --")

    best_ind = hof[0]
    return best_ind, logbook


def plot_ga_logbook(lb: tools.Logbook):
    generations = lb.select("gen")
    distances = lb.select("best_individual_distance")
    normalized_distances = [float(d.values[0]) for d in distances]
    normalized_generations = [int(g) for g in generations]
    plt.scatter(normalized_generations, normalized_distances, color="orange")
    plt.plot(normalized_generations, normalized_distances)
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.title("Genetic Algorithm Evolution")
    plt.show()


if __name__ == '__main__':
    st = time.time()
    # run for a limited amount of time or generations
    best_ind, logbook = run_ga(0.6, 0.3, 500, 500, 60*20)
    path = utils.normalize_final_path(best_ind)
    print("==== Best Result ====")
    print(f'Path: {path}')
    print(f'Distance: {float(best_ind.fitness.values[0]):.2f}')
    # plot_ga_logbook(logbook)
    print(f"Took {time.time() - st:.2f} seconds to execute.")
