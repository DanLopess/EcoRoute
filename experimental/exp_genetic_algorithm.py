from math import sin, sqrt
import random
from deap import base
from deap import creator
from deap import tools


def z1(x1, x2):
    return sqrt(x1**2 + x2**2)


def z2(x1, x2):
    return sqrt((x1 - 1)**2 + (x2 + 1)**2)


def f1(individual):
    x1 = individual[0]
    x2 = individual[1]
    _z1 = z1(x1, x2)
    _z2 = z2(x1, x2)
    try:
        return (sin(4 * _z1) / _z1) + (sin(2.5 * _z2) / _z2)
    except:
        print(f'Failed for values: X1={x1} X2={x2}')
        exit()


def mutate(individual, indpb):
    prob = random.random()
    if prob < indpb:
        pos_to_mutate = random.randint(0, 1)
        individual[pos_to_mutate] += random.uniform(-1, 1)


creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.01, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", f1)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", mutate, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

hof = tools.HallOfFame(1)


def main():
    random.seed(64)
    pop = toolbox.population(n=100)
    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = [fit]

    print("  Evaluated %i individuals" % len(pop))
    g = 0

    while g < 100:
        g = g + 1
        print("-- Generation %i --" % g)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = [fit]

        print("  Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring

        hof.update(pop)

    print(f"-- End of (successful) evolution (Generations: {g}) --")

    best_ind = hof[0]
    fitness = f1(best_ind)
    print("Best individual is %s: %s" % (best_ind, fitness))


main()
