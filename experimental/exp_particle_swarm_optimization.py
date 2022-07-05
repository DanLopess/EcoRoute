import operator
import random
import numpy
import math
from deap import base
from deap import tools
from deap import creator
from math import sin, sqrt


def z1(x1, x2):
    return sqrt(x1**2 + x2**2)


def z2(x1, x2):
    return sqrt((x1 - 1)**2 + (x2 + 1)**2)


def f1(individual):
    if len(individual) == 2:
        x1 = individual[0]
        x2 = individual[1]
        _z1 = z1(x1, x2)
        _z2 = z2(x1, x2)
        try:
            return [(sin(4 * _z1) / _z1) + (sin(2.5 * _z2) / _z2)]
        except:
            print(f'Failed for values: X1={x1} X2={x2}')
            exit()
    exit()


creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Particle",
               list,
               fitness=creator.FitnessMax,
               speed=list,
               smin=None,
               smax=None,
               best=None)


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(
        map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2,
                 pmin=-6, pmax=6, smin=-3, smax=3)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", f1)


def main():
    pop = toolbox.population(n=100)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 1000
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    return pop, logbook, best


pop, logbook, best = main()

fitness = f1(best)
print("Best individual is %s: %s" % (best, fitness))
