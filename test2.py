import random
from deap import creator, base, tools, algorithms
import time
start = time.time()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    ideal_sequence = list(range(1, 11))
    differences = [abs(a - b) for a, b in zip(individual, ideal_sequence)]
    fitness_value = 1 / (sum(differences) + .001)
    return fitness_value,

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=4)

population = toolbox.population(n=100)

if __name__=="__main__":
    import multiprocessing
    num_processes=4
    pool = multiprocessing.Pool(processes=num_processes)
    toolbox.register("map", pool.map)
    NGEN = 20000
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
                
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
                
        population = toolbox.select(offspring, k=len(population))
                
    top10 = tools.selBest(population, k=10)
    stop = time.time()-start
    print(top10[0])
    print(evalOneMax(top10[0]))
    print(stop)
