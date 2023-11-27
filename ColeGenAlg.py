import random

def generate_list(length):
    return [random.randint(1, length) for _ in range(length)]

def generate_initial_population(population_size,list_length):
    return [generate_list(list_length) for _ in range(population_size)]

def fitness(lst):
    total_sum = 0

    for index, value in enumerate(lst):
        difference = abs(value - (index+1))
        total_sum += difference

    return total_sum

def choose_parents(population,num_parents):
    parents = sorted(population, key=fitness)[:num_parents]
    return parents

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(child, mutation_rate, length):
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] = random.randint(1, length)
    return child

def evolve_population(population, generations, mutation_rate, list_length):
    for generation in range(generations):
        population.sort(key=fitness)
        
        if len(population) < 2:
            continue  

        parents = choose_parents(population, 2)
        child = crossover(parents[0], parents[1])
        child = mutate(child, mutation_rate, list_length)
        population = population[:-1] 
        population.append(child) 

    population.sort(key=fitness)
    return population

def GA_sort(list_length, population_size, generations, mutation_rate):
    population = generate_initial_population(population_size, list_length)
    evolved_population = evolve_population(population, generations, mutation_rate, list_length)
    return evolved_population[0]

list_length = 10
population_size = 1000
generations = 10000
mutation_rate = 0.1

sorted_list = GA_sort(list_length, population_size, generations, mutation_rate)
print("Sorted List:", sorted_list)