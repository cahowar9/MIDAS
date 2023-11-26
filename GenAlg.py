import random

def generate_random_list(length):
    return [random.randint(1, 100) for _ in range(length)]

def create_population(population_size, list_length):
    return [generate_random_list(list_length) for _ in range(population_size)]

def calculate_fitness(lst):
    # Fitness is calculated based on the number of incorrectly sorted pairs
    return sum([1 for i in range(len(lst) - 1) if lst[i] > lst[i + 1]])

def select_parents(population, num_parents):
    parents = sorted(population, key=calculate_fitness)[:num_parents]
    return parents

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(child, mutation_rate):
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] = random.randint(1, 100)
    return child

def evolve_population(population, generations, mutation_rate):
    for generation in range(generations):
        population.sort(key=calculate_fitness)
        
        if len(population) < 2:
            continue  # Skip this iteration if there are not enough individuals for crossover

        parents = select_parents(population, 2)
        child = crossover(parents[0], parents[1])
        child = mutate(child, mutation_rate)
        population = population[:-1]  # Remove the least fit individual
        population.append(child)  # Add the new child to the population

    # Final sorting
    population.sort(key=calculate_fitness)
    return population

def genetic_sort(list_length, population_size, generations, mutation_rate):
    population = create_population(population_size, list_length)
    evolved_population = evolve_population(population, generations, mutation_rate)
    return evolved_population[0]

# Example usage
list_length = 5
population_size = 1000
generations = 1000
mutation_rate = 0.1

sorted_list = genetic_sort(list_length, population_size, generations, mutation_rate)
print("Sorted List:", sorted_list)