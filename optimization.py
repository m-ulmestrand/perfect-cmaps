import numpy as np
from numba import njit


@njit
def fitness_function(individual: np.ndarray, l_profile: np.ndarray, l_min: np.ndarray, l_max: np.ndarray):
    l_corrected = l_profile * individual[0] + individual[1]
    if np.logical_and(l_corrected > l_min, l_corrected < l_max).all():
        return individual[0]
    
    return 0.0

def initialize_population(pop_size, gene_limits):
    population = np.random.uniform(gene_limits[:, 0], gene_limits[:, 1], (pop_size, 2))
    return population

# Selection of parents (tournament selection)
@njit
def tournament_selection(population, fitnesses, tournament_size):
    selected = np.zeros_like(population)
    pop_size = len(population)
    for i in range(pop_size):
        participants_idx = np.random.randint(0, pop_size, tournament_size)
        winner_idx = participants_idx[np.argmax(fitnesses[participants_idx])]
        selected[i] = population[winner_idx]
    return selected

@njit
def crossover(parent1, parent2):
    alpha = np.random.rand()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2

@njit
def mutate(individual, gene_limits, mutation_rate=0.5):
    for i in range(2):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.uniform(gene_limits[i, 0], gene_limits[i, 1])
    return individual

# Main genetic algorithm
def genetic_algorithm(
    pop_size: int, 
    generations: int, 
    gene_limits: np.ndarray, 
    l_profile: np.ndarray,
    l_min: np.ndarray, 
    l_max: np.ndarray, 
    mutation_rate: float = 0.5, 
    elitism: bool = True
):
    # Initialize population
    population = initialize_population(pop_size, gene_limits)
    
    for gen in range(generations):
        # Evaluate fitness
        fitnesses = np.array([fitness_function(ind, l_profile, l_min, l_max) for ind in population])
        
        # Elitism: Keep the best individual
        if elitism:
            elite_idx = np.argmax(fitnesses)
            elite = population[elite_idx]
        
        # Selection
        selected_population = tournament_selection(population, fitnesses, tournament_size=3)
        
        # Crossover and mutation to create new population
        new_population = np.zeros_like(population)
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population[i] = mutate(child1, gene_limits, mutation_rate)
            new_population[i + 1] = mutate(child2, gene_limits, mutation_rate)
        
        if elitism:
            # Replace a random individual with the elite one
            random_idx = np.random.randint(0, pop_size)
            new_population[random_idx] = elite
        
        population = new_population
    
    # Return the best individual
    final_fitnesses = np.array([fitness_function(ind, l_profile, l_min, l_max) for ind in population])
    best_idx = np.argmax(final_fitnesses)
    return population[best_idx], final_fitnesses[best_idx]


# Pre-compile numba functions
genetic_algorithm(
    2, 
    1, 
    np.array([[0.0, 1.0], [-100.0, 100.0]]), 
    np.linspace(0, 100, 5), 
    np.linspace(20, 30, 5),
    np.linspace(30, 50, 5) 
)

if __name__ == "__main__":
    # Define the gene limits and run the GA
    gene_limits = np.array([[0.0, 1.0], [-100.0, 100.0]])
    n = 100
    l_min = np.linspace(0, 20, n)
    l_max = np.linspace(50, 40, n)

    base_envelope = np.linspace(0, 100, 100)
    best_individual, best_fitness = genetic_algorithm(100, 200, gene_limits, base_envelope, l_min, l_max)
    print(best_individual, best_fitness)

    from matplotlib import pyplot as plt
    plt.plot(base_envelope * best_individual[0] + best_individual[1], color="black")
    plt.plot(l_max, color="red", linestyle="--")
    plt.plot(l_min, color="red", linestyle="--")
    plt.show()