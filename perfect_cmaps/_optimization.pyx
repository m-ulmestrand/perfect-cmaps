# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time

np.import_array()


cdef double random_uniform() noexcept nogil:
    return <double>rand() / <double>RAND_MAX


cdef double random_uniform_range(double low, double high) noexcept nogil:
    return low + (high - low) * random_uniform()


cdef int random_int(int low, int high) noexcept nogil:
    return low + <int>(random_uniform() * (high - low))


cdef double fitness_function(
    double[:] individual,
    double[:] l_profile,
    double[:] l_min,
    double[:] l_max,
    int n
) noexcept nogil:
    cdef int i
    cdef double l_corrected

    for i in range(n):
        l_corrected = l_profile[i] * individual[0] + individual[1]
        if l_corrected <= l_min[i] or l_corrected >= l_max[i]:
            return 0.0

    return individual[0]


cdef void tournament_selection(
    double[:, :] population,
    double[:] fitnesses,
    double[:, :] selected,
    int pop_size,
    int tournament_size
) noexcept nogil:
    cdef int i, j, participant_idx, winner_idx
    cdef double best_fitness

    for i in range(pop_size):
        winner_idx = random_int(0, pop_size)
        best_fitness = fitnesses[winner_idx]

        for j in range(1, tournament_size):
            participant_idx = random_int(0, pop_size)
            if fitnesses[participant_idx] > best_fitness:
                best_fitness = fitnesses[participant_idx]
                winner_idx = participant_idx

        selected[i, 0] = population[winner_idx, 0]
        selected[i, 1] = population[winner_idx, 1]


cdef void crossover(
    double[:] parent1,
    double[:] parent2,
    double[:] child1,
    double[:] child2
) noexcept nogil:
    cdef double alpha = random_uniform()

    child1[0] = alpha * parent1[0] + (1.0 - alpha) * parent2[0]
    child1[1] = alpha * parent1[1] + (1.0 - alpha) * parent2[1]
    child2[0] = alpha * parent2[0] + (1.0 - alpha) * parent1[0]
    child2[1] = alpha * parent2[1] + (1.0 - alpha) * parent1[1]


cdef void mutate(
    double[:] individual,
    double[:, :] gene_limits,
    double mutation_rate
) noexcept nogil:
    if random_uniform() < mutation_rate:
        individual[0] = random_uniform_range(gene_limits[0, 0], gene_limits[0, 1])
    if random_uniform() < mutation_rate:
        individual[1] = random_uniform_range(gene_limits[1, 0], gene_limits[1, 1])


def genetic_algorithm(
    int pop_size,
    int generations,
    double[:, :] gene_limits,
    double[:] l_profile,
    double[:] l_min,
    double[:] l_max,
    double mutation_rate = 0.5,
    bint elitism = True
):
    srand(<unsigned int>time(NULL))

    cdef int n = l_profile.shape[0]
    cdef int gen, i, elite_idx, best_idx, random_idx
    cdef double best_fitness, elite_fitness

    # Allocate arrays
    cdef np.ndarray[double, ndim=2] population = np.empty((pop_size, 2), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] selected = np.empty((pop_size, 2), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] new_population = np.empty((pop_size, 2), dtype=np.float64)
    cdef np.ndarray[double, ndim=1] fitnesses = np.empty(pop_size, dtype=np.float64)

    # Typed memoryviews for fast access
    cdef double[:, :] pop_view = population
    cdef double[:, :] sel_view = selected
    cdef double[:, :] new_pop_view = new_population
    cdef double[:] fit_view = fitnesses

    # Initialize population
    for i in range(pop_size):
        pop_view[i, 0] = random_uniform_range(gene_limits[0, 0], gene_limits[0, 1])
        pop_view[i, 1] = random_uniform_range(gene_limits[1, 0], gene_limits[1, 1])

    cdef double[2] elite

    for gen in range(generations):
        # Evaluate fitness
        for i in range(pop_size):
            fit_view[i] = fitness_function(pop_view[i], l_profile, l_min, l_max, n)

        # Elitism: Keep the best individual
        if elitism:
            elite_idx = 0
            for i in range(1, pop_size):
                if fit_view[i] > fit_view[elite_idx]:
                    elite_idx = i
            elite[0] = pop_view[elite_idx, 0]
            elite[1] = pop_view[elite_idx, 1]

        # Selection
        tournament_selection(pop_view, fit_view, sel_view, pop_size, 3)

        # Crossover and mutation
        for i in range(0, pop_size, 2):
            crossover(sel_view[i], sel_view[i + 1], new_pop_view[i], new_pop_view[i + 1])
            mutate(new_pop_view[i], gene_limits, mutation_rate)
            mutate(new_pop_view[i + 1], gene_limits, mutation_rate)

        if elitism:
            random_idx = random_int(0, pop_size)
            new_pop_view[random_idx, 0] = elite[0]
            new_pop_view[random_idx, 1] = elite[1]

        # Swap populations
        population, new_population = new_population, population
        pop_view = population
        new_pop_view = new_population

    # Find best in final population
    for i in range(pop_size):
        fit_view[i] = fitness_function(pop_view[i], l_profile, l_min, l_max, n)

    best_idx = 0
    for i in range(1, pop_size):
        if fit_view[i] > fit_view[best_idx]:
            best_idx = i

    result = np.array([pop_view[best_idx, 0], pop_view[best_idx, 1]])
    return result, fit_view[best_idx]
