import numpy as np
from perfect_cmaps._optimization import genetic_algorithm


if __name__ == "__main__":
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