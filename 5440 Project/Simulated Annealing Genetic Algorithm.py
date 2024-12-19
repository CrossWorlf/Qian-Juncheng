import numpy as np
import random
import json
from typing import List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class SimulatedAnnealingGeneticAlgorithm:
    UNIFORM_CROSSOVER = 1
    BIT_FLIP_MUTATION = 2

    def __init__(self, data: dict):
        self.cycle = data['cycle']
        self.genome_size = data['genome_size']
        self.init_pop_size = data['init_pop_size']
        self.crossover_scheme = data['crossover_scheme']
        self.mutation_scheme = data['mutation_scheme']
        self.knapsack_object = data['knapsack_object']
        self.initial_temperature = data['initial_temperature']
        self.cooling_rate = data['cooling_rate']
        self.min_temperature = data['min_temperature']
        self.max_iterations = data['max_iterations']
        self.restart_threshold = data['restart_threshold']
        self.tournament_size = data['tournament_size']
        self.adaptive_cooling = data['adaptive_cooling']
        self.elite_size = data['elite_size']

        self.population = self.generate_population(self.init_pop_size)
        self.best_genome = max(self.population, key=self.fitness_func)
        self.target_fitness = sum(self.knapsack_object.values)
        self.no_improvement_count = 0
        self.best_fitness_history = []
        self.diversity_history = []
        self.temperature_history = []

    def generate_population(self, pop_size: int) -> List[int]:
        population = []
        for _ in range(pop_size):
            num_ones = random.randint(self.genome_size // 4, 3 * self.genome_size // 4)
            genome = 0
            positions = random.sample(range(self.genome_size), num_ones)
            for pos in positions:
                genome |= (1 << pos)
            population.append(genome)
        return population

    def crossover(self, genome1: int, genome2: int, temperature: float) -> Tuple[int, int]:
        if self.crossover_scheme == self.UNIFORM_CROSSOVER:
            child1, child2 = 0, 0
            fitness1 = self.fitness_func(genome1)
            fitness2 = self.fitness_func(genome2)

            for i in range(self.genome_size):
                bit1 = (genome1 >> i) & 1
                bit2 = (genome2 >> i) & 1

                if bit1 == bit2:
                    if bit1:
                        child1 |= (1 << i)
                        child2 |= (1 << i)
                else:
                    crossover_rate = 0.5 + (0.3 * (temperature / self.initial_temperature) ** 0.5)
                    if random.random() < crossover_rate:
                        child1 |= (bit1 << i)
                        child2 |= (bit2 << i)
                    else:
                        child1 |= (bit2 << i)
                        child2 |= (bit1 << i)

            return child1, child2
        else:
            raise NotImplementedError

    def mutate(self, genome: int, temperature: float) -> int:
        if self.mutation_scheme == self.BIT_FLIP_MUTATION:
            mutation_rate = self.adaptive_mutation_rate(temperature)
            current_fitness = self.fitness_func(genome)
            mutated = genome

            for i in range(self.genome_size):
                if random.random() < mutation_rate:
                    test_genome = mutated ^ (1 << i)
                    test_fitness = self.fitness_func(test_genome)
                    if test_fitness > current_fitness or \
                            random.random() < np.exp((test_fitness - current_fitness) / temperature):
                        mutated = test_genome
                        current_fitness = test_fitness

            return mutated
        else:
            raise NotImplementedError

    def adaptive_mutation_rate(self, temperature: float) -> float:
        base_rate = 0.25
        temp_factor = (temperature / self.initial_temperature) ** 0.5
        diversity_factor = (1 - self.population_diversity()) ** 0.5

        if len(self.best_fitness_history) > 1:
            if self.best_fitness_history[-1] <= self.best_fitness_history[-2]:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = max(0, self.no_improvement_count - 1)

        stagnation_factor = min(0.6, self.no_improvement_count / self.restart_threshold)
        return min(0.6, base_rate * (1 + temp_factor + stagnation_factor + diversity_factor))

    def select(self, population: List[int], temperature: float) -> int:
        tournament = random.sample(population, self.tournament_size)

        if random.random() < 0.25:
            return random.choice(tournament)

        best = max(tournament, key=self.fitness_func)
        for candidate in tournament:
            if candidate != best:
                delta = self.fitness_func(best) - self.fitness_func(candidate)
                if delta > 0 and random.random() < np.exp(-delta / (temperature * 2)):
                    return candidate
        return best

    def fitness_func(self, genome: int) -> int:
        weight = 0
        value = 0
        for i in range(self.genome_size):
            if (genome >> i) & 1:
                weight += self.knapsack_object.weights[i]
                value += self.knapsack_object.values[i]

        if weight > self.knapsack_object.capacity:
            return int(value * (self.knapsack_object.capacity / weight) ** 1.2)
        return value

    def population_diversity(self) -> float:
        bit_sum = [0] * self.genome_size
        for genome in self.population:
            for i in range(self.genome_size):
                bit_sum[i] += (genome >> i) & 1

        diversity = sum(min(count, len(self.population) - count)
                        for count in bit_sum) / (self.genome_size * len(self.population) / 2)
        return diversity

    def visualize_metrics(self):
        iterations = range(len(self.best_fitness_history))

        avg_fitness_history = []
        for i in range(len(self.best_fitness_history)):
            avg_fitness = sum(self.fitness_func(genome) for genome in self.population) / len(self.population)
            avg_fitness_history.append(avg_fitness)

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, avg_fitness_history, '-', label='Average Profit', color='blue')
        plt.plot(iterations, self.best_fitness_history, '--', label='Maximum Profit', color='orange')
        plt.xlabel('Iterations')
        plt.ylabel('Profit')
        plt.title('Evolution of Profit Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.show()

    def driver(self) -> int:
        temperature = self.initial_temperature
        best_genome = self.best_genome
        current_best_fitness = self.fitness_func(best_genome)
        self.best_fitness_history = [current_best_fitness]
        self.diversity_history = [self.population_diversity()]
        self.temperature_history = [temperature]

        for iteration in tqdm(range(self.cycle), desc="Optimization Progress"):
            population_with_fitness = [(genome, self.fitness_func(genome))
                                       for genome in self.population]
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)
            elite = [genome for genome, _ in population_with_fitness[:self.elite_size]]

            new_population = elite.copy()

            while len(new_population) < self.init_pop_size:
                parent1 = self.select(self.population, temperature)
                parent2 = self.select(self.population, temperature)
                child1, child2 = self.crossover(parent1, parent2, temperature)
                child1 = self.mutate(child1, temperature)
                child2 = self.mutate(child2, temperature)
                new_population.extend([child1, child2])

            self.population = new_population[:self.init_pop_size]

            current_best = max(self.population, key=self.fitness_func)
            current_fitness = self.fitness_func(current_best)

            if current_fitness > current_best_fitness:
                best_genome = current_best
                current_best_fitness = current_fitness
                self.best_genome = best_genome
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            self.best_fitness_history.append(current_best_fitness)
            self.diversity_history.append(self.population_diversity())
            self.temperature_history.append(temperature)

            if self.no_improvement_count >= self.restart_threshold:
                if iteration < self.cycle // 3:
                    temperature = self.initial_temperature * 0.9
                elif iteration < 2 * self.cycle // 3:
                    temperature = self.initial_temperature * 0.7
                else:
                    temperature = self.initial_temperature * 0.5

                new_population = self.generate_population(self.init_pop_size - self.elite_size)
                new_population.extend(elite)
                self.population = new_population
                self.no_improvement_count = 0
            else:
                temperature = max(self.min_temperature,
                                  temperature * self.cooling_rate)

        return best_genome


class Knapsack:
    def __init__(self, n: int, json_fname: str = None):
        self.n = n
        if json_fname:
            with open(json_fname, 'r') as f:
                data = json.load(f)
                self.values = np.array(data['values'])
                self.weights = np.array(data['weights'])
                self.capacity = data['capacity']
        else:
            self.values = np.random.randint(1, 100, size=n)
            self.weights = np.random.randint(1, 50, size=n)
            self.capacity = int(np.sum(self.weights) * 0.6)

    def to_numpy(self):
        self.values = np.array(self.values)
        self.weights = np.array(self.weights)


if __name__ == "__main__":
    fname = "./values/30_values.json"
    knapsack_object = Knapsack(30, json_fname=fname)
    knapsack_object.to_numpy()

    genetic_algo_data = {
        'cycle': 200,  # Changed from 400 to 200
        'genome_size': knapsack_object.n,
        'init_pop_size': knapsack_object.n * 15,
        'crossover_scheme': SimulatedAnnealingGeneticAlgorithm.UNIFORM_CROSSOVER,
        'mutation_scheme': SimulatedAnnealingGeneticAlgorithm.BIT_FLIP_MUTATION,
        'knapsack_object': knapsack_object,
        'initial_temperature': 500.0,
        'cooling_rate': 0.9999,
        'min_temperature': 1.0,
        'max_iterations': 7000,
        'restart_threshold': 400,
        'tournament_size': 3,
        'adaptive_cooling': True,
        'elite_size': 1
    }

    saga = SimulatedAnnealingGeneticAlgorithm(genetic_algo_data)
    best_genome = saga.driver()

    sequence = [(best_genome >> i) & 1 for i in range(knapsack_object.n)]
    print(f"Sequence: {sequence}")
    print(f"Genome Value: {best_genome}")
    print(f"Profit: {saga.fitness_func(best_genome)}")
    print(f"Capacity Used: {sum(w * s for w, s in zip(knapsack_object.weights, sequence))}")

    saga.visualize_metrics()

