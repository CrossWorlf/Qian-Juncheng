from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
from standard_genetic_algorithm import GeneticAlgorithm
from generate_data import Knapsack
import matplotlib.pyplot as plt


def get_genome_value(genome: np.ndarray):
    return int('0b' + ''.join([str(i) for i in genome.tolist()]), 2)


def get_genome_sequence(code: int, padding: int):
    return np.array([int(x) for x in np.binary_repr(code, padding)])


def greedy_solution(knapsack_obj):
    ratio = [(v / w, i) for i, (v, w) in enumerate(zip(knapsack_obj.values, knapsack_obj.weights))]
    ratio.sort(reverse=True)  # 按价值/重量比降序排序

    genome = [0] * knapsack_obj.n
    total_weight = 0
    for _, i in ratio:
        if total_weight + knapsack_obj.weights[i] <= knapsack_obj.capacity:
            genome[i] = 1
            total_weight += knapsack_obj.weights[i]
    return genome


def fitness_func(code: int, knapsack_obj: Knapsack):
    # get genome sequence
    genome = get_genome_sequence(code, knapsack_obj.n)
    # check if total load of genome fits in capacity
    if np.dot(genome, knapsack_obj.weights) <= knapsack_obj.capacity:
        # return the profit
        return np.dot(genome, knapsack_obj.values)
    # return Negative Infinity
    return -np.inf
class HybridIslandRestartGA(GeneticAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_fitness_per_cycle = []
        self.average_fitness_per_cycle = []
        self.migration_interval = 5  # 迁移间隔
        self.restart_rate = 0.995    # 重启概率

    def evolve_subpopulation(self, population):
        if len(population) == 0:
            population = self.init_population()  # 如果种群为空，重新初始化

        # 子种群演化：选择、交叉、变异
        population = self.selection(population)
        if len(population) == 0:
            population = self.init_population()

        population = self.crossover(population)
        if len(population) == 0:
            population = self.init_population()

        population = self.mutation(population)
        if len(population) == 0:
            population = self.init_population()

        return np.unique(population)

    def migrate(self, populations):
        best_individuals = [max(pop, key=lambda g: self.fitness_func(g)) for pop in populations if len(pop) > 0]
        for i, pop in enumerate(populations):
            if len(pop) > 0 and len(best_individuals) > 0:  # 确保目标种群非空
                pop = np.append(pop, best_individuals[(i + 1) % len(best_individuals)])
                populations[i] = self.selection(np.unique(pop))  # 选择后维持平衡
        return populations

    def driver(self, num_islands=5, threshold_vector=None, threshold_value=None):
        populations = [self.init_population() for _ in range(num_islands)]
        winner_genomes = []

        for cycle in tqdm(range(self.cycle), leave=False):
            # 并行演化各子种群
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.evolve_subpopulation, pop) for pop in populations]
                populations = [f.result() for f in as_completed(futures)]

            # 迁移机制
            if cycle % self.migration_interval == 0:
                populations = self.migrate(populations)

            if np.random.rand() > self.restart_rate:
                for i in range(num_islands):
                    if not populations[i].size:  # 重新初始化空子种群
                        populations[i] = self.init_population()
                    else:
                        elite_individuals = self.selection(populations[i])[:5]
                        if len(elite_individuals) == 0:  # 确保精英个体非空
                            populations[i] = self.init_population()
                        else:
                            new_population = self.mutation(elite_individuals)
                            populations[i] = np.append(populations[i], new_population)
                            populations[i] = self.selection(populations[i])

            # 记录适应度
            best_fitness = max(
                max(self.fitness_func(g) for g in pop) for pop in populations if len(pop) > 0
            )

            # 过滤掉适应度为 -np.inf 的个体，确保平均适应度有效
            valid_fitness_values = [
                [self.fitness_func(g) for g in pop if self.fitness_func(g) != -np.inf]
                for pop in populations if len(pop) > 0
            ]

            if any(len(fitnesses) > 0 for fitnesses in valid_fitness_values):
                avg_fitness = np.mean([np.mean(fitnesses) for fitnesses in valid_fitness_values if len(fitnesses) > 0])
            else:
                avg_fitness = 0  # 设置默认值，避免空数组问题

            self.best_fitness_per_cycle.append(best_fitness)
            self.average_fitness_per_cycle.append(avg_fitness)

        # 选取全局最优解
        best_individuals = [max(pop, key=lambda g: self.fitness_func(g)) for pop in populations]
        best_genome = max(best_individuals, key=lambda g: self.fitness_func(g))
        return best_genome

    def visualize_metrics(self):
        # 绘制适应度曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_per_cycle, label='Best Fitness')
        plt.plot(self.average_fitness_per_cycle, label='Average Fitness', linestyle='--')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.title('Hybrid GA with Island Model and Restart')
        plt.legend()
        plt.show()

        # 验证数据输出
        print("Best Fitness Per Cycle:", self.best_fitness_per_cycle)
        print("Average Fitness Per Cycle:", self.average_fitness_per_cycle)


if __name__ == "__main__":
    # 初始化背包问题
    fname = "./values/30_values.json"
    knapsack_object = Knapsack(30, json_fname=fname)
    knapsack_object.to_numpy()

    # 定义算法参数
    hybrid_ga_data = {
        'cycle': 200,
        'genome_size': knapsack_object.n,
        'init_pop_size': knapsack_object.n ** 2,
        'crossover_scheme': GeneticAlgorithm.UNIFORM_CROSSOVER,
        'mutation_scheme': GeneticAlgorithm.BIT_FLIP_MUTATION,
        'fitness_func': lambda genome: fitness_func(genome, knapsack_object),
        'seed_range': (0, 2 ** knapsack_object.n - 1),
        'encode': get_genome_value,
        'decode': lambda genome: get_genome_sequence(genome, knapsack_object.n)
    }

    # 运行混合算法
    ga = HybridIslandRestartGA(**hybrid_ga_data)
    winner_genome = ga.driver(num_islands=25, threshold_vector=knapsack_object.weights,
                              threshold_value=0.95*knapsack_object.capacity)

    # 输出结果
    print("Sequence: {}\nGenome Value: {}\nProfit: {}\nCapacity Used: {}".format(
        get_genome_sequence(winner_genome, knapsack_object.n),
        winner_genome,
        fitness_func(winner_genome, knapsack_object),
        np.dot(get_genome_sequence(winner_genome, knapsack_object.n), knapsack_object.weights)))
    ga.visualize_metrics()