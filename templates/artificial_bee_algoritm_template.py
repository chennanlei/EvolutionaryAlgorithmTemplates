"""
-*- coding: utf-8 -*-
@Time: 2021/4/3 15:31
@File: artificial_bee_algoritm_template.py
@Version: 1.0
@Author: chennanlei
@Contact: chennanlei@gmail.com
@Last Modified by: chennanlei
@Last Modified time: 2021/4/3 15:31
@Description：artificial_bee_algoritm_template
"""

from pymoo.problems.single.flowshop_scheduling import create_random_flowshop_problem, visualize
import numpy as np
import random
from pymoo.model.sampling import Sampling
from pymoo.util.display import Display
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.optimize import minimize
from pymoo.model.callback import Callback
from pymoo.model.population import Population
from pymoo.model.individual import Individual
import matplotlib.pyplot as plt
from algorithms.artificial_bee_algorithm import ArtificialBeeAlgorithm
from util.neighborhood_search import NeighborhoodSearch


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        data = []
        n_machines, n_jobs = problem.data.shape
        for _ in range(n_samples):
            x = [i for i in range(n_jobs)]
            random.shuffle(x)
            data.append(x)
        return np.row_stack(data)


class MyDisplay(Display):
    def __init__(self):
        super().__init__()
        self.best_fitness = 10000
        self.best_gen = 0

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        pop_f = algorithm.pop.get('F')
        self.output.append("min", "{:.2f}".format(np.min(pop_f)))
        self.output.append("max", "{:.2f}".format(np.max(pop_f)))
        self.output.append("mean", "{:.2f}".format(np.mean(pop_f)))
        if np.min(pop_f) < self.best_fitness:
            self.best_fitness = np.min(pop_f)
            self.best_gen = algorithm.n_gen
        self.output.append("best_gen", self.best_gen)


class MyCallback(Callback):
    def __init__(self):
        super(MyCallback, self).__init__()
        self.data['min'] = []
        self.data['max'] = []
        self.data['mean'] = []
        self.data['best_gen'] = []

    def notify(self, algorithm, **kwargs):
        pop_f = algorithm.pop.get('F')
        best_gen = algorithm.display.best_gen
        self.data['min'].append(np.min(pop_f))
        self.data['max'].append(np.max(pop_f))
        self.data['mean'].append(np.mean(pop_f))
        self.data['best_gen'].append(best_gen)


class MySearch(NeighborhoodSearch):
    def _do(self, problem, pop, **kwargs):
        new_pop = Population(len(pop))
        for index, individual in enumerate(pop):
            x = individual.get('X')  # 直接取值，可以理解为已经deepcopy(x)
            # 随机选择两个位置
            i, j = np.random.choice(len(x), 2, replace=False)
            # 交换
            x[i], x[j] = x[j], x[i]
            new_pop[index].set('X', x)
        return new_pop


def test():
    seed = 1
    problem = create_random_flowshop_problem(5, 30, seed=seed)
    import time
    t = 1000 * time.time()  # current time in milliseconds
    np.random.seed(int(t) % 2 ** 32)  # 重设随机种子
    algorithm = ArtificialBeeAlgorithm(
        pop_size=50,
        n_onlookers=20,
        sampling=MySampling(),
        neighborhood_search=MySearch(),
        eliminate_duplicates=False,
        display=MyDisplay()
    )
    termination = SingleObjectiveDefaultTermination(n_last=200, n_max_gen=200)

    res = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination,
        verbose=True,
        callback=MyCallback()
    )
    mean_data = res.algorithm.callback.data['mean']
    max_data = res.algorithm.callback.data['max']
    min_data = res.algorithm.callback.data['min']

    best_gen = res.algorithm.callback.data['best_gen'][-1]
    print(best_gen)
    plt.plot(np.arange(len(mean_data)), np.column_stack([min_data, mean_data, max_data]))
    plt.show()
    visualize(problem, res.X)


if __name__ == '__main__':
    test()
