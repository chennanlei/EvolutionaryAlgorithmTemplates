import numpy as np
from pymoo.factory import get_sampling, get_termination
from pymoo.optimize import minimize
from pymoo.problems.single.flowshop_scheduling import create_random_flowshop_problem, visualize
from pymoo.model.callback import Callback
from operators.repair.perm_repair import SequentialPermRepair, PermRepairByP
from algorithms.shuffled_algorithm import ShuffledAlgorithm
import matplotlib.pyplot as plt
from pymoo.util.display import Display
from operators.search.global_search import GlobalSearchByBest
from pymoo.model.population import Population, Individual
from operators.search.group_search import GroupSearch

np.random.seed(10)


class FJPCallback(Callback):
    def __init__(self):
        super(FJPCallback, self).__init__()
        self.data['best'] = []
        self.data['worst'] = []

    def notify(self, algorithm, **kwargs):
        f = algorithm.pop.get("F")
        self.data['best'].append(f.min())
        self.data['worst'].append(f.max())


class FJPDisplay(Display):
    def __init__(self):
        super().__init__()
        self.best_fitness = float("inf")
        self.best_gen = 0

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        pop_f = algorithm.pop.get('F')
        self.output.append("min", "{:.2f}".format(np.min(pop_f)))
        self.output.append("max", "{:.2f}".format(np.max(pop_f)))
        self.output.append("mean", "{:.2f}".format(np.mean(pop_f)))


class SFLASearch(GroupSearch):
    def __init__(self, sampling, repair, evaluator=None, **kwargs):
        super(SFLASearch, self).__init__(sampling, repair, evaluator, **kwargs)

    def renew(self, pw: Individual, pb: Individual, **kwargs) -> Population:
        d_max = kwargs.get("d_max")
        new_pb = []
        rand = np.random.random()
        for elem_w, elem_b in zip(pw.get('X'), pb.get('X')):
            ds = int(rand * (elem_b - elem_w))
            if ds > 0:
                elem = elem_w + min(ds, d_max)
            else:
                elem = elem_w + max(ds, -d_max)
            new_pb.append(elem)

        return Population.new("X", np.atleast_2d(np.array(new_pb)))


class SFLAGlobalSearch(GlobalSearchByBest):
    def __init__(self, generations, d_max, **kwargs):
        super(SFLAGlobalSearch, self).__init__(generations, **kwargs)
        self.d_max = d_max

    def search_for_groups(self, problem, pops, **kwargs):
        pops = super(SFLAGlobalSearch, self).search_for_groups(problem, pops, d_max=self.d_max, **kwargs)
        return pops


def main():
    n_machines = 5
    n_jobs = 20

    problem = create_random_flowshop_problem(n_machines=n_machines, n_jobs=n_jobs)
    search = SFLASearch(sampling=get_sampling('perm_random'),
                        repair=PermRepairByP())
    mating = SFLAGlobalSearch(search=search,
                              meme_size=5,
                              d_max=3,
                              generations=5)
    algorithm = ShuffledAlgorithm(pop_size=20,
                                  mating=mating,
                                  eliminate_duplicates=True,
                                  repair=SequentialPermRepair(),
                                  sampling=get_sampling('perm_random'))

    # termination = SingleObjectiveDefaultTermination(n_max_gen=100)
    termination = get_termination('n_gen', 300)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   display=FJPDisplay(),
                   verbose=True,
                   callback=FJPCallback())
    visualize(problem, res.X)

    best_val = res.algorithm.callback.data['best']
    worst_val = res.algorithm.callback.data['worst']
    plt.plot(np.arange(len(best_val)), best_val, label='best')
    plt.plot(np.arange(len(worst_val)), worst_val, label='worst')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
