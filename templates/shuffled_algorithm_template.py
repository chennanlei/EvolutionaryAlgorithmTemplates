import numpy as np
from pymoo.factory import get_sampling, get_termination
from pymoo.optimize import minimize
from pymoo.problems.single.flowshop_scheduling import create_random_flowshop_problem, visualize
from pymoo.model.callback import Callback
from operators.repair.perm_repair import SequentialPermRepair, PermRepairByP
from algorithms.shuffled_algorithm import ShuffledAlgorithm
import matplotlib.pyplot as plt
from pymoo.util.display import Display
from operators.search.global_search import GlobalSearch
from abc import abstractmethod
from pymoo.model.population import Population, Individual
from pymoo.model.evaluator import Evaluator
from util.neighborhood_search import NeighborhoodSearch


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


class Search(NeighborhoodSearch):
    def __init__(self, sampling, repair, evaluator=None):
        self.sampling = sampling
        self.evaluator = evaluator if evaluator is not None else Evaluator(skip_already_evaluated=False)
        self.repair = repair

    def _do(self, problem, pop, **kwargs):
        # global best
        pg = kwargs.get("pg")

        # get the best and worst individual in meme_group
        pb = pop[np.nanargmin(pop.get("F"))]
        pw = pop[np.nanargmax(pop.get("F"))]

        flag = self.replace(problem, pop, pw, pb=pb, **kwargs)

        if flag:
            return pop

        flag = self.replace(problem, pop, pw, pb=pg, **kwargs)
        if flag:
            return pop

        self.replace(problem, pop, pw, **kwargs)

        return pop

    def replace(self, problem, pop, pw, **kwargs) -> bool:
        d_max = kwargs.get("d_max")
        pb = kwargs.get("pb")

        k = np.nanargmax(pop.get("F"))
        if pb is None:
            ind = self.sampling.do(problem, 1)[0]
            self.evaluator.eval(problem, ind)
            pop[k] = ind
        else:
            _pb = self.renew(pw, pb, d_max)
            _pb = self.repair.do(problem, _pb, P=pb.get("X"))

            self.evaluator.eval(problem, _pb)
            ind = _pb[0]

            if ind.get("F") < pw.get("F"):
                pop[k] = ind
                return True
        return False

    @ abstractmethod
    def renew(self, pw: Individual, pb: Individual, d_max) -> Population:
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


class SFLAGlobalSearch(GlobalSearch):
    def __init__(self, d_max, lx, **kwargs):
        super(SFLAGlobalSearch, self).__init__(**kwargs)
        self.d_max = d_max
        self.lx = lx

    def search_for_groups(self, problem, pops, **kwargs):
        algorithm = kwargs.get("algorithm")

        # record the global best individual
        pg = algorithm.opt[0]

        for _ in range(self.lx):
            pops = [self.search.do(problem, pop, pg=pg, d_max=self.d_max) for pop in pops]
        return pops


def main():
    n_machines = 5
    n_jobs = 20

    problem = create_random_flowshop_problem(n_machines=n_machines, n_jobs=n_jobs)
    search = Search(sampling=get_sampling('perm_random'),
                    repair=PermRepairByP())
    mating = SFLAGlobalSearch(search=search,
                              meme_size=5,
                              d_max=3,
                              lx=5)
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
