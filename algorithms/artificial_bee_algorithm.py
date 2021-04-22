"""
-*- coding: utf-8 -*-
@Time: 2021/4/3 15:28
@File: artificial_bee_algorithm.py
@Version: 1.0
@Author: chennanlei
@Contact: chennanlei@gmail.com
@Last Modified by: chennanlei
@Last Modified time: 2021/4/3 15:28
@Description：artificial_bee_algorithm
"""
import numpy as np
from pymoo.model.population import Population
from pymoo.model.algorithm import Algorithm, filter_optimum
from pymoo.model.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.model.repair import NoRepair
from pymoo.model.initialization import Initialization
from pymoo.util.dominator import Dominator
from util.neighborhood_search import NeighborhoodSearch
from util.individual_termination import IndividualTermination
from operators.roulette_wheel_selection import RouletteWheelSelection


class BeeTermination(IndividualTermination):
    def __init__(self, max_trials):
        super(BeeTermination, self).__init__()
        # max number of trials without any improvement
        self.max_trials = max_trials

    def _do_continue(self, bee, **kwargs):
        algorithm = kwargs.get("algorithm")
        best_bee = kwargs.get("best_bees")[0]
        return algorithm.compare(bee, best_bee) >= 0 or algorithm.n_gen - bee.get('created_gen') < self.max_trials


def get_relation(individual_a, individual_b):
    """
        compare individual a and b, a > b : 1; a < b : -1; a == b: 0
        if the number of objectives > 1, use Dominator.get_relation
    Args:
        individual_a(Individual): individual a
        individual_b(Individual): individual b

    Returns:
        int: 1 means a is better than b; -1: worse; 0: equal

    """
    a_f = individual_a.get('F')
    b_f = individual_b.get('F')
    if not hasattr(a_f, "__len__"):
        a_f = [a_f]
        b_f = [b_f]
    return Dominator.get_relation(a_f, b_f)


class ArtificialBeeAlgorithm(Algorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=None,
                 n_onlookers=None,
                 selection=RouletteWheelSelection(larger_is_better=False),
                 compare=get_relation,
                 neighborhood_search=NeighborhoodSearch(),
                 bee_termination=BeeTermination(20),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 repair=None,
                 **kwargs):

        super(ArtificialBeeAlgorithm, self).__init__(**kwargs)

        # the bee population size used
        self.pop_size = pop_size

        # the number of onlooker bees
        self.n_onlookers = n_onlookers
        if self.n_onlookers is None:
            self.n_onlookers = self.pop_size // 2

        # individual termination  criterion by max_trials
        self.bee_termination = bee_termination

        # set the duplicate detection class
        # a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        # simply set the no repair object if it is None
        self.repair = repair if repair is not None else NoRepair()

        self.initialization = Initialization(sampling,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)
        # selection method for onlookers
        self.selection = selection
        # the instance of Neighborhood_search class
        self.neighborhood_search = neighborhood_search
        # compare method for two fitness of individuals
        self.compare = compare
        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.pop = None

    def create_new(self, pop_size):
        if pop_size == 0:
            return Population()
        # create the population
        pop = self.initialization.do(self.problem, pop_size, algorithm=self)
        # set the created generation
        pop.set("created_gen", self.n_gen)
        # then evaluate using the objective function
        self.evaluator.eval(self.problem, pop, **{"algorithm": self})
        return pop

    def _initialize(self):
        # create the initial population
        self.pop = self.create_new(self.pop_size)

    def _next(self):
        # 1. employees phase
        self.send_employee(np.arange(self.pop_size))

        # 2. onlookers phase
        self.send_onlookers()

        # 3. scouts phase
        self.send_scouts()

    def send_employee(self, indexes):
        if len(indexes) == 0:
            return
        # do neighborhood search for bees
        new_pop = self.neighborhood_search.do(self.problem, self.pop[indexes], algorithm=self)
        # set created generation for new bees
        new_pop.set('created_gen', self.n_gen)
        # evaluate new bee
        self.evaluator.eval(self.problem, new_pop, **{"algorithm": self})
        for i, new_bee in enumerate(new_pop):
            if self.compare(self.pop[indexes[i]], new_bee) == -1:
                self.pop[indexes[i]] = new_bee

    def send_onlookers(self):
        selected_indexes = self.selection.do(self.pop, n_select=self.n_onlookers, n_parents=1).reshape(-1)
        self.send_employee(selected_indexes)

    def send_scouts(self):
        best_bees = filter_optimum(self.pop, least_infeasible=True)
        to_be_replaced = []
        for index, bee in enumerate(self.pop):
            if self.bee_termination.has_terminated(bee, best_bees=best_bees, algorithm=self):
                to_be_replaced.append(index)
        new_pop = self.create_new(len(to_be_replaced))
        for i, index in enumerate(to_be_replaced):
            self.pop[index] = new_pop[i]

