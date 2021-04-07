import numpy as np
from abc import abstractmethod
from functools import reduce
from pymoo.model.population import Population
from pymoo.model.evaluator import Evaluator
from pymoo.model.infill import InfillCriterion
from util.neighborhood_search import NeighborhoodSearch
from pymoo.model.duplicate import DefaultDuplicateElimination, NoDuplicateElimination


class Partition:
    def do(self, problem, pop, n_groups=1, **kwargs):
        return self._do(problem, pop, n_groups, **kwargs)

    @abstractmethod
    def _do(self, problem, pop, n_groups=1, **kwargs):
        pass


class EqualPartition(Partition):
    def _do(self, problem, pop, n_groups=1, **kwargs):
        n = len(pop)

        n_groups = min(n_groups, n)
        pops = np.array_split(pop, n_groups)

        return pops


class IsometricPartition(Partition):
    def _do(self, problem, pop, n_groups=1, **kwargs):
        n = len(pop)

        n_groups = min(n_groups, n)
        pops = []
        for i in range(n_groups):
            pops.append(pop[i::n_groups])

        return pops


class IsometricFitnessPartition(IsometricPartition):
    def _do(self, problem, pop, n_groups=1, **kwargs):
        pop = Population.create(*sorted(pop, key=lambda x: x.get('F')[0], reverse=True))

        return super(IsometricFitnessPartition, self)._do(problem, pop, n_groups, **kwargs)


class GlobalSearch(InfillCriterion):
    def __init__(self,
                 partition=None,
                 search=None,
                 meme_size=1,
                 **kwargs):
        super().__init__(**kwargs)

        self.partition = partition if partition is not None else IsometricFitnessPartition()
        self.search = search
        self.meme_size = meme_size if meme_size is not None else 1

    @abstractmethod
    def search_for_groups(self, problem, pops, **kwargs):
        return pops

    def _do(self, problem, pop, n_offsprings, **kwargs):
        # Group large populations into small populations
        pops = self.partition.do(problem, pop, self.meme_size)

        # the local search in small population
        pops = self.search_for_groups(problem, pops, **kwargs)

        pop = reduce(lambda x, y: Population.merge(x, y), pops)

        return pop
