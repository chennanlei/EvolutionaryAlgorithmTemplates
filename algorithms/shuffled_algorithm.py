from pymoo.model.algorithm import Algorithm
from pymoo.model.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.model.repair import NoRepair
from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival
from pymoo.model.initialization import Initialization
from pymoo.model.population import Population
from operators.search.global_search import GlobalSearch
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm


class ShuffledAlgorithm(GeneticAlgorithm):
    def __init__(self,
                 pop_size=None,
                 meme_size=None,
                 sampling=None,
                 repair=None,
                 survival=None,
                 min_infeas_pop_size=None,
                 mating=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 **kwargs
                 ):
        super(ShuffledAlgorithm, self).__init__(
            pop_size=pop_size,
            sampling=sampling,
            repair=repair,
            survival=survival,
            min_infeas_pop_size=min_infeas_pop_size,
            mating=mating,
            eliminate_duplicates=eliminate_duplicates,
            **kwargs
        )
        if self.survival is None:
            self.survival = FitnessSurvival()
        if mating is None:
            self.mating = GlobalSearch(meme_size=meme_size)

    def _next(self):
        self.off = self.mating.do(self.problem, self.pop, len(self.pop), algorithm=self)
        self.off.set("n_gen", self.n_gen)
        if len(self.off) == 0:
            self.termination.force_termination = True
            return

        self.pop = Population.merge(self.pop, self.off)
        if self.survival:
            self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self,
                                        n_min_infeas_survive=self.min_infeas_pop_size)
