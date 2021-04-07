from pymoo.model.algorithm import Algorithm
from pymoo.model.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.model.repair import NoRepair
from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival
from pymoo.model.initialization import Initialization
from pymoo.model.population import Population
from operators.search.global_search import GlobalSearch


class ShuffledAlgorithm(Algorithm):

    def __init__(self,
                 pop_size=None,
                 meme_size=None,
                 sampling=None,
                 repair=None,
                 survival=None,
                 min_infeas_pop_size=None,
                 mating=None,
                 search=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        meme_size:the size of meme_group
        d_max:Maximum allowable step size
        sampling : {sampling}

        """

        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.meme_size = meme_size

        # set the duplicate detection class - a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        # simply set the no repair object if it is None
        self.repair = repair if repair is not None else NoRepair()

        self.survival = survival if survival is not None else FitnessSurvival()

        # minimum number of individuals surviving despite being infeasible - by default disabled
        self.min_infeas_pop_size = min_infeas_pop_size

        self.initialization = Initialization(sampling,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)

        if mating is None:
            mating = GlobalSearch(search=search,
                                  repair=self.repair)

        self.mating = mating

        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.pop = None
        self.off = None

    def _initialize(self):
        # create the initial population
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        pop.set("n_gen", self.n_gen)

        # then evaluate using the objective function
        self.evaluator.eval(self.problem, pop)

        # that call is a dummy survival to set attributes that are necessary for the mating selection
        if self.survival:
            pop = self.survival.do(self.problem, pop, len(pop), algorithm=self,
                                   n_min_infeas_survive=self.min_infeas_pop_size)

        self.pop, self.off = pop, pop

    def _next(self):

        self.off = self.mating.do(self.problem, self.pop, len(self.pop), meme_size=self.meme_size, algorithm=self)

        self.off.set("n_gen", self.n_gen)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.off) == 0:
            self.termination.force_termination = True
            return

        # # merge the offsprings with the current population
        self.pop = Population.merge(self.pop, self.off)

        # then do survival selection
        if self.survival:
            self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self,
                                        n_min_infeas_survive=self.min_infeas_pop_size)
