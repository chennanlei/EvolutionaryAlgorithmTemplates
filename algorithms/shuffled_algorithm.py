from pymoo.model.duplicate import DefaultDuplicateElimination
from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival
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
