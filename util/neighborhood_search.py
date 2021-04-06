"""
neighborhood search

"""
from abc import abstractmethod
from pymoo.model.population import Population
from pymoo.model.individual import Individual
import numpy as np


class NeighborhoodSearch:
    """
        领域搜索
    """

    def do(self, problem, pop, **kwargs):
        """
        执行领域搜索, 若有algorithm.repair 则修复不合法解
        Args:
            problem(Problem): a problem
            pop(Individual|Population|np.ndarray): basic population or individual
            **kwargs: kwargs used for self._do

        Returns:
            Individual|Population|np.ndarray: new population or individual

        """
        is_individual = isinstance(pop, Individual)
        is_numpy_array = isinstance(pop, np.ndarray) and not isinstance(pop, Population)

        # make sure the object is a population
        if is_individual or is_numpy_array:
            pop = Population().create(pop)
        new_pop = self._do(problem, pop, **kwargs)
        algorithm = kwargs.get('algorithm')
        if algorithm and algorithm.repair:
            new_pop = algorithm.repair.do(problem, new_pop)
        if is_individual:
            return new_pop[0]
        if is_numpy_array:
            if len(new_pop) == 1:
                new_pop = new_pop[0]
            return new_pop
        return new_pop

    @staticmethod
    @abstractmethod
    def _do(problem, pop, **kwargs):
        """
        根据problem，对pop中的每一个个体进行进行领域搜索
        Args:
            problem(Problem): 用于获取问题相关信息
            pop(Population): 群体
            **kwargs:

        Returns:
            Population: new Population
        """
        return pop.copy(deep=True)
