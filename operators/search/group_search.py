#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2021/4/27 20:34
# Author: Zheng Shaoxiang
# @Email: zhengsx95@163.com
# Description:
from util.neighborhood_search import NeighborhoodSearch
from pymoo.model.evaluator import Evaluator
from pymoo.model.population import Population
from pymoo.model.individual import Individual
import numpy as np
from abc import abstractmethod


class GroupSearch(NeighborhoodSearch):
    def __init__(self, sampling, repair, evaluator=None, **kwargs):
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

    def replace(self, problem, pop, pw, pb=None, **kwargs) -> bool:
        k = np.nanargmax(pop.get("F"))
        if pb is None:
            ind = self.sampling.do(problem, 1)[0]
            self.evaluator.eval(problem, ind)
            pop[k] = ind
        else:
            _pb = self.renew(pw, pb, **kwargs)
            _pb = self.repair.do(problem, _pb, P=pb.get("X"))

            self.evaluator.eval(problem, _pb)
            ind = _pb[0]

            if ind.get("F") < pw.get("F"):
                pop[k] = ind
                return True
        return False

    @abstractmethod
    def renew(self, pw: Individual, pb: Individual, **kwargs) -> Population:
        return Population.create(pb)


if __name__ == '__main__':
    pass
