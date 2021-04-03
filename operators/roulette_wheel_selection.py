"""
-*- coding: utf-8 -*-
@Time: 2021/4/3 16:22
@File: roulette_wheel_selection.py
@Version: 1.0
@Author: chennanlei
@Contact: chennanlei@gmail.com
@Last Modified by: chennanlei
@Last Modified time: 2021/4/3 16:22
@Description：roulette_wheel_selection
"""
import numpy as np

from pymoo.model.selection import Selection
from pymoo.util.roulette import RouletteWheelSelection as BasicSelection


class RouletteWheelSelection(Selection):
    """
      The Roulette wheel selection is used to simulated a roulette wheel between individuals.
      Return selected indexes
    """

    def __init__(self, larger_is_better=False):
        """

        Args:
            larger_is_better(bool): larger is better
        """

        super(RouletteWheelSelection, self).__init__()
        self.larger_is_better = larger_is_better

    def _do(self, pop, n_select, n_parents=1, **kwargs):
        f = pop.get('F')
        if f.shape[1] > 1:
            raise Exception("The number of objectives for the roulette wheel selection must be one!")
        selected_indexes = BasicSelection(f, self.larger_is_better).next(n_select * n_parents)
        return np.reshape(selected_indexes, (n_select, n_parents))
