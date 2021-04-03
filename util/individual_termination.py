"""
-*- coding: utf-8 -*-
@Time: 2021/4/3 15:26
@File: individual_termination.py
@Version: 1.0
@Author: chennanlei
@Contact: chennanlei@gmail.com
@Last Modified by: chennanlei
@Last Modified time: 2021/4/3 15:26
@Description：individual_termination
"""
from abc import abstractmethod


class IndividualTermination:
    def __init__(self) -> None:
        """
        Base class for the implementation of a termination criterion for an individual.
        """
        super().__init__()

        # the individual can be forced to terminate by setting this attribute to true
        self.force_termination = False

    def do_continue(self, individual, **kwargs):
        """

        Whenever the individual objects wants to know whether it should continue or not it simply
        asks the termination criterion for it.

        Parameters
        ----------
        individual : class
            The individual object that is asking if it has terminated or not.

        Returns
        -------
        do_continue : bool
            Whether the individual has terminated or not.

        """

        if self.force_termination:
            return False
        return self._do_continue(individual, **kwargs)

    # the concrete implementation of the individual
    @abstractmethod
    def _do_continue(self, individual, **kwargs):
        pass

    def has_terminated(self, individual, **kwargs):
        """
        Instead of asking if the individual should continue it can also ask if it has terminated.
        (just negates the continue method.)
        """
        return not self.do_continue(individual, **kwargs)


if __name__ == '__main__':
    pass
