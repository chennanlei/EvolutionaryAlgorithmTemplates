
from pymoo.model.repair import Repair
from pymoo.model.population import Population
import numpy as np
from abc import abstractmethod
from pymoo.util.misc import at_least_2d_array


def repeated_element_index(arr):
    assert np.ndim(arr) == 1, f"The dimension of the input must be 1, {np.ndim(arr)} obtained, {arr=}"

    ele2ind = {}  # record the elements in arr and their first index
    repeated_index = []
    for ind, elem in enumerate(arr):
        if elem in ele2ind:
            repeated_index.append(ind)
        else:
            ele2ind[elem] = ind

    return np.array(repeated_index)


def sequential_replace(X, replaced_index, replaced_elem):
    if len(replaced_index) != 0:
        X[replaced_index] = replaced_elem


class PermutationRepair(Repair):

    def _do(self,
            problem,
            pop_or_X,
            **kwargs):

        is_array = not isinstance(pop_or_X, Population)

        X = pop_or_X if is_array else pop_or_X.get("X")
        self.perm_replacement(problem, X, **kwargs)

        if is_array:
            return X
        else:
            pop_or_X.set("X", X)
            return pop_or_X

    @ abstractmethod
    def perm_replacement(self, problem, X, **kwargs):
        pass


class SequentialPermRepair(PermutationRepair):
    def superfluous_elements(self, arr, **kwargs):
        n = len(arr)
        elem_set = set(arr)

        super_elem = []
        for i in range(n):
            if i not in elem_set:
                super_elem.append(i)
        return np.array(super_elem)

    def perm_replacement(self, problem, X, **kwargs):
        for k in range(len(X)):
            index_arr = repeated_element_index(X[k])
            superfluous_elem = self.superfluous_elements(X[k], **kwargs)
            sequential_replace(X[k], index_arr, superfluous_elem)


class PermRepairByP(SequentialPermRepair):
    def superfluous_elements(self, arr, **kwargs):
        P = kwargs.get('P')
        elem_set = set(arr)

        super_elem = []
        for i in P:
            if i not in elem_set:
                super_elem.append(i)
        return np.array(super_elem)
