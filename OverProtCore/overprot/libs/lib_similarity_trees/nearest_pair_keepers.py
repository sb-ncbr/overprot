from collections import defaultdict
from typing import Generic, TypeVar, Tuple, Iterable, Callable
import itertools
import math

from ..lib import PriorityQueue

from .abstract_similarity_tree import AbstractSimilarityTree
from .ghtree import GHTree
from .mtree import MTree

K = TypeVar('K')
V = TypeVar('V')


class TreeNearestPairKeeper(Generic[K, V]):
    '''Data structure for keeping the nearest pairs amongst a dynamic set of elements, based on a metric Tree'''
    def __init__(self, tree: AbstractSimilarityTree[K, V], elements: Iterable[Tuple[K, V]] = []):
        self._tree = tree
        self._nearest_to = defaultdict(list)  # _nearest_to[i] contains all elements for which i has been enrolled as the nearest element
        queue_items = []
        for key, value in elements:
            self._tree.insert(key, value)
            if len(self) >= 2:
                nearest_to_key = self._tree.kNN_query(key, 1)[0]
                dist = self._tree.get_distance(key, nearest_to_key)
                queue_items.append((dist, (key, nearest_to_key)))
                self._nearest_to[nearest_to_key].append(key)
        self._priority_queue = PriorityQueue[float, Tuple[K, K]](queue_items)  # contains the nearest older element for each element as (newer, older)

    def insert(self, key: K, value: V) -> None:
        self._tree.insert(key, value)
        self._enroll_nearest_element(key)

    def delete(self, key: K) -> None:
        self._tree.delete(key)
        if key in self._nearest_to:
            nearest_to = self._nearest_to.pop(key)
            for elem in nearest_to:
                if elem in self:
                    self._enroll_nearest_element(elem)

    def _enroll_nearest_element(self, key: K) -> None:
        if len(self) >= 2:
            nearest_to_key = self._tree.kNN_query(key, 1)[0]
            dist = self._tree.get_distance(key, nearest_to_key)
            self._priority_queue.add(dist, (key, nearest_to_key))
            self._nearest_to[nearest_to_key].append(key)

    def pop_nearest_pair(self) -> Tuple[K, K, float]:
        if len(self) < 2:
            raise ValueError('This GHTNearestPairKeeper contains less than 2 elements.')
        popped = self._priority_queue.pop_min_which(self.contains_pair)
        assert popped is not None
        dist, (i, j) = popped
        self.delete(i)
        self.delete(j)
        return i, j, dist

    def contains_pair(self, pair: Tuple[K, K]) -> bool:
        return pair[0] in self and pair[1] in self

    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._tree)

    def __contains__(self, element) -> bool:
        '''Decide if element is present'''
        return element in self._tree

    def get_statistics(self) -> str:
        return self._tree.get_statistics()

class GHTNearestPairKeeper(TreeNearestPairKeeper[K, V]):
    def __init__(self, distance_function: Callable[[V, V], float], leaf_size=8, elements: Iterable[Tuple[K, V]] = []):
        tree = GHTree[K, V](distance_function, leaf_size=leaf_size)
        super().__init__(tree, elements=elements)


class MTNearestPairKeeper(TreeNearestPairKeeper[K, V]):
    def __init__(self, distance_function: Callable[[V, V], float], fork_arity=4, leaf_arity=4, elements: Iterable[Tuple[K, V]] = []):
        tree = MTree[K, V](distance_function, fork_arity=fork_arity, leaf_arity=leaf_arity)
        super().__init__(tree, elements=elements)


class DumbNearestPairFinder:  # TODO replace with some M-tree, GHT or something like that
    '''Naive implementation for keeping the nearest pairs amongst a set of elements (for testing GHTNearestPairKeeper)'''
    def __init__(self, distance_function):
        self.distance_function = distance_function
        self.items = {}
        self.distances = {}
        self.n_calculated_distances = 0
    def __len__(self):
        return len(self.items)
    def insert(self, key, item):
        for other_key, other_item in self.items.items():
            distance = self.distance_function(item, other_item)
            self.distances[(key, other_key)] = distance
            self.n_calculated_distances += 1
        self.items[key] = item
    def delete(self, key):
        self.items.pop(key)
        for other_key in self.items.keys():
            self.distances.pop((key, other_key), None)
            self.distances.pop((other_key, key), None)
    def __getitem__(self, key):
        return self.items[key]
    def get_nearest_pair(self):
        if len(self) < 2:
            raise ValueError('Must contain at least 2 items to find the nearest pair')
        distance, (i, j) = min((distance, pair) for pair, distance in self.distances.items())
        return i, j, distance
    def pop_nearest_pair(self):
        i, j, distance = self.get_nearest_pair()
        self.delete(i)
        self.delete(j)
        return i, j, distance
