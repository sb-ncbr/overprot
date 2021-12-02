'''Dummy kNN searcher'''

import math
import statistics
from typing import Generic, TypeVar, List, Tuple, Dict, Set, Optional, Callable, Final, Iterator, Any, Counter, Sequence, Literal, Deque, Iterable
from dataclasses import dataclass, field
import numpy as np
import json

from .abstract_similarity_tree import AbstractSimilarityTree, K, V
from .caches import FunctionCache, DistanceCache, MinFinder
from ..lib import PriorityQueue, ProgressBar


class DummySearcher(AbstractSimilarityTree[K, V]):
    _elements: List[K]
    _distance_cache: DistanceCache[K, V]

    def __init__(self, distance_function: Callable[[V, V], float], keys_values: Sequence[Tuple[K, V]] = ()) -> None:
        self._distance_function = distance_function
        self._distance_cache = DistanceCache(distance_function)
        for k, v in keys_values:
            self._distance_cache.insert(k, v)
        self._elements = [k for k, v in keys_values]
        n = len(self._elements)
        with ProgressBar(n, title=f'Loading {n} elements to DummySearcher') as bar:
            for k1 in self._elements:
                for k2 in self._elements:
                    self.get_distance(k1, k2)
                bar.step()
    
    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._elements)

    def __contains__(self, element) -> bool:
        '''Decide if element is present in the tree'''
        return element in self._elements

    def get_distance(self, key1: K, key2: K) -> float:
        '''key1 can be element or pivot, key2 must be element'''
        return self._distance_cache.get_distance(key1, key2)

    def get_statistics(self):
        result = f'''Dummy searcher statistics:
        Elements: {len(self)}
        {self._distance_cache.get_statistics()}'''
        return result
    
    def kNN_query_by_value(self, query_value: V, k: int, distance_low_bounds: List[Callable[[V, V], float]] = []) -> List[Tuple[float, K]]:
        besties = MinFinder[K](n=k)
        current_range = besties.top_size()
        query_dist = FunctionCache[K, float](lambda key: self._distance_cache.get_distance_to_value(key, query_value))
        for i_elem, elem in enumerate(self._elements):
            for pivot in self._elements[:i_elem]:
                # TODO use only if already calculated
                if pivot in query_dist:
                    d_e_p = self.get_distance(elem, pivot)
                    d_p_q = query_dist[pivot]
                    if abs(d_e_p - d_p_q) >= current_range:
                        break
            else:
                d_e_q = query_dist[elem]
                besties.bubble_in(d_e_q, elem)
                current_range = besties.top_size()

        result = besties.pop_all_not_none()
        return result
