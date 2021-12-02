import sys
from typing import List, Tuple, Dict, Generic, Callable

from ..lib import PriorityQueue
from .abstract_similarity_tree import K, V
from .caches import MinFinder

INFINITY = sys.maxsize

class Target(Generic[K, V]):
    _bins: List[Tuple[int, int]]
    _elements_in_bins: List[List[K]]
    _values: Dict[K, V]

    def __init__(self, elements_values: List[Tuple[K, V]], distance_function: Callable[[V, V], float], length_function: Callable[[V], int], bin_size: int = 10) -> None:
        print('INFINITY:', INFINITY)
        lenghts = [length_function(value) for key, value in elements_values]
        max_length = max(lenghts, default=0)
        self._bins = [(low, low+bin_size) for low in range(0, max_length, bin_size)]
        last_high = self._bins[-1][1] if len(self._bins) > 0 else 0
        self._bins.append((last_high, INFINITY))
        self._n_bins = len(self._bins)
        self._elements_in_bins = [[] for i in range(self._n_bins)]
        self._values = {}
        for (elem, value), length in zip(elements_values, lenghts):
            i_bin = self._get_bin(length)
            self._elements_in_bins[i_bin].append(elem)
            self._values[elem] = value
        self._distance_function = distance_function
        self._length_function = length_function
        self._distance_function_counter = 0

    def _get_bin(self, length: int) -> int:
        i_bin = next(i for i, (low, high) in enumerate(self._bins) if length >= low)  # TODO enforce same-width bins and make everything easier?
        return i_bin
    
    def kNN_query_by_value(self, query_value: V, k: int) -> List[Tuple[float, K]]:
        query_length = self._length_function(query_value)
        the_bin = self._get_bin(query_length)
        queue = PriorityQueue[float, int]()
        for i, (low, high) in enumerate(self._bins):
            dmin = 0.0 if i == the_bin else min(abs(low - query_length), abs(high-1 - query_length))  # high-1 because length must be int ;)
            dmin = dmin / 2  # because of how distance_function works (abs(m-n)/2 <= distance <= (m+n)/2))
            queue.add(dmin, i)
        besties = MinFinder[K](n=k)
        current_range = besties.top_size()
        while len(queue) > 0:
            best = queue.pop_min()
            assert best is not None
            dmin, i_bin = best
            if dmin >= current_range:
                continue
            for elem in self._elements_in_bins[i_bin]:
                value = self._values[elem]
                d_e_q_low = 0.5 * abs(self._length_function(value) - query_length)  # because of how distance_function works (abs(m-n)/2 <= distance <= (m+n)/2))
                if d_e_q_low >= current_range:
                    continue
                d_e_q = self._distance_function(value, query_value)
                self._distance_function_counter += 1
                besties.bubble_in(d_e_q, elem)
                current_range = besties.top_size()
            
        result = besties.pop_all_not_none()
        return result
                
