'''VPT, Vantage Point Tree'''

import math
from typing import List, Tuple, Callable, Sequence, Iterable
import numpy as np
from multiprocessing import Pool

from .abstract_similarity_tree import AbstractSimilarityTree, K, V
from .caches import FunctionCache, DistanceCache, MinFinder
from ..lib import ProgressBar


ZERO_ELEMENT = '-'
TOLERANCE = 0.0
TOLERANCE_RATIO = 1.0


class LAESA(AbstractSimilarityTree[K, V]):
    # _distance_function: Callable[[V, V], float]
    _distance_cache: DistanceCache[K, V]
    _elements: List[K]
    _pivots: List[K]
    _distance_array: np.ndarray

    def __init__(self, distance_function: Callable[[V, V], float], keys_values: Sequence[Tuple[K, V]] = (), n_pivots: int = -1, n_processes: int = 1) -> None:
        if n_pivots == -1:
            n_pivots = len(keys_values)
        self._distance_function = distance_function
        self._distance_cache = DistanceCache(distance_function)
        n = len(keys_values)
        self._elements = [k for k, v in keys_values]
        assert len(set(self._elements)) == n, 'Duplicate keys are not allowed'
        assert n >= n_pivots
        self._pivots = self._elements[:n_pivots]
        self._pivot_set = set(self._pivots)
        for k, v in keys_values:
            self._distance_cache.insert(k, v)
        if n_processes == 1:
            with ProgressBar(n*n_pivots, title=f'Bulk-loading {n} elements, {n_pivots} pivots') as bar:
                for pivot in self._pivots:
                    for elem in self._elements:
                        self.get_distance(pivot, elem)
                        bar.step()
        else:
            with Pool(n_processes) as pool:
                pairs = [(pivot, elem) for pivot in self._pivots for elem in self._elements]
                self._distance_cache.calculate_distances(pairs, pool=pool, with_progress_bar=True)
        self._distance_array = np.empty((n_pivots, n))
        for ip, pivot in enumerate(self._pivots):
            for ie, elem in enumerate(self._elements):
                self._distance_array[ip, ie] = self.get_distance(pivot, elem)  
    
    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._elements)

    def __contains__(self, element) -> bool:
        '''Decide if element is present in the tree'''
        return element in self._elements

    def _get_distance_to_value(self, key1: K, value2: V) -> float:
        return self._distance_cache.get_distance_to_value(key1, value2)

    def get_distance(self, key1: K, key2: K) -> float:
        return self._distance_cache.get_distance(key1, key2)

    def get_statistics(self):
        n_elements = len(self._elements)
        result = f'''LAESA with {len(self._pivots)} pivots statistics:
        Elements: {n_elements}
        {self._distance_cache.get_statistics()}'''
        return result
    
    def kNN_query_by_value(self, query_value: V, k: int, distance_low_bounds: List[Callable[[V, V], float]] = [], the_range: float = math.inf) -> List[Tuple[float, K]]:
        besties = MinFinder[K](n=k)
        query_dist = FunctionCache[K, float](lambda key: self._get_distance_to_value(key, query_value))
        current_range = min(besties.top_size(), the_range)
        min_dists = {key: 0.0 for key in self._elements}
        candidates = set(self._elements)
        for pivot in self._pivots:
            d_p_q = query_dist[pivot]
            current_range = min(besties.bubble_in(d_p_q, pivot), the_range)
            candidates.discard(pivot)
            to_discard = []
            for elem in candidates:
                assert self._distance_cache.is_calculated(pivot, elem)
                d_p_k = self.get_distance(pivot, elem)
                dmin_e_q = abs(d_p_q - d_p_k)
                min_dists[elem] = max(min_dists[elem], dmin_e_q)
                if min_dists[elem] > current_range:
                    to_discard.append(elem)
            for elem in to_discard:
                candidates.discard(elem)
        sorted_keys = sorted((min_dists[elem], elem) for elem in candidates)
        for dmin, elem in sorted_keys:
            if elem in candidates and dmin < current_range:
                d_e_q = query_dist[elem]
                current_range = min(besties.bubble_in(d_e_q, elem), the_range)
        result = besties.pop_all_not_none()
        return result

    def kNN_query_by_value3(self, query_value: V, k: int, distance_low_bounds: List[Callable[[V, V], float]] = [], the_range: float = math.inf) -> List[Tuple[float, K]]:
        query_dist = FunctionCache[K, float](lambda key: self._get_distance_to_value(key, query_value))
        d_q_p = np.array([query_dist[pivot] for pivot in self._pivots])
        d_q_p_e_low = np.abs(d_q_p.reshape((-1, 1)) - self._distance_array)
        d_q_e_low = np.max(d_q_p_e_low, axis=0)
        
        besties = MinFinder[K](n=k)
        for ip, pivot in enumerate(self._pivots):
            current_range = min(besties.bubble_in(d_q_p[ip], pivot), the_range)
        
        best_order: np.ndarray[int] = np.argsort(d_q_e_low)  # type: ignore
        for ie in best_order:
            if d_q_e_low[ie] > current_range*TOLERANCE_RATIO:
                break
            elem = self._elements[ie]
            if elem not in self._pivot_set:
                d_q_e = query_dist[elem]
                # dx = max(abs(query_dist[p] - self.get_distance(p, elem)) for p in self._pivots)
                # assert d_q_e_low[ie] == dx, f'{d_q_e_low[ie]} == {dx}'
                # if not d_q_e_low[ie] <= d_q_e:
                #     print(f'{d_q_e_low[ie]} <= {d_q_e}')
                current_range = min(besties.bubble_in(d_q_e, elem), the_range)
        result = besties.pop_all_not_none()
        return result

    def _can_be_under(self, elem: K, query_distance: FunctionCache[K, float], through: Iterable[K], limit: float) -> float:
        '''Decide if the distance between elem and query can be less than limit, using elements from through as pivots.'''
        # d_p_q = query_distance[node.pivot]
        for t in through:
            if t in query_distance:
                d_e_t = self.get_distance(elem, t)
                d_t_q = query_distance[t]
                d_e_q_low = abs(d_e_t - d_t_q)
                # assert d_p_q_low <= d_p_q, f'{d_p_q_low} <= {d_p_q}'
                if d_e_q_low >= limit:
                    return False
        return True
    