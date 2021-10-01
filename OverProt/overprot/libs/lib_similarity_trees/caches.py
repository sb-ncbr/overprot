import math
import heapq
from multiprocessing import Pool
from typing import NamedTuple, Generic, Dict, Callable, Mapping, Iterator, Iterable, Sized, Container, Optional, List, Tuple

from .. import lib
from .abstract_similarity_tree import K, V, X, Y

class FunctionCache(Generic[X, Y], Mapping[X, Y]):
    def __init__(self, function: Callable[[X], Y]) -> None:
        self._function: Callable[[X], Y] = function
        self._cache: Dict[X, Y] = {}

    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._cache)

    def __contains__(self, x: X) -> bool:  # type: ignore
        '''Decide if element is present here'''
        return x in self._cache

    def __getitem__(self, x: X) -> Y:
        '''Get f(x)'''
        if x not in self._cache:
            self._cache[x] = self._function(x)
        return self._cache[x]

    def __iter__(self) -> Iterator[X]:
        return iter(self._cache.keys())


class DistanceCache(Generic[K, V], Sized, Container):
    def __init__(self, distance_function: Callable[[V, V], float]):
        self._distance_function: Callable[[V, V], float] = distance_function
        self._elements: Dict[K, V] = {}  # mapping of element keys to values (i.e. objects on which distance function is defined)
        self._cache: Dict[K, Dict[K, float]] = {}  # 
        self._calculated_distances_counter = 0  # really calculated distances
        self._worst_calculated_distances_counter = 0  # theoretical number of calculated distances in worst case (i.e. each-to-each)

    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._elements)

    def __contains__(self, element) -> bool:
        '''Decide if element is present here'''
        return element in self._elements

    def insert(self, key: K, value: V) -> None:
        '''Insert a new element with given key and value. The key must be unique.'''
        assert key not in self, f'Key {key} is already here'
        self._worst_calculated_distances_counter += len(self._elements)
        self._elements[key] = value
        self._cache[key] = {}
    
    def delete(self, key: K) -> None:
        assert key in self, f'Key {key} is not here'    
        self._elements.pop(key)
        dists = self._cache.pop(key)
        for other in dists.keys():
            self._cache[other].pop(key)

    def get_distance(self, key1: K, key2: K) -> float:
        '''Get the distance between two objects present in the tree'''
        if key1 == key2:
            return 0.0
        else:
            assert key1 in self, f'Key {key1} not here'
            assert key2 in self, f'Key {key2} not here'
            try: 
                return self._cache[key1][key2]
            except KeyError:
                dist = self._distance_function(self._elements[key1], self._elements[key2])
                self._cache[key1][key2] = dist
                self._cache[key2][key1] = dist
                self._calculated_distances_counter += 1
                return dist

    def set_distance(self, key1: K, key2: K, distance: float) -> None:
        '''Set the distance between two objects present in the tree, without calculating it.'''
        assert key1 in self, f'Key {key1} not here'
        assert key2 in self, f'Key {key2} not here'
        if not self.is_calculated(key1, key2):
            self._calculated_distances_counter += 1
        self._cache[key1][key2] = distance
        self._cache[key2][key1] = distance

    def get_distance_to_value(self, key1: K, value2: V) -> float:
        '''Get the distance between two objects present in the tree'''
        assert key1 in self, f'Key {key1} not here'
        dist = self._distance_function(self._elements[key1], value2)
        self._calculated_distances_counter += 1
        return dist

    def is_calculated(self, key1: K, key2: K) -> bool:
        return key1 == key2 or key2 in self._cache[key1]

    def can_be_under(self, key1: K, key2: K, through: K, r: float, force_calculations: bool = False) -> bool:
        '''Decide if d(key1, key2) can be less than r, using through as auxiliary point for distance approximation.
        If force_calculations==True, proceed even if d(key1, through) or d(key2, through) are not calculated yet.'''
        # if force_calculations:
        #     assert self._is_calculated(key1, through) and self._is_calculated(key2, through)
        if force_calculations or self.is_calculated(key1, through) and self.is_calculated(key2, through):
            d_1_t = self.get_distance(key1, through)
            d_2_t = self.get_distance(key2, through)
            dmin_1_2 = abs(d_1_t - d_2_t)
            return dmin_1_2 < r
        else:
            return True

    def get_statistics(self) -> str:
        '''Return string with basic statistics about the cache'''
        n_elements = len(self._elements)
        n_distance_cache_entries = sum(len(dic) for dic in self._cache.values())
        if self._worst_calculated_distances_counter > 0:
            percent_calculated_distances = self._calculated_distances_counter / self._worst_calculated_distances_counter * 100
        else: 
            percent_calculated_distances = 100
        result = f'''Distance cache statistics:
        Elements: {n_elements}, Entries in distance cache: {n_distance_cache_entries}
        Calculated distances: {self._calculated_distances_counter} / {self._worst_calculated_distances_counter} ({percent_calculated_distances:#.3g}%) '''
        return result
    
    def _is_under(self, key1: K, key2: K, r: float) -> bool:
        '''Decide if distance(key1, key2) <= r.
        Try to use all already-computed distances to avoid new computation.'''
        if key1 == key2:
            return 0.0 < r
        cache1 = self._cache[key1]
        if key2 in cache1:
            return cache1[key2] < r
        cache2 = self._cache[key2]
        if len(cache1) > len(cache2):
            cache1, cache2 = cache2, cache1
        for p, d_p_k1 in cache1.items():
            if p in cache2:
                d_p_k2 = cache2[p]
                if r <= abs(d_p_k1 - d_p_k2):  # <= d(key1, key2)
                    return False
        return self.get_distance(key1, key2) < r

    def json(self) -> object:
        result_dict = {}
        for key1, cache in self._cache.items():
            for key2, distance in cache.items():
                kmin = min(key1, key2)  # type: ignore
                kmax = max(key1, key2)  # type: ignore
                result_dict[(kmin, kmax)] = distance
        result = [(key1, key2, distance) for (key1, key2), distance in result_dict.items()]
        return result

    def calculate_distances(self, pairs: List[Tuple[K, K]], pool: Optional[Pool] = None, overwrite: bool = False, with_progress_bar: bool = False) -> None:
        # print(f'Calculating {len(pairs)} in {pool}')
        jobs = []
        index = {}
        for i, (key1, key2) in enumerate(pairs):
            if not self.is_calculated(key1, key2) or overwrite:
                value1 = self._elements[key1]
                value2 = self._elements[key2]
                name = str(i)
                index[name] = (key1, key2)
                job = lib.Job(name=name, func=self._distance_function, args=(value1, value2), kwargs={}, stdout=None, stderr=None)
                jobs.append(job)
        job_results = lib.run_jobs_with_multiprocessing(jobs, n_processes=1, pool=pool, progress_bar=with_progress_bar)
        for result in job_results:
            key1, key2 = index[result.job.name]
            value1 = self._elements[key1]
            value2 = self._elements[key2]
            distance = result.result
            self.set_distance(key1, key2, distance)


class Range(NamedTuple):
    low: float
    high: float


class DistanceCacheWithRanges(DistanceCache[K, V]):
    def __init__(self, distance_function: Callable[[V, V], float]):
        super().__init__(distance_function)
        self._range_cache: Dict[K, Dict[K, Range]] = {}
    
    def insert(self, key: K, value: V) -> None:
        '''Insert a new element with given key and value. The key must be unique.'''
        super().insert(key, value)
        self._range_cache[key] = {}
    
    def delete(self, key: K) -> None:
        ranges = self._range_cache.pop(key)
        for other in ranges.keys():
            self._range_cache[other].pop(key)

    def get_distance(self, key1: K, key2: K) -> float:
        '''Get the distance between two objects present in the tree'''
        new = not self.is_calculated(key1, key2)
        distance = super().get_distance(key1, key2)
        if new:
            self._range_cache[key1][key2] = self._range_cache[key2][key1] = Range(distance, distance)
        return distance

    def get_range(self, key1: K, key2: K) -> Range:
        '''Get the distance range between two objects present in the tree'''
        assert key1 in self, f'Key {key1} not here'
        assert key2 in self, f'Key {key2} not here'
        if key1 == key2:
            return Range(0.0, 0.0)
        try: 
            return self._range_cache[key1][key2]
        except KeyError:
            return Range(0.0, math.inf)

    def can_be_under(self, p: K, q: K, through: K, r: float, force_calculations: bool = False) -> bool:
        '''Decide if d(p, q) can be less than r, using through as auxiliary point for distance approximation.
        If force_calculations==True, proceed even if d(p, through) or d(q, through) are not calculated yet.'''
        # if force_calculations:
        #     assert self._is_calculated(key1, through) and self._is_calculated(key2, through)
        if force_calculations:
            d_pt = self.get_distance(p, through)
            d_qt = self.get_distance(q, through)
        dmin, dmax = self.approximate_distance_range(p, q, through)
        if dmin >= r:
            return False
        # if force_calculations:
        #     d_pt = self.get_distance(p, through)
        #     d_qt = self.get_distance(q, through)
        #     dmin, dmax = self.approximate_distance_range(p, q, through)
        #     # dmin_pq = abs(d_pt - d_qt)
        #     return dmin < r
        return True

    def approximate_distance_range(self, p: K, q: K, through: K) -> Range:
        low_pq, high_pq = self.get_range(p, q)
        low_pt, high_pt = self.get_range(p, through)
        low_qt, high_qt = self.get_range(q, through)
        # print(f'd_{p}_{q} ({through}): {low_pq:.2f}-{high_pq:.2f} => ', end='')
        low_pq = max(low_pq, low_pt - high_qt, low_qt - high_pt)
        high_pq = min(high_pq, high_pt + high_qt)
        # print(f'{low_pq:.2f}-{high_pq:.2f}')
        range_pq = Range(low_pq, high_pq)
        self._range_cache[p][q] = self._range_cache[q][p] = range_pq
        return range_pq

    def get_statistics(self) -> str:
        '''Return string with basic statistics about the cache'''
        n_elements = len(self._elements)
        n_distance_cache_entries = sum(len(dic) for dic in self._cache.values())
        n_range_cache_entries = sum(len(dic) for dic in self._range_cache.values())
        if self._worst_calculated_distances_counter > 0:
            percent_calculated_distances = self._calculated_distances_counter / self._worst_calculated_distances_counter * 100
        else: 
            percent_calculated_distances = 100
        result = f'''Distance cache statistics:
        Elements: {n_elements}, Entries in distance cache: {n_distance_cache_entries}, Entries in range cache: {n_range_cache_entries}
        Calculated distances: {self._calculated_distances_counter} / {self._worst_calculated_distances_counter} ({percent_calculated_distances:#.3g}%) '''
        return result
   

class MinFinder(Generic[V]):
    '''This class serves for maintaining a set of n smallest elements of type V (their size is a float).
    It is implemented using a max heap.'''
    def __init__(self, keys_elements: Optional[Iterable[Tuple[float, Optional[V]]]] = None, n: Optional[int] = None) -> None:
        if keys_elements is None:
            n = n or 0
            keys_elements = [(math.inf, None) for i in range(n)]
        self._heap = [(-k, -i, v) for i, (k, v) in enumerate(keys_elements)]
        self._seq = -len(self._heap)
        heapq.heapify(self._heap)
        if n is None:
            n = len(self)
        else:
            if len(self) < n:
                raise ValueError('len(keys_elements) must be at least size')
            while len(self) > n:
                heapq.heappop(self._heap)
    def __len__(self) -> int:
        '''Return current number of elements.'''
        return len(self._heap)
    def top(self) -> Tuple[float, Optional[V]]:
        '''Return the size of the max element and the element itself.'''
        if len(self) == 0:
            raise ValueError
        k_neg, seq, v = self._heap[0]
        return -k_neg, v
    def top_size(self) -> float:
        '''Return the size of the max element.'''
        size, elem = self.top()
        return size
    def bubble_in(self, key: float, element: V) -> float:
        '''Iff key is smaller than current top, insert element and remove current top; otherwise do nothing.
        Return the new top size.'''
        if len(self) == 0:
            return
        k_top_neg, _, _ = self._heap[0]
        k_neg = -key
        if k_neg > k_top_neg:
            heapq.heapreplace(self._heap, (k_neg, self._seq, element))
            self._seq -= 1
        return self.top_size()
    def pop_all(self) -> List[Tuple[float, Optional[V]]]:
        '''Remove and return all elements (min to max).'''
        result = []
        while len(self) > 0:
            k_neg, seq, value = heapq.heappop(self._heap)
            result.append((-k_neg, value))
        result.reverse()
        return result
    def pop_all_not_none(self, raise_on_none: bool = True) -> List[Tuple[float, V]]:
        '''Remove and return all elements (min to max) that are not None. If raise_on_none, then raise if any element is None.'''
        result = []
        while len(self) > 0:
            k_neg, seq, value = heapq.heappop(self._heap)
            if value is None:
                if raise_on_none:
                    raise ValueError('None encountered')
                else:
                    continue
            result.append((-k_neg, value))
        result.reverse()
        return result
