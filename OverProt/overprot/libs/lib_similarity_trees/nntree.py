'''NNT, Nearest Neighbor Tree
A tree structure useful for popping the two nearest elements from the tree.
The tree has simple invariants:
- Each node holds one element and can have unlimited number of children.
- The oldest element is in the root.
- Children are younger than their parent.
- Each element (except for the root) is a child of its nearest older neighbor.
'''

from collections import defaultdict
from typing import NamedTuple, Generic, TypeVar, List, Tuple, Dict, Set, Union, Optional, Callable, Iterable, Iterator, Any, Counter, ClassVar, Sized, Container
from dataclasses import dataclass, field
import itertools
import math

from .. import lib
from .abstract_similarity_tree import AbstractSimilarityTree

K = TypeVar('K')  # Type of keys
V = TypeVar('V')  # Type of values


GNAT_RANGES = False


@dataclass(order=True)
class _Node():
    pivot: int
    rc: float = 0.0
    parent: Optional['_Node'] = field(default=None, repr=False)
    children: List['_Node'] = field(default_factory=list, repr=False)
    gnat_ranges: Dict[int, Tuple[float, float]] = field(default_factory=dict, repr=False)  # node_i.gnat_ranges[j] = (r_low(p_i, S_j), r_high(p_i, S_j))

    def __eq__(self, other: object) -> bool:
        if type(self) != type(other):
            return False
        return self.pivot == other.pivot  # type: ignore
    
    def __str__(self) -> str:
        return f'_Node(pivot={self.pivot}, rc={self.rc}, n_children={len(self.children)})'


class _NNTree(Generic[V]):
    def __init__(self, distance_function: Callable[[V, V], float], with_nearest_pair_queue: bool = False, **options):
        self._counter = 0
        self._elements: Set[int] = set()
        self._home_nodes: Dict[int, _Node] = {}
        self._root: Optional[_Node] = None
        self._distance_cache: DistanceCache[int, V] = DistanceCache(distance_function)
        self._nearest_pair_queue: Optional[lib.PriorityQueue[float, Tuple[int, int]]] = lib.PriorityQueue() if with_nearest_pair_queue else None
        self._options = {opt for opt, on in options.items() if on}

    def get_distance(self, key1: int, key2: int) -> float:
        return self._distance_cache.get_distance(key1, key2)

    def _is_calculated(self, key1: int, key2: int) -> float:
        return self._distance_cache.is_calculated(key1, key2)

    def _str_lines(self, node: Optional[_Node], indent: int) -> Iterator[str]:
        if node is None:
            yield 'Empty tree'
        else:
            yield '    ' * indent + str(node)
            for child in node.children:
                yield from self._str_lines(child, indent + 1)
        
    def __str__(self):
        return '\n'.join(self._str_lines(self._root, 0))

    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._elements)

    def __contains__(self, element) -> bool:
        '''Decide if element is present here'''
        return element in self._elements

    def json(self) -> Optional[dict]:
        if self._root is None:
            return None
        else:
            return {str(self._root): self._json_subtree(self._root)}

    def _json_subtree(self, node: _Node) -> dict:
        return {str(child): self._json_subtree(child) for child in node.children}
    
    def _nearest_older_node(self, query: int, root: Optional[_Node], initial_guess: Optional[_Node] = None) -> Optional[_Node]:
        PIVOT_PIVOT_CONSTRAINT = True
        PIVOT_GUESS_CONSTRAINT = True
        PIVOT_SIBLING_CONSTRAINT = True
        USE_IS_UNDER = False
        if root is None:
            return None
        assert query > root.pivot
        d_pq = self.get_distance(root.pivot, query)
        dmin = max(0, d_pq - root.rc)
        current_nn, current_range = root, d_pq
        if initial_guess is not None:
            d_gq = self.get_distance(initial_guess.pivot, query)
            if d_gq < d_pq:
                current_nn, current_range = initial_guess, d_gq
        queue = lib.PriorityQueue[float, _Node]([(dmin, root)])
        while not queue.is_empty():
            best = queue.pop_min()
            assert best is not None
            dmin, node = best
            if dmin >= current_range:
                break
            d_pq = self.get_distance(node.pivot, query)
            for child in node.children:
                if child.pivot > query:
                    break
                if self._is_calculated(child.pivot, query):
                    if self.get_distance(child.pivot, query) >= current_range + child.rc:
                        continue
                else:
                    if PIVOT_GUESS_CONSTRAINT and not self._can_be_under(child.pivot, query, current_nn.pivot, current_range + child.rc):
                        continue
                    if PIVOT_SIBLING_CONSTRAINT and not self._pivot_sibling_constraints(query, child, current_range):
                        continue
                    if PIVOT_PIVOT_CONSTRAINT and not self._pivot_pivot_constraints(query, child, current_range):
                        continue
                    if GNAT_RANGES and not self._gnat_constraints(query, child, current_range):
                        continue
                if USE_IS_UNDER:
                    if self._distance_cache._is_under(child.pivot, query, child.rc + current_range):
                        d_cq = self.get_distance(child.pivot, query)
                        if d_cq < current_range:
                            current_nn, current_range = child, d_cq
                        dmin = max(0, d_cq - child.rc)
                        queue.add(dmin, child)
                else:
                    d_cq = self.get_distance(child.pivot, query)
                    if d_cq < current_range:
                        current_nn, current_range = child, d_cq
                    dmin = max(0, d_cq - child.rc)
                    if dmin < current_range:
                        queue.add(dmin, child)
        return current_nn
    
    def _distance_upper_bound(self, i: int, j: int, throughs: Iterable[int]) -> float:
        if self._is_calculated(i, j):
            return self.get_distance(i, j)
        bound = math.inf
        for t in throughs:
            if self._is_calculated(i, t) and self._is_calculated(t, j):
                bound = min(bound, self.get_distance(i, t) + self.get_distance(t, j))
        return bound

    def _gen_ancestors(self, node: _Node, include_self: bool = False) -> Iterator[_Node]:
        ancestor = node if include_self else node.parent
        while ancestor is not None:
            yield ancestor
            ancestor = ancestor.parent            

    def _can_be_under(self, key1: int, key2: int, through: int, r: float, force_calculations: bool = False) -> bool:
        '''Decide if d(key1, key2) can be less than r, using through as auxiliary point for distance approximation.
        If force_calculations==True, proceed even if d(key1, through) or d(key2, through) are not calculated yet.'''
        # if force_calculations:
        #     assert self._is_calculated(key1, through) and self._is_calculated(key2, through)
        return self._distance_cache.can_be_under(key1, key2, through, r, force_calculations=force_calculations)
        # if force_calculations or self._is_calculated(key1, through) and self._is_calculated(key2, through):
        #     d_1_t = self.get_distance(key1, through)
        #     d_2_t = self.get_distance(key2, through)
        #     dmin_1_2 = abs(d_1_t - d_2_t)
        #     return dmin_1_2 < r
        # else:
        #     return True

    def _pivot_pivot_constraints(self, query: int, node: _Node, range_: float) -> bool:
        for ancestor in self._gen_ancestors(node):
            if not self._can_be_under(node.pivot, query, ancestor.pivot, range_ + node.rc, force_calculations=True):
                return False
            # return True  # non-recursive
        return True

    def _pivot_sibling_constraints(self, query: int, node: _Node, range_: float) -> bool:
        if node.parent is not None:
            siblings = node.parent.children
            for sibling in siblings:
                if 'ps_all' in self._options:
                    if sibling.pivot == query:
                        continue
                else:
                    if sibling.pivot >= query:
                        break
                if not self._can_be_under(node.pivot, query, sibling.pivot, range_ + node.rc, force_calculations=False):
                    return False
        return True
    
    def _gnat_constraints(self, query: int, node: _Node, range_: float) -> bool:
        assert node.parent is not None
        for sibling in node.parent.children:
            if sibling != node and self._is_calculated(sibling.pivot, query):
                d_sq = self.get_distance(sibling.pivot, query)
                dlow_so, dhigh_so = sibling.gnat_ranges[node.pivot]
                if d_sq >= dhigh_so + range_ or d_sq <= dlow_so - range_:
                    return False
        return True

    def insert(self, value: V, nearest_element_guess: Optional[int] = None) -> int:
        key = self._counter
        self._counter += 1
        self._elements.add(key)
        self._distance_cache.insert(key, value)
        new_node = _Node(pivot=key)
        self._home_nodes[key] = new_node
        nearest_node_guess: Optional[_Node] = self._home_nodes.get(nearest_element_guess, None)  # type: ignore
        parent = self._nearest_older_node(key, self._root, initial_guess=nearest_node_guess)
        self._link(parent, new_node)
        return key

    def _link(self, parent: Optional[_Node], child: _Node) -> None:
        if parent is None:
            child.parent = None
            self._root = child
        else:
            child.parent = parent
            if GNAT_RANGES:
                for sib in parent.children:
                    child.gnat_ranges[sib.pivot] = self._appr_gnat_range(child.pivot, sib)
                anc = child
            self._bubble_into_sorted_list(parent.children, child)
            if self._nearest_pair_queue is not None:
                self._nearest_pair_queue.add(self.get_distance(parent.pivot, child.pivot), (parent.pivot, child.pivot))
            while parent is not None:
                new_min_rc = self.get_distance(parent.pivot, child.pivot) + child.rc
                parent.rc = max(parent.rc, new_min_rc)
                if GNAT_RANGES:
                    siblings = [c for c in parent.children if c != anc]
                    for sib in siblings:
                        d_sc = self.get_distance(sib.pivot, child.pivot)
                        if anc.pivot not in sib.gnat_ranges:
                            assert anc == child
                            sib.gnat_ranges[anc.pivot] = (d_sc - child.rc, d_sc + child.rc)
                        else:
                            assert anc != child
                            low, high = sib.gnat_ranges[anc.pivot]
                            sib.gnat_ranges[anc.pivot] = (min(low, d_sc - child.rc), max(high, d_sc + child.rc))
                    anc = parent
                parent = parent.parent
        
    def _appr_gnat_range(self, pivot: int, node: _Node) -> Tuple[float, float]:
        d_pn = self.get_distance(pivot, node.pivot)
        low = d_pn
        high = d_pn
        for child in node.children:
            d_pc = self.get_distance(pivot, child.pivot)
            low = min(low, d_pc - child.rc)
            high = max(high, d_pc + child.rc)
        return low, high

    def _check_invariants(self):
        for key in self._elements:
            older_nn = min((other for other in self._elements if other < key), key = lambda k: self.get_distance(key, k), default=None)
            parent = self._home_nodes[key].parent
            parent_key = parent.pivot if parent is not None else None
            assert older_nn == parent_key, f'{older_nn}, {parent_key}'
        self._check_if_elements_present(self._root)
    
    def _check_if_elements_present(self, node: _Node):
        if node is not None:
            assert node.pivot in self
            for child in node.children:
                self._check_if_elements_present(child)

    def delete(self, key: int) -> None:
        assert key in self
        self._elements.remove(key)
        node = self._home_nodes.pop(key)
        self._distance_cache.delete(key)

        parent = node.parent
        if parent is None:
            self._root = None
        else:
            parent.children.remove(node)
        self._upper_bound_rcs(parent)
        for child in node.children:
            new_parent = self._nearest_older_node(child.pivot, self._root, initial_guess=parent)
            self._link(new_parent, child)

    def delete_parent_and_child(self, parent_key: int, child_key: int) -> None:
        assert parent_key in self
        assert child_key in self
        self._elements.remove(parent_key)
        self._elements.remove(child_key)
        parent_node = self._home_nodes.pop(parent_key)
        child_node = self._home_nodes.pop(child_key)
        self._distance_cache.delete(parent_key)
        self._distance_cache.delete(child_key)

        grandparent = parent_node.parent
        if grandparent is None:
            self._root = None
        else:
            grandparent.children.remove(parent_node)
        parent_node.children.remove(child_node)
        orphans = sorted(parent_node.children + child_node.children)
        self._upper_bound_rcs(grandparent)
        for orphan in orphans:
            new_parent = self._nearest_older_node(orphan.pivot, self._root, initial_guess=grandparent)
            self._link(new_parent, orphan)

    def _upper_bound_rcs(self, node: Optional[_Node]) -> None:
        '''Decrease covering radii if excessive in and all its ancestors'''
        if node is not None:
            rc_upper_bound = max((self.get_distance(node.pivot, child.pivot) + child.rc for child in node.children), default=0.0)
            node.rc = min(node.rc, rc_upper_bound)
            self._upper_bound_rcs(node.parent)
    
    def pop_nearest_pair(self) -> Tuple[int, int, float]:
        if self._nearest_pair_queue is None:
            raise ValueError('Nearest pair queue not initialized. Use with_nearest_pair_queue=True in initializer.')
        if len(self) < 2:
            raise ValueError('This tree contains less than 2 elements.')
        popped = self._nearest_pair_queue.pop_min_which(self._contains_pair)
        assert popped is not None
        dist, (i, j) = popped
        # self.delete(j)  # delete younger first!
        # self.delete(i)
        self.delete_parent_and_child(i, j)
        return i, j, dist

    def _contains_pair(self, pair: Tuple[int, int]) -> bool:
        return pair[0] in self and pair[1] in self

    def get_statistics(self) -> str:
        return self._distance_cache.get_statistics()

    @staticmethod
    def _bubble_into_sorted_list(lst: List[Any], new_elem: Any) -> None:
        '''Add new_elem to a sorted list lst and keep it sorted.'''
        lst.append(new_elem)
        for i in range(len(lst)-1, 0, -1):
            if lst[i-1] > lst[i]:
                lst[i], lst[i-1] = lst[i-1], lst[i]
            else:
                break


class NNTree(Generic[K, V], AbstractSimilarityTree[K, V]):
    def __init__(self, distance_function: Callable[[V, V], float], with_nearest_pair_queue: bool = False, **kwargs):
        self._tree = _NNTree(distance_function, with_nearest_pair_queue=with_nearest_pair_queue, **kwargs)
        self._key2index: Dict[K, int] = {}
        self._index2key: Dict[int, K] = {}

    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._tree)

    def __contains__(self, element) -> bool:
        '''Decide if element is present here'''
        return element in self._tree

    def insert(self, key: K, value: V, nearest_element_guess: Optional[K] = None) -> None:
        '''Insert a new element with given key and value into the tree. The key must be unique.'''
        nearest_index_guess: Optional[int] = self._key2index.get(nearest_element_guess, None)  # type: ignore
        index = self._tree.insert(value, nearest_element_guess=nearest_index_guess)
        self._key2index[key] = index
        self._index2key[index] = key

    def delete(self, key: K) -> None:
        '''Delete an existing element from the tree.'''
        index = self._key2index.pop(key)
        self._index2key.pop(index)
        self._tree.delete(index)
    
    def kNN_query(self, query: K, k: int, include_query: bool = False) -> List[K]:
        '''Perform classical kNN query, return the k nearest neighbors to query (including itself iff include_query==True). query must be present in the tree.'''
        raise NotImplementedError

    # def nearest_older_neighbor(self, key: K) -> Optional[K]:
    #     nn_index = self._tree.nearest_older_neighbor(self._key2index[key])
    #     return self._index2key[nn_index] if nn_index is not None else None
    
    def get_distance(self, key1: K, key2: K) -> float:
        '''Get the distance between two elements present in the tree'''
        return self._tree.get_distance(self._key2index[key1], self._key2index[key2])

    def get_statistics(self) -> str:
        return self._tree.get_statistics()

    def pop_nearest_pair(self) -> Tuple[K, K, float]:
        idx_i, idx_j, dist = self._tree.pop_nearest_pair()
        key_i = self._index2key.pop(idx_i)
        key_j = self._index2key.pop(idx_j)
        self._key2index.pop(key_i)
        self._key2index.pop(key_j)
        return key_i, key_j, dist
 
    def _check_invariants(self):
        self._tree._check_invariants()

class _MagicNNTree(Generic[K, V], AbstractSimilarityTree[K, V]):
    def __init__(self, *args, **kwargs):
        self._main_tree = NNTree(*args, **kwargs)
        self._helping_tree = NNTree(*args, **kwargs)

    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._main_tree)

    def __contains__(self, element) -> bool:
        '''Decide if element is present here'''
        return element in self._main_tree

    def insert(self, key: K, value: V, nearest_element_guess: Optional[K] = None) -> None:
        self._helping_tree.insert(key, value, nearest_element_guess)
        idx = self._helping_tree._key2index[key]
        parent_node = self._helping_tree._tree._home_nodes[idx].parent
        parent = self._helping_tree._index2key[parent_node.pivot] if parent_node is not None else nearest_element_guess
        self._main_tree.insert(key, value, nearest_element_guess=parent)

    def delete(self, key: K) -> None:
        self._main_tree.delete(key)
        self._helping_tree.delete(key)

    def pop_nearest_pair(self) -> Tuple[K, K, float]:
        self._helping_tree.pop_nearest_pair()
        return self._main_tree.pop_nearest_pair()

    def get_statistics(self) -> str:
        main_stats = self._main_tree.get_statistics()
        helper_stats = self._helping_tree.get_statistics()
        main_calcs = self._main_tree._tree._distance_cache._calculated_distances_counter
        helper_calcs = self._helping_tree._tree._distance_cache._calculated_distances_counter
        saving = (helper_calcs - main_calcs) / helper_calcs
        return f'Main tree:\n{main_stats}\nHelper tree:\n{helper_stats}\nSaving: {100*saving:.0f}%'
    
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


class Range(NamedTuple):
    low: float
    high: float


class DistanceCacheWithRanges(Generic[K, V], DistanceCache[K, V]):
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
   