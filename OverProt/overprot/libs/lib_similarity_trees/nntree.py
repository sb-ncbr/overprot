'''NNT, Nearest Neighbor Tree
A tree structure useful for popping the two nearest elements from the tree.
The tree has simple invariants:
- Each node holds one element and can have unlimited number of children.
- The oldest element is in the root.
- Children are younger than their parent.
- Each element (except for the root) is a child of its nearest older neighbor.
'''

from typing import Generic, TypeVar, List, Tuple, Dict, Set, Optional, Callable, Mapping, Iterable, Iterator, Any
from dataclasses import dataclass, field
import math

from .. import lib
from .abstract_similarity_tree import AbstractSimilarityTree
from .caches import DistanceCache, FunctionCache

K = TypeVar('K')  # Type of keys
V = TypeVar('V')  # Type of values


GNAT_RANGES = False
PIVOT_PIVOT_CONSTRAINT = True
PIVOT_GUESS_CONSTRAINT = True
PIVOT_SIBLING_CONSTRAINT = True
USE_IS_UNDER = False


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
    
    def kNN_query(self, query_value: V, k: int) -> List[Tuple[float, int]]:
        '''Perform classical kNN query, return the k nearest neighbors to query.'''
        assert len(self) >= k
        if k == 0:
            return []
        query_dist = FunctionCache[int, float](lambda key: self._distance_cache.get_distance_to_value(key, query_value))
        root = self._root
        assert root is not None
        d_rq = query_dist[root.pivot]
        dmin = max(0, d_rq - root.rc)
        NOTHING = -1
        worklist = k * [(math.inf, NOTHING)]  # sorted kNN with their distances from query
        worklist[0] = (d_rq, root.pivot)
        current_range = worklist[-1][0]
        queue = lib.PriorityQueue[float, _Node]()
        queue.add(dmin, root)  # TODO what if the queue is sorted by d_pq instead of dmin? (I might have tried it and it might have not worked)
        while not queue.is_empty():
            best = queue.pop_min()
            assert best is not None
            dmin, node = best
            if dmin >= current_range:
                break
            for child in node.children:
                if PIVOT_PIVOT_CONSTRAINT and not self._pivot_pivot_constraints2(child, query_dist, current_range):
                    continue
                if PIVOT_SIBLING_CONSTRAINT and not self._pivot_sibling_constraints2(child, query_dist, current_range):
                    continue
                if USE_IS_UNDER:
                    print('Warning: Not using USE_IS_UNDER')
                d_cq = query_dist[child.pivot]
                if d_cq < current_range:
                    self._bubble_into_sorted_list(worklist, (d_cq, child.pivot), keep_size=True)
                    current_range = worklist[-1][0]
                dmin = max(0, d_cq - child.rc)
                if dmin < current_range:
                    queue.add(dmin, child)
        result = [dk for dk in worklist if dk[1] != NOTHING]
        assert len(result) == k, f'Requested {k} nearest neighbors, found only {len(result)} ({query_value}, {worklist})'
        return result

    def _nearest_older_node(self, query: int, root: Optional[_Node], initial_guess: Optional[_Node] = None) -> Optional[_Node]:
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
            # if d_pq == 0.0:
            #     break
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
        return self._distance_cache.can_be_under(key1, key2, through, r, force_calculations=force_calculations)

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

    def _pivot_pivot_constraints2(self, node: _Node, query_distance: Mapping[int, float], range_: float) -> bool:
        for ancestor in self._gen_ancestors(node):
            assert self._is_calculated(node.pivot, ancestor.pivot)
            assert ancestor.pivot in query_distance
            d_nq_min = abs(self.get_distance(node.pivot, ancestor.pivot) - query_distance[ancestor.pivot])
            if d_nq_min > range_:
                return False
        return True

    def _pivot_sibling_constraints2(self, node: _Node, query_distance: Mapping[int, float], range_: float) -> bool:
        if node.parent is not None:
            siblings = node.parent.children
            for sibling in siblings:
                if sibling.pivot == node.pivot:
                    break
                if self._is_calculated(node.pivot, sibling.pivot) and sibling.pivot in query_distance:
                # if True:  # debug
                    d_nq_min = abs(self.get_distance(node.pivot, sibling.pivot) - query_distance[sibling.pivot])
                    if d_nq_min > range_:
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
    def _bubble_into_sorted_list(lst: List[Any], new_elem: Any, keep_size: bool = False) -> None:
        '''Add new_elem to a sorted list lst and keep it sorted. 
        If keep_size==True, then also remove the new maximum, thus keeping the same length.'''
        lst.append(new_elem)
        for i in range(len(lst)-1, 0, -1):
            if lst[i-1] > lst[i]:
                lst[i], lst[i-1] = lst[i-1], lst[i]
            else:
                break
        if keep_size:
            lst.pop()


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
    
    def kNN_query_by_value(self, query_value: V, k: int) -> List[Tuple[float, K]]:
        '''Perform classical kNN query, return the k nearest neighbors to query (including itself iff include_query==True). query must be present in the tree.'''
        result = self._tree.kNN_query(query_value, k)
        return [(dist, self._index2key[idx]) for dist, idx in result]

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

