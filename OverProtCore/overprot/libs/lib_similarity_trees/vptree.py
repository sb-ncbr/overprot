'''VPT, Vantage Point Tree'''

from __future__ import annotations
import math
import statistics
from typing import Generic, Collection, List, Tuple, Dict, Set, Union, Optional, Callable, Iterator, Sequence
from dataclasses import dataclass, field

from .abstract_similarity_tree import AbstractSimilarityTree, K, V
from .caches import FunctionCache, DistanceCache, MinFinder
from ..lib import PriorityQueue, ProgressBar


@dataclass
class _VPFork(Generic[K]):
    parent: Optional['_VPFork[K]'] = field(repr=False)
    # order_in_parent: OrderInParent  # 1 if this is the first child of the parent, 2 if this is the second child of the parent, 0 if the parent is root
    pivot: K
    subtree_in: '_VPNode[K]' = field(repr=False)
    subtree_out: '_VPNode[K]' = field(repr=False)
    r: float = 0.0
    rc: float = 0.0
    # TODO include also covering radius?
    # TODO covering radius for leaves?

@dataclass
class _VPLeaf(Generic[K]):
    parent: Optional['_VPFork[K]'] = field(repr=False)
    # order_in_parent: OrderInParent = ONLY_CHILD  # 1 if this is the first child of the parent, 2 if this is the second child of the parent, 0 if the parent is root
    elements: List[K] = field(default_factory=list)

_VPNode = Union[_VPFork[K], _VPLeaf[K]]


REUSE_PIVOTS = True


class VPTree(AbstractSimilarityTree[K, V]):
    _leaf_size: int
    # _distance_function: Callable[[V, V], float]
    _distance_cache: DistanceCache[K, V]
    _elements: Set[K]
    _home_leaves: Dict[K, _VPLeaf[K]]
    _root: _VPNode[K]

    def __init__(self, distance_function: Callable[[V, V], float], keys_values: Sequence[Tuple[K, V]] = (), leaf_size: int = 8) -> None:
        assert leaf_size >= 1
        self._leaf_size = leaf_size
        self._distance_function = distance_function
        self._distance_cache = DistanceCache(distance_function)
        self._elements = set()  # keys of elements currently present in the tree (pivot is not element)
        self._home_leaves = {}  # mapping of elements to their accommodating leaves
        self._root = _VPLeaf(parent=None)
        self._bulk_load(keys_values, with_progress_bar=True)
    
    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._elements)

    def __contains__(self, element) -> bool:
        '''Decide if element is present in the tree'''
        return element in self._elements

    def _bulk_load(self, keys_values: Sequence[Tuple[K, V]], with_progress_bar: bool = False) -> None:
        elements = [k for k, v in keys_values]
        n = len(elements)
        assert len(self) == 0, 'Bulk-load only works on an empty tree'
        assert len(set(elements)) == n, 'Duplicate keys are not allowed'
        if n == 0: 
            return
        self._elements.update(elements)
        for k, v in keys_values:
            self._distance_cache.insert(k, v)
        bar = ProgressBar(len(elements), title=f'Bulk-loading {n} elements into VPTree').start() if with_progress_bar else None
        self._root = self._create_node_from_bulk(elements, progress_bar=bar)
        if bar is not None:
            bar.finalize()
        print(f'Bulk-loaded {n} elements')
    
    def _create_node_from_bulk(self, elements: List[K], parent_pivot: Optional[K] = None, ancestor_pivots: Optional[List[K]] = None, progress_bar: Optional[ProgressBar] = None) -> _VPNode[K]:
        if len(elements) <= self._leaf_size:
            new_leaf = _VPLeaf(parent=None, elements=elements)
            for elem in elements:
                self._home_leaves[elem] = new_leaf
            if progress_bar is not None:
                progress_bar.step(len(elements))
            return new_leaf
        else:
            pivot = self._select_pivot1(elements, parent_pivot)
            # ancestor_pivots = ancestor_pivots or []
            # pivot = self._select_pivot2(elements, ancestor_pivots)
            # ancestor_pivots.append(pivot)
            elems_in, elems_out, r, r_max = self._partition(pivot, elements)
            subtree_in = self._create_node_from_bulk(elems_in, parent_pivot=pivot, ancestor_pivots=ancestor_pivots, progress_bar=progress_bar)
            subtree_out = self._create_node_from_bulk(elems_out, parent_pivot=pivot, ancestor_pivots=ancestor_pivots, progress_bar=progress_bar)
            # ancestor_pivots.pop()
            new_fork = _VPFork(parent=None, pivot=pivot, subtree_in=subtree_in, subtree_out=subtree_out, r=r, rc=r_max)
            subtree_in.parent = new_fork
            subtree_out.parent = new_fork
            return new_fork   

    def _select_pivot1(self, elements: List[K], parent_pivot: Optional[K]) -> K:
        if parent_pivot is None:
            parent_pivot = elements[0]
        d_max, pivot = max((self.get_distance(parent_pivot, k), k) for k in elements)
        return pivot

    def _select_pivot2(self, elements: List[K], ancestor_pivots: Collection[K]) -> K:
        if len(ancestor_pivots) == 0:
            ancestor_pivots = (elements[0],)
        d_max, pivot = max((min(self.get_distance(a, k) for a in ancestor_pivots), k) for k in elements)
        return pivot

    def _partition(self, pivot: K, elements: List[K]) -> Tuple[List[K], List[K], float, float]:
        elems_in = []
        elems_out = []
        distances = [self.get_distance(pivot, elem) for elem in elements]
        r = statistics.median(distances)
        r_max = max(distances)
        IN, OUT = 1, 2
        last_tie = OUT
        for elem, dist in zip(elements, distances):
            if dist < r:
                to = IN
            elif dist > r:
                to = OUT
            elif last_tie == IN:
                to = last_tie = OUT
            else:  # last_tie == OUT
                to = last_tie = IN
            if to == IN:
                elems_in.append(elem)
            else:  # to == OUT
                elems_out.append(elem)
        return elems_in, elems_out, r, r_max

    def get_distance(self, key1: K, key2: K) -> float:
        '''key1 can be element or pivot, key2 must be element'''
        return self._distance_cache.get_distance(key1, key2)

    def get_statistics(self):
        n_elements = len(self._elements)
        forks, leaves = self.get_forks_and_leaves()
        n_forks = len(forks)
        n_leaves = len(leaves)
        result = f'''VPT tree statistics:
        Elements: {n_elements}, Forks: {n_forks}, Leaves: {n_leaves}
        {self._distance_cache.get_statistics()}
        {self.leaf_diameters(leaves)}
        {self.leaf_distances(leaves)}'''
        return result
    
    def leaf_diameters(self, leaves: List[_VPLeaf]) -> str:
        diameters = []
        for leaf in leaves:
            n = len(leaf.elements)
            diameter = max((self.get_distance(leaf.elements[i], leaf.elements[j]) for i in range(n) for j in range(i)), default=0.0)
            diameters.append(diameter)
        return f'leaf diameters: {min(diameters):.3f}-{max(diameters):.3f}, mean: {statistics.mean(diameters):.3f} median: {statistics.median(diameters):.3f}'

    def leaf_distances(self, leaves: List[_VPLeaf]) -> str:
        dists: list[float] = []
        for leaf in leaves:
            n = len(leaf.elements)
            dists.extend(self.get_distance(leaf.elements[i], leaf.elements[j]) for i in range(n) for j in range(i))
        return f'in-leaf distances: {min(dists):.3f}-{max(dists):.3f}, mean: {statistics.mean(dists):.3f} median: {statistics.median(dists):.3f}'

    def get_forks_and_leaves(self) -> Tuple[List[_VPFork], List[_VPLeaf]]:
        forks: List[_VPFork[K]] = []
        leaves: List[_VPLeaf[K]] = []
        self._collect_nodes(self._root, forks, leaves)
        return forks, leaves

    def _collect_nodes(self, node: _VPNode[K], out_forks: List[_VPFork[K]], out_leaves: List[_VPLeaf[K]]):
        if isinstance(node, _VPLeaf):
            out_leaves.append(node)
        elif isinstance(node, _VPFork):
            out_forks.append(node)
            self._collect_nodes(node.subtree_in, out_forks, out_leaves)
            self._collect_nodes(node.subtree_out, out_forks, out_leaves)
        else:
            raise AssertionError

    def kNN_query_by_value(self, query_value: V, k: int, distance_low_bounds: List[Callable[[V, V], float]] = []) -> List[Tuple[float, K]]:
        besties = MinFinder[K](n=k)
        self._kNN_query_by_value(query_value, besties, distance_low_bounds=distance_low_bounds)
        result = besties.pop_all_not_none()
        return result

    def _kNN_query_by_value(self, query_value: V, besties: MinFinder[K], distance_low_bounds: List[Callable[[V, V], float]] = []) -> None:
        query_dist = FunctionCache[K, float](lambda key: self._distance_cache.get_distance_to_value(key, query_value))
        queue = PriorityQueue[float, _VPNode]()
        queue.add(0.0, self._root)
        QAE = False#True
        QAP = False
        LC = False
        n_leaves = 0
        while not queue.is_empty():
            best = queue.pop_min()
            assert best is not None
            dmin, node = best
            current_range, _ = besties.top()
            if dmin >= current_range:
                continue
            if isinstance(node, _VPLeaf):
                n_leaves += 1
                for elem in node.elements:           
                    if LC:
                        if 0.5 * abs(self._distance_cache._elements[elem].n - query_value.n) >= current_range:
                            continue
                    if QAE:
                        d_e_q_low, d_e_q_high = self._distance_bounds_of_element_from_ancestors(elem, query_dist)
                        if d_e_q_low >= current_range:
                            continue
                    elem_value = self._distance_cache._elements[elem]
                    if any(d(elem_value, query_value) >= current_range for d in distance_low_bounds):
                        continue
                    besties.bubble_in(query_dist[elem], elem)
            elif isinstance(node, _VPFork):
                if QAP:
                    d_p_q_low, d_p_q_high = self._distance_bounds_from_ancestors(node, query_dist)
                    # if query_dist[node.pivot] > 0.0:
                    #     print(d_p_q_low, query_dist[node.pivot], d_p_q_high, current_range, sep='\t')
                    can_be_in = d_p_q_low < node.r + current_range and query_dist[node.pivot] < node.r + current_range
                    can_be_out = d_p_q_high > node.r - current_range and query_dist[node.pivot] > node.r - current_range
                else:
                    d_p_q = query_dist[node.pivot]
                    can_be_in = d_p_q < node.r + current_range
                    can_be_out = d_p_q > node.r - current_range and d_p_q < node.rc + current_range
                if can_be_in:
                    dmin = max(dmin, query_dist[node.pivot] - node.r)
                    queue.add(dmin, node.subtree_in)
                if can_be_out:
                    dmin = max(dmin, node.r - query_dist[node.pivot], query_dist[node.pivot] - node.rc)
                    # using the same dmin variable as reassigned in if can_be_in. TODO correct!
                    queue.add(dmin, node.subtree_out)
            else:
                raise AssertionError('node must be one of: _VPFork, _VPLeaf')
        # print('n_leaves', n_leaves)

    def _gen_ancestors(self, node: _VPFork[K], include_self: bool = False) -> Iterator[K]:
        if include_self:
            yield node.pivot
        ancestor_node = node.parent
        while ancestor_node is not None:
            yield ancestor_node.pivot
            ancestor_node = ancestor_node.parent

    def _gen_ancestors_of_element(self, element: K) -> Iterator[K]:
        leaf = self._home_leaves[element]
        if leaf.parent is not None:
            yield from self._gen_ancestors(leaf.parent, include_self=True)

    def _distance_bounds_of_element_from_ancestors(self, element: K, query_distance: FunctionCache[K, float]) -> Tuple[float, float]:
        '''Return lower and upper bound for distance element-query, using all its ancestors 
        (it is expected that all ancestor-query and ancestor-element distances are already known, otherwise they will be calculated).''' 
        d_p_q_low, d_p_q_high = 0.0, math.inf
        for ancestor in self._gen_ancestors_of_element(element):
            d_p_a = self.get_distance(element, ancestor)
            d_a_q = query_distance[ancestor]
            d_p_q_low = max(d_p_q_low, abs(d_p_a - d_a_q))
            d_p_q_high = min(d_p_q_high, d_p_a + d_a_q)
        return d_p_q_low, d_p_q_high

    def _distance_bounds_from_ancestors(self, node: _VPFork[K], query_distance: FunctionCache[K, float]) -> Tuple[float, float]:
        '''Return lower and upper bound for distance pivot-query, using all its ancestors 
        (it is expected that all ancestor-query and ancestor-pivot distances are already known, otherwise they will be calculated).''' 
        d_p_q_low, d_p_q_high = 0.0, math.inf
        for ancestor in self._gen_ancestors(node, include_self=False):
            d_p_a = self.get_distance(node.pivot, ancestor)
            d_a_q = query_distance[ancestor]
            d_p_q_low = max(d_p_q_low, abs(d_p_a - d_a_q))
            d_p_q_high = min(d_p_q_high, d_p_a + d_a_q)
        return d_p_q_low, d_p_q_high
