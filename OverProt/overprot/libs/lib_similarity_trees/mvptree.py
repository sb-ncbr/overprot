'''VPT, Vantage Point Tree'''

import math
import statistics
from datetime import timedelta
from typing import Generic, TypeVar, List, Tuple, Dict, Set, Union, Optional, Callable, Final, Iterator, Any, Counter, Sequence, Literal, Deque, Iterable
from dataclasses import dataclass, field
import numpy as np
import json
from multiprocessing import Pool

from .abstract_similarity_tree import AbstractSimilarityTree, K, V
from .caches import FunctionCache, DistanceCache, MinFinder
from .. import lib
from ..lib import PriorityQueue, ProgressBar, Timing


ZERO_ELEMENT = '-'
# TOLERANCE = 0.0


@dataclass
class _MVPFork(Generic[K]):
    parent: Optional['_MVPFork[K]'] = field(repr=False)
    pivot: K
    radii: List[float] = field(default_factory=list)
    children: List['_MVPNode[K]'] = field(default_factory=list)

@dataclass
class _MVPLeaf(Generic[K]):
    parent: Optional['_MVPFork[K]'] = field(repr=False)
    elements: List[K] = field(default_factory=list)

_MVPNode = Union[_MVPFork[K], _MVPLeaf[K]]


class MVPTree(Generic[K, V], AbstractSimilarityTree[K, V]):
    _K: int  # arity
    _leaf_size: int
    # _distance_function: Callable[[V, V], float]
    _distance_cache: DistanceCache[K, V]
    _elements: Set[K]
    _home_leaves: Dict[K, _MVPLeaf[K]]
    _root: _MVPNode[K]

    def __init__(self, distance_function: Callable[[V, V], float], keys_values: Sequence[Tuple[K, V]] = (), arity: int = 4, leaf_size: int = 8, 
                 n_processes: int = 1, root_element: Optional[Tuple[K, V]] = None) -> None:
        assert leaf_size >= 1
        self._K = arity
        self._leaf_size = leaf_size
        self._distance_function = distance_function
        self._distance_cache = DistanceCache(distance_function)
        self._elements = set()  # keys of elements currently present in the tree (pivot is not necessarily element)
        # self._home_leaves = {}  # mapping of elements to their accommodating leaves
        self._root = _MVPLeaf(parent=None)
        self._bulk_load(keys_values, with_progress_bar=True, n_processes=n_processes, root_element=root_element)
    
    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._elements)

    def __contains__(self, element) -> bool:
        '''Decide if element is present in the tree'''
        return element in self._elements

    def _bulk_load(self, keys_values: Sequence[Tuple[K, V]], with_progress_bar: bool = False, n_processes: int = 1, root_element: Optional[Tuple[K, V]] = None) -> None:
        elements = [k for k, v in keys_values]
        n = len(elements)
        assert len(self) == 0, 'Bulk-load only works on an empty tree'
        assert len(set(elements)) == n, 'Duplicate keys are not allowed'
        if n == 0: 
            return
        self._elements.update(elements)
        if root_element is not None:
            self._distance_cache.insert(*root_element)
            root_pivot, _ = root_element
        else:
            root_pivot = None
        for k, v in keys_values:
            self._distance_cache.insert(k, v)
        bar = ProgressBar(len(elements), title=f'Bulk-loading {n} elements into {self._K}-way MVPTree').start() if with_progress_bar else None
        if n_processes == 1:
            self._root = self._create_node_from_bulk(elements, pivot=root_pivot, progress_bar=bar)
        else:
            with Pool(n_processes) as pool:
                self._root = self._create_node_from_bulk(elements, pivot=root_pivot, progress_bar=bar, pool=pool)
        if bar is not None:
            bar.finalize()
    
    def _create_node_from_bulk(self, elements: List[K], pivot: Optional[K] = None, parent_pivot: Optional[K] = None, progress_bar: Optional[ProgressBar] = None, pool: Optional[Pool] = None) -> _MVPNode[K]:
        if len(elements) <= self._leaf_size:
            new_leaf = _MVPLeaf(parent=None, elements=elements)
            if progress_bar is not None:
                progress_bar.step(len(elements))
            return new_leaf
        else:
            if pivot is None:
                pivot = self._select_pivot0(elements, parent_pivot)
                # pivot = self._select_pivot1(elements, parent_pivot)
                elements.remove(pivot)
            bins, rs = self._partition(pivot, elements, pool=pool)
            new_fork = _MVPFork(parent=None, pivot=pivot, radii=rs)
            for bin in bins:
                subtree = self._create_node_from_bulk(bin, parent_pivot=pivot, progress_bar=progress_bar, pool=pool)
                new_fork.children.append(subtree)
                subtree.parent = new_fork
            return new_fork

    def _select_pivot0(self, elements: List[K], parent_pivot: Optional[K]) -> K:
        return elements[0]

    def _select_pivot1(self, elements: List[K], parent_pivot: Optional[K]) -> K:
        if parent_pivot is None:
            parent_pivot = elements[0]
        d_max, pivot = max((self.get_distance(parent_pivot, k), k) for k in elements)
        return pivot

    def _partition(self, pivot: K, elements: List[K], pool: Optional[Pool] = None) -> Tuple[List[List[K]], List[float]]:
        bins = [[] for i in range(self._K)]
        self._distance_cache.calculate_distances([(pivot, elem) for elem in elements], pool=pool)
        distances = [self.get_distance(pivot, elem) for elem in elements]
        rs = statistics.quantiles(distances, n=self._K)
        rs.insert(0, min(distances))
        rs.append(max(distances))
        # print('distances', sorted(distances))
        # print('rs', rs)
        for elem, dist in zip(elements, distances):
            for i_bin in range(self._K):
                if dist < rs[i_bin+1]:
                    the_bin = i_bin
                    break
                elif dist == rs[i_bin+1]:
                    if i_bin == self._K - 1:
                        # end od last bin
                        the_bin = i_bin
                        break
                    else:
                        # tie
                        if len(bins[i_bin]) <= len(bins[i_bin+1]):
                            the_bin = i_bin
                            break
                        else:
                            the_bin = i_bin+1
                            break
            else:
                raise AssertionError('No bin selected')
            bins[the_bin].append(elem)
        return bins, rs

    def _get_distance_to_value(self, key1: K, value2: K) -> float:
        # if key1 == ZERO_ELEMENT:
        #     return self._distance_from_zero(value2)
        # else:
        #     return self._distance_cache.get_distance_to_value(key1, value2)
        return self._distance_cache.get_distance_to_value(key1, value2)

    def get_distance(self, key1: K, key2: K) -> float:
        # if key1 == key2 == ZERO_ELEMENT:
        #     return 0.0
        # elif key1 == ZERO_ELEMENT:
        #     return self._distance_from_zero(self._distance_cache._elements[key2])
        # elif key2 == ZERO_ELEMENT:
        #     return self._distance_from_zero(self._distance_cache._elements[key1])
        # else:
        #     return self._distance_cache.get_distance(key1, key2)
        return self._distance_cache.get_distance(key1, key2)

    def get_statistics(self):
        n_elements = len(self._elements)
        forks, leaves = self.get_forks_and_leaves()
        n_forks = len(forks)
        n_leaves = len(leaves)
        result = f'''{self._K}-way MVPT tree statistics:
        Elements: {n_elements}, Forks: {n_forks}, Leaves: {n_leaves}
        {self._distance_cache.get_statistics()}'''
        return result
    
    def leaf_diameters(self, leaves: List[_MVPLeaf]) -> None:
        diameters = []
        for leaf in leaves:
            n = len(leaf.elements)
            diameter = max((self.get_distance(leaf.elements[i], leaf.elements[j]) for i in range(n) for j in range(i)), default=0.0)
            diameters.append(diameter)
        return f'leaf diameters: {min(diameters):.3f}-{max(diameters):.3f}, mean: {statistics.mean(diameters):.3f} median: {statistics.median(diameters):.3f}'

    def leaf_distances(self, leaves: List[_MVPLeaf]) -> None:
        dists = []
        for leaf in leaves:
            n = len(leaf.elements)
            dists.extend(self.get_distance(leaf.elements[i], leaf.elements[j]) for i in range(n) for j in range(i))
        min_dist = min(dists, default=math.nan)
        max_dist = max(dists, default=math.nan)
        mean = statistics.mean(dists) if len(dists) > 0 else math.nan
        median = statistics.median(dists) if len(dists) > 0 else math.nan
        return f'in-leaf distances: {min_dist:.3f}-{max_dist:.3f}, mean: {mean:.3f} median: {median:.3f}'

    def get_forks_and_leaves(self) -> Tuple[List[_MVPFork], List[_MVPLeaf]]:
        forks: List[_MVPFork[K]] = []
        leaves: List[_MVPLeaf[K]] = []
        self._collect_nodes(self._root, forks, leaves)
        return forks, leaves

    def _collect_nodes(self, node: _MVPNode[K], out_forks: List[_MVPFork[K]], out_leaves: List[_MVPLeaf[K]]):
        if isinstance(node, _MVPLeaf):
            out_leaves.append(node)
        elif isinstance(node, _MVPFork):
            out_forks.append(node)
            for subtree in node.children:
                self._collect_nodes(subtree, out_forks, out_leaves)
        else:
            raise AssertionError

    def kNN_query_by_value(self, query_value: V, k: int, distance_low_bounds: List[Callable[[V, V], float]] = [], the_range: float = math.inf) -> List[Tuple[float, K]]:
        besties = MinFinder[K](n=k)
        query_dist = FunctionCache[K, float](lambda key: self._get_distance_to_value(key, query_value))
        queue = PriorityQueue[float, _MVPNode]()
        queue.add(0.0, self._root)
        current_range = min(besties.top_size(), the_range)
        seen = set()
        QAE = True
        n_leaves = 0
        while not queue.is_empty():
            best = queue.pop_min()
            assert best is not None
            dmin, node = best
            if dmin >= current_range:
                continue
            if isinstance(node, _MVPLeaf):
                n_leaves += 1
                ancestors = list(self._gen_ancestors_of_leaf(node))
                for elem in node.elements:
                    assert elem not in seen
                    if elem in seen:
                        continue
                    if QAE:
                        if not self._can_be_under(elem, query_dist, ancestors, current_range):
                            continue
                    elem_value = self._distance_cache._elements[elem]
                    if any(d(elem_value, query_value) >= current_range for d in distance_low_bounds):
                        continue
                    current_range = min(besties.bubble_in(query_dist[elem], elem), the_range)
                    seen.add(elem)
            elif isinstance(node, _MVPFork):
                d_p_q = query_dist[node.pivot]
                assert node.pivot not in seen
                if node.pivot != ZERO_ELEMENT and node.pivot not in seen:
                    current_range = min(besties.bubble_in(d_p_q, node.pivot), the_range)
                    seen.add(node.pivot)
                for i in range(self._K):
                    dmin_child = max(dmin, node.radii[i] - d_p_q, d_p_q - node.radii[i+1])
                    if dmin_child < current_range:
                        queue.add(dmin_child, node.children[i])
            else:
                raise AssertionError('node must be one of: _MVPFork, _MVPLeaf')
        result = besties.pop_all_not_none()
        return result

    def _gen_ancestors(self, node: _MVPFork[K], include_self: bool = False) -> Iterator[K]:
        if include_self:
            yield node.pivot
        ancestor_node = node.parent
        while ancestor_node is not None:
            yield ancestor_node.pivot
            ancestor_node = ancestor_node.parent

    def _gen_ancestors_of_leaf(self, leaf: _MVPLeaf[K]) -> Iterator[K]:
        if leaf.parent is not None:
            yield from self._gen_ancestors(leaf.parent, include_self=True)

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
    
    def _node_to_json(self, node: _MVPNode[K]) -> object:
        if isinstance(node, _MVPLeaf):
            return node.elements
        elif isinstance(node, _MVPFork):
            return {
                'pivot': node.pivot,
                'radii': node.radii,
                'children': [self._node_to_json(child) for child in node.children]
            }
        else:
            raise AssertionError

    def _node_from_json(self, js: object) -> _MVPNode[K]:
        if isinstance(js, list):
            return _MVPLeaf(parent=None, elements=js)
        elif isinstance(js, dict):
            fork = _MVPFork(parent=None, pivot=js['pivot'], radii=js['radii'], children=[])
            for c in js['children']:
                child = self._node_from_json(c)
                fork.children.append(child)
                child.parent = fork
            return fork
        else:
            raise AssertionError
    
    def save(self, file: str, with_cache: bool = False, **json_kwargs) -> None:
        js = {
            'TYPE': 'MVPT', 
            'ARITY': self._K, 
            'LEAF_SIZE': self._leaf_size, 
            'root': self._node_to_json(self._root),
            'distance_cache': self._distance_cache.json() if with_cache else []
        }
        kwargs = {'indent': 1}
        kwargs.update(json_kwargs)
        with open(file, 'w') as w:
            json.dump(js, w, **kwargs)

    @staticmethod
    def load(file: str, distance_function: Callable[[V, V], float], get_value: Callable[[K], V]) -> 'MVPTree[K, V]':
        with open(file) as r:
            js = json.load(r)
        assert isinstance(js, dict)
        leaf_size = js['LEAF_SIZE']
        arity = js['ARITY']
        result = MVPTree[K, V](distance_function, arity=arity, leaf_size=leaf_size)
        root = result._node_from_json(js['root'])
        result._root = root
        forks, leaves = result.get_forks_and_leaves()
        for fork in forks:
            if fork.pivot != ZERO_ELEMENT:
                result._elements.add(fork.pivot)
        for leaf in leaves:
            for elem in leaf.elements:
                result._elements.add(elem)
        for elem in result._elements:
            result._distance_cache.insert(elem, get_value(elem))
        for k1, k2, dist in js['distance_cache']:
            result._distance_cache.set_distance(k1, k2, dist)
        return result
