'''GHT, Generalized Hyperplane Tree'''

import math
import itertools
from typing import Generic, TypeVar, List, Tuple, Dict, Set, Union, Optional, Callable, Final, Iterator, Any, Counter, Sequence, Literal, Deque, Iterable
from dataclasses import dataclass, field
import heapq

from .abstract_similarity_tree import AbstractSimilarityTree
from .caches import FunctionCache, DistanceCache
from ..lib import PriorityQueue

K = TypeVar('K')
V = TypeVar('V')

OrderInParent = Literal[0, 1, 2]
ONLY_CHILD: Final = 0
FIRST_CHILD: Final = 1
SECOND_CHILD: Final = 2

@dataclass
class _GHRoot(Generic[K]):
    subtree: Union['_GHFork[K]', '_GHLeaf[K]'] = field(repr=False)
    def __init__(self):
        self.subtree = _GHLeaf(parent=self)

@dataclass
class _GHFork(Generic[K]):
    parent: Union['_GHRoot[K]', '_GHFork[K]'] = field(repr=False)
    order_in_parent: OrderInParent  # 1 if this is the first child of the parent, 2 if this is the second child of the parent, 0 if the parent is root
    pivot1: K
    pivot2: K
    subtree1: Union['_GHFork[K]', '_GHLeaf[K]'] = field(repr=False)
    subtree2: Union['_GHFork[K]', '_GHLeaf[K]'] = field(repr=False)
    rc1: float = 0.0
    rc2: float = 0.0

@dataclass
class _GHLeaf(Generic[K]):
    parent: Union['_GHRoot[K]', '_GHFork[K]'] = field(repr=False)
    order_in_parent: OrderInParent = ONLY_CHILD  # 1 if this is the first child of the parent, 2 if this is the second child of the parent, 0 if the parent is root
    elements: List[K] = field(default_factory=list)

_GHNode = Union[_GHRoot[K], _GHFork[K], _GHLeaf[K]]


class GHTree(Generic[K, V], AbstractSimilarityTree[K, V]):
    def __init__(self, distance_function: Callable[[V, V], float], keys_values: Sequence[Tuple[K, V]] = (), leaf_size: int = 8):
        assert leaf_size >= 1
        self._leaf_size: int = leaf_size
        self._distance_function: Callable[[V, V], float] = distance_function
        self._pivots = Counter[K]()  # keys of pivots anywhere in the tree, and the number of nodes in which they are pivot
        self._elements: Set[K] = set()  # keys of elements currently present in the tree (pivot is not element)
        self._home_leaves: Dict[K, _GHLeaf[K]] = {}  # mapping of elements to their accommodating leaves
        self._distance_cache = DistanceCache[K, V](distance_function)
        self._root = _GHRoot[K]()
        self._bulk_load(keys_values)
        # TODO implement distance_cache, values etc. by lists instead of dicts? (insert() will return the new sequentially assigned key instead of taking it)
    
    def _bulk_load(self, keys_values: Sequence[Tuple[K, V]]) -> None:
        elements = [k for k, v in keys_values]
        n = len(elements)
        assert len(self) == 0, 'Bulk-load only works on an empty tree'
        assert len(set(elements)) == n, 'Duplicate keys are not allowed'
        if n == 0: 
            return
        self._elements.update(elements)
        for k, v in keys_values:
            self._distance_cache.insert(k, v)
        self._root = _GHRoot()
        self._root.subtree = self._create_node_from_bulk(elements)
        # refs from the subtree to the root are set by default in _create_node_from_bulk()
        print(f'Bulk-loaded {n} elements')
    
    def insert(self, key: K, value: V) -> None:
        '''Insert a new element with given key and value into the tree. The key must be unique (including the deleted elements).'''
        assert key not in self._elements, f'Key {key} is already in the tree'
        assert key not in self._pivots, f'Key {key} was in the tree in the past and may never be inserted again'
        self._elements.add(key)
        self._distance_cache.insert(key, value)
        self._insert_to_node(self._root, key)
    
    def _create_node_from_bulk(self, elements: List[K]) -> Union[_GHFork, _GHLeaf]:
        if len(elements) <= self._leaf_size:
            return _GHLeaf(parent=self._root, order_in_parent=ONLY_CHILD, elements=elements)
        else:
            p0 = elements[0]
            d_max, p1 = max((self.get_distance(p0, k), k) for k in elements)
            d_max, p2 = max((self.get_distance(p1, k), k) for k in elements if k != p1)
            # TODO solve special case of len(keys_values)==1
            elems1, elems2, rc1, rc2 = self._partition(p1, p2, elements)
            subtree1 = self._create_node_from_bulk(elems1)
            subtree2 = self._create_node_from_bulk(elems2)
            new_fork = _GHFork(parent=self._root, order_in_parent=ONLY_CHILD, pivot1=p1, pivot2=p2, subtree1=subtree1, subtree2=subtree2, rc1=rc1, rc2=rc2)
            subtree1.parent = new_fork
            subtree1.order_in_parent = FIRST_CHILD
            subtree2.parent = new_fork
            subtree2.order_in_parent = SECOND_CHILD
            self._pivots[p1] += 1
            self._pivots[p2] += 1
            return new_fork        

    def _insert_to_node(self, node: _GHNode[K], key: K) -> None:
        if isinstance(node, _GHRoot):
            self._insert_to_node(node.subtree, key)
        elif isinstance(node, _GHFork):
            # TODO divide half-half in case of ties
            if self.get_distance(node.pivot1, key) <= self.get_distance(node.pivot2, key):
                self._insert_to_node(node.subtree1, key)
                node.rc1 = max(node.rc1, self.get_distance(node.pivot1, key))
            else:
                self._insert_to_node(node.subtree2, key)
                node.rc2 = max(node.rc2, self.get_distance(node.pivot2, key))
        elif isinstance(node, _GHLeaf):
            elements = node.elements
            elements.append(key)
            self._home_leaves[key] = node
            if len(elements) > self._leaf_size:
                parent = node.parent
                p1, p2 = self._select_pivots(elements)
                self._pivots[p1] += 1
                self._pivots[p2] += 1
                elems1, elems2, rc1, rc2 = self._partition(p1, p2, elements)
                new_leaf1 = _GHLeaf[K](parent=parent, order_in_parent=0, elements=elems1)
                new_leaf2 = _GHLeaf[K](parent=parent, order_in_parent=0, elements=elems2)
                # new_rc1 = max((self.get_distance(p1, e) for e in elems1), default=0.0)
                # new_rc2 = max((self.get_distance(p2, e) for e in elems2), default=0.0)
                for elem in elems1:
                    self._home_leaves[elem] = new_leaf1
                for elem in elems2:
                    self._home_leaves[elem] = new_leaf2
                new_fork = _GHFork[K](parent=parent, order_in_parent=node.order_in_parent, pivot1=p1, pivot2=p2, subtree1=new_leaf1, subtree2=new_leaf2, rc1=rc1, rc2=rc2)
                new_leaf1.parent = new_fork
                new_leaf1.order_in_parent = 1
                new_leaf2.parent = new_fork
                new_leaf2.order_in_parent = 2
                if node.order_in_parent == 0:
                    assert isinstance(parent, _GHRoot)
                    parent.subtree = new_fork
                elif node.order_in_parent == 1:
                    assert isinstance(parent, _GHFork)
                    parent.subtree1 = new_fork
                elif node.order_in_parent == 2:
                    assert isinstance(parent, _GHFork)
                    parent.subtree2 = new_fork
                else: 
                    raise AssertionError('GHLeaf.order_in_parent must be 0 or 1 or 2')
        else: 
            raise AssertionError('GHNode must be one of: GHRoot, GHFork, GHLeaf')
    
    def delete(self, key: K) -> None:
        self._elements.remove(key)
        leaf = self._home_leaves.pop(key)
        if key not in self._pivots:
            self._distance_cache.delete(key)
        leaf.elements.remove(key)
        if len(leaf.elements) == 0:
            parent = leaf.parent
            if isinstance(parent, _GHFork):
                # Remove parent and replace it by sibling
                greatparent = parent.parent
                sibling = parent.subtree2 if leaf.order_in_parent == 1 else parent.subtree1
                if parent.order_in_parent == 1:
                    assert isinstance(greatparent, _GHFork)
                    greatparent.subtree1 = sibling
                elif parent.order_in_parent == 2:
                    assert isinstance(greatparent, _GHFork)
                    greatparent.subtree2 = sibling
                else:
                    assert isinstance(greatparent, _GHRoot)
                    greatparent.subtree = sibling
                sibling.parent = greatparent
                sibling.order_in_parent = parent.order_in_parent
                p1 = parent.pivot1
                p2 = parent.pivot2
                self._pivots[p1] -= 1
                if self._pivots[p1] == 0:
                    self._pivots.pop(p1)
                self._pivots[p2] -= 1
                if self._pivots[p2] == 0:
                    self._pivots.pop(p2)

    def _select_pivots(self, elements: List[K]) -> Tuple[K, K]:
        # Strategy select the two farthest elements
        best_dist, p1, p2 = max((self.get_distance(i, j), i, j) for i, j in itertools.combinations(elements, 2))
        return p1, p2

    def _partition(self, p1: K, p2: K, elements: List[K]) -> Tuple[List[K], List[K], float, float]:
        elems1 = []
        elems2 = []
        rc1 = 0.0
        rc2 = 0.0
        last_tie = 2
        for elem in elements:
            d_p1_e = self.get_distance(p1, elem)
            d_p2_e = self.get_distance(p2, elem)
            if d_p1_e < d_p2_e:
                to = 1
            elif d_p1_e > d_p2_e:
                to = 2
            elif last_tie == 1:  # tie
                to = last_tie = 2
            else:  # last_tie == 2
                to = last_tie = 1
            if to == 1:
                elems1.append(elem)
                rc1 = max(rc1, d_p1_e)
            else:  # to == 2
                elems2.append(elem)
                rc2 = max(rc2, d_p2_e)
            # if d_p1_e <= d_p2_e:
            #     elems1.append(elem)
            #     rc1 = max(rc1, d_p1_e)
            # else: 
            #     elems2.append(elem)
            #     rc2 = max(rc2, d_p2_e)
        return elems1, elems2, rc1, rc2

    def get_distance(self, key1: K, key2: K) -> float:
        '''key1 can be element or pivot, key2 must be element'''
        return self._distance_cache.get_distance(key1, key2)
    
    def _str_lines(self, node: _GHNode[K], indent: int = 0) -> Iterator[str]:
        '''Generate the lines of str(self)'''
        yield '    ' * indent + str(node)
        if isinstance(node, _GHRoot):
            yield from self._str_lines(node.subtree, indent=indent+1)
        elif isinstance(node, _GHFork):
            yield from self._str_lines(node.subtree1, indent=indent+1)
            yield from self._str_lines(node.subtree2, indent=indent+1)
        elif isinstance(node, _GHLeaf):
            pass
        else: 
            raise AssertionError('GHNode must be one of: GHRoot, GHFork, GHLeaf')
    
    def __str__(self) -> str:
        return '\n'.join(self._str_lines(self._root))
    
    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._elements)

    def __contains__(self, element) -> bool:
        '''Decide if element is present in the tree'''
        return element in self._elements

    def _recursive_template(self, key: K, value: V, node: _GHNode[K]) -> None:
        # TODO add docstring
        if isinstance(node, _GHRoot):
            raise NotImplementedError
        elif isinstance(node, _GHFork):
            raise NotImplementedError
        elif isinstance(node, _GHLeaf):
            raise NotImplementedError
        else: 
            raise AssertionError('GHNode must be one of: GHRoot, GHFork, GHLeaf')

    def range_query(self, query: K, range_: float) -> List[K]:
        if query not in self._elements:
            raise NotImplementedError('Implemented only for queries present in the tree')
        # counter = [0]
        return list(self._range_query(self._root, query, range_))

    def _range_query(self, node: _GHNode[K], query: K, range_: float) -> Iterator[K]:
        # TODO add docstring
        if isinstance(node, _GHRoot):
            yield from self._range_query(node.subtree, query, range_)
        elif isinstance(node, _GHFork):
            # Double-pivot constraint
            d_p1_q = self.get_distance(node.pivot1, query)
            d_p2_q = self.get_distance(node.pivot2, query)
            if d_p1_q - range_ <= d_p2_q + range_ and d_p1_q <= node.rc1 + range_:
                yield from self._range_query(node.subtree1, query, range_)
            if d_p1_q + range_ >= d_p2_q - range_ and d_p2_q <= node.rc2 + range_:
                yield from self._range_query(node.subtree2, query, range_)
        elif isinstance(node, _GHLeaf):
            for e in node.elements:
                if self.get_distance(e, query) <= range_:
                    yield e
        else: 
            raise AssertionError('GHNode must be one of: GHRoot, GHFork, GHLeaf')

    def _retrieve_path(self, key: K) -> List[_GHNode[K]]:
        '''Return the full path on which the key is located = list of nodes ordered from leaf to root.'''
        path: List[_GHNode[K]] = []
        node: _GHNode[K] = self._home_leaves[key]
        while not isinstance(node, _GHRoot):
            path.append(node)
            node = node.parent
        path.append(node)
        return path

    def kNN_query(self, query: K, k: int, include_query: bool = False) -> List[K]:
        '''Perform a modified version of kNN query 
        (start in the leaf where the query is located, continue toward the root and search all side-branches by classical kNN).
        query must be present in the tree!'''
        # TODO implement 1-NN query separately - might be a bit faster?
        if include_query and len(self._elements) < k:
            raise ValueError('Tree contains less than k elements')
        if not include_query and len(self._elements) < k+1:
            raise ValueError('Tree contains less than k+1 elements')
        path = self._retrieve_path(query)
        path_length = len(path)
        candidates: List[Tuple[float, Optional[K]]] = [(math.inf, None) for i in range(k)]
        leaf = path[0]
        self._kNN_query_top_down(leaf, query, candidates, include_query=include_query)
        for i in range(1, path_length - 1):
            node = path[i]
            child = path[i-1]
            assert isinstance(node, _GHFork)
            assert isinstance(child, (_GHFork, _GHLeaf))
            if child.order_in_parent == 1:
                self._kNN_query_top_down(node, query, candidates, skip_subtree1=True)
            else:
                self._kNN_query_top_down(node, query, candidates, skip_subtree2=True)
        return [elem for dist, elem in candidates if elem is not None]
   
    def kNN_query_classical_by_value(self, query_value: V, k: int) -> List[Tuple[float, K]]:
        '''Perform classical kNN query starting in the root.'''
        # TODO test on more real dataset (random subset of PDB/non-redundant-PDB?)
        # TODO try using priority queue?
        # TODO try sibling-sibling-query and pivot-ancestor-query constraints
        # TODO using cheaper distance approximations
        besties = MinFinder[K](n=k)
        query_dist = FunctionCache[K, float](lambda key: self._distance_cache.get_distance_to_value(key, query_value))
        queue: List[Union[_GHFork, _GHLeaf]] = []
        queue.append(self._root.subtree)
        while len(queue) > 0:
            node = queue.pop()
            range_, _ = besties.top()
            if isinstance(node.parent, _GHFork):
                d_p1_q, d_p2_q = query_dist[node.parent.pivot1], query_dist[node.parent.pivot2]
                rc1, rc2 = node.parent.rc1, node.parent.rc2
                if node.order_in_parent == FIRST_CHILD:
                    if d_p1_q - range_ >= d_p2_q + range_ or d_p1_q >= rc1 + range_:
                        continue
                else:
                    if d_p2_q - range_ >= d_p1_q + range_ or d_p2_q >= rc2 + range_:
                        continue
            if isinstance(node, _GHLeaf):
                for elem in node.elements:
                    besties.bubble_in(query_dist[elem], elem)
            elif isinstance(node, _GHFork):
                # TODO include this constraint: d_q_p2 >= d_p1_p2 - d_p1_q
                # TODO try to include pivot-ancestor-query constraint?
                d_p1_q = query_dist[node.pivot1]
                d_p2_q = query_dist[node.pivot2]
                if d_p1_q <= d_p2_q:
                    if d_p1_q + range_ >= d_p2_q - range_ and d_p2_q <= node.rc2 + range_:  # Double-pivot constraint
                        queue.append(node.subtree2)
                    if d_p1_q <= node.rc1 + range_:
                        queue.append(node.subtree1)
                else:
                    if d_p1_q - range_ <= d_p2_q + range_ and d_p1_q <= node.rc1 + range_:  # Double-pivot constraint
                        queue.append(node.subtree1)
                    if d_p2_q <= node.rc2 + range_:
                        queue.append(node.subtree2)
            else:
                raise AssertionError('node must be one of: _GHFork, _GHLeaf')
        result = besties.pop_all_not_none()
        return result

    def kNN_query_classical_by_value_with_priority_queue(self, query_value: V, k: int) -> List[Tuple[float, K]]:
        '''Perform classical kNN query starting in the root.'''
        # TODO test on more real dataset (random subset of PDB/non-redundant-PDB?)
        # TODO try using priority queue?
        # TODO try sibling-sibling-query and pivot-ancestor-query constraints
        # TODO using cheaper distance approximations
        besties = MinFinder[K](n=k)
        query_dist = FunctionCache[K, float](lambda key: self._distance_cache.get_distance_to_value(key, query_value))
        queue = PriorityQueue[float, Union[_GHFork, _GHLeaf]]()
        queue.add(0.0, self._root.subtree)
        while not queue.is_empty():
            best = queue.pop_min()
            assert best is not None
            dmin, node = best
            range_, _ = besties.top()
            if isinstance(node.parent, _GHFork):
                d_p1_q, d_p2_q = query_dist[node.parent.pivot1], query_dist[node.parent.pivot2]
                rc1, rc2 = node.parent.rc1, node.parent.rc2
                if node.order_in_parent == FIRST_CHILD:
                    if d_p1_q - range_ >= d_p2_q + range_ or d_p1_q >= rc1 + range_:
                        continue
                else:
                    if d_p2_q - range_ >= d_p1_q + range_ or d_p2_q >= rc2 + range_:
                        continue
            if isinstance(node, _GHLeaf):
                for elem in node.elements:
                    besties.bubble_in(query_dist[elem], elem)
            elif isinstance(node, _GHFork):
                # TODO include this constraint: d_q_p2 >= d_p1_p2 - d_p1_q
                # TODO try to include pivot-ancestor-query constraint?
                d_p1_q = query_dist[node.pivot1]
                d_p2_q = query_dist[node.pivot2]
                if d_p1_q <= d_p2_q:
                    if d_p1_q + range_ >= d_p2_q - range_ and d_p2_q <= node.rc2 + range_:  # Double-pivot constraint
                        queue.add(max(0.0, d_p2_q-node.rc2), node.subtree2)
                    if d_p1_q <= node.rc1 + range_:
                        queue.add(max(0.0, d_p1_q-node.rc1), node.subtree1)
                else:
                    if d_p1_q - range_ <= d_p2_q + range_ and d_p1_q <= node.rc1 + range_:  # Double-pivot constraint
                        queue.add(max(0.0, d_p1_q-node.rc1), node.subtree1)
                    if d_p2_q <= node.rc2 + range_:
                        queue.add(max(0.0, d_p2_q-node.rc2), node.subtree2)
            else:
                raise AssertionError('node must be one of: _GHFork, _GHLeaf')
        result = besties.pop_all_not_none()
        return result

    def kNN_query_classical(self, query: K, k: int, include_query: bool = False) -> List[Tuple[float, K]]:
        '''Perform classical kNN query starting in the root.'''
        work_list: List[Tuple[float, Optional[K]]] = [(math.inf, None) for i in range(k)]
        self._kNN_query_top_down(self._root, query, work_list, include_query=include_query)
        assert all(k is not None for d, k in work_list)
        result = [(d, k) for d, k in work_list if k is not None]
        assert len(result) == k, f'Found less hits ({len(result)}) than required ({k}).'
        return result

    def _kNN_query_top_down(self, node: _GHNode[K], query: K, work_list: List[Tuple[float, Optional[K]]], skip_subtree1: bool = False, skip_subtree2: bool = False, include_query: bool = False) -> None:
        '''Perform classical kNN query starting in node, optionally skipping one of the subtrees. 
        work_list must contain k nearest elements found so far and their distances and will be updated with newly found elements. 
        Use (math.inf, None) if elements are not found yet.'''
        if isinstance(node, _GHLeaf):
            for elem in node.elements:
                if elem != query or include_query:
                    dist = self.get_distance(query, elem)
                    self.bubble_into_sorted_list(work_list, (dist, elem))
        elif isinstance(node, _GHFork):
            d_p1_q = self.get_distance(node.pivot1, query)
            d_p2_q = self.get_distance(node.pivot2, query)
            range_ = work_list[-1][0]
            if not skip_subtree1 and not skip_subtree2:
                if d_p1_q <= d_p2_q:
                    if d_p1_q <= node.rc1 + range_:
                        self._kNN_query_top_down(node.subtree1, query, work_list, include_query=include_query)
                        range_ = work_list[-1][0]
                    if d_p1_q + range_ >= d_p2_q - range_ and d_p2_q <= node.rc2 + range_:  # Double-pivot constraint
                        self._kNN_query_top_down(node.subtree2, query, work_list, include_query=include_query)
                else:
                    if d_p2_q <= node.rc2 + range_:
                        self._kNN_query_top_down(node.subtree2, query, work_list, include_query=include_query)
                        range_ = work_list[-1][0]
                    if d_p1_q - range_ <= d_p2_q + range_ and d_p1_q <= node.rc1 + range_:  # Double-pivot constraint
                        self._kNN_query_top_down(node.subtree1, query, work_list, include_query=include_query)
            elif not skip_subtree1:
                if d_p1_q - range_ <= d_p2_q + range_ and d_p1_q <= node.rc1 + range_:  # Double-pivot constraint
                    self._kNN_query_top_down(node.subtree1, query, work_list, include_query=include_query)
            elif not skip_subtree2:
                if d_p1_q + range_ >= d_p2_q - range_ and d_p2_q <= node.rc2 + range_:  # Double-pivot constraint
                    self._kNN_query_top_down(node.subtree2, query, work_list, include_query=include_query)
        elif isinstance(node, _GHRoot):
            self._kNN_query_top_down(node.subtree, query, work_list, include_query=include_query)
        else: 
            raise AssertionError('GHNode must be one of: GHRoot, GHFork, GHLeaf')
  
    @staticmethod
    def bubble_into_sorted_list(lst: List[Any], new_elem: Any) -> None:
        '''Add new_elem to a sorted list lst, keep it sorted, and remove the new largest element.'''
        if new_elem < lst[-1]:
            n = len(lst)
            lst[-1] = new_elem
            for i in range(n-1, 0, -1):
                if lst[i] < lst[i-1]:
                    lst[i-1], lst[i] = lst[i], lst[i-1]

    def get_statistics(self):
        n_elements = len(self._elements)
        n_pivots = len(self._pivots)
        forks = []
        leaves = []
        self._collect_nodes(self._root, forks, leaves)
        n_forks = len(forks)
        n_leaves = len(leaves)
        # for leaf in leaves:
        #     assert len(leaf.elements) <= self._leaf_size
        result = f'''GHT tree statistics:
        Elements: {n_elements}, Pivots: {n_pivots},
        Forks: {n_forks}, Leaves: {n_leaves}
        {self._distance_cache.get_statistics()}'''
        return result
    
    def _collect_nodes(self, node: _GHNode[K], out_forks: List[_GHFork[K]], out_leaves: List[_GHLeaf[K]]):
        if isinstance(node, _GHLeaf):
            out_leaves.append(node)
        elif isinstance(node, _GHFork):
            out_forks.append(node)
            self._collect_nodes(node.subtree1, out_forks, out_leaves)
            self._collect_nodes(node.subtree2, out_forks, out_leaves)
        elif isinstance(node, _GHRoot):
            self._collect_nodes(node.subtree, out_forks, out_leaves)
        else:
            raise AssertionError


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
    def bubble_in(self, key: float, element: V) -> None:
        '''Iff key is smaller than current top, insert element and remove current top; otherwise do nothing.'''
        if len(self) == 0:
            return
        k_top_neg, _, _ = self._heap[0]
        k_neg = -key
        if k_neg > k_top_neg:
            heapq.heapreplace(self._heap, (k_neg, self._seq, element))
            self._seq -= 1
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
        
