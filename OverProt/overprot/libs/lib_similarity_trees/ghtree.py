'''GHT, Generalized Hyperplane Tree'''

import math
import itertools
from collections import defaultdict
from typing import NamedTuple, Generic, TypeVar, List, Tuple, Dict, Set, Union, Optional, Callable, Iterable, Iterator, Any, Counter, Sized, Container
from dataclasses import dataclass, field

from .abstract_similarity_tree import AbstractSimilarityTree

K = TypeVar('K')
V = TypeVar('V')


@dataclass
class GHRoot(Generic[K]):
    # subtree: 'GHNode[K]' = field(repr=False)
    subtree: Union['GHFork[K]', 'GHLeaf[K]'] = field(repr=False)
    def __init__(self):
        self.subtree = GHLeaf(parent=self)

@dataclass
class GHFork(Generic[K]):
    # parent: 'GHNode[K]' = field(repr=False)
    parent: Union['GHRoot[K]', 'GHFork[K]'] = field(repr=False)
    order_in_parent: int  # 1 if this is the first child of the parent, 2 if this is the second child of the parent, 0 if the parent is root
    pivot1: K
    pivot2: K
    # subtree1: 'GHNode[K]' = field(repr=False)
    # subtree2: 'GHNode[K]' = field(repr=False)
    subtree1: Union['GHFork[K]', 'GHLeaf[K]'] = field(repr=False)
    subtree2: Union['GHFork[K]', 'GHLeaf[K]'] = field(repr=False)

@dataclass
class GHLeaf(Generic[K]):
    # parent: 'GHNode[K]' = field(repr=False)
    parent: Union['GHRoot[K]', 'GHFork[K]'] = field(repr=False)
    order_in_parent: int = 0  # 1 if this is the first child of the parent, 2 if this is the second child of the parent, 0 if the parent is root
    elements: List[K] = field(default_factory=list)

GHNode = Union[GHRoot[K], GHFork[K], GHLeaf[K]]

class GHTree(Generic[K, V], AbstractSimilarityTree[K, V]):
    def __init__(self, distance_function: Callable[[V, V], float], leaf_size: int = 8):
        assert leaf_size >= 1
        self._leaf_size: int = leaf_size
        self._distance_function: Callable[[V, V], float] = distance_function
        # self._pivots: Set[K] = set()  # keys of pivots anywhere in the tree
        self._pivots = Counter[K]()  # keys of pivots anywhere in the tree, and the number of nodes in which they are pivot
        self._elements: Set[K] = set()  # keys of elements currently present in the tree (pivot is not element)
        self._values: Dict[K, V] = {}  # mapping of element/pivot keys to objects on which distance function is defined
        self._home_leaves: Dict[K, GHLeaf[K]] = {}  # mapping of elements to their accommodating leaves
        self._distance_cache: Dict[K, Dict[K, float]] = defaultdict(dict)  # distance_cache[i][j] keeps distance between i and j, where i is element, j is element or pivot
        self._root = GHRoot[K]()
        self._calculated_distances_counter = 0  # really calculated distances
        self._worst_calculated_distances_counter = 0  # theoretical number of calculated distances in worst case (i.e. each-to-each)
        # TODO implement distance_cache, values etc. by lists instead of dicts? (insert() will return the new sequentially assigned key instead of taking it)
    
    def insert(self, key: K, value: V) -> None:
        '''Insert a new element with given key and value into the tree. The key must be unique (including the deleted elements).'''
        assert key not in self._elements, f'Key {key} is already in the tree'
        assert key not in self._pivots, f'Key {key} was in the tree in the past and may never be inserted again'
        self._worst_calculated_distances_counter += len(self._elements)
        self._elements.add(key)
        self._values[key] = value
        self._insert_to_node(self._root, key)
    
    def _insert_to_node(self, node: GHNode[K], key: K) -> None:
        if isinstance(node, GHRoot):
            self._insert_to_node(node.subtree, key)
        elif isinstance(node, GHFork):
            if self.get_distance(node.pivot1, key) <= self.get_distance(node.pivot2, key):
                self._insert_to_node(node.subtree1, key)
            else:
                self._insert_to_node(node.subtree2, key)
        elif isinstance(node, GHLeaf):
            elements = node.elements
            elements.append(key)
            self._home_leaves[key] = node
            if len(elements) > self._leaf_size:
                parent = node.parent
                p1, p2 = self._select_pivots(elements)
                self._pivots[p1] += 1
                self._pivots[p2] += 1
                elems1, elems2 = self._partition(p1, p2, elements)
                new_leaf1 = GHLeaf[K](parent=parent, order_in_parent=0, elements=elems1)
                new_leaf2 = GHLeaf[K](parent=parent, order_in_parent=0, elements=elems2)
                for elem in elems1:
                    self._home_leaves[elem] = new_leaf1
                for elem in elems2:
                    self._home_leaves[elem] = new_leaf2
                new_fork = GHFork[K](parent=parent, order_in_parent=node.order_in_parent, pivot1=p1, pivot2=p2, subtree1=new_leaf1, subtree2=new_leaf2)
                new_leaf1.parent = new_fork
                new_leaf1.order_in_parent = 1
                new_leaf2.parent = new_fork
                new_leaf2.order_in_parent = 2
                if node.order_in_parent == 0:
                    assert isinstance(parent, GHRoot)
                    parent.subtree = new_fork
                elif node.order_in_parent == 1:
                    assert isinstance(parent, GHFork)
                    parent.subtree1 = new_fork
                elif node.order_in_parent == 2:
                    assert isinstance(parent, GHFork)
                    parent.subtree2 = new_fork
                else: 
                    raise AssertionError('GHLeaf.order_in_parent must be 0 or 1 or 2')
        else: 
            raise AssertionError('GHNode must be one of: GHRoot, GHFork, GHLeaf')
    
    def delete(self, key: K) -> None:
        self._elements.remove(key)
        leaf = self._home_leaves.pop(key)
        calc_distances = self._distance_cache.pop(key)
        if key not in self._pivots:
            self._values.pop(key)
            for mate in calc_distances.keys():
                if mate in self._elements:
                    self._distance_cache[mate].pop(key)
        leaf.elements.remove(key)
        if len(leaf.elements) == 0:
            parent = leaf.parent
            if isinstance(parent, GHFork):
                # Remove parent and replace it by sibling
                greatparent = parent.parent
                sibling = parent.subtree2 if leaf.order_in_parent == 1 else parent.subtree1
                if parent.order_in_parent == 1:
                    assert isinstance(greatparent, GHFork)
                    greatparent.subtree1 = sibling
                elif parent.order_in_parent == 2:
                    assert isinstance(greatparent, GHFork)
                    greatparent.subtree2 = sibling
                else:
                    assert isinstance(greatparent, GHRoot)
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
        # Strategy select the two farther elements
        best_dist, p1, p2 = max((self.get_distance(i, j), i, j) for i, j in itertools.combinations(elements, 2))
        return p1, p2

    def _partition(self, p1: K, p2: K, elements: List[K]) -> Tuple[List[K], List[K]]:
        elems1 = []
        elems2 = []
        for elem in elements:
            if self.get_distance(p1, elem) <= self.get_distance(p2, elem):
                elems1.append(elem)
            else: 
                elems2.append(elem)
        return elems1, elems2

    def get_distance(self, key1: K, key2: K) -> float:
        '''key1 can be element or pivot, key2 must be element'''
        assert key1 in self._elements or key1 in self._pivots
        assert key2 in self._elements
        if key1 == key2:
            return 0.0
        try: 
            return self._distance_cache[key2][key1]
        except KeyError:
            dist = self._distance_function(self._values[key1], self._values[key2])
            self._calculated_distances_counter += 1
            # print(f'd_{key1}_{key2}')
            if key1 in self._elements:
                self._distance_cache[key1][key2] = dist
            self._distance_cache[key2][key1] = dist
            return dist
    
    def _str_lines(self, node: GHNode[K], indent: int = 0) -> Iterator[str]:
        '''Generate the lines of str(self)'''
        yield '    ' * indent + str(node)
        if isinstance(node, GHRoot):
            yield from self._str_lines(node.subtree, indent=indent+1)
        elif isinstance(node, GHFork):
            yield from self._str_lines(node.subtree1, indent=indent+1)
            yield from self._str_lines(node.subtree2, indent=indent+1)
        elif isinstance(node, GHLeaf):
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

    def _recursive_template(self, key: K, value: V, node: GHNode[K]) -> None:
        # TODO add docstring
        if isinstance(node, GHRoot):
            raise NotImplementedError
        elif isinstance(node, GHFork):
            raise NotImplementedError
        elif isinstance(node, GHLeaf):
            raise NotImplementedError
        else: 
            raise AssertionError('GHNode must be one of: GHRoot, GHFork, GHLeaf')

    def range_query(self, query: K, range_: float) -> List[K]:
        if query not in self._elements:
            raise NotImplementedError('Implemented only for queries present in the tree')
        # counter = [0]
        return list(self._range_query(self._root, query, range_))

    def _range_query(self, node: GHNode[K], query: K, range_: float) -> Iterator[K]:
        # TODO add docstring
        if isinstance(node, GHRoot):
            yield from self._range_query(node.subtree, query, range_)
        elif isinstance(node, GHFork):
            # Double-pivot constraint
            d_p1_q = self.get_distance(node.pivot1, query)
            d_p2_q = self.get_distance(node.pivot2, query)
            if d_p1_q - range_ <= d_p2_q + range_:
                yield from self._range_query(node.subtree1, query, range_)
            if d_p1_q + range_ >= d_p2_q - range_:
                yield from self._range_query(node.subtree2, query, range_)
        elif isinstance(node, GHLeaf):
            for e in node.elements:
                if self.get_distance(e, query) <= range_:
                    yield e
        else: 
            raise AssertionError('GHNode must be one of: GHRoot, GHFork, GHLeaf')

    def _retrieve_path(self, key: K) -> List[GHNode[K]]:
        '''Return the full path on which the key is located = list of nodes ordered from leaf to root.'''
        path: List[GHNode[K]] = []
        node: GHNode[K] = self._home_leaves[key]
        while not isinstance(node, GHRoot):
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
            assert isinstance(node, GHFork)
            assert isinstance(child, (GHFork, GHLeaf))
            if child.order_in_parent == 1:
                self._kNN_query_top_down(node, query, candidates, skip_subtree1=True)
            else:
                self._kNN_query_top_down(node, query, candidates, skip_subtree2=True)
        return [elem for dist, elem in candidates if elem is not None]
   
    def _kNN_query_top_down(self, node: GHNode[K], query: K, work_list: List[Tuple[float, Optional[K]]], skip_subtree1: bool = False, skip_subtree2: bool = False, include_query: bool = False) -> None:
        '''Perform classical kNN query starting in node, optionally skipping one of the subtrees. 
        work_list must contain k nearest elements found so far and their distances and will be updated with newly found elements. 
        Use (math.inf, None) if elements are not found yet.'''
        if isinstance(node, GHLeaf):
            for elem in node.elements:
                if elem != query or include_query:
                    dist = self.get_distance(query, elem)
                    self.bubble_into_sorted_list(work_list, (dist, elem))
        elif isinstance(node, GHFork):
            d_p1_q = self.get_distance(node.pivot1, query)
            d_p2_q = self.get_distance(node.pivot2, query)
            if not skip_subtree1 and not skip_subtree2:
                if d_p1_q <= d_p2_q:
                    self._kNN_query_top_down(node.subtree1, query, work_list)
                    range_ = work_list[-1][0]
                    if d_p1_q + range_ >= d_p2_q - range_:  # Double-pivot constraint
                        self._kNN_query_top_down(node.subtree2, query, work_list)
                else:
                    self._kNN_query_top_down(node.subtree2, query, work_list)
                    range_ = work_list[-1][0]
                    if d_p1_q - range_ <= d_p2_q + range_:  # Double-pivot constraint
                        self._kNN_query_top_down(node.subtree1, query, work_list)
            else:
                range_ = work_list[-1][0]
                if not skip_subtree1 and d_p1_q - range_ <= d_p2_q + range_:  # Double-pivot constraint
                    self._kNN_query_top_down(node.subtree1, query, work_list)
                elif not skip_subtree2 and d_p1_q + range_ >= d_p2_q - range_:  # Double-pivot constraint
                    self._kNN_query_top_down(node.subtree2, query, work_list)
        elif isinstance(node, GHRoot):
            self._kNN_query_top_down(node.subtree, query, work_list)
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
        n_distance_cache_entries = sum(len(dic) for dic in self._distance_cache.values())
        percent_calculated_distances = self._calculated_distances_counter / self._worst_calculated_distances_counter * 100
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
        Entries in distance cache: {n_distance_cache_entries}
        Calculated distances: {self._calculated_distances_counter} / {self._worst_calculated_distances_counter} ({percent_calculated_distances:.0f}%) '''
        return result
    
    def _collect_nodes(self, node: GHNode[K], out_forks: List[GHFork[K]], out_leaves: List[GHLeaf[K]]):
        if isinstance(node, GHLeaf):
            out_leaves.append(node)
        elif isinstance(node, GHFork):
            out_forks.append(node)
            self._collect_nodes(node.subtree1, out_forks, out_leaves)
            self._collect_nodes(node.subtree2, out_forks, out_leaves)
        elif isinstance(node, GHRoot):
            self._collect_nodes(node.subtree, out_forks, out_leaves)
        else:
            raise AssertionError
