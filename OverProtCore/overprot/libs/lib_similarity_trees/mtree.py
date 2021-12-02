'''M-tree with Delete'''

from __future__ import annotations
from typing import NamedTuple, Generic, TypeVar, List, Tuple, Dict, Set, Union, Optional, Callable, Iterable, Iterator, Any, Counter, ClassVar
from dataclasses import dataclass, field
import itertools
import math

from .. import lib
from .abstract_similarity_tree import AbstractSimilarityTree

K = TypeVar('K')  # Type of keys
V = TypeVar('V')  # Type of values


@dataclass
class _LeafEntry(Generic[K]):
    parent: '_ForkEntry[K]' = field(repr=False)
    key: K
    dist_to_parent: float = math.inf
    rc: ClassVar[float] = 0.0
    @property
    def pivot(self) -> K:
        return self.key

class _Leaf(List[_LeafEntry[K]], Generic[K]):
    pass

@dataclass
class _ForkEntry(Generic[K]):
    parent: Optional['_ForkEntry[K]'] = field(default=None, repr=False)
    pivot: Optional[K] = None
    rc: float = math.inf # covering radius
    dist_to_parent: float = math.inf
    children: '_Node[K]' = field(default_factory=_Leaf, repr=False)

class _Fork(List[_ForkEntry[K]], Generic[K]):
    pass

_Node = Union[_Fork[K], _Leaf[K]]

_Entry = Union[_ForkEntry[K], _LeafEntry[K]]

class MTree(AbstractSimilarityTree[K, V]):
    def __init__(self, distance_function: Callable[[V, V], float], fork_arity: int = 4, leaf_arity: int = 4, update_pivots_on_insert: bool = False):
        self._distance_function: Callable[[V, V], float] = distance_function
        self._fork_arity: int = fork_arity
        self._leaf_arity: int = leaf_arity
        self._root: _ForkEntry[K] = _ForkEntry()
        self._elements: Dict[K, V] = {}  # mapping of element keys to values (i.e. objects on which distance function is defined)
        self._home_entries: Dict[K, _LeafEntry[K]] = {}  # mapping of element key to their accommodating leaf entries
        self._distance_cache: Dict[K, Dict[K, float]] = {}  # defaultdict(dict)  # distance_cache[i][j] keeps distance between i and j, where i is element, j is element or pivot
        self._calculated_distances_counter = 0  # really calculated distances
        self._worst_calculated_distances_counter = 0  # theoretical number of calculated distances in worst case (i.e. each-to-each)
        self._update_pivots_on_insert = update_pivots_on_insert
    
    def __str__(self) -> str:
        return '\n'.join(self._str_lines(self._root))
    
    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._elements)

    def __contains__(self, element) -> bool:
        '''Decide if element is present in the tree'''
        return element in self._elements

    def insert(self, key: K, value: V) -> None:
        '''Insert a new element with given key and value into the tree. The key must be unique.'''
        assert key not in self, f'Key {key} is already in the tree'
        self._worst_calculated_distances_counter += len(self._elements)
        self._elements[key] = value
        self._distance_cache[key] = {}
        self._insert_to_entry(self._root, key)
        # TODO try replacing pivot by the new element, if it decreases rc / bubbling pivots up to decrease rc? in insert() or after bulk_load()
    
    def delete(self, key: K) -> None:
        assert key in self, f'Key {key} is not in the tree'
        leaf_entry = self._home_entries[key]

        entry = leaf_entry.parent
        children = entry.children
        assert isinstance(children, _Leaf)
        children.remove(leaf_entry)
        if len(children) >= 1:
            # Check pivots and rcs
            self._update_pivots_upwards(entry, key)
        else:
            # Remove entry and consolide bottom-up and check pivots and rcs
            parent = entry.parent
            if parent is not None:  # parent is not root
                assert isinstance(parent.children, _Fork)
                parent.children.remove(entry)
                self._update_pivots_upwards(parent, removed_key=key)
                self._consolidate(parent, removed_key=key)
                self._update_pivots_upwards(parent, removed_key=key)
    
        self._elements.pop(key)
        self._home_entries.pop(key)
        dists = self._distance_cache.pop(key)
        for other in dists.keys():
            self._distance_cache[other].pop(key)

    def kNN_query(self, query: K, k: int, include_query: bool = False, pivot_prox=False) -> List[K]:
        '''Perform classical kNN query.'''
        work_list: List[Tuple[float, Optional[K]]] = [(math.inf, None)] * k
        self._kNN_query_classical(self._root, query, work_list, include_query=include_query, pivot_prox=pivot_prox)
        # print(work_list)
        return [key for dist, key in work_list if key is not None]
    
    def get_distance(self, key1: Optional[K], key2: Optional[K]) -> float:
        '''Get the distance between two objects present in the tree'''
        assert not (key1 is None and key2 is None)
        if key1 is None or key2 is None:
            return math.inf
        elif key1 == key2:
            return 0.0
        else:
            try: 
                return self._distance_cache[key1][key2]
            except KeyError:
                dist = self._distance_function(self._elements[key1], self._elements[key2])
                self._distance_cache[key1][key2] = dist
                self._distance_cache[key2][key1] = dist
                self._calculated_distances_counter += 1
                return dist

    def get_statistics(self) -> str:
        '''Return string with basic statistics about the tree'''
        n_elements = len(self)
        n_distance_cache_entries = sum(len(dic) for dic in self._distance_cache.values())
        percent_calculated_distances = self._calculated_distances_counter / self._worst_calculated_distances_counter * 100
        fork_entries, leaf_entries = self._collect_entries_by_type()
        n_forks = len([f for f in fork_entries if isinstance(f.children, _Fork)])
        n_leaves = len([f for f in fork_entries if isinstance(f.children, _Leaf)])
        leaf_occupancy = n_elements / n_leaves
        fork_occupancy = (n_forks + n_leaves - 1) / n_forks if n_forks > 0 else math.nan
        result = f'''M-tree statistics:
        Elements: {n_elements}, Forks: {n_forks}, Leaves: {n_leaves}
        Fork occupancy: {fork_occupancy:.1f}/{self._fork_arity}, Leaf occupancy: {leaf_occupancy:.1f}/{self._leaf_arity}
        Entries in distance cache: {n_distance_cache_entries}
        Calculated distances: {self._calculated_distances_counter} / {self._worst_calculated_distances_counter} ({percent_calculated_distances:#.3g}%) '''
        return result
    
    def json(self, entry: Optional[_Entry[K]] = None) -> Any:
        '''Return JSON-convertible object representing the tree'''
        if entry is None:
            entry = self._root
        if isinstance(entry, _ForkEntry):
            key = str(entry)
            if isinstance(entry.children, _Fork):
                value = {}
                for child in entry.children:
                    value.update(self.json(child))
                return {key: value}
            else:
                value_ = []
                for child_ in entry.children:
                    value_.append(self.json(child_))
                return {key: value_}
        elif isinstance(entry, _LeafEntry):
            return str(entry)
        else: 
            raise AssertionError('MEntry must be one of: MForkEntry, MLeafEntry')
    
    def _get_distance_bounds(self, key1: Optional[K], key2: Optional[K]) -> Tuple[float, float]:
        assert not (key1 is None and key2 is None)
        if key1 is None or key2 is None:
            return (math.inf, math.inf)
        elif key1 == key2:
            return (0.0, 0.0)
        else:
            cache1 = self._distance_cache[key1]
            if key2 in cache1:
                dist = cache1[key2]
                return (dist, dist)
            cache2 = self._distance_cache[key2]
            if len(cache1) > len(cache2):
                cache1, cache2 = cache2, cache1
            lower, upper = 0.0, math.inf
            for p, d_p_k1 in cache1.items():
                if p in cache2:
                    d_p_k2 = cache2[p]
                    lower = max(lower, abs(d_p_k1 - d_p_k2))
                    upper = min(upper, d_p_k1 + d_p_k2)
            return (lower, upper)

    def _is_under_checked(self, key1: Optional[K], key2: Optional[K], r: float) -> bool:
        result = self._is_under(key1, key2, r)
        assert result == (self.get_distance(key1, key2) < r)
        return result

    def _is_under(self, key1: Optional[K], key2: Optional[K], r: float) -> bool:
        '''Decide if distance(key1, key2) <= r.
        Try to use all already-computed distances to avoid new computation.'''
        assert not (key1 is None and key2 is None)
        if key1 is None or key2 is None:
            return math.inf  < r
        if key1 == key2:
            return 0.0 < r
        cache1 = self._distance_cache[key1]
        if key2 in cache1:
            return cache1[key2] < r
        cache2 = self._distance_cache[key2]
        if len(cache1) > len(cache2):
            cache1, cache2 = cache2, cache1
        for p, d_p_k1 in cache1.items():
            if p in cache2:
                d_p_k2 = cache2[p]
                if r <= abs(d_p_k1 - d_p_k2):  # <= d(key1, key2)
                    return False
        return self.get_distance(key1, key2) < r

    def _test_get_distance_bound(self) -> None:
        qs = []
        for i, j in itertools.combinations(self._elements.keys(), 2):
            lower, upper = self._get_distance_bounds(i, j)
            dist = self._distance_function(self._elements[i], self._elements[j])
            assert lower <= dist <= upper
            qs.append(lower/dist)
        import numpy
        qsa = numpy.array(qs)
        print(numpy.median(qsa), numpy.mean(qsa), numpy.std(qsa))

    def _all_get_distance_bound(self) -> None:
        for i, j in itertools.combinations(self._elements.keys(), 2):
            lower, upper = self._get_distance_bounds(i, j)
            # dist = self._distance_function(self._elements[i], self._elements[j])
            # assert lower <= dist <= upper

    def _str_lines(self, entry: _Entry[K], indent: int = 0) -> Iterator[str]:
        '''Generate the lines of str(self)'''
        yield '    ' * indent + str(entry)
        if isinstance(entry, _ForkEntry):
            for child in entry.children:
                yield from self._str_lines(child, indent=indent+1)
        elif isinstance(entry, _LeafEntry):
            pass
        else: 
            raise AssertionError('MEntry must be one of: MForkEntry, MLeafEntry')
    
    def _insert_to_entry(self, entry: _ForkEntry[K] , key: K) -> None:
        # TODO add docstring
        node = entry.children
        if self._update_pivots_on_insert and entry.parent is not None:
            new_rc_bound = self._calculate_rc_upper_bound(key, node)
            if new_rc_bound < entry.rc:
                entry.pivot = key
                entry.rc = new_rc_bound
                entry.dist_to_parent = self.get_distance(entry.parent.pivot, key)
                for child in node:
                    child.dist_to_parent = self.get_distance(key, child.pivot)
        if isinstance(node, _Leaf):
            new_entry = _LeafEntry(key=key, dist_to_parent=self.get_distance(entry.pivot, key), parent=entry)
            self._home_entries[key] = new_entry
            self._link(entry, new_entry)
            if len(node) > self._leaf_arity:
                self._split_entry(entry)
            if self._update_pivots_on_insert:
                self._upper_bound_rcs(entry)
        elif isinstance(node, _Fork):
            n_entries = len(node)
            distances = [self.get_distance(e.pivot, key) for e in node]
            covering = [i for i in range(n_entries) if distances[i] <= node[i].rc]
            if len(covering) > 0:
                best = min(covering, key = lambda i: distances[i])
            else:
                rc_extensions = [distances[i] - node[i].rc for i in range(n_entries)]
                best = min(range(n_entries), key = lambda i: rc_extensions[i])
                node[best].rc = distances[best]
            self._insert_to_entry(node[best], key)

    def _consolidate(self, entry: _ForkEntry[K], removed_key: K) -> None:
        '''If entry has only one child, eliminate entry and replace by the child'''
        children = entry.children
        assert isinstance(children, _Fork)
        if len(children) == 1:
            parent = entry.parent
            child = children[0]
            if entry.rc < child.rc:
                # change pivot in children
                assert entry.pivot is not None
                self._change_pivot(child, entry.pivot, entry.rc)
            if parent is None:
                self._set_root(child)
            else:
                assert isinstance(parent.children, _Fork)
                parent.children.remove(entry)
                self._link(parent, child)
                self._update_pivots_upwards(parent, removed_key=removed_key)

    def _update_pivots_upwards(self, entry: _ForkEntry[K], removed_key: K) -> None:
        '''Select the best new pivot from entry's children, or keep the old one (if best and not removed).
        Do this also for all ancestors. Also update rcs all the way up.'''
        if entry.parent is not None:
            if entry.pivot == removed_key:
                best_pivot, best_rc = self._select_pivot(entry)
                self._change_pivot(entry, best_pivot, best_rc)
            else:
                self._upper_bound_rcs(entry, recursive=False)
                best_pivot, best_rc = self._select_pivot(entry)
                if best_rc < entry.rc:
                    self._change_pivot(entry, best_pivot, best_rc)
            self._update_pivots_upwards(entry.parent, removed_key)

    def _select_pivot(self, entry: _ForkEntry[K]) -> Tuple[K, float]:
        '''Select the best pivot from the entry's children, return the pivot and rc.'''
        best_rc, best_pivot = min((self._calculate_rc_upper_bound(child.pivot, entry.children), child.pivot)
                                  for child in entry.children)
        assert best_pivot is not None
        return best_pivot, best_rc

    def _fork_entry_from_fork_children(self, pivot: K, children: List[_ForkEntry[K]]) -> _ForkEntry:
        assert len(children) > 0
        if len(children) == 1:
            return children[0]
        else:
            new_entry = _ForkEntry(pivot=pivot, rc=math.nan, dist_to_parent=math.inf, children=_Fork[K](), parent=None)
            for child in children:
                self._link(new_entry, child)
            new_entry.rc = max(child.dist_to_parent + child.rc for child in children)
            return new_entry

    def _fork_entry_from_leaf_children(self, pivot: K, rc: float, children: List[_LeafEntry[K]]) -> _ForkEntry:
        assert len(children) > 0
        new_entry = _ForkEntry(pivot=pivot, rc=rc, dist_to_parent=math.inf, children=_Leaf[K](), parent=self._root)
        for child in children:
            self._link(new_entry, child)
        return new_entry

    def _split_entry(self, full_entry: _ForkEntry[K]) -> None:
        # Create new entries
        node = full_entry.children
        if isinstance(node, _Fork):
            assert len(node) > self._fork_arity  # debug
            elements_rcs = [] #(entry.pivot, entry.rc) for entry in node if entry.pivot is not None]
            for entry in node:
                assert entry.pivot is not None
                elements_rcs.append((entry.pivot, entry.rc))
            p1, p2, children1, children2, rc1, rc2 = self._select_pivots(elements_rcs)
            new_entry1 = self._fork_entry_from_fork_children(p1, [node[i] for i in children1])
            new_entry2 = self._fork_entry_from_fork_children(p2, [node[i] for i in children2])
        elif isinstance(node, _Leaf):
            assert len(node) > self._leaf_arity  # debug
            elements_rcs = [(entry.key, 0.0) for entry in node]
            p1, p2, children1, children2, rc1, rc2 = self._select_pivots(elements_rcs)
            new_entry1 = self._fork_entry_from_leaf_children(p1, rc1, [node[i] for i in children1])
            new_entry2 = self._fork_entry_from_leaf_children(p2, rc2, [node[i] for i in children2])
        else: 
            raise AssertionError('MNode must be one of: MFork, MLeaf')

        # Place new node into the tree
        if full_entry.parent is None:  # root
            self._root = _ForkEntry(children=_Fork[K]())
            self._link(self._root, new_entry1)
            self._link(self._root, new_entry2)
        else:  # non-root
            parent = full_entry.parent
            assert isinstance(parent.children, _Fork)
            parent.children.remove(full_entry)
            self._link(parent, new_entry1)
            self._link(parent, new_entry2)
            if len(parent.children) > self._fork_arity:
                self._split_entry(parent)
            else:
                self._upper_bound_rcs(parent)

    def _upper_bound_rcs(self, bottom_entry: _ForkEntry[K], recursive=True) -> None:
        '''Decrease covering radii if excessive in bottom_entry. If recursive, continue with all its ancestors'''
        rc_upper_bound = max(child.dist_to_parent + child.rc for child in bottom_entry.children)
        bottom_entry.rc = min(bottom_entry.rc, rc_upper_bound)
        if recursive and bottom_entry.parent is not None:
            self._upper_bound_rcs(bottom_entry.parent)
    
    def _calculate_rc_upper_bound(self, pivot: Optional[K], children: _Node[K]) -> float:
        return max(self.get_distance(pivot, child.pivot) + child.rc for child in children)

    def _link(self, parent: _ForkEntry[K], child: _ForkEntry[K]|_LeafEntry[K]) -> None:
        if isinstance(child, _ForkEntry):
            assert isinstance(parent.children, _Fork), 'Cannot link fork entry into leaf'
            parent.children.append(child)
            child.parent = parent
            child.dist_to_parent = self.get_distance(parent.pivot, child.pivot)
        elif isinstance(child, _LeafEntry):
            assert isinstance(parent.children, _Leaf), 'Cannot link leaf entry into fork'
            parent.children.append(child)
            child.parent = parent
            child.dist_to_parent = self.get_distance(parent.pivot, child.key)
        else:
            raise AssertionError('MNode must be one of: MFork, MLeaf')

    def _set_root(self, new_root: _ForkEntry[K]) -> None:
        self._root = new_root
        new_root.parent = None
        new_root.pivot = None
        new_root.rc = math.inf
        for child in new_root.children:
            child.dist_to_parent = math.inf

    def _change_pivot(self, entry: _ForkEntry[K], new_pivot: K, new_rc: float) -> None:
        assert entry.parent is not None
        entry.pivot, entry.rc = new_pivot, new_rc
        entry.dist_to_parent = self.get_distance(entry.parent.pivot, entry.pivot)
        for child in entry.children:
            child.dist_to_parent = self.get_distance(entry.pivot, child.pivot)

    def _select_pivots(self, elements_rcs: List[Tuple[K, float]]) -> Tuple[K, K, List[int], List[int], float, float]:
        '''Select two pivots by minMaxRC criterion, return them, indices of their children , and covering radii'''
        assert len(elements_rcs) >= 2
        best_result: Tuple[K, K, List[int], List[int], float, float]
        best_result = elements_rcs[0][0], elements_rcs[1][0], [], [], math.inf, math.inf  # will be overwritten
        best_max_rc = math.inf
        for (i, rc_i), (j, rc_j) in itertools.combinations(elements_rcs, 2):
            new_children_i, new_children_j, new_rc_i, new_rc_j = self._partition(i, j, elements_rcs)
            new_max_rc = max(new_rc_i, new_rc_j)
            if new_max_rc < best_max_rc:
                best_result = i, j, new_children_i, new_children_j, new_rc_i, new_rc_j
                best_max_rc = new_max_rc
        return best_result
    
    def _partition(self, pivot1: K, pivot2: K, elements_rcs: Iterable[Tuple[K, float]]) -> Tuple[List[int], List[int], float, float]:
        '''Return list of element indices belonging to the two pivots and estimation of their covering radii'''
        children1 = []
        children2 = []
        rc1 = 0.0
        rc2 = 0.0
        last_tie = 2
        for i, (elem, rc_elem) in enumerate(elements_rcs):
            d_p1_elem = self.get_distance(pivot1, elem)
            d_p2_elem = self.get_distance(pivot2, elem)
            if d_p1_elem < d_p2_elem:
                to = 1
            elif d_p1_elem > d_p2_elem:
                to = 2
            elif last_tie == 1:  # tie
                to = last_tie = 2
            else:
                to = last_tie = 1
            if to == 1:
                children1.append(i)
                rc1 = max(rc1, d_p1_elem + rc_elem)
            else:  # to == 2
                children2.append(i)
                rc2 = max(rc2, d_p2_elem + rc_elem)
            # if d_p1_elem <= d_p2_elem:
            #     children1.append(i)
            #     rc1 = max(rc1, d_p1_elem + rc_elem)
            # else: 
            #     children2.append(i)
            #     rc2 = max(rc2, d_p2_elem + rc_elem)
        assert len(children1) > 0
        assert len(children2) > 0
        return children1, children2, rc1, rc2

    def _check_invariants(self) -> None:
        fork_entries, leaf_entries = self._collect_entries_by_type()
        # Pointers parent-child
        for fork in fork_entries:
            for child in fork.children:
                assert child.parent == fork
            if fork.parent is not None:
                fork in fork.parent.children
        for leaf in leaf_entries:
            leaf in leaf.parent.children
        
        # Distances 
        for fork in fork_entries:
            for child in fork.children:
                child_pivot = child.key if isinstance(child, _LeafEntry) else child.pivot
                assert child.dist_to_parent == self.get_distance(fork.pivot, child_pivot), f'\n{fork}\n-> {child}'

        # Node arity and no useless nodes
        for fork in fork_entries:
            node = fork.children
            if isinstance(node, _Fork):
                assert 1 <= len(node) <= self._fork_arity
                assert 2 <= len(node)  # Would be useless node
            else:
                if len(self) > 0:
                    assert 1 <= len(node) <= self._leaf_arity

        # Covering radii (from objects or from children)
        for fork in fork_entries:
            if fork.pivot is not None:
                objs = [leafe.key for leafe in self._collect_entries_by_type(start_entry=fork)[1]]
                rc_objs = max(self.get_distance(fork.pivot, obj) for obj in objs)
                if isinstance(fork.children, _Leaf):
                    assert fork.rc == rc_objs, f'{fork.rc}, {rc_objs}'
                elif isinstance(fork.children, _Fork):
                    rc_children = max(self.get_distance(fork.pivot, child.pivot) + child.rc for child in fork.children)
                    assert rc_objs <= fork.rc <= rc_children, f'{fork.pivot}: {rc_objs} <= {fork.rc} <= {rc_children}'
        # print('OK')

    def _collect_entries(self, start_entry: _Entry[K]) -> Iterator[_Entry[K]]:
        '''Iterate over all entries in the tree'''
        yield start_entry
        if isinstance(start_entry, _ForkEntry):
            for child in start_entry.children:
                yield from self._collect_entries(child)

    def _collect_entries_by_type(self, start_entry: _Entry[K] = None) -> Tuple[List[_ForkEntry[K]], List[_LeafEntry[K]]]:
        '''Return list of fork- and leaf-entries. By default start in the root.'''
        start_entry = start_entry or self._root
        forks, leafs = [], []
        for entry in self._collect_entries(start_entry):
            if isinstance(entry, _ForkEntry):
                forks.append(entry)
            elif isinstance(entry, _LeafEntry):
                leafs.append(entry)
            else: 
                raise AssertionError('MEntry must be one of: MForkEntry, MLeafEntry')
        return forks, leafs

    def _kNN_query_classical(self, start_entry: _ForkEntry[K], query: K, work_list: List[Tuple[float, Optional[K]]], include_query: bool = False, pivot_prox=False) -> None:
        '''Perform classical kNN query starting in start_entry, optionally skipping one of the children.
        work_list must contain k nearest elements found so far and their distances and will be updated with newly found elements. 
        Use (math.inf, None) if elements are not found yet.'''
        dmin = max(0, self.get_distance(start_entry.pivot, query) - start_entry.rc)
        queue = lib.PriorityQueue([(dmin, start_entry)])
        if pivot_prox:
            self._kNN_query_process_queue_pp(query, queue, work_list, include_query)
        else:
            self._kNN_query_process_queue(query, queue, work_list, include_query)
        
    def kNN_query_classical_with_bottom_guess(self, query: K, k: int, include_query: bool = False) -> List[K]:
        assert query in self
        work_list: List[Tuple[float, Optional[K]]] = [(math.inf, None)] * k

        entry = self._home_entries[query].parent
        dmin = max(0, self.get_distance(entry.pivot, query) - entry.rc)
        queue = lib.PriorityQueue([(dmin, entry)])
        self._kNN_query_process_queue(query, queue, work_list, include_query)

        while entry.parent is not None:
            for sibling in entry.parent.children:
                if sibling is not entry:
                    assert isinstance(sibling, _ForkEntry)
                    dmin = max(0, self.get_distance(sibling.pivot, query) - sibling.rc)
                    queue.add(dmin, sibling)
            entry = entry.parent
        self._kNN_query_process_queue(query, queue, work_list, include_query)

        return [key for dist, key in work_list if key is not None]
        
    def kNN_query_bottom_up(self, query: K, k: int, include_query: bool = False) -> List[K]:
        assert query in self
        work_list: List[Tuple[float, Optional[K]]] = [(math.inf, None)] * k

        entry = self._home_entries[query].parent
        found = False
        for le in entry.children:
            assert isinstance(le, _LeafEntry)
            if le.key == query:
                found = True
        assert found
        dmin = max(0, self.get_distance(entry.pivot, query) - entry.rc)
        queue = lib.PriorityQueue([(dmin, entry)])
        self._kNN_query_process_queue(query, queue, work_list, include_query)

        while entry.parent is not None:
            for sibling in entry.parent.children:
                if sibling is not entry:
                    assert isinstance(sibling, _ForkEntry)
                    dmin = max(0, self.get_distance(sibling.pivot, query) - sibling.rc)
                    queue.add(dmin, sibling)
            entry = entry.parent
            self._kNN_query_process_queue(query, queue, work_list, include_query)

        return [key for dist, key in work_list if key is not None]
        
    def _kNN_query_process_queue(self, query, queue, work_list, include_query) -> None:
        PIVOT_PIVOT_CONSTRAINT = True
        USE_IS_UNDER = False # True
        while not queue.is_empty():
            current_range = work_list[-1][0]
            best = queue.pop_min()
            assert best is not None
            dmin, entry = best
            # print('range', current_range, '   popped', dmin, entry)
            if USE_IS_UNDER:
                dmin =  max(0, self.get_distance(entry.pivot, query) - entry.rc)
            if dmin >= current_range:
                break
            node = entry.children
            if isinstance(node, _Fork):
                for child in node:
                    if PIVOT_PIVOT_CONSTRAINT:
                        d_pp_q = self.get_distance(entry.pivot, query)
                        d_pp_p = self.get_distance(entry.pivot, child.pivot)
                        if abs(d_pp_q - d_pp_p) - child.rc > current_range:
                            continue
                    if USE_IS_UNDER:
                        if self._is_under(child.pivot, query, child.rc + current_range):
                            dmin = max(0, self.get_distance(child.pivot, query) - child.rc)
                            queue.add(dmin, child)
                    else:
                        dmin = max(0, self.get_distance(child.pivot, query) - child.rc)
                        if dmin < current_range:
                            queue.add(dmin, child)
            elif isinstance(node, _Leaf):
                for leaf_entry in node:
                    if leaf_entry.key != query or include_query:
                        if PIVOT_PIVOT_CONSTRAINT:
                            d_p_q = self.get_distance(entry.pivot, query)
                            d_p_o = self.get_distance(entry.pivot, leaf_entry.key)                        
                            if abs(d_p_q - d_p_o) > current_range:
                                continue
                        if USE_IS_UNDER:
                            if self._is_under(query, leaf_entry.key, current_range):
                                dist = self.get_distance(query, leaf_entry.key)
                                self._bubble_into_sorted_list(work_list, (dist, leaf_entry.key))
                        else:
                            dist = self.get_distance(query, leaf_entry.key)
                            self._bubble_into_sorted_list(work_list, (dist, leaf_entry.key))
            else:
                raise AssertionError('MNode must be one of: MFork, MLeaf')

    def _kNN_query_process_queue_pp(self, query, queue, work_list, include_query) -> None:
        while not queue.is_empty():
            current_range = work_list[-1][0]
            best = queue.pop_min()
            assert best is not None
            d_pq, entry = best
            dmin = max(0, self.get_distance(entry.pivot, query) - entry.rc)
            if dmin >= current_range:
                continue
            node = entry.children
            if isinstance(node, _Fork):
                for child in node:
                    d_pq =self.get_distance(child.pivot, query)
                    queue.add(d_pq, child)
            elif isinstance(node, _Leaf):
                for leaf_entry in node:
                    if leaf_entry.key != query or include_query:
                        dist = self.get_distance(query, leaf_entry.key)
                        self._bubble_into_sorted_list(work_list, (dist, leaf_entry.key))
            else:
                raise AssertionError('MNode must be one of: MFork, MLeaf')

            
       
        # if isinstance(start_entry.children, MLeaf):
        #     for leaf_entry in start_entry.children:
        #         if leaf_entry.key != query or include_query:
        #             dist = self.get_distance(query, leaf_entry.key)
        #             self.bubble_into_sorted_list(work_list, (dist, leaf_entry.key))
       
  
    def _recursive_template_entries(self, entry: _Entry[K], key: K) -> None:
        # TODO add docstring
        if isinstance(entry, _ForkEntry):
            raise NotImplementedError
        elif isinstance(entry, _LeafEntry):
            raise NotImplementedError
        else: 
            raise AssertionError('MEntry must be one of: MForkEntry, MLeafEntry')

    @staticmethod
    def _bubble_into_sorted_list(lst: List[Any], new_elem: Any) -> None:
        '''Add new_elem to a sorted list lst, keep it sorted, and remove the new largest element.'''
        if new_elem < lst[-1]:
            n = len(lst)
            lst[-1] = new_elem
            for i in range(n-1, 0, -1):
                if lst[i] < lst[i-1]:
                    lst[i-1], lst[i] = lst[i], lst[i-1]

# @dataclass
class CacheEntry(NamedTuple):
    exact: bool
    lower: float
    upper: float

class DistanceEstimator(Generic[K, V]):
    def __init__(self, distance_function: Callable[[V, V], float]):
        self._distance_function: Callable[[V, V], float] = distance_function
        self._elements: Dict[K, V] = {}  # mapping of element keys to values (i.e. objects on which distance function is defined)
        self._distance_cache: Dict[K, Dict[K, CacheEntry]] = {}  # distance_cache[i][j] = (is_exact, lower_bound, upper_bound)
        self._calculated_distances_counter = 0  # really calculated distances
        self._worst_calculated_distances_counter = 0  # theoretical number of calculated distances in worst case (i.e. each-to-each)
    
    def insert(self, key: K, value: V) -> None:
        '''Insert a new element with given key and value. The key must be unique.'''
        assert key not in self._elements, f'Key {key} is already in the tree'
        self._worst_calculated_distances_counter += len(self._elements)
        self._elements[key] = value
        self._distance_cache[key] = {}
    
    def get_distance(self, key1: K, key2: K) -> float:
        '''Get the distance between two objects'''
        if key1 == key2:
            return 0.0
        try: 
            exact, lower, upper = self._distance_cache[key1][key2]
            if exact:
                return lower
        except KeyError:
            pass
        dist = self._distance_function(self._elements[key1], self._elements[key2])
        self._distance_cache[key1][key2] = self._distance_cache[key2][key1] = CacheEntry(True, dist, dist)
        self._calculated_distances_counter += 1
        return dist

    def _is_under(self, i: K, j: K, r: float) -> bool:
        '''Decide if distance(key1, key2) <= r.
        Try to use all already-computed distances to avoid new computation.'''
        if i == j:
            return 0.0 < r
        cache1 = self._distance_cache[i]
        if j in cache1:
            exact, lower, upper = cache1[j]
            if upper < r:
                return True
            if r <= lower:
                return False
        cache2 = self._distance_cache[j]
        if len(cache1) > len(cache2):
            cache1, cache2 = cache2, cache1
        # for p, (e_pi, d_pi_L, d_pi_H) in cache1.items():
        #     if p in cache2:
        #         (e_pj, d_pj_L, d_pj_H) = cache2[p]
        #         d_ij_L = max(0, d_pi_L - d_pj_H, d_pj_L - d_pi_H)
        #         d_ij_H = d_pi_H + d_pj_H
        #         # TODO continue here
        #         if r <= abs(d_p_k1 - d_p_k2):  # <= d(key1, key2)
        #             return False
        return self.get_distance(i, j) < r
