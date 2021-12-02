'''Bubble tree?'''

from typing import Generic, TypeVar, List, Tuple, Dict, Set, Optional, Callable, Final, Iterator, Any, Counter, Sequence, Literal, Deque, Iterable
from dataclasses import dataclass, field
import numpy as np
import json

from .abstract_similarity_tree import AbstractSimilarityTree, K, V
from .caches import FunctionCache, DistanceCache, MinFinder
from ..lib import PriorityQueue, ProgressBar, Timing


ZERO_ELEMENT = '-'
TOLERANCE = 0.0


@dataclass
class _BNode(Generic[K]):
    pivot: Optional[K]
    radius: float
    rc: float = 0.0
    children: List['_BNode[K]'] = field(default_factory=list)
    parent: Optional['_BNode[K]'] = field(default=None, repr=False)


class BubbleTree(AbstractSimilarityTree[K, V]):
    _distance_cache: DistanceCache[K, V]
    _root: Optional[_BNode[K]]
    _MAX_RADIUS: float = 1024.0
    _MIN_RADIUS: float = 1.0  # 1.0? 0.5? 0.25?
    _Q: float = 2  # what other quotient? 1.618? :)

    def __init__(self, distance_function: Callable[[V, V], float], keys_values: Sequence[Tuple[K, V]] = ()) -> None:
        self._distance_cache = DistanceCache(distance_function)
        self._root = _BNode(pivot=ZERO_ELEMENT, radius=self._MAX_RADIUS)
        n = len(keys_values)
        self.insert(ZERO_ELEMENT, None)
        with ProgressBar(n, title=f'Loading {n} elements into BubbleTree') as bar, Timing(f'Loading {n} elements into BubbleTree'):
            for k, v in keys_values:
                self.insert(k, v)
                bar.step()

    def _distance_from_zero(self, value: V) -> float:
        assert type(value).__name__ == 'StructInfo'
        return 0.5 * value.n
    
    def insert(self, key: K, value: V) -> None:
        if key != ZERO_ELEMENT:
            self._distance_cache.insert(key, value)
        self._insert_to_node2(self._root, key)
    
    def _insert_to_node2(self, node: _BNode[K], key: K) -> None:
        if node.pivot is None:
            node.pivot = key
        d_p_k = self.get_distance(node.pivot, key)
        assert d_p_k <= node.radius
        node.rc = max(node.rc, d_p_k)
        subradius = node.radius / self._Q
        if subradius < self._MIN_RADIUS:
            # creating leaf
            new_leaf = _BNode(pivot=key, radius=0, parent=node)
            node.children.append(new_leaf)
        else:
            if len(node.children) == 0:
                new_child = _BNode(pivot=node.pivot, radius=subradius, parent=node)
                node.children.append(new_child)
            for child in node.children:
                d_c_k = self.get_distance(child.pivot, key)
                if d_c_k <= subradius:
                    self._insert_to_node2(child, key)
                    break
            else:
                new_child = _BNode(pivot=key, radius=subradius, parent=node)
                node.children.append(new_child)
                self._insert_to_node2(new_child, key)
        assert d_p_k <= node.rc <= node.radius
    
    def __len__(self) -> int:
        '''Return number of elements'''
        return len(self._distance_cache)

    def __contains__(self, element) -> bool:
        '''Decide if element is present in the tree'''
        return element in self._distance_cache

    def get_distance(self, key1: K, key2: K) -> float:
        if key1 == key2 == ZERO_ELEMENT:
            return 0.0
        elif key1 == ZERO_ELEMENT:
            return self._distance_from_zero(self._distance_cache._elements[key2])
        elif key2 == ZERO_ELEMENT:
            return self._distance_from_zero(self._distance_cache._elements[key1])
        else:
            return self._distance_cache.get_distance(key1, key2)

    def _get_distance_to_value(self, key1: K, value2: K) -> float:
        if key1 == ZERO_ELEMENT:
            return self._distance_from_zero(value2)
        else:
            return self._distance_cache.get_distance_to_value(key1, value2)

    def get_statistics(self):
        result = f'''Bubble tree statistics:
        Elements: {len(self)}
        {self._distance_cache.get_statistics()}'''
        return result
    
    def kNN_query_by_value(self, query_value: V, k: int, distance_low_bounds: List[Callable[[V, V], float]] = []) -> List[Tuple[float, K]]:
        query_dist = FunctionCache[K, float](lambda key: self._get_distance_to_value(key, query_value))
        besties = MinFinder[K](n=k)
        queue = PriorityQueue[float, _BNode]()
        seen = set()
        root = self._root
        if root.pivot is None:  # empty tree
            return besties.pop_all_not_none()
        d_r_q = query_dist[root.pivot]
        current_range = besties.bubble_in(d_r_q, root.pivot)
        seen.add(root.pivot)
        queue.add(0.0, root)
        while not queue.is_empty():
            best = queue.pop_min()
            assert best is not None
            dmin, node = best
            if dmin >= current_range + TOLERANCE:
                continue
            d_excluded = 0.0
            for child in node.children:
                assert child.pivot is not None
                # if not self._can_be_under(child, query_dist, [node], current_range + child.rc + TOLERANCE):
                #     continue
                # if not self._can_be_under(child, query_dist, self._gen_ancestors(child), current_range + child.rc + TOLERANCE):
                #     continue
                if not self._can_be_under(child, query_dist, self._gen_older_siblings(child), current_range + child.rc + TOLERANCE):  # better than uncles
                    continue
                # if not self._can_be_under(child, query_dist, self._gen_older_uncles(child), current_range + child.rc + TOLERANCE):
                #     continue
                d_c_q = query_dist[child.pivot]
                if child.pivot not in seen:
                    current_range = besties.bubble_in(d_c_q, child.pivot)
                    seen.add(child.pivot)
                dmin_child = max(dmin, d_c_q - child.rc, d_excluded)
                if dmin_child < current_range + TOLERANCE:
                    queue.add(dmin_child, child)
                d_excluded = max(d_excluded, child.radius - d_c_q)
                if d_excluded > current_range + TOLERANCE:
                    break
        result = besties.pop_all_not_none()
        return result

    def _distance_bound_from_ancestors(self, node: _BNode[K], query_distance: FunctionCache[K, float]) -> float:
        '''Return lower bound for distance pivot-query, using all its ancestors 
        (it is expected that all ancestor-query and ancestor-pivot distances are already known, otherwise they will be calculated).''' 
        d_p_q_low = 0.0 
        ancestor = node.parent
        while ancestor is not None:
            d_p_a = self.get_distance(node.pivot, ancestor.pivot)
            d_a_q = query_distance[ancestor.pivot]
            d_p_q_low = max(d_p_q_low, abs(d_p_a - d_a_q))
            ancestor = ancestor.parent
        return d_p_q_low

    def _distance_bound_from_siblings(self, node: _BNode[K], query_distance: FunctionCache[K, float]) -> float:
        '''Return lower bound for distance pivot-query, using all its older siblings (if their distance to query is known).'''
        d_p_q_low = 0.0 
        siblings = node.parent.children if node.parent is not None else ()
        for sibling in siblings:
            if sibling.pivot == node.pivot:
                break
            if sibling.pivot in query_distance:
                d_p_s = self.get_distance(node.pivot, sibling.pivot)
                d_s_q = query_distance[sibling.pivot]
                d_p_q_low = max(d_p_q_low, abs(d_p_s - d_s_q))
        return d_p_q_low

    def _distance_bound(self, node: _BNode[K], query_distance: FunctionCache[K, float], through: Iterable[_BNode[K]]) -> float:
        d_p_q_low = 0.0 
        for t in through:
            if t.pivot in query_distance:
                d_p_t = self.get_distance(node.pivot, t.pivot)
                d_t_q = query_distance[t.pivot]
                d_p_q_low = max(d_p_q_low, abs(d_p_t - d_t_q))
        return d_p_q_low

    def _can_be_under(self, node: _BNode[K], query_distance: FunctionCache[K, float], through: Iterable[_BNode[K]], limit: float) -> float:
        '''Decide if the distance between node.pivot and query can be less than limit, using nodes from through as pivots.'''
        # d_p_q = query_distance[node.pivot]
        for t in through:
            if t.pivot in query_distance:
                d_p_t = self.get_distance(node.pivot, t.pivot)
                d_t_q = query_distance[t.pivot]
                d_p_q_low = abs(d_p_t - d_t_q)
                # assert d_p_q_low <= d_p_q, f'{d_p_q_low} <= {d_p_q}'
                if d_p_q_low >= limit:
                    return False
        return True
    
    def _gen_ancestors(self, node: _BNode[K]) -> Iterator[_BNode]:
        ancestor = node.parent
        while ancestor is not None:
            yield ancestor
            ancestor = ancestor.parent

    def _gen_older_siblings(self, node: _BNode[K]) -> Iterator[_BNode]:
        if node.parent is not None:
            siblings = node.parent.children if node.parent is not None else ()
            for sibling in siblings:
                if sibling.pivot == node.pivot:
                    break
                yield sibling

    def _gen_older_uncles(self, node: _BNode[K]) -> Iterator[_BNode]:
        yield from self._gen_older_siblings(node)
        for ancestor in self._gen_ancestors(node):
            yield ancestor
            yield from self._gen_older_siblings(ancestor)                

    def _node_to_json(self, node: _BNode[K]) -> Dict[str, Any]:
        return {
            'pivot': node.pivot,
            'radius': node.radius,
            'rc': node.rc,
            'n_children': len(node.children),  # TODO eventually remove
            'children': [self._node_to_json(child) for child in node.children]
        }

    def _node_from_json(self, js: Dict[str, Any]) -> object:
        ...
    
    def json(self, with_cache: bool = False, **kwargs) -> str:
        result = {
            'MAX_RADIUS': self._MAX_RADIUS, 
            'MIN_RADIUS': self._MIN_RADIUS, 
            'Q': self._Q, 
            'root': self._node_to_json(self._root),
            'distance_cache': self._distance_cache.json() if with_cache else []
        }
        # TODO try saving only needed cached distances (pivot-ancestor, pivot-sibling...)
        return json.dumps(result, **kwargs)

    def save(self, file: str, with_cache: bool = False) -> None:
        with open(file, 'w') as w:
            w.write(self.json(with_cache=with_cache, indent=1))

