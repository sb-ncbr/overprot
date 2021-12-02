'''
Graph theory and embedding directed acyclic graphs.
'''

from collections import defaultdict
from typing import List, Dict, Set, Tuple, Iterable, Sequence, Optional, NamedTuple, Literal
from dataclasses import dataclass, field
import itertools
import numpy as np

from .lib_logging import Timing


@dataclass
class Dag(object):
    levels: List[List[int]]
    edges: List[Tuple[int, int]]

    _vertices: Optional[List[int]] = field(init=False, default=None)
    _in_neighbors: Optional[Dict[int, List[int]]] = field(init=False, default=None)

    @property
    def vertices(self) -> List[int]:
        '''Return the list of all vertices in all levels (ordered by the levels).'''
        if self._vertices is None:
            self._vertices = [v for level in self.levels for v in level]
        return self._vertices

    @property
    def in_neighbors(self) -> Dict[int, List[int]]:
        '''Return dictionary d, such that d[v] is the list of all vertices having an edge to v.'''
        if self._in_neighbors is None:
            self._in_neighbors = {v: [] for v in self.vertices}
            for u, v, *_ in self.edges:
                self._in_neighbors[v].append(u)
        return self._in_neighbors
   
    def _get_slice(self, level_from: int, level_to: int) -> 'Dag':
        slice_levels = self.levels[level_from:level_to]
        slice_edges = [(u, v) for lev in slice_levels[1:] for v in lev for u in self.in_neighbors[v]]
        return Dag(slice_levels, slice_edges)

    def slices_backup(self) -> List['Dag']:
        '''Find all n "slice-vertices" v[i] such that 
            Vertex v is a slice-vertex <==> foreach vertex u. exists path u->v or exists path v-> or u==v
        Find all n+1 "slices" S[i] such that
            u in S[0] <==> exists path         u->v[0]
            u in S[i] <==> exists path v[i-1]->u->v[i]
            u in S[n] <==> exists path v[n-1]->u
        Return list S[0], v[0], S[1], v[1], ... v[n-1], S[n]. (ommitting empty S[i])
        '''
        ancestors: Dict[int, Set[int]] = {}
        seen_vertices = 0
        slices = []
        last_slice_vertex = -1
        for i_level, level in enumerate(self.levels):
            for vertex in level:
                ancestors[vertex] = set(self.in_neighbors[vertex]).union(*(ancestors[u] for u in self.in_neighbors[vertex]))
            if len(level) == 1 and len(ancestors[level[0]]) == seen_vertices:
                # This is a slice-vertex
                slice_vertex = level[0]
                if i_level > last_slice_vertex + 1:
                    slices.append(self._get_slice(last_slice_vertex+1, i_level))
                slices.append(Dag([[slice_vertex]], []))
                ancestors.clear()
                ancestors[slice_vertex] = set()
                last_slice_vertex = i_level
            seen_vertices += len(level)
        if len(self.levels) > last_slice_vertex+1:
            slices.append(self._get_slice(last_slice_vertex+1, len(self.levels)))
        assert {v for s in slices for v in s.vertices} == set(self.vertices)
        return slices
   
    def slices(self) -> List['Dag']:
        '''Find all n "slice-vertices" v[i] such that 
            Vertex v is a slice-vertex <==> foreach vertex u. exists path u->v or exists path v-> or u==v
        Find all n+1 "slices" S[i] such that
            u in S[0] <==> exists path         u->v[0]
            u in S[i] <==> exists path v[i-1]->u->v[i]
            u in S[n] <==> exists path v[n-1]->u
        Return list S[0], v[0], S[1], v[1], ... v[n-1], S[n]. (ommitting empty S[i])
        '''
        ancestors: Dict[int, Set[int]] = {}
        seen_vertices = 0
        slices = []
        last_slice_vertex = -1
        for i_level, level in enumerate(self.levels):
            # print('level', i_level, level)
            for vertex in level:
                ancestors[vertex] = set(self.in_neighbors[vertex]).union(*(ancestors[u] for u in self.in_neighbors[vertex]))
            # print('    ancestors', ancestors)
            if len(level) == 1 and len(ancestors[level[0]]) == seen_vertices:
                # This is a slice-vertex
                slice_vertex = level[0]
                # print('    slice_vertex', slice_vertex)
                if i_level > last_slice_vertex + 1:
                    slices.append(self._get_slice(last_slice_vertex+1, i_level))
                slices.append(Dag([[slice_vertex]], []))
                ancestors.clear()
                ancestors[slice_vertex] = set()
                last_slice_vertex = i_level
                seen_vertices = 1
            else:
                seen_vertices += len(level)
        if len(self.levels) > last_slice_vertex+1:
            slices.append(self._get_slice(last_slice_vertex+1, len(self.levels)))
        assert {v for s in slices for v in s.vertices} == set(self.vertices)
        return slices
   
    @staticmethod
    def from_precedence_matrix(precedence: np.ndarray) -> 'Dag': #Tuple[List[List[int]], List[Tuple[int, int]]]:
        '''Create minimal dag from precedence matrix and return vertex levels and list of edges.'''
        levels: List[List[int]] = []
        edges = []
        n = precedence.shape[0]
        # built_precedence = np.zeros(precedence.shape, dtype=bool)  # precedence which is given by transitivity of edges

        def are_transitively_connected(from_level, from_vertex, to_vertex, in_neighbours_of_to_vertex):
            for level in reversed(levels[from_level+1:]):
                for v in level:
                    if precedence[from_vertex, v] and v in in_neighbours_of_to_vertex:
                        return True
            return False

        todo = set(range(n))
        while len(todo) > 0:
            mins = [i for i in todo if not any(i!=j and precedence[j, i] for j in todo)]
            if len(mins)==0:
                raise Exception('Cyclic graph.')
            for v in mins:
                todo.remove(v)
            for v in mins:
                in_neighbours_of_v: List[int] = []
                for i in reversed(range(len(levels))):
                    for u in levels[i]:
                        must_join = precedence[u, v] and not are_transitively_connected(i, u, v, in_neighbours_of_v)
                        if must_join:
                            edges.append((u, v))
                            in_neighbours_of_v.append(u)
            levels.append(mins)
        return Dag(levels, edges)
    
    @staticmethod
    def from_path(vertices: List[int]) -> 'Dag':
        levels = [[v] for v in vertices]
        edges = list(zip(vertices[:-1], vertices[1:]))
        return Dag(levels, edges)

class XY(NamedTuple):
    x: float
    y: float
    def __add__(self, other: tuple) -> 'XY':
        if isinstance(other, XY):
            return XY(self.x + other.x, self.y + other.y)
        else:
            raise TypeError('Adding XY and other tuple type is ambigous.')
    def round(self, ndigits=None) -> 'XY':
        return XY(round(self.x, ndigits), round(self.y, ndigits))

class Size(NamedTuple):
    width: float
    height: float

class Box(NamedTuple):
    '''Boundaries of a box with center in [0,0].
    All dimensions are positive. Center means geometrical center along x, center of gravity along y.'''
    left: float
    right: float
    top: float
    bottom: float
    weight: float

    def add_margins(self, left_margin: float, right_margin: float, top_margin: float, bottom_margin: float) -> 'Box':
        return Box(self.left + left_margin, self.right + right_margin, self.top + top_margin, self.bottom + bottom_margin, self.weight)

def implicit_vertices(edges: List[Tuple[int,int]]) -> Set[int]:
    vertices = set()
    for u, v, *_ in edges:
        vertices.add(u)
        vertices.add(v)
    return vertices

def connected_components(vertices: Optional[Iterable[int]], edges: List[Tuple[int,int]]) -> List[List[int]]:
    '''Get connected components of a graph (efficient implementation with shallow tree).
    If vertices is None, the set of vertices is inferred from the set of edges.
    Connected components and the vertices within them are sorted.'''
    if vertices is None:
        vertices = implicit_vertices(edges)
    shallow_tree = _ShallowTree(vertices)
    for u, v, *_ in edges:
        shallow_tree.join(u, v)
    return shallow_tree.sets(sort=True)

def dag_connected_components(dag: Dag) -> List[Dag]:
    components = connected_components(dag.vertices, dag.edges)
    n_components = len(components)
    vertex2component = {v: c for c, component in enumerate(components) for v in component}
    dags = [Dag([], []) for i in range(n_components)]
    for i_level, level in enumerate(dag.levels):
        for vertex in level:
            i_comp = vertex2component[vertex]
            levs = dags[i_comp].levels
            if len(levs) <= i_level:
                levs.append([])
            assert len(levs) == i_level+1
            levs[i_level].append(vertex)
    for edge in dag.edges:
        i_comp = vertex2component[edge[0]]
        assert i_comp == vertex2component[edge[1]]
        dags[i_comp].edges.append(edge)
    return dags

class _ShallowTree(object):
    '''Efficient representation of a set of disjoint sets. 
    Each set is identified by its 'root', which is its smallest element.'''
    parents: Dict[int, int]

    def __init__(self, elements: Iterable[int]) -> None:
        '''Initialize with some elements, so that each element forms its own set.'''
        self.parents = {i: i for i in elements}
    def root(self, element: int) -> int:
        '''Get the root of the set containing element.'''
        if self.parents[element] == element:
            return element  # This element is the root
        else:
            the_root = self.root(self.parents[element])
            self.parents[element] = the_root
            return the_root
    def join(self, i: int, j: int) -> int:
        '''Join the set containing i and the set containing j, return the root of the new set. (Do nothing if i and j already are in the same set.)'''
        root_i = self.root(i)
        root_j = self.root(j)
        new_root = min(root_i, root_j)
        self.parents[root_i] = self.parents[root_j] = new_root
        return new_root
    def sets(self, sort=False) -> List[List[int]]:
        '''Return the list of sets, each set itself is a list of elements. 
        If sort==True, then each set is sorted (its root is first) and the sets are sorted by their root.'''
        set_dict = defaultdict(list)  # map roots to their sets
        for elem in self.parents:
            set_dict[self.root(elem)].append(elem)
        set_list = list(set_dict.values())
        if sort:
            for s in set_list:
                s.sort()
            set_list.sort()
        return set_list

def _align_top_left(embedding: Tuple[Box, Dict[int, XY]]) -> Tuple[Box, Dict[int, XY]]:
    '''Shift embedding so that the top left corner of the bounding box is at [0, 0].'''
    box, positions = embedding
    shift = XY(box.left, box.top)
    new_positions = {i: pos + shift for i, pos in positions.items()}
    total_width = box.left + box.right
    total_height = box.top + box.bottom
    new_box = Box(0.0, total_width, 0.0, total_height, box.weight)
    return new_box, new_positions

def embed_dag(dag: Dag, vertex_sizes: Dict[int, Size], *, 
              x_padding=0.0, y_padding=0.0, 
              left_margin=0.0, right_margin=0.0, top_margin=0.0, bottom_margin=0.0, align_top_left=False) -> Tuple[Box, Dict[int, XY]]:
    '''Place vertices in a plane so that all edges go from left to right, use heuristics to make it look nice.
    Return the bounding box and positions of individual vertices.'''
    if len(dag.vertices) == 0:
        box = Box(0.0, 0.0, 0.0, 0.0, 0.0)
        positions: Dict[int, XY] = {}
    else:
        box, positions = _embed_dag(dag, vertex_sizes, x_padding=x_padding, y_padding=y_padding)
    box = box.add_margins(left_margin, right_margin, top_margin, bottom_margin)
    embedding = (box, positions)
    if align_top_left:
        embedding = _align_top_left(embedding)
    return embedding

def _embed_dag(dag: Dag, vertex_sizes: Dict[int, Size], *, x_padding=0.0, y_padding=0.0) -> Tuple[Box, Dict[int, XY]]:
    '''Embed DAG with possibly more than one component.'''
    components = dag_connected_components(dag)
    embeddings = [_embed_connected_dag(comp, vertex_sizes, x_padding=x_padding, y_padding=y_padding) for comp in components]
    grand_embedding = _combine_embeddings(embeddings, direction='vertical_rearranged', y_padding=y_padding)
    return grand_embedding

def _embed_connected_dag(dag: Dag, vertex_sizes: Dict[int, Size], *, x_padding=0.0, y_padding=0.0) -> Tuple[Box, Dict[int, XY]]:
    '''Embed DAG with exactly one component.'''
    slices = dag.slices()
    if len(slices) == 1:
        return _embed_unsliceable_dag(slices[0], vertex_sizes, x_padding=x_padding, y_padding=y_padding)
    else:
        embeddings = [_embed_dag(s, vertex_sizes, x_padding=x_padding, y_padding=y_padding) for s in slices]
        grand_embedding = _combine_embeddings(embeddings, direction='horizontal', x_padding=x_padding)
        return grand_embedding

def _embed_levels(levels: Sequence[Sequence[int]], vertex_sizes: Dict[int, Size], *, x_padding=0.0, y_padding=0.0) -> Tuple[Box, Dict[int, XY]]:
    level_embeddings = []
    for level in levels:
        vertex_embeddings = []
        for vertex in level:
            w, h = vertex_sizes[vertex]
            box = Box(w/2, w/2, h/2, h/2, w*h)
            positions = {vertex: XY(0.0, 0.0)}
            vertex_embeddings.append((box, positions))
        level_embedding = _combine_embeddings(vertex_embeddings, direction='vertical', y_padding=y_padding)
        level_embeddings.append(level_embedding)
    grand_embedding = _combine_embeddings(level_embeddings, direction='horizontal', x_padding=x_padding)
    return grand_embedding

def _embed_unsliceable_dag(dag: Dag, vertex_sizes: Dict[int, Size], *, x_padding=0.0, y_padding=0.0, max_allowed_permutations=1024) -> Tuple[Box, Dict[int, XY]]:
    '''Embed DAG with exactly one component, without trying to slice it.'''
    n_perm = _permutation_number(dag.levels, max_allowed=max_allowed_permutations+1)
    if len(dag.vertices) == 1 or n_perm > max_allowed_permutations:
        # Do not permutate (not needed or too expensive)
        return _embed_levels(dag.levels, vertex_sizes, x_padding=x_padding, y_padding=y_padding)
    else:
        # Permutate all levels to find the best tuple of permutations
        level_permutations = [itertools.permutations(level) for level in dag.levels]
        permutations = itertools.product(*level_permutations)
        best_embedding = None
        best_penalty = np.inf
        perm: Sequence[Sequence[int]]
        for perm in permutations:
            embedding = _embed_levels(perm, vertex_sizes, x_padding=x_padding, y_padding=y_padding)
            box, positions = embedding
            penalty = sum(abs(positions[u].y - positions[v].y) for u, v in dag.edges)
            if penalty < best_penalty:
                best_embedding = embedding
                best_penalty = penalty
        assert best_embedding is not None
        return best_embedding

def _permutation_number(levels: List[List[int]], max_allowed=np.inf) -> int:
    '''Return the cardinality of the cartesian product of permutations of each level, 
    or return max_allowed if it becomes clear that the result would be > max_allowed.'''
    result = 1
    for level in levels:
        for i in range(len(level), 0, -1):
            result *= i
            if result > max_allowed:
                return max_allowed
    return result

def _combine_embeddings(embeddings: List[Tuple[Box, Dict[int, XY]]], *, direction: Literal['horizontal', 'vertical', 'vertical_rearranged'], x_padding=0.0, y_padding=0.0) -> Tuple[Box, Dict[int, XY]]:
    boxes, positions = zip(*embeddings)
    if direction == 'horizontal':
        grand_box, box_places = _stack_boxes_horizontally(boxes, x_padding=x_padding)
    elif direction == 'vertical':
        grand_box, box_places = _stack_boxes_vertically(boxes, y_padding=y_padding)
    elif direction == 'vertical_rearranged':
        grand_box, box_places = _stack_boxes_vertically(boxes, y_padding=y_padding, rearrange=True)
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    grand_positions = {}
    for box_place, poss in zip(box_places, positions):
        for vertex, xy in poss.items():
            grand_positions[vertex] = box_place + xy
    return grand_box, grand_positions

def _stack_boxes_horizontally(boxes: Sequence[Box], *, x_padding=0.0, y_padding=0.0) -> Tuple[Box, List[XY]]:
    '''Return the grand box containing all boxes, and the list of positions of boxes in the grand box.'''
    grand_width = sum(box.left + box.right for box in boxes) + (len(boxes) - 1) * x_padding
    grand_left = grand_right = grand_width / 2
    box_places = []
    x = -grand_left
    for box in boxes:
        x += box.left
        box_places.append(XY(x, 0.0))
        x += box.right
        x += x_padding
    grand_top = max(box.top for box in boxes)
    grand_bottom = max(box.bottom for box in boxes)
    sum_w = sum(box.weight for box in boxes)
    grand_box = Box(grand_left, grand_right, grand_top, grand_bottom, sum_w)
    return grand_box, box_places

def _stack_boxes_vertically(boxes: Sequence[Box], *, x_padding=0.0, y_padding=0.0, rearrange=False) -> Tuple[Box, List[XY]]:
    '''Return the grand box containing all boxes, and the list of positions of boxes in the grand box.'''
    n_boxes = len(boxes)
    sum_w = 0.0
    sum_wy = 0.0
    y = 0.0
    rearranged_boxes = _rearrange_boxes_from_middle(boxes) if rearrange else list(enumerate(boxes))
    for i, box in rearranged_boxes:
        y += box.top
        sum_w += box.weight
        sum_wy += box.weight * y
        y += box.bottom
        y += y_padding
    mean_y = sum_wy / sum_w
    box_places: List[XY] = [XY(0.0, 0.0)] * n_boxes
    y = -mean_y
    for i, box in rearranged_boxes:
        y += box.top
        box_places[i] = XY(0.0, y)
        y += box.bottom
        y += y_padding
    grand_top = mean_y
    grand_bottom = y - y_padding
    grand_left = max(box.left for box in boxes)
    grand_right = max(box.right for box in boxes)
    grand_box = Box(grand_left, grand_right, grand_top, grand_bottom, sum_w)
    return grand_box, box_places

def _rearrange_boxes_from_middle(boxes: Sequence[Box]) -> List[Tuple[int, Box]]:
    '''Change the order of the boxes so that the biggest are in the middle, e.g. 1 3 5 7 8 6 4 2.
    Return the list of tuples (original_index, box).'''
    indexed_boxes = sorted(enumerate(boxes), key=lambda t: t[1].weight, reverse=True)  # type: ignore
    even = indexed_boxes[0::2]
    odd = indexed_boxes[1::2]
    odd.reverse()
    total = odd + even
    return total
    


class LabeledEdge(NamedTuple):
    invertex: int
    outvertex: int
    label: int

def sort_dag(vertices, does_precede):
    todo = list(vertices)
    done = []
    while len(todo) > 0:
        mins = [i for i in todo if not any(i!=j and does_precede(j, i) for j in todo)]
        if len(mins)==0:
            raise Exception('Cyclic graph.')
        done.extend(mins)
        todo = [i for i in todo if i not in mins]
        # if len(mins) > 1:
        # 	log('Sort DAG: uncomparable vertices:', *mins)
    return done

def left_subdag_tipsets(n_vertices: int, edges: List[Tuple[int,int]], precedence_matrix: Optional[np.ndarray]=None) -> Tuple[List[Tuple[int,...]], List[LabeledEdge]]:
    '''Return the list of tipsets of all left subdags of dag H and all successor left subdag pairs (as index tuples).
    G is a left subdag of dag H iff:
        foreach u-v path in H. if v in G then u in G
    Tipset of dag G is the minimal vertex subset S such that:
        foreach u in G. exists v in S. exists path u-v
    G, G' are a successor left subdag pair of dag H iff:
        G, G' are left subdags of H and G can be obtained by removing one vertex from G'.
    '''
    vertices = range(n_vertices)
    if precedence_matrix is None:
        with Timing('precedence_matrix', mute=True):
            precedence_matrix = precedence_matrix_from_edges_dumb(n_vertices, edges)
    with Timing('left_subdag_tipsets', mute=True):
        nontips = {u for (u, v) in edges}
        tipset = tuple(v for v in vertices if v not in nontips)
        active_tipsets = {tipset: 0}
        tipset_counter = 1
        all_tipsets: List[Tuple[int,...]] = []
        tipset_edges = []
        while len(active_tipsets) > 0:
            all_tipsets.extend(active_tipsets.keys())
            new_tipsets = {}
            for tipset, tipset_id in active_tipsets.items():
                incoming_edges = [(u,v) for (u,v) in edges if v in tipset]
                for tip in tipset:
                    potential_subtips = [u for (u,v) in incoming_edges if v == tip]
                    subtips = []
                    for u in potential_subtips:
                        if any(precedence_matrix[u,v] and v != tip for v in tipset):  # TODO this will cause problem if there is subgraph [A->B, B->C, A->C]
                            continue
                        subtips.append(u)
                    new_tipset = tuple(sorted(subtips + [t for t in tipset if t != tip]))
                    if new_tipset not in new_tipsets:
                        new_tipset_id = tipset_counter
                        new_tipsets[new_tipset] = new_tipset_id
                        tipset_counter += 1
                    else:
                        new_tipset_id = new_tipsets[new_tipset]
                    tipset_edges.append(LabeledEdge(new_tipset_id, tipset_id, tip))
            active_tipsets = new_tipsets
    n_tipsets = len(all_tipsets)
    all_tipsets = all_tipsets[::-1]
    tipset_edges = [LabeledEdge(n_tipsets-1-i, n_tipsets-1-j, tip) for (i, j, tip) in tipset_edges]
    assert are_edge_labels_unique_within_each_oriented_path(tipset_edges)
    return all_tipsets, tipset_edges

def are_edge_labels_unique_within_each_oriented_path(edges: List[LabeledEdge]):
    edge_starts = {u for u, v, lab in edges}
    edge_ends = {v for u, v, lab in edges}
    vertices = list(edge_starts | edge_ends)
    inflowing_labels: Dict[int, Set[int]] = {v: set() for v in vertices}
    while True:
        changed = False
        for u, v, lab in edges:
            if lab not in inflowing_labels[v]:
                changed = True
                inflowing_labels[v].add(lab)
            if lab in inflowing_labels[u]:
                return False
        if not changed:
            return True

def precedence_matrix_from_edges_dumb(n_vertices, edges):
    # n_vertices = len(vertices)
    # assert all(vertices[i] < vertices[i+1] for i in range(n_vertices-1))
    vertices = range(n_vertices)
    precedence = np.eye(n_vertices, dtype=bool)
    while True:
        changed = False
        for u, v in edges:
            for w in vertices:
                if not precedence[u, w] and precedence[v, w]:
                    changed = True
                    precedence[u, w] = True
        if not changed:
            break
    for v in vertices:
        precedence[v, v] = False
    return precedence

def test_left_subdag_tipsets():
    # n_vertices = 10
    # edges = [(i, i+1) for i in range(n_vertices-1)]
    # edges = [(0,1), (1,2), (2,3), (3,4), (1,5), (1,6), (4,7), (5,7), (7,8), (6,8), (8,9)]
    n_vertices = 13
    edges = [(0,1), (1,2), (2,3), (3,4), (4,10), (10,11), (11,12), (1,5), (5,6), (6,4), (6,7), (7,8), (8,9), (9,11)]
    tipsets, tipset_edges = left_subdag_tipsets(n_vertices, edges)
    print(tipsets)
    s = set()
    for (i,j,tip) in tipset_edges:
        s.add((tipsets[i], tipsets[j], tip))
    for g, h, t in sorted(s):
        print(f'{g} + {t} -> {h}')
    print(len(tipsets))
