'''AcyclicClusteringSimple and related functions'''


from __future__ import annotations
import sys
from pathlib import Path
import json
import itertools
import numpy as np
from collections import defaultdict, Counter
from typing import Tuple, List, Dict, Optional, Sequence, Mapping, Any, Collection, overload
from datetime import datetime
from numba import jit  # type: ignore

from . import lib
from . import lib_sses
from .lib_sses import Sse, SseType, LadderType
from . import lib_clustering
from . import lib_graphs
from .lib_logging import ProgressBar

from . import superimpose3d
from .lib_domains import Domain


# USE_NUMPY_HEAP = True  # ad-hoc implementation of min-heap, saves memory, but is slower than heapq
SSE_COUNT_THRESHOLD_FOR_NUMPY_HEAP = 8000  # ad-hoc implementation of min-heap, saves memory, but is slower than heapq (0 = always use, inf = never use)

UNLABELLED = -1  # label for samples that are not classified to any class

NONMATCHED = -1
FROM_DIAG, FROM_LEFT, FROM_TOP = 1, 2, 3  # FROM_DIAG will appear the worst when maximum is a tie

Edge = Tuple  #[int, int, Any, ...]
# EdgeIIL = Tuple[int, int, LadderType]
WeightedBetaEdge = Tuple[int, int, lib_sses.LadderType, float]

Match = Tuple[int, int]
Matching = List[Match]


def read_sses_simple(domains: List[Domain], directory: Path, length_thresholds: Optional[Dict[str, int]] = None
                     )-> tuple[list[int], list[Sse], np.ndarray[np.float64], np.ndarray[np.int8], list[Edge]]:
    offsets = [0]  # counting a beta-ladder side as one SSE (for preference matrix)
    sses = []
    edges: list[Edge] = []
    # lib_sses.compute_ssa(domains, directory, skip_if_exists=True)
    for domain in domains:
        with open(directory / f'{domain.name}.sses.json') as f:
            annot = json.load(f)[domain.name]
        sses_here = annot['secondary_structure_elements']
        for sse in sses_here:
            sse['domain'] = domain.name
        # Filter SSEs by length:
        if length_thresholds is not None:
            sses_here = [sse for sse in sses_here if lib_sses.long_enough(sse, length_thresholds)]
        label_index = { sse['label']: i for i, sse in enumerate(sses_here, start=offsets[-1]) }
        connectivity = annot.get('beta_connectivity', [])
        for strand1, strand2, orientation in connectivity:
            if strand1 == strand2:
                print(f'WARNING: Beta connectivity in {domain.name} contains self-connection {strand1}-{strand2}. Removing.', file=sys.stderr)
                continue
            i1 = label_index.get(strand1, None)
            i2 = label_index.get(strand2, None)
            if i1 is not None and i2 is not None:
                i1, i2 = sorted((i1, i2))
                edges.append((i1, i2, orientation))
        sses.extend(sses_here)
        offsets.append(len(sses))
    n_sses = len(sses)

    lib.log('Extracting coordinates...')
    coordinates = np.zeros((n_sses, 6), dtype=np.float64)
    for i, sse in enumerate(sses):
        coordinates[i,:] = sse['start_vector'] + sse['end_vector']

    type_vector = np.array([lib_sses.two_class_type_int(sse) for sse in sses], dtype=np.int8)

    return offsets, sses, coordinates, type_vector, edges


def edges_from_neighbours_lists(neighbours_lists: List[List[int]]) -> List[Edge]:
    edges: List[Edge] = []
    for i, neighs in enumerate(neighbours_lists):
        for j in neighs:
            if i < j:
                edges.append((i, j))
    return edges

def neighbours_lists_from_edges(n_vertices: int, edges: List[Edge]) -> List[List[int]]:
    neighbours_lists: List[List[int]] = [ [] for i in range(n_vertices) ]
    for u, v, *_ in edges:
        neighbours_lists[u].append(v)
        neighbours_lists[v].append(u)
    return neighbours_lists

def edge_lists_from_edges(n_vertices: int, edges: List[Edge]) -> List[List[Edge]]:
    edge_lists: List[List[Edge]] = [ [] for i in range(n_vertices) ]
    for u, v, *rest in edges:
        edge_lists[u].append((u, v, *rest))
        edge_lists[v].append((v, u, *rest))
    return edge_lists

def cluster_edges(edges: List[Edge], labels: Sequence[int]|np.ndarray, threshold=0.5) -> List[Edge]:
    vertex_counter: Dict[int, int] = defaultdict(int)
    for li in labels:
        if li != UNLABELLED:
            vertex_counter[li] += 1
    edge_counter: Dict[Edge, int] = defaultdict(int)
    for i, j, *rest in edges:
        li, lj = labels[i], labels[j]
        if li != UNLABELLED and lj != UNLABELLED:
            edge_counter[(li, lj, *rest)] += 1
    superedges = []
    for superedge, edge_count in sorted(edge_counter.items()):
        supervertex1, supervertex2, *rest = superedge
        # edginess = 2 * edge_count / (vertex_counter[supervertex1] + vertex_counter[supervertex2])
        edginess = edge_count / min(vertex_counter[supervertex1], vertex_counter[supervertex2])
        # lib.log(f'{str(superedge):^12}: {vertex_counter[supervertex1]:3d} {vertex_counter[supervertex2]:3d} {edge_count:3d} | {edginess:5.3f} {edginess>=threshold}')
        if edginess >= threshold:
            superedges.append(superedge)
    # return sorted(superedges)
    return sorted(strong_subgraph(vertex_counter, { edge: count/threshold for edge, count in edge_counter.items() }))

def strong_subgraph(vertex_weights: Mapping[int, float], edge_weights: Dict[Edge, float]) -> List[Edge]:
    '''Creates a subgraph H of vertex-weighted and edge-weighted graph G, s.t. for each edge uv of H holds:
    w(uv) >= min{max{w(a)|vertex a in A}, max{w(b)|vertex b in B}}, where A, B are connected components u, v.
    (A == B iff uv lies on a cycle. There can be more inclusion-maximal graphs H, this algorithm adds edges from highest-weight edge...).
    Returns the list of edges of H.''' 
    # component_vertices = [ {vertex} for vertex in vertex_weights.keys() ]
    # component_max_vertex_weight = list(vertex_weights.values())
    # component_min_edge_weight = [np.inf] * len(vertex_weights)
    components = { vertex: (v_weight, np.inf, {vertex}) for vertex, v_weight in vertex_weights.items() }
    # components are stored in a dict, where key is one representant vertex and values is triple (max. vertex weight, min. cycle-edge weight, vertex set)
    edges_by_weight = sorted(edge_weights.items(), key=lambda kv: kv[1], reverse=True)
    selected_edges = []
    for edge, e_weight in edges_by_weight:
        v1, v2, *_ = edge
        lead1, maxvw1, mincew1, vertices1 = next( (lead, maxvw, mincew, vertices) for lead, (maxvw, mincew, vertices) in components.items() if v1 in vertices )
        lead2, maxvw2, mincew2, vertices2 = next( (lead, maxvw, mincew, vertices) for lead, (maxvw, mincew, vertices) in components.items() if v2 in vertices )
        if e_weight >= min(maxvw1, maxvw2) and min(mincew1, mincew2) >= max(maxvw1, maxvw2):
            if lead1 == lead2:  # v1, v2 lie in the same component
                mincew = min(mincew1, e_weight)
                components[lead1] = (maxvw1, mincew, vertices1)
            else:  # v1, v2 lie in different components
                maxvw = max(maxvw1, maxvw2)
                mincew = min(mincew1, mincew2)
                vertices = vertices1 | vertices2
                components[lead1] = (maxvw, mincew, vertices)
                components.pop(lead2)
            selected_edges.append(edge)
    return selected_edges

def distance_matrix(coords, coords2=None):
    '''Calculate Euclidean distance between each pair of rows from coords and coords2
    coords -- matrix m*3
    coords2 -- matrix n*3 (default = coords)
    Return matrix m*n
    '''
    if coords2 is None:
        coords2 = coords
    m = coords.shape[0]
    n = coords2.shape[0]
    assert coords.shape[1] == coords2.shape[1] == 3
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=float)
    diff = coords.reshape((-1,1,3)) - coords2.reshape((1,-1,3))
    distance = np.sqrt(np.sum(diff**2, axis=2))
    return distance

def sse_distance_matrix(start_end_coords, start_end_coords2=None):
    '''start_end_coords - matrix n*6'''
    if start_end_coords2 is None:
        m = n = start_end_coords.shape[0]
        assert start_end_coords.shape[1] == 6
        start_distance = distance_matrix(start_end_coords[:, 0:3])
        end_distance = distance_matrix(start_end_coords[:, 3:6])
    else:
        m = start_end_coords.shape[0]
        n = start_end_coords2.shape[0]
        assert start_end_coords.shape[1] == start_end_coords2.shape[1] == 6
        start_distance = distance_matrix(start_end_coords[:, 0:3], start_end_coords2[:, 0:3])
        end_distance = distance_matrix(start_end_coords[:, 3:6], start_end_coords2[:, 3:6])
    distance = start_distance + end_distance
    assert distance.shape == (m, n), f'{m}, {n}, {distance.shape}'
    return distance

def iterative_superimposition(sse_coords1, sse_coords2, type_vector1, type_vector2, edges1=None, edges2=None, 
        weights1=None, max_iterations=np.inf, warning_on_empty_matching=None) -> Tuple[np.ndarray, np.ndarray, Matching, np.ndarray]:
    min_segment_lengths = lib.each_to_each(np.minimum, segment_lengths(sse_coords1), segment_lengths(sse_coords2))

    sse_coords1_rotated = sse_coords1.copy()
    old_total = -np.inf
    iterations = 0
    R = np.eye(3)
    t = np.zeros((3, 1))

    while True:
        distance = sse_distance_matrix(sse_coords1_rotated, sse_coords2)
        if edges1 is not None and edges2 is not None:
            distance = include_min_ladder_in_distance_matrix_from_local_edges(distance, edges1, edges2)
        if iterations == 0:
            distance0 = distance
        DYNPROG_SCORE_MAX = 30
        score = linearly_decreasing_score(distance, type_vector1, type_vector2, intercept=DYNPROG_SCORE_MAX) * min_segment_lengths
        if weights1 is not None:
            score *= weights1.reshape((-1, 1))
        matching, total = dynprog_align(score)
        # lib.log_debug('Total:', total, 'Matching:', matching)
        if not total > old_total or iterations == max_iterations:
            break
        indices1, indices2 = zip(*matching) if len(matching) > 0 else ((), ())
        # lib.log_debug('Indices1:', indices1, 'Indices2:', indices2)
        A = np.vstack((sse_coords1[indices1, 0:3], sse_coords1[indices1, 3:6])).transpose()
        B = np.vstack((sse_coords2[indices2, 0:3], sse_coords2[indices2, 3:6])).transpose()
        weights = np.array([ min_segment_lengths[i, j] for (i, j) in matching ] * 2)
        if len(matching) == 0:
            if warning_on_empty_matching is not None:
                lib.log(f'WARNING: iterative_superimposition(): Empty matching in iteration {iterations}. ({warning_on_empty_matching})')
                return  np.eye(3), np.zeros((3, 1)), matching, distance0
            else:
                raise Exception(f'iterative_superimposition(): Empty matching in iteration {iterations}.')
        # lib.log_debug(A.shape, B.shape)
        # lib.log_debug('weights', weights)
        R, t = superimpose3d.optimal_rotation_translation(A, B, weights=weights)
        sse_coords1_rotated[:, 0:3] = superimpose3d.rotate_and_translate(sse_coords1[:, 0:3].transpose(), R, t).transpose()
        sse_coords1_rotated[:, 3:6] = superimpose3d.rotate_and_translate(sse_coords1[:, 3:6].transpose(), R, t).transpose()
        old_total = total
        iterations += 1
    # lib.log_debug('Iterations:', iterations)
    return  R, t, matching, distance

def distance_matrix_with_iterative_superimposition_many(sse_coords, type_vector, offsets, edges=None, max_iterations=np.inf):
    n_domains = len(offsets) - 1
    n_sses = offsets[-1]
    distance = np.empty((n_sses, n_sses), dtype=float)
    for i, j in itertools.combinations_with_replacement(range(n_domains), 2):
        ifrom, ito = offsets[i:i+2]
        jfrom, jto = offsets[j:j+2]
        if edges is not None:
            local_edges1 = [ (u-ifrom, v-ifrom, etc) for (u, v, etc) in edges if ifrom <= u < ito ]
            local_edges2 = [ (u-jfrom, v-jfrom, etc) for (u, v, etc) in edges if jfrom <= u < jto ]
        else:
            local_edges1 = local_edges2 = None
        # lib.log_debug(f'\nIter. distance {i} - {j}')
        R, t, matching, d = iterative_superimposition(
            sse_coords[ifrom:ito, :], 
            sse_coords[jfrom:jto, :],
            type_vector[ifrom:ito],
            type_vector[jfrom:jto],
            edges1=local_edges1,
            edges2=local_edges2,
            max_iterations=max_iterations,
            warning_on_empty_matching=f'distance_matrix_with_iterative_superimposition_many(): {i} <-> {j}')
        distance[ifrom:ito, jfrom:jto] = d
        distance[jfrom:jto, ifrom:ito] = d.transpose()
    return distance

def segment_length_difference_matrix(start_end_coords):
    '''start_end_coords - matrix n*6'''
    lengths = segment_lengths(start_end_coords)
    diff = np.abs(lib.each_to_each(np.subtract, lengths))
    return diff

def segment_lengths(start_end_coords):
    '''start_end_coords - matrix n*6'''
    vector_diff = start_end_coords[:, 3:6] - start_end_coords[:, 0:3]
    lengths = np.sqrt(np.sum(vector_diff**2, axis=1))
    return lengths

def include_min_ladder_in_distance_matrix(distance, edges: List[Edge]):
    n = distance.shape[0]
    neighbours_lists = neighbours_lists_from_edges(n, edges)
    new_distance = distance.copy()
    strands = [ i for i, neighs in enumerate(neighbours_lists) if len(neighs) > 0 ]  # here some strands can act like helices, if their partners have been filtered out by length filter
    for i, j in itertools.combinations(strands, 2):
        min_neigh_distance = min( distance[neigh_i, neigh_j] for neigh_i, neigh_j in itertools.product(neighbours_lists[i], neighbours_lists[j]) )
        new_distance[i, j] = new_distance[j, i] = (distance[i, j] + min_neigh_distance) / 2
    return new_distance

def include_min_ladder_in_distance_matrix_from_local_edges(distance, edges1: List[Edge], edges2: List[Edge]):
    n1, n2 = distance.shape
    neighbours_lists1 = neighbours_lists_from_edges(n1, edges1)
    neighbours_lists2 = neighbours_lists_from_edges(n2, edges2)
    new_distance = distance.copy()
    strands1 = [ i for i, neighs in enumerate(neighbours_lists1) if len(neighs) > 0 ]  # here some strands can act like helices, if their partners have been filtered out by length filter
    strands2 = [ i for i, neighs in enumerate(neighbours_lists2) if len(neighs) > 0 ]  # here some strands can act like helices, if their partners have been filtered out by length filter
    for i, j in itertools.product(strands1, strands2):
        min_neigh_distance = min( distance[neigh_i, neigh_j] for neigh_i, neigh_j in itertools.product(neighbours_lists1[i], neighbours_lists2[j]) )
        new_distance[i, j] = (distance[i, j] + min_neigh_distance) / 2
    return distance
    # TODO return new_distance and test!

def include_protein_distance(sse_distance, protein_distance, offsets, overwrite=False):
    if not overwrite:
        sse_distance = sse_distance.copy()
    n_proteins = len(offsets) - 1
    for i in range(n_proteins):
        for j in range(n_proteins):
            sse_distance[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] += protein_distance[i, j]
    return sse_distance

def dynprog_matrices(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Return cumulative scores matrix (shape (m+1, n+1)) and direction matrix (shape(m, n)) for tracing the best matching.'''
    m, n = scores.shape
    score_type = scores.dtype.type
    cumulative = np.zeros((m+1, n+1), dtype=score_type)
    direction = np.zeros((m, n), dtype=np.int32)
    for i in range(1, m+1):
        for j in range(1, n+1):
            on_left = cumulative[i, j-1]
            on_top = cumulative[i-1, j]
            on_diag = cumulative[i-1, j-1]
            score = scores[i-1, j-1]
            cumulative[i, j], direction[i-1, j-1] = max((on_left, FROM_LEFT), (on_top, FROM_TOP), (on_diag + score, FROM_DIAG))
    return cumulative, direction

@jit(nopython=True)
def dynprog_matrices_diagonal_method(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Return cumulative scores matrix (shape (m+1, n+1)) and direction matrix (shape(m, n)) for tracing the best matching.
    Equivalent to dynprog_matrices(), but faster.'''
    score_type = scores.dtype.type
    direction_type = np.int8
    # INT_TYPE = np.int32
    m, n = scores.shape
    M, N = m+1, n+1  # size for cumulative matrices
    if m == 0:  # Special case, matching is empty, cumulative is zeros.
        cumulative = np.zeros((M, N), dtype=score_type)
        direction = np.full((m, n), FROM_LEFT, dtype=direction_type)
        return cumulative, direction
    if n == 0:  # Special case, matching is empty, cumulative is zeros.
        cumulative = np.zeros((M, N), dtype=score_type)
        direction = np.full((m, n), FROM_TOP, dtype=direction_type)
        return cumulative, direction
    N1 = N - 1
    MN = M*N
    N1N1 = N1*N1
    n1 = n - 1
    mn = m*n
    n1n1 = n1*n1
    if n1 == 0:  # Special case when scores.shape == (m, 1); the line assigning to on_diag would raise ValueError('slice step cannot be zero') without these 2 lines
        n1 = 1
    cumulative = np.empty((M, N), dtype=score_type)
    direction = np.empty((m, n), dtype=direction_type)
    cumulative[0, :] = cumulative[:, 0] = 0
    # first row and column in direction will never be read (hopefully), but I filled them just to be sure
    cumulative_ = cumulative.ravel()
    direction_ = direction.ravel()
    scores_ = scores.ravel()
    for D in range(2, M+N-1):
        d = D - 2 
        if D < N:  # start on top
            START = D + N1
            start = d
        else:  # start on right
            START = N*D - N1N1
            start = n*d - n1n1
        if D < M:  # end on left
            END = N*D + 1 - N1
            end = n*d + 1
        else:  # end on bottom
            END = MN - M - N + 2 + D
            end = mn - m - n + 2 + d
        on_left = cumulative_[START-1 : END-1 : N1]
        on_top = cumulative_[START-N : END-N : N1]
        on_diag = cumulative_[START-N-1 : END-N-1 : N1] + scores_[start : end : n1]
        direc = np.where(on_top > on_left, FROM_TOP, FROM_LEFT)
        maximum = np.maximum(on_top, on_left)
        direction_[start:end:n1] = np.where(on_diag > maximum, FROM_DIAG, direc)
        cumulative_[START:END:N1] = np.maximum(on_diag, maximum)
    return cumulative, direction

@jit(nopython=True)
def trace_direction_matrix(direction: np.ndarray, include_nonmatched: bool) -> Matching:
    matching = []
    m, n = direction.shape
    i, j = m-1, n-1
    while i >= 0 and j >= 0:
        if direction[i, j] == FROM_LEFT:
            if include_nonmatched:
                matching.append((NONMATCHED, j))
            j -= 1
        elif direction[i, j] == FROM_TOP:
            if include_nonmatched:
                matching.append((i, NONMATCHED))
            i -= 1
        else:  # FROM_DIAG
            matching.append((i, j))
            i -= 1
            j -= 1
    if include_nonmatched:
        while i >= 0:
            matching.append((i, NONMATCHED))
            i -= 1
        while j >= 0:
            matching.append((NONMATCHED, j))
            j -= 1
    matching.reverse()
    return matching

def dynprog_align(scores: np.ndarray, include_nonmatched=False) -> Tuple[Matching, float]:
    cumulative, direction = dynprog_matrices_diagonal_method(scores)
    matching = trace_direction_matrix(direction, include_nonmatched)
    total_score = cumulative[-1, -1]
    return matching, total_score


def dynprog_matrices_diagonal_method2(scores: np.ndarray) -> Tuple[float, np.ndarray]:
    ''' Do the same as dynprog_matrices(), but faster using numpy broadcasting (cca 4x faster on matrices 64*64)'''
    m, n = scores.shape
    score_type = scores.dtype.type
    M, N = m+1, n+1  # size for cumulative matrices
    if m == 0:  # Special case, matching is empty, cumulative is zeros.
        cumulative = np.zeros((M, N), dtype=score_type)
        direction = np.full((M, N), FROM_LEFT, dtype=np.int32)
        return 0, direction
    if n == 0:  # Special case, matching is empty, cumulative is zeros.
        cumulative = np.zeros((M, N), dtype=score_type)
        direction = np.full((M, N), FROM_TOP, dtype=np.int32)
        return 0, direction
    anti_cum = Antidiagonals((M, N))
    anti_cum.fill_first_row(0)
    anti_cum.fill_first_column(0)
    anti_dir = Antidiagonals((M, N), dtype=np.int32)
    anti_scores = Antidiagonals.from_matrix(scores)
    for D in range(2, anti_cum.n_diags):
        on_left = anti_cum.get_left_from_inner(D)
        on_top = anti_cum.get_up_from_inner(D)
        on_diag = anti_cum.get_leftup_from_inner(D) + anti_scores.get(D-2)
        direc = anti_dir.get_inner(D)
        maximum = anti_cum.get_inner(D)
        direc[:] = np.where(on_top > on_left, FROM_TOP, FROM_LEFT)
        np.maximum(on_top, on_left, out=maximum)  # type: ignore
        direc[:] = np.where(on_diag > maximum, FROM_DIAG, direc)
        np.maximum(on_diag, maximum, out=maximum)  # type: ignore
    return anti_cum.get(anti_cum.n_diags-1)[0], anti_dir.retrieve_matrix()

class Antidiagonals(object):
    def __init__(self, shape, dtype=float):
        m, n = shape
        self.m = m
        self.n = n
        self.n_diags = m + n - 1
        ds = np.arange(self.n_diags)
        self.starts = np.empty_like(ds)
        self.starts[:n] = ds[:n]
        self.starts[n:] = n * ds[n:] - (n-1)**2
        self.ends = np.empty_like(ds)
        self.ends[:m] = n * ds[:m] + 1
        self.ends[m:] = m*n - m - n + 2 + ds[m:]
        self.lengths = np.minimum(np.minimum(ds + 1, self.n_diags - ds), min(m, n))
        self.step = n - 1 if n > 1 else 1
        self.offsets = np.empty(self.n_diags+1, dtype=int)
        self.offsets[0] = 0
        np.cumsum(self.lengths, out=self.offsets[1:])
        self.values = np.empty(m*n, dtype=dtype)

    def get(self, i: int) -> np.ndarray:
        '''Get i-th antidiagonal'''
        return self.values[self.offsets[i]:self.offsets[i+1]]

    def get_inner(self, i: int) -> np.ndarray:
        '''Get i-th antidiagonal, excluding the first row and column'''
        fro = self.offsets[i] + 1 if i < self.n else self.offsets[i]
        to = self.offsets[i+1] - 1 if i < self.m else self.offsets[i+1]
        return self.values[fro:to]

    def get_up_from_inner(self, i: int) -> np.ndarray:
        '''Get elements one above the i-th inner antidiagonal'''
        fro = self.offsets[i-1]
        to = self.offsets[i] - 1
        return self.values[fro:to]

    def get_left_from_inner(self, i: int) -> np.ndarray:
        '''Get elements one left from the i-th inner antidiagonal'''
        fro = self.offsets[i-1] + 1
        to = self.offsets[i]
        return self.values[fro:to]

    def get_leftup_from_inner(self, i: int) -> np.ndarray:
        '''Get elements one left and one up from the i-th inner antidiagonal'''
        fro = self.offsets[i-2] if i <= self.n else self.offsets[i-2] + 1
        to = self.offsets[i-1] if i <= self.m  else self.offsets[i-1] - 1
        return self.values[fro:to]

    def fill_first_row(self, value) -> None:
        self.values[self.offsets[:self.n]] = value
    
    def fill_first_column(self, value) -> None:
        self.values[self.offsets[1:self.m+1] - 1] = value
    
    def retrieve_matrix(self) -> np.ndarray:
        matrix = np.empty((self.m, self.n), dtype=self.values.dtype)
        flat_matrix = matrix.ravel()
        values = self.values
        offsets = self.offsets
        starts = self.starts
        ends = self.ends
        step = self.step
        for i in range(offsets.size - 1):
            flat_matrix[starts[i]:ends[i]:step] = values[offsets[i]:offsets[i+1]]
        return matrix

    def fill_with_matrix(self, matrix) -> None:
        assert matrix.shape == (self.m, self.n)
        flat_matrix = matrix.ravel()
        for i in range(self.offsets.size - 1):
            self.values[self.offsets[i]:self.offsets[i+1]] = flat_matrix[self.starts[i]:self.ends[i]:self.step]

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'Antidiagonals':
        anti = cls(matrix.shape, matrix.dtype)
        anti.fill_with_matrix(matrix)
        return anti


def antidiagonals(matrix: np.ndarray):
    '''Place elements of matrix into antidiags, so that antidiags[offsets[i]:offsets[i+1]] contains i-th antidiagonal of matrix.'''
    m, n = matrix.shape
    matrix_ = matrix.ravel()
    n_diags = m+n-1
    ds = np.arange(n_diags)
    starts = np.empty_like(ds)
    starts[:n] = ds[:n]
    starts[n:] = n * ds[n:] - (n-1)**2
    ends = np.empty_like(ds)
    ends[:m] = n * ds[:m] + 1
    ends[m:] = m*n - m - n + 2 + ds[m:]
    lengths = np.minimum(np.minimum(ds + 1, n_diags - ds), min(m, n))  # type: ignore
    step = n - 1
    offsets = np.empty(n_diags+1, dtype=int)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])  # type: ignore
    antidiags = np.empty(m*n, dtype=matrix.dtype)  # type: ignore
    for i in range(n_diags):
        antidiags[offsets[i]:offsets[i+1]] = matrix_[starts[i]:ends[i]:step]
    return antidiags, offsets
    # TODO test

def from_antidiagonals(m, n, antidiags):
    n_diags = m+n-1
    ds = np.arange(n_diags)
    starts = np.empty_like(ds)
    starts[:n] = ds[:n]
    starts[n:] = n * ds[n:] - (n-1)**2
    ends = np.empty_like(ds)
    ends[:m] = n * ds[:m] + 1
    ends[m:] = m*n - m - n + 2 + ds[m:]
    lengths = np.minimum(np.minimum(ds + 1, n_diags - ds), min(m, n))
    step = n - 1
    offsets = np.empty(n_diags+1, dtype=int)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    result = np.empty((m, n), dtype=antidiags.dtype)
    result_ = result.ravel()
    for i in range(offsets.size - 1):
        result_[starts[i]:ends[i]:step] = antidiags[offsets[i]:offsets[i+1]]
    return result
    
def antidiagonal(X: np.ndarray, d: int):
    '''Extracts d-th antidiagonal from a matrix X (0th antidiagonal is X[0,0], the last is X[-1,-1]).'''
    m, n = X.shape
    Y = X.reshape(-1)
    if d > m + n - 2:
        raise IndexError()
    if d < n:  # start on top
        start = d
    else:  # start on right
        start = n*d - (n-1)*(n-1)
    if d < m:  # end on left
        end = n*d + 1
    else:  # end on bottom
        end = m*n - m - n + 2 + d
    return Y[start:end:n-1]

def antidiagonal_indices(X: np.ndarray, d: int, skip_top_and_left=False) -> Tuple[int, int, int]:
    '''Return (start, end, step) such that X[start:end:step] is the d-th antidiagonal of matrix X (0th antidiagonal is [X[0,0]], the last is [X[-1,-1]]).'''
    M, N = X.shape
    N1 = N - 1
    if d > M + N - 2:
        raise IndexError()
    if d < N:  # start on top
        start = d + N1 if skip_top_and_left else d
    else:  # start on right
        start = N*d - N1*N1
    if d < M:  # end on left
        end = N*d + 1 - N1 if skip_top_and_left else d
    else:  # end on bottom
        end = M*N - M - N + 2 + d
    return start, end, N1

def n_antidiagonals(X: np.ndarray) -> int:
    m, n = X.shape
    return m + n - 1

# def dynprog_align2(scores: np.ndarray, include_nonmatched=False) -> Tuple[Matching, float]:
#     m, n = scores.shape
#     with lib.Timing('new'):
#         total_score, direction = dynprog_matrices_diagonal_method2(scores)
#     matching = trace_direction_matrix(direction, include_nonmatched)
#     # total_score = cumulative[m, n]
#     return matching, total_score

def dynprog_total_scores_each_to_each(scores, offsets):
    n = len(offsets) - 1
    total_scores = np.zeros((n, n), dtype=float)
    n_combinations = n * (n + 1) / 2
    with ProgressBar(n_combinations, title='Dynprog...') as bar:
        for i, j in itertools.combinations_with_replacement(range(n), 2):
            ifrom, ito = offsets[i:i+2]
            jfrom, jto = offsets[j:j+2]
            matching, total_score = dynprog_align(scores[ifrom:ito, jfrom:jto])
            total_scores[i, j] = total_score
            total_scores[j, i] = total_score
            bar.step()
    return total_scores

def dynprog_fuckup_indices_each_to_each(scores, offsets):
    n = len(offsets) - 1
    fuckups = np.empty_like(scores)
    n_combinations = n * (n + 1) / 2
    with ProgressBar(n_combinations, title='Dynprog...') as bar:
        for i, j in itertools.combinations_with_replacement(range(n), 2):
            ifrom, ito = offsets[i:i+2]
            jfrom, jto = offsets[j:j+2]
            fuckup = fuckup_indices(scores[ifrom:ito, jfrom:jto])
            fuckups[ifrom:ito, jfrom:jto] = fuckup
            fuckups[jfrom:jto, ifrom:ito] = fuckup.transpose()
            bar.step()
    return fuckups

def fuckup_indices(scores):
    m, n = scores.shape
    cumulative,	direction = dynprog_matrices(scores)
    cumulative_rev, direction_rev = [ np.flip(M) for M in dynprog_matrices(np.flip(scores)) ]
    best_score = cumulative[m, n]
    alternative_scores = cumulative[:-1, :-1] + cumulative_rev[1:, 1:] + scores
    fuckup_indices = best_score - alternative_scores
    return fuckup_indices

def dynprog_realign(scores, offsets, reference_weights=None):
    '''Indices >= offsets[-1] correspond to reference SSEs'''
    n = len(offsets) - 1
    n_normal_sses = offsets[-1]
    total_scores = np.zeros((n,), dtype=float)
    new_labels = np.full((n_normal_sses,), UNLABELLED, dtype=int)
    with ProgressBar(n, title='Dynprog realign...') as bar:
        for i in range(n):
            ifrom, ito = offsets[i:i+2]
            sc = scores[ifrom:ito, n_normal_sses:]
            if reference_weights is not None:
                sc *= reference_weights
            matching, total_score = dynprog_align(sc)
            for normal_sse, ref_sse in matching:
                new_labels[ifrom+normal_sse] = ref_sse
            total_scores[i] = total_score
            bar.step()
    return new_labels, total_scores

def iterative_rematching(sse_coords: np.ndarray, type_vector: np.ndarray, init_labels: np.ndarray, edges, offsets, adhesion=0, max_iterations=np.inf) -> np.ndarray:
    if len(init_labels) == 0:
        return init_labels.copy()
    n_clusters = init_labels.max() + 1
    n_domains = len(offsets) - 1
    n_sses = offsets[-1]
    labels = np.empty((2, *init_labels.shape), dtype=init_labels.dtype)  # type: ignore
    new, old, final = 0, 1, 0
    labels[new] = init_labels
    # labels = init_labels.copy()
    # seen_labels = []  # for cycle detection
    # old_labels = np.full_like(labels, UNLABELLED)
    sse_coords_rotated = sse_coords.copy()

    local_edges = []
    for i in range(n_domains):
        ifrom, ito = offsets[i:i+2]
        local_edges.append([ (u-ifrom, v-ifrom, *etc) for (u, v, *etc) in edges if ifrom <= u < ito ])

    ref_type_vector = np.zeros(n_clusters, dtype=np.int32)
    for i, label in enumerate(init_labels):
        ref_type_vector[label] = type_vector[i]  # will write same value more times, I don't care
    
    iterations = 0
    f = anova_f(sse_coords, labels[new])
    self_p = np.zeros(2)
    self_p[new] = self_classification_probabilities(sse_coords, type_vector, labels[new])[2]
    lib.log(f'Iterations, ANOVA-F, self_prob:\t{iterations}\t{f}\t{self_p[new]}')
    # while not any( np.all(labs == labels) for labs in seen_labels ) and iterations < max_iterations:  # convergence detection + cycle detection to avoid infinite loops
    while iterations < max_iterations:  # convergence detection by local maximum of self_prob - implemented at the end of the cycle
        new, old = old, new
        t0 = datetime.now()
        # Calculate reference coordinates
        ref_coords = np.zeros((n_clusters, 6))
        member_counts = np.zeros(n_clusters, dtype=np.int32)
        for j in range(n_sses):
            label = labels[old, j]
            if label != UNLABELLED:  #  label UNLABELLED means non-matched SSE
                ref_coords[label, :] += sse_coords[j, :]
                # ref_coords[label, :] += sse_coords_rotated[j, :]
                member_counts[label] += 1
        nonempty_clusters = member_counts > 0
        ref_coords[nonempty_clusters, :] /= member_counts[nonempty_clusters].reshape((-1, 1))  
        # Calculate distances from reference coordinates
        ref_edges = cluster_edges(edges, labels[old], threshold=1e-9) # trying a benevolent threshold here
        ref_weights = member_counts**adhesion if adhesion > 0 else np.where(nonempty_clusters, 1, 0)
        for i in range(n_domains):
            ifrom, ito = offsets[i:i+2]
            R, t, matching, distance = iterative_superimposition(ref_coords, sse_coords[ifrom:ito, :], ref_type_vector, type_vector[ifrom:ito], 
                edges1=ref_edges, edges2=local_edges[i], weights1=ref_weights, warning_on_empty_matching=f'iterative_rematching(): domain {i}')
            # R, t, matching, distance = iterative_superimposition(ref_coords, sse_coords[ifrom:ito, :], ref_type_vector,type_vector[ifrom:ito], 
            # 	edges1=ref_edges, edges2=local_edges[i], weights1=ref_weights, max_iterations=0)
            new_labels_here = labels[new, ifrom:ito]
            new_labels_here.fill(UNLABELLED)
            for (ref, query) in matching:
                new_labels_here[query] = ref
            sse_coords_rotated[ifrom:ito, 0:3] = superimpose3d.rotate_and_translate(sse_coords[ifrom:ito, 0:3].transpose(), R, t).transpose()
            sse_coords_rotated[ifrom:ito, 3:6] = superimpose3d.rotate_and_translate(sse_coords[ifrom:ito, 3:6].transpose(), R, t).transpose()
        iterations += 1
        t1 = datetime.now()
        f = anova_f(sse_coords, labels[new])
        t2 = datetime.now()
        self_p[new] = self_classification_probabilities(sse_coords, type_vector, labels[new])[2]
        t3 = datetime.now()
        lib.log(f'Iterations, ANOVA-F, self_prob:\t{iterations}\t{f}\t{self_p[new]}')
        # lib.log_debug(f'Times (iter, anova_f, self_prob): {t1-t0} {t2-t1} {t3-t2}')
        if self_p[new] > self_p[old]:
            final = new
        else:
            final = old
            break
    # cycle_size = next(( len(seen_labels) - i for i, labs in enumerate(seen_labels) if np.all(labs == labels) ), 0)
    lib.log('Iterative rematching iterations:', iterations)
    return labels[final]

def rematch_with_SecStrAnnotator(domains: List[Domain], directory: Path, sses, offsets, extra_options='') -> np.ndarray:
    lib_sses.annotate_all_with_SecStrAnnotator(domains, directory, extra_options=extra_options)
    labels = np.full(len(sses), UNLABELLED, dtype=np.int32)
    for domain, fro, to in zip(domains, offsets[:-1], offsets[1:]):
        index = { (sses[i][lib_sses.CHAIN], sses[i][lib_sses.START], sses[i][lib_sses.END]): i for i in range(fro, to) }
        with open(directory / f'{domain.name}-annotated.sses.json') as r:
            annotated_sses = json.load(r)[domain.name]['secondary_structure_elements']
        for annot_sse in annotated_sses:
            key = (annot_sse[lib_sses.CHAIN], annot_sse[lib_sses.START], annot_sse[lib_sses.END])
            label = lib.integer_suffix(annot_sse[lib_sses.LABEL])
            labels[index[key]] = label
    return labels

def filter_labels_by_count(labels: np.ndarray, min_count: int) -> np.ndarray:
    counts = Counter(labels)
    labels = np.array([ label if counts[label] >= min_count else UNLABELLED for label in labels ])
    return labels

@overload
def relabel_without_gaps(labels: np.ndarray, class_precedence_matrix: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    ...
@overload
def relabel_without_gaps(labels: np.ndarray, class_precedence_matrix: None) -> Tuple[int, np.ndarray, None]: 
    ...
def relabel_without_gaps(labels, class_precedence_matrix):
    '''Map class labels to new class labels, so that no number is missing, and then map the sample labels.
    e.g. [0,0,2,UNLABELLED,3,2] -> [0,0,1,UNLABELLED,2,1]
    '''
    valid_labels = set(labels)
    valid_labels.discard(UNLABELLED)
    n_classes = len(valid_labels)
    old2new = { label: i for i, label in enumerate(sorted(valid_labels)) }
    old2new[UNLABELLED] = UNLABELLED
    # new_labels = np.full_like(labels, UNLABELLED)
    # for i, old in enumerate(labels):
    # 	new_labels[i] = old2new[old]
    new_labels = np.array([old2new[old] for old in labels])
    if class_precedence_matrix is not None:
        new_precedence = np.empty((n_classes, n_classes), dtype=class_precedence_matrix.dtype)  # type: ignore
        for i_old, i_new in old2new.items():
            for j_old, j_new in old2new.items():
                new_precedence[i_new, j_new] = class_precedence_matrix[i_old, j_old]
    else:
        new_precedence = None
    return n_classes, new_labels, new_precedence

def anova_f(coords, labels) -> float:
    ''' coords: n*k, labels: n '''
    n_total = len(labels)
    if n_total == 0:
        return np.nan
    # Add unique classes to unclassified samples (label UNLABELLED)
    current_classes = max(labels) + 1
    labels = labels.copy()
    for i, label in enumerate(labels):
        if label == UNLABELLED:
            labels[i] = current_classes
            current_classes += 1
    # Build index
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    indices = list(label_to_indices.values())
    # lib.log_debug('indices:', indices)
    n_classes = len(indices)
    df_within = n_total - n_classes
    df_between = n_classes - 1
    means = np.array([ coords[idcs].mean(axis=0) for idcs in indices ])
    the_mean = coords.mean(axis=0)
    sumsqs_within = [ np.sum((coords[idcs] - mean)**2) for idcs, mean in zip(indices, means) ]
    sumsqs_between = [ np.sum((mean - the_mean)**2) * len(idcs) for idcs, mean in zip(indices, means) ]
    ss_within = sum(sumsqs_within)
    ss_between = sum(sumsqs_between)
    ss_total = np.sum((coords - the_mean)**2)
    # lib.log_debug(ss_total, ss_between, ss_within)
    # lib.log_debug(n_total - 1, df_between, df_within)
    # lib.log_debug(ss_total/(n_total - 1), ss_between/df_between, ss_within/df_within)
    if df_within != 0 and df_between != 0:
        f = (ss_between * df_within) / (ss_within * df_between)
    else:
        f = np.nan
    return f

def self_classification_probabilities(sse_coords, type_vector, labels, sse_weights=None):
    '''Calculate self-probability of each sample (SSE) as the probability that a Bayesian model would classify the sample correctly.
    The model assumes Gaussian distribution with spherical variance for each class.
    Samples with label == UNLABELLED are treated as unclassified, thus their self_prob == 0.
    Additionally calculate weighted self_probs for each class and the total self_prob.
    Params shapes: sse_coords (n, ...), type_vector: (n,), labels: (n,), sse_weights: (n,)
    Return shapes: self_probs: (n,), class_self_probs: (n_classes,), total_self_prob: () '''
    
    n_sses = len(labels)

    if n_sses == 0:
        self_probs = np.zeros((0,))
        class_self_probs = np.zeros((0,))
        total_self_prob = np.nan
        return self_probs, class_self_probs, total_self_prob
    
    n_classes = labels.max() + 1
    coords_dims = tuple(range(sse_coords.ndim)) 
    
    # Get type and size of each class
    class_type_vector = np.zeros(n_classes, dtype=int)
    member_counts = np.zeros(n_classes, dtype=int)
    for i, label in enumerate(labels):
        if label != UNLABELLED:
            class_type_vector[label] = type_vector[i]  # will write same value more times, I don't care
            member_counts[label] += 1

    # Approximate each class with spherical Gaussian distribution N(mu, sigma)
    mu = np.zeros((n_classes, *sse_coords.shape[1:]), dtype=float)
    sigma = np.zeros(n_classes, dtype=float)
    for j in range(n_classes):
        if member_counts[j] > 0:
            xs = sse_coords[labels == j]
            mu[j] = xs.mean(axis=0)
            sigma[j] = np.sqrt(xs.var(axis=0).sum() + 1e-9)  # + 1e-9 is to avoid the degenerated case of sigma==0 (e.g. one-member class)
    
    # Get self_prob of each sample (SSE)
    self_probs = np.zeros(n_sses, dtype=float)
    for i in range(n_sses):
        if labels[i] != UNLABELLED:  # else: self_probs[i] = 0
            x = sse_coords[i]
            # class_dependent_probs = np.zeros(n_classes, dtype=float)
            # for j in range(n_classes):
            # 	if class_type_vector[j] == type_vector[i] and member_counts[j] > 0:
            # 		class_dependent_probs[j] = 1/sigma[j] * np.exp(-((x - mu[j, :])**2).sum() / (2 * sigma[j]**2))
            # 	else:
            # 		class_dependent_probs[j] = 0
            class_valid = np.logical_and(class_type_vector == type_vector[i], member_counts > 0)
            class_dependent_probs = np.zeros(n_classes, dtype=float)
            class_dependent_probs[class_valid] = 1/sigma[class_valid] * np.exp(-np.sum((x - mu[class_valid])**2, axis=coords_dims[1:]) / (2 * sigma[class_valid]**2))
            total_prob = (class_dependent_probs * member_counts).sum()
            self_probs[i] = class_dependent_probs[labels[i]] * member_counts[labels[i]] / total_prob
    
    # Get weighted-average self_prob of each class and total
    class_self_probs = np.zeros(n_classes, dtype=float)
    for j in range(n_classes):
        if member_counts[j] > 0:
            class_self_probs[j] = self_probs[labels == j].mean() if sse_weights is None else (self_probs[labels==j] * sse_weights[labels==j]).sum() / (sse_weights[labels==j]).sum()
    total_self_prob = self_probs.mean() if sse_weights is None else (self_probs * sse_weights).sum() / sse_weights.sum()
    return self_probs, class_self_probs, total_self_prob

def labelling_agreement(labels1, labels2, allow_matching=False, include_both_unclassified=True):
    n = len(labels1)
    if len(labels2) != n:
        raise ValueError('labels1 and labels2 must have the same length')
    if n == 0:
        return np.nan
    n_classes1 = max(labels1) + 1
    n_classes2 = max(labels2) + 1
    contingency = np.zeros((n_classes1, n_classes2), dtype=int)
    n_both_unclassified = 0
    for lab1, lab2 in zip(labels1, labels2):
        if lab1 != UNLABELLED and lab2 != UNLABELLED:
            contingency[lab1, lab2] += 1
        elif lab1 == UNLABELLED and lab2 == UNLABELLED:
            n_both_unclassified += 1
    if allow_matching:
        matching, n_matched = dynprog_align(contingency)
    else:
        n_matched = contingency.trace()
    if include_both_unclassified:
        return (n_matched + n_both_unclassified) / n
    else:
        return n_matched / n

def make_precedence_matrix(offsets, sses=None):
    n_domains = len(offsets) - 1
    n_sses = offsets[-1]
    precedence = np.zeros((n_sses, n_sses), dtype=bool)
    for p in range(n_domains):
        for i, j in itertools.combinations(range(offsets[p], offsets[p+1]), 2):
            if sses is None or sses[i] != sses[j]:  # filter out cases when there is twice the same SSE (a strand = multiple sides)
                precedence[i, j] = True
    return precedence

def linearly_decreasing_score(distance, type_vector, type_vector2=None, intercept=30):
    score = intercept - distance
    type_mismatch = lib.each_to_each(np.not_equal, type_vector, type_vector2)
    score[type_mismatch] = 0
    score[score < 0] = 0
    return score

def get_labels_from_memberlists(memberlists: List[List[int]]) -> np.ndarray:
    flat_members = [m for members in memberlists for m in members]
    n_samples = max(flat_members) + 1 if len(flat_members) > 0 else 0
    labels = np.full(n_samples, UNLABELLED)
    for label, members in enumerate(memberlists):
        labels[members] = label
    return labels

def dynprog_match_dags__old(scores: np.ndarray, edges1, edges2) -> Tuple[Matching, float]:
    '''Contains a bug (sorting vertices 1 doesn't guarantee the best matching).'''
    n_vertices1, n_vertices2 = scores.shape
    precedence1 = lib_graphs.precedence_matrix_from_edges_dumb(n_vertices1, edges1)
    precedence2 = lib_graphs.precedence_matrix_from_edges_dumb(n_vertices2, edges2)
    sorted_vertices1 = lib_graphs.sort_dag(range(n_vertices1), lambda u,v: precedence1[u,v])
    tipsets2, tipset_edges2 = lib_graphs.left_subdag_tipsets(n_vertices2, edges2, precedence_matrix=precedence2)
    # lib.log_debug('n1 n2 m2', len(sorted_vertices1), len(tipsets2), len(tipset_edges2))
    m, n = len(sorted_vertices1) + 1, len(tipsets2)
    cumulative = np.zeros((m, n))
    direction = np.zeros((m, n), dtype=np.int32)  # 0 = move 1 up (skip tip1), -k = move k left (skip one of tipset2), +k = move 1 up and k left (match tip1 to one of tipset2)
    tip2_matrix = np.empty((m, n), dtype=np.int32)  
    tipset_edges2_by_outvertex = defaultdict(list)
    for edge in tipset_edges2:
        tipset_edges2_by_outvertex[edge.outvertex].append(edge)
    for i in range(1, m):
        tip1 = sorted_vertices1[i-1]
        for j in range(1, n):
            tipset2 = tipsets2[j]
            inedges2 = tipset_edges2_by_outvertex[j]
            # Skip tip1:
            cumulative[i,j] = cumulative[i-1,j]
            direction[i,j] = 0
            # Skip any of tipset2:
            for j_predecessor, _, tip2 in inedges2:
                candidate_cum = cumulative[i, j_predecessor]
                if candidate_cum > cumulative[i,j]:
                    cumulative[i,j] = candidate_cum
                    direction[i,j] = j_predecessor - j  # always negative
            # Match tip1 to any of tipset2:
            for j_predecessor, _, tip2 in inedges2:
                candidate_cum = cumulative[i-1, j_predecessor] + scores[tip1, tip2]
                if candidate_cum > cumulative[i,j]:
                    cumulative[i,j] = candidate_cum
                    direction[i,j] = j - j_predecessor  # always positive
                    tip2_matrix[i,j] = tip2
    # lib.print_matrix(cumulative, 'tmp/cumulative.tsv')
    # lib.print_matrix(direction, 'tmp/direction.tsv')

    # Reconstruct path (matching):
    i, j = m-1, n-1
    matching: Matching = []
    while i > 0 and j > 0:
        direct = direction[i, j]
        if direct == 0:
            i -= 1
        elif direct < 0:
            j += direct
        else:
            tip1 = sorted_vertices1[i-1]
            tip2 = tip2_matrix[i, j]
            matching.append((tip1, tip2))
            i -= 1
            j -= direct
    matching.reverse()
    assert lib_graphs.are_edge_labels_unique_within_each_oriented_path(tipset_edges2)
    assert lib.are_unique(i for i, j in matching)
    assert lib.are_unique(j for i, j in matching)
    total_score = cumulative[m-1, n-1]
    return matching, total_score
    # TODO try to vectorize computation of cumulative and direction
    # TODO put larger (or more branched) dag on rows, smaller on columns
    # TODO keep edge list or precedence matrix (decide which if more efficient)

def dynprog_match_dags(scores: np.ndarray, edges1, edges2) -> Tuple[Matching, float]:
    n_vertices1, n_vertices2 = scores.shape
    precedence1 = lib_graphs.precedence_matrix_from_edges_dumb(n_vertices1, edges1)
    precedence2 = lib_graphs.precedence_matrix_from_edges_dumb(n_vertices2, edges2)
    # sorted_vertices1 = lib_graphs.sort_dag(range(n_vertices1), lambda u,v: precedence1[u,v])
    tipsets1, tipset_edges1 = lib_graphs.left_subdag_tipsets(n_vertices1, edges1, precedence_matrix=precedence1)
    tipsets2, tipset_edges2 = lib_graphs.left_subdag_tipsets(n_vertices2, edges2, precedence_matrix=precedence2)
    assert lib_graphs.are_edge_labels_unique_within_each_oriented_path(tipset_edges1)
    assert lib_graphs.are_edge_labels_unique_within_each_oriented_path(tipset_edges2)
    # lib.log_debug('n1 n2 m2', len(sorted_vertices1), len(tipsets2), len(tipset_edges2))
    m, n = len(tipsets1), len(tipsets2)
    cumulative = np.full((m, n), -1.0)
    cumulative[0, :] = 0
    cumulative[:, 0] = 0
    # direction = np.zeros((m, n), dtype=np.int32)  # 0 = move 1 up (skip tip1), -k = move k left (skip one of tipset2), +k = move 1 up and k left (match tip1 to one of tipset2)
    tip_matrix = np.empty((m, n, 2), dtype=np.int32)  
    tipset_edges1_by_outvertex = defaultdict(list)
    tipset_edges2_by_outvertex = defaultdict(list)
    dir = np.full((m, n, 2), NONMATCHED, dtype=np.int32)
    for edge in tipset_edges1:
        tipset_edges1_by_outvertex[edge.outvertex].append(edge)
    for edge in tipset_edges2:
        tipset_edges2_by_outvertex[edge.outvertex].append(edge)
    for i in range(1, m):
        tipset1 = tipsets1[i]
        inedges1 = tipset_edges1_by_outvertex[i]
        for j in range(1, n):
            tipset2 = tipsets2[j]
            inedges2 = tipset_edges2_by_outvertex[j]
            # Skip any of tipset1:
            for i_predecessor, _, tip1 in inedges1:
                if cumulative[i_predecessor, j] > cumulative[i, j]:
                    cumulative[i, j] = cumulative[i_predecessor, j]
                    dir[i, j] = (i_predecessor, j)
            # Skip any of tipset2:
            for j_predecessor, _, tip2 in inedges2:
                if cumulative[i, j_predecessor] > cumulative[i, j]:
                    cumulative[i, j] = cumulative[i, j_predecessor]
                    dir[i, j] = (i, j_predecessor)
            # Match any of tipset1 to any of tipset2:
            for i_predecessor, _, tip1 in inedges1:
                for j_predecessor, _, tip2 in inedges2:
                    candidate_cum = cumulative[i_predecessor, j_predecessor] + scores[tip1, tip2]
                    if candidate_cum > cumulative[i,j]:
                        cumulative[i,j] = candidate_cum
                        dir[i,j] = (i_predecessor, j_predecessor)
                        tip_matrix[i, j] = (tip1, tip2)
    # lib.print_matrix(cumulative, 'tmp/cumulative.tsv')
    # lib.print_matrix(direction, 'tmp/direction.tsv')

    # Reconstruct path (matching):
    i, j = m-1, n-1
    matching: Matching = []
    while i > 0 and j > 0:
        i_predecessor, j_predecessor = dir[i, j]
        if i_predecessor != i and j_predecessor != j:
            tip1, tip2 = tip_matrix[i, j]
            matching.append((tip1, tip2))
        i, j = i_predecessor, j_predecessor
    matching.reverse()
    assert lib.are_unique(i for i, j in matching)
    assert lib.are_unique(j for i, j in matching)
    total_score = cumulative[m-1, n-1]
    return matching, total_score
    # TODO try to vectorize computation of cumulative and direction
    # TODO put larger (or more branched) dag on rows, smaller on columns
    # TODO keep edge list or precedence matrix (decide which if more efficient)

def merge_dags(n_vertices1: int, n_vertices2: int, edges1, edges2, matching: Matching, draw_result=False):
    n_matches = len(matching)
    n_vertices_new = n_vertices1 + n_vertices2 - n_matches
    matched1 = {i for i, j in matching}
    matched2 = {j for i, j in matching}
    new_vertices: List[Tuple[int, int]] = []
    vertices1_to_new = np.empty(n_vertices1, dtype=int)
    vertices2_to_new = np.empty(n_vertices2, dtype=int)
    for i in range(n_vertices1):
        if i not in matched1:
            vertices1_to_new[i] = len(new_vertices)
            new_vertices.append((i, NONMATCHED))
    for j in range(n_vertices2):
        if j not in matched2:
            vertices2_to_new[j] = len(new_vertices)
            new_vertices.append((NONMATCHED, j))
    for i, j in matching:
        vertices1_to_new[i] = len(new_vertices)
        vertices2_to_new[j] = len(new_vertices)
        new_vertices.append((i, j))
    # lib.log_debug('unmatched2', sorted(j for j in range(n_vertices2) if j not in matched2))
    # lib.log_debug('matching', len(matching), matching)
    # lib.log_debug('matched1 matched2', len(matched1), len(matched2))
    # lib.log_debug('matched2', Counter(j for i, j in matching))
    # lib.log_debug('v1_to_new', n_vertices1, vertices1_to_new)
    # lib.log_debug('v2_to_new', n_vertices2, vertices2_to_new)
    # lib.log_debug('new_vertices', len(new_vertices), new_vertices)
    assert len(new_vertices) == n_vertices_new, f'{len(new_vertices)} != {n_vertices_new}'
    new_edge_set = set()
    for u1, v1 in edges1:
        u = vertices1_to_new[u1]
        v = vertices1_to_new[v1]
        new_edge_set.add((u,v))
    for u2, v2 in edges2:
        u = vertices2_to_new[u2]
        v = vertices2_to_new[v2]
        new_edge_set.add((u,v))
    new_edges = list(new_edge_set)
    precedence = lib_graphs.precedence_matrix_from_edges_dumb(n_vertices_new, new_edges)
    new_edges_nonredundant = lib_graphs.Dag.from_precedence_matrix(precedence).edges  # TODO try to do this more efficiently without creating precedence matrix?
    # lib.log_debug('edges', len(edges_new), sorted(edges_new))
    # lib.log_debug('nonred_edges', len(edges_new_nonredundant), edges_new_nonredundant)
    # lib.log_debug('precedence', precedence)
    if draw_result:
        import networkx as nx  # type: ignore
        from matplotlib import pyplot as plt
        DGnew = nx.DiGraph()
        DGnew.add_nodes_from(new_vertices)
        DGnew.add_edges_from((new_vertices[u], new_vertices[v]) for u,v in new_edges_nonredundant)
        nx.draw_kamada_kawai(DGnew, with_labels=True, node_color='red')
        plt.savefig('tmp/dag.png')
        plt.close()  # type: ignore
    return new_vertices, new_edges_nonredundant

def aggregate_stuff_in_merged_vertices(new_vertices: Matching, stuffs1, stuffs2, aggregation_function):
    return [stuffs1[u] if v==NONMATCHED else stuffs2[v] if u==NONMATCHED else aggregation_function(stuffs1[u], stuffs2[v]) for (u,v) in new_vertices]

def local_edges_from_global_edges(global_edges: Collection[Edge], offsets: list[int]) -> list[list[Edge]]:
    n = len(offsets) - 1
    local_edges: list[list[Edge]] = [[] for i in range(n)]
    i_structure = 0
    for u, v, orientation in sorted(global_edges):
        while u >= offsets[i_structure+1]:
            i_structure += 1
        fro = offsets[i_structure]
        to = offsets[i_structure+1]
        assert fro <= u < to
        assert fro <= v < to
        local_edges[i_structure].append((u - fro, v - fro, orientation))
    return local_edges

def include_min_ladder_in_score_matrix_weighted(score: np.ndarray, weights1: np.ndarray, weights2: np.ndarray, edges1: List[Edge], edges2: List[Edge]) -> np.ndarray:
    '''Edges must have form (vertex, vertex, orientation, ..., weight)'''
    n1, n2 = score.shape
    edge_lists1 = edge_lists_from_edges(n1, edges1)
    edge_lists2 = edge_lists_from_edges(n2, edges2)
    new_score = score.copy()
    strands1 = [ i for i, edge_list in enumerate(edge_lists1) if sum(weight for *_, weight in edge_list) >= weights1[i] ]  # here some strands can act like helices, if their partners have been filtered out by length filter
    strands2 = [ i for i, edge_list in enumerate(edge_lists2) if sum(weight for *_, weight in edge_list) >= weights2[i] ]  # here some strands can act like helices, if their partners have been filtered out by length filter
    for ui in strands1:
        for uj in strands2:
            weight_ui = weights1[ui]
            weight_uj = weights2[uj]
            edges_i = edge_lists1[ui]
            edges_j = edge_lists2[uj]
            remaining_edge_weights_i = [w/weight_ui for *_, w in edges_i]
            remaining_edge_weights_j = [w/weight_uj for *_, w in edges_j]
            remaining_weight = 1.0
            acquired_score = 0.0
            scored_edge_pairs = sorted((score[vi, vj], ei, ej) for ei, (_ui, vi, oi, *_) in enumerate(edges_i) for ej, (_uj, vj, oj, *_) in enumerate(edges_j) if oi == oj)
            while remaining_weight > 0 and len(scored_edge_pairs) > 0:
                best_score, ei, ej = scored_edge_pairs.pop()
                weight = min(remaining_weight, remaining_edge_weights_i[ei], remaining_edge_weights_j[ej])
                if weight > 0:
                    acquired_score += weight * best_score
                    remaining_weight -= weight
                    remaining_edge_weights_i[ei] -= weight
                    remaining_edge_weights_j[ej] -= weight
            new_score[ui, uj] = (score[ui, uj] + acquired_score) / 2
    return new_score


def test_dynprog_match_dags():
    n_vertices1 = 10
    edges1 = [(i, i+1) for i in range(n_vertices1 - 1)]
    # edges = [(0,1), (1,2), (2,3), (3,4), (1,5), (1,6), (4,7), (5,7), (7,8), (6,8), (8,9)]
    n_vertices2 = 13
    edges2 = [(0,1), (1,2), (2,3), (3,4), (4,10), (10,11), (11,12), (1,5), (5,6), (6,4), (6,7), (7,8), (8,9), (9,11)]
    # n_vertices = 6
    # edges = [(0,1), (1,2), (2,3), (0,4), (4,5), (5,3)]
    matching, score = dynprog_match_dags(np.ones((n_vertices1, n_vertices2)), edges1, edges2)
    lib.log(matching, score)


class AcyclicClusteringSimple:
    def __init__(self, aggregate_function=lib_clustering.average_linkage_aggregate_function, max_joining_distance=np.inf):
        # aggregate_function - how to compute a value (of distance) for the cluster joined from two clusters value1,size1,value2,size2 => result_value
        self.aggregate_function = aggregate_function
        self.max_joining_distance = max_joining_distance

    def fit(self, distance_matrix, precedence_matrix, type_vector=None, domain_names=None, p_offsets=None, 
            member_count_threshold=0, sses=None, output_dir: Optional[Path]=None):
        # only samples with the same type can be linked
        n_D = distance_matrix.shape[0] # number of samples = number of leaves
        # assert n_D > 0
        if n_D == 0:
            # raise NotImplementedError('Cannot cluster 0 samples.')
            sys.stderr.write('WARNING: AcyclicClusteringSimple.fit(): Clustering 0 samples.\n')
            m = 0
            self.distances = np.full(m, 0.0) # distances between the children of each internal node
            self.children = np.full((m, 2), lib_clustering.NO_CHILD) # matrix of children pairs
            self.final_members = []
            self.n_clusters = 0
            self.labels = np.full(n_D, UNLABELLED)
            self.cluster_distance_matrix = lib.submatrix_int_indexing(distance_matrix, [], [])
            self.cluster_precedence_matrix = lib.submatrix_int_indexing(precedence_matrix, [], [])
            return
        else:
            m = 2*n_D - 1 # max. possible number of all nodes
        if distance_matrix.shape[1] != n_D:
            raise Exception('distance_matrix must be a square matrix')
        if precedence_matrix.shape != (n_D, n_D):
            raise Exception('precedence_matrix must have the same shape as distance_matrix')
        curr_n_nodes = n_D
        leader = list(range(n_D))
        members = [[i] for i in range(n_D)]
        children = np.full((m, 2), lib_clustering.NO_CHILD) # matrix of children pairs
        self.distances = np.full(m, 0.0) # distances between the children of each internal node
        self.children = children
        active_nodes = set(range(n_D))
        D = distance_matrix # no copying (to spare memory)
        P = precedence_matrix # no copying (to spare memory)
        T = type_vector if type_vector is not None else np.zeros(n_D, dtype=np.int8) # type vector

        def can_join(ij):
            i, j = ij
            if i in active_nodes and j in active_nodes:
                li = leader[i]
                lj = leader[j]
                if T[li] == T[lj] and not P[li, lj] and not P[lj, li]:
                    return True
            return False

        distance_queue: Any
        if n_D > SSE_COUNT_THRESHOLD_FOR_NUMPY_HEAP:
            lib.log('Using Numpy Heap')
            type_counts: dict[int, int] = Counter(type_vector)
            # capacity = m * (m-1)
            capacity = sum( c*(c-1) for c in type_counts.values() )  # TODO this capacity guess might not be sufficient!!! (use better guess or dynamic allocation)
            distance_queue = lib.NumpyMinHeap((2,), int, capacity, key_type=float, 
                keys_values=( (D[i, j], (i, j)) for (i, j) in itertools.combinations(active_nodes, 2) if can_join((i, j)) ))
        else:
            lib.log('Using Heapq')
            distance_queue = lib.PriorityQueue( (D[i, j], (i, j)) for (i, j) in itertools.combinations(active_nodes, 2) if can_join((i, j)) )
        

        with ProgressBar(n_D-1, title=f'Clustering {n_D} samples') as bar:
            while len(active_nodes) >= 2:
                best_pair = distance_queue.pop_min_which(can_join)
                if best_pair == None:
                    break  # no joinable pairs, algorithm has converged
                distance, (p, q) = best_pair
                if distance > self.max_joining_distance:
                    break  # no joinable pairs with lower distance than the limit, algorithm has converged
                p_leader, q_leader = leader[p], leader[q]
                new_leader = min(p_leader,  q_leader)
                new = curr_n_nodes
                curr_n_nodes += 1
                leader.append(p_leader)
                members.append(members[p] + members[q])  # TODO this can take much memory
                children[new,:] = [p, q]
                self.distances[new] = distance
                active_nodes.remove(p)
                active_nodes.remove(q)
                active_nodes.add(new)

                # Update precedence matrix (apply transitivity)
                active_leaders = [ leader[i] for i in active_nodes ]
                self.update_precedence(P, active_leaders, p_leader, q_leader)

                # Update distance matrix (calculate distances from the new cluster)
                for i in active_nodes:
                    if i != new:
                        i_leader = leader[i]
                        new_distance = self.aggregate_function(D[p_leader, i_leader], len(members[p]), D[q_leader, i_leader], len(members[q]))
                        D[new_leader, i_leader] = new_distance
                        D[i_leader, new_leader] = new_distance
                        if can_join((i, new)):
                            distance_queue.add(new_distance, (i, new))
                bar.step()

        # Filter by cluster size (number of members)
        active_nodes_list = [ node for node in active_nodes if len(members[node]) >= member_count_threshold ]
        active_nodes_list = lib_graphs.sort_dag(active_nodes_list, lambda i, j: P[leader[i], leader[j]] )
        active_leaders = [ leader[node] for node in active_nodes_list ]
        
        self.final_members = [ members[node] for node in active_nodes_list ]
        
        self.n_clusters = len(active_nodes_list)
        self.labels = np.full(n_D, UNLABELLED)
        for label in range(self.n_clusters):
            self.labels[self.final_members[label]] = label

        self.cluster_distance_matrix = lib.submatrix_int_indexing(D, active_leaders, active_leaders)
        self.cluster_precedence_matrix = lib.submatrix_int_indexing(P, active_leaders, active_leaders)

        names = [ ('H' if type_vector[l] == 0 else 'E') + str(i) for i, l in enumerate(active_leaders) ]
        if sses is not None:
            lengths = np.zeros((len(active_nodes_list), len(domain_names)), dtype=np.int32)
            for i_domain in range(len(domain_names)):
                fro, to = p_offsets[i_domain:i_domain+2]
                for i_sse in range(fro, to):
                    label = self.labels[i_sse]
                    if label != UNLABELLED:  # label UNLABELLED means excluded (too small) cluster
                        lengths[label, i_domain] = sses[i_sse]['length']
            if output_dir is not None:
                lib.print_matrix(lengths, output_dir/'lengths.tsv', row_names=names, col_names=domain_names)
        if output_dir is not None:
            lib.print_matrix(self.cluster_distance_matrix, output_dir/'cluster_distance_matrix.tsv', row_names=names, col_names=names)
            lib.print_matrix(self.cluster_precedence_matrix, output_dir/'cluster_precedence_matrix.tsv', row_names=names, col_names=names)

        return

    @staticmethod
    def update_precedence(P, active_leaders, p, q):  # joins clusters p and q in precedence matrix P, with active leaders p_active_leaders (+ formerly p and q)
        # P[new_leader, :] = np.logical_or(P[p_leader, :], P[q_leader, :])
        # P[:, new_leader] = np.logical_or(P[:, p_leader], P[:, q_leader])
        to_p = [l for l in active_leaders if P[l, p] and not P[l, q]] + [p]
        to_q = [l for l in active_leaders if P[l, q] and not P[l, p]] + [q]
        from_p = [l for l in active_leaders if P[p, l] and not P[q, l]] + [p]
        from_q = [l for l in active_leaders if P[q, l] and not P[p, l]] + [q]
        for i in to_p:
            for j in from_q:
                P[i, j] = True
        for i in to_q:
            for j in from_p:
                P[i, j] = True
        # TODO rewrite more numpyically

class GuidedAcyclicClustering:
    def __init__(self):
        pass
    
    def fit(self, sample_coords, score_function, sample_aggregation_function, offsets, guide_tree_children, 
            beta_connections=[], ladder_correction=False, weight_scores=True):
        '''
        sample_coords: d-dimensional coordinates of each sample (array[N,d])
        score_function: takes coords of two sample sets and their weights, returns score matrix for matching them (array[n1,d], array[n1],array[n2,d], array[n2] => array[n1,n2])
        sample_aggregation_function: takes coords of two samples and their weights, returns coords and weights of the resulting aggregated sample (array[d], float, array[d], float => array[d], float)
        offsets: index of the first sample of each sample path, the last number is the total number of samples (array[n_leaves+1])
        guide_tree_children: binary tree for guiding the order of merging sample paths (each row contains indices of one node's children, first n_leaves rows correspond to leaves (~sample paths)) array[2*n_leaves-1, 2]
        where n_leaves = number of sample paths (~proteins), N = total number of samples (~SSEs), d = number of coordinates describing each sample
        '''
        n_leaves = len(offsets) - 1

        coords = []
        weights = []
        precedence_edges = []
        members = []
        beta_edges: list[list[WeightedBetaEdge]] = [[(*edge, 1.0) for edge in local_edges] for local_edges in local_edges_from_global_edges(beta_connections, offsets)]  # last item (1.0) in each edge is weight

        # Process leaves (~sample paths)
        for i in range(n_leaves):
            fro = offsets[i]
            to = offsets[i+1]
            n_samples_here = to - fro
            coords.append(sample_coords[fro:to, :])
            weights.append(np.ones(n_samples_here))
            precedence_edges.append([(u, u+1) for u in range(n_samples_here - 1)])
            members.append([[sample] for sample in range(fro, to)])
        
        # Process internal nodes (~aggregated dags)
        with ProgressBar(n_leaves - 1, title='Guided clustering - merging nodes') as bar:
            for i in range(n_leaves, 2*n_leaves - 1):
                left, right = guide_tree_children[i]
                coords1 = coords[left]
                coords2 = coords[right]
                weights1 = weights[left]
                weights2= weights[right]
                prec_edges1 = precedence_edges[left]
                prec_edges2 = precedence_edges[right]
                beta_edges1 = beta_edges[left]
                beta_edges2 = beta_edges[right]
                n1 = coords1.shape[0]
                n2 = coords2.shape[0]
                scores = score_function(coords1, weights1, coords2, weights2)
                # lib.print_matrix(scores, 'tmp/scores.tsv')
                if ladder_correction:
                    scores = include_min_ladder_in_score_matrix_weighted(scores, weights1, weights2, beta_edges1, beta_edges2)
                    # lib.print_matrix(scores, 'tmp/scores_with_ladder_correction.tsv')
                if weight_scores:
                    scores = scores * weights1.reshape((-1, 1)) * weights2.reshape((1, -1))
                matching, matching_score = dynprog_match_dags(scores, prec_edges1, prec_edges2)
                new_vertices, new_prec_edges = merge_dags(n1, n2, prec_edges1, prec_edges2, matching, draw_result=False and (i==2*n_leaves-2))
                new_coords, new_weights = self.aggregate_coords_and_weights(new_vertices, coords1, coords2, weights1, weights2, sample_aggregation_function)
                new_members = self.aggregate_members(new_vertices, members[left], members[right])
                new_beta_edges = self.aggregate_edges(new_vertices, beta_edges1, beta_edges2)
                coords.append(new_coords)
                weights.append(new_weights)
                precedence_edges.append(new_prec_edges)
                members.append(new_members)
                beta_edges.append(new_beta_edges)
                # Save memory
                coords[left] = weights[left] = precedence_edges[left] = members[left] = beta_edges[left] = None
                coords[right] = weights[right] = precedence_edges[right] = members[right] = beta_edges[right] = None
                bar.step()

        final_members = members[-1]
        final_precedence_edges = precedence_edges[-1]
        n_clusters = len(final_members)
        cluster_precedence_matrix = lib_graphs.precedence_matrix_from_edges_dumb(n_clusters, final_precedence_edges)

        sorted_cluster_indices = lib_graphs.sort_dag(range(n_clusters), lambda i, j: cluster_precedence_matrix[i,j])
        final_members = [final_members[v] for v in sorted_cluster_indices]
        old_to_new = { o: n for n, o in enumerate(sorted_cluster_indices) }
        final_precedence_edges = [(old_to_new[u], old_to_new[v], *rest) for (u, v, *rest) in final_precedence_edges]
        cluster_precedence_matrix = lib_graphs.precedence_matrix_from_edges_dumb(n_clusters, final_precedence_edges)

        self.final_members = final_members
        self.n_clusters = n_clusters
        self.labels = get_labels_from_memberlists(final_members)
        self.cluster_precedence_matrix = cluster_precedence_matrix
    
    @staticmethod
    def aggregate_coords_and_weights(new_vertices: Matching, coords1, coords2, weights1, weights2, aggregation_function) -> Tuple[np.ndarray, np.ndarray]:
        n = len(new_vertices)
        d = coords1.shape[1]
        new_coords = np.empty((n, d))
        new_weights = np.empty((n,))
        for i, (u, v) in enumerate(new_vertices):
            if v == NONMATCHED:
                new_coords[i] = coords1[u]
                new_weights[i] = weights1[u]
            elif u == NONMATCHED:
                new_coords[i] = coords2[v]
                new_weights[i] = weights2[v]
            else:
                new_coords[i], new_weights[i] = aggregation_function(coords1[u], weights1[u], coords2[v], weights2[v])
        return new_coords, new_weights

    @staticmethod
    def aggregate_members(new_vertices: Matching, members1: List[List[int]], members2: List[List[int]]) -> List[List[int]]:
        new_members = [
            members1[u] if v == NONMATCHED 
            else members2[v] if u == NONMATCHED 
            else members1[u] + members2[v] 
            for (u, v) in new_vertices]
        return new_members
        
    @staticmethod
    def aggregate_edges(new_vertices: Matching, edges1: List[WeightedBetaEdge], edges2: List[WeightedBetaEdge]) -> List[WeightedBetaEdge]:
        old1_to_new = {u: i for i, (u, v) in enumerate(new_vertices)}
        old2_to_new = {v: i for i, (u, v) in enumerate(new_vertices)}
        new_edge_weights: Dict[Tuple[int, int, lib_sses.LadderType], float] = defaultdict(float)
        for p, q, orient, weight in edges1:
            edge = old1_to_new[p], old1_to_new[q], orient
            new_edge_weights[edge] += weight
        for p, q, orient, weight in edges2:
            edge = old2_to_new[p], old2_to_new[q], orient
            new_edge_weights[edge] += weight
        new_edges = [(p, q, orient, weight) for (p, q, orient), weight in sorted(new_edge_weights.items())]
        return new_edges
    