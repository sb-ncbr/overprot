import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List, Dict, Any, Optional
import itertools
import numpy as np  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from numba import jit  # type: ignore

import Bio  # type: ignore
# from Bio.SubsMat import MatrixInfo
from ete3 import Tree  # type: ignore  # sudo apt install python3-pyqt5.qtsvg

from . import lib
from . import lib_clustering
from . import lib_acyclic_clustering_simple
from . import superimpose3d
from .lib_logging import Timing, ProgressBar
from .lib_structure import Structure
from . import lib_pymol
from . import lib_domains
from .lib_similarity_trees.nearest_pair_keepers import DumbNearestPairFinder, GHTNearestPairKeeper, MTNearestPairKeeper
from .lib_similarity_trees.nntree import NNTree


ALPHABET = list('ACDEFGHIKLMNPQRSTVWXY')
ALPHABET_INDEX = {letter: i for i, letter in enumerate(ALPHABET)}

class WeightedCoordinates:
    coords: np.ndarray  # point coordinates, coords.shape == (dim, n_points)
    relative_weigths: np.ndarray  # relative weight of each point, relative_weigths.shape == (n_points, ), 0 < relative_weigths <= 1
    absolute_weight: int  # absolute weight of the whole object (absolute weight of point i == absolute_weight * relative_weights[i])

    def __init__(self, coords: np.ndarray, relative_weights=None, absolute_weight=None):
        '''WeightedCoordinates with weights 1, coords.shape == (dim, n_points). Do not use optional arguments.'''
        dim, n_points = coords.shape
        self.coords = coords
        self.relative_weights = lib.not_none(relative_weights, np.ones(n_points))
        self.absolute_weight = lib.not_none(absolute_weight, 1)
    
    def __len__(self):
        return self.relative_weights.shape[0]

    @classmethod
    def combine(cls, A: 'WeightedCoordinates', B: 'WeightedCoordinates', matching: List[Tuple[int, int]]) -> 'WeightedCoordinates':
        '''Combine two WeightedCoordinates objects into one "sum" object'''
        NONMATCHED = lib_acyclic_clustering_simple.NONMATCHED
        # lib.log_debug('lengths A B matching:', len(A), len(B), len(matching))
        # lib.log_debug('matching:', *matching)
        # lib.log_debug('indices A:', *[i for i, j in matching if i != NONMATCHED])
        # lib.log_debug('indices B:', *[j for i, j in matching if j != NONMATCHED])
        assert [i for i, j in matching if i != NONMATCHED] == list(range(len(A)))
        assert [j for i, j in matching if j != NONMATCHED] == list(range(len(B)))
        n = len(matching)
        dim = A.coords.shape[0]

        # i indices apply to A, j to B, k to the result
        matching_ = np.array(matching)  # type: ignore
        k_onlyA = matching_[:, 1] == NONMATCHED
        k_onlyB = matching_[:, 0] == NONMATCHED
        k_matched = np.logical_not(np.logical_or(k_onlyA, k_onlyB))  # type: ignore
        i_onlyA = matching_[k_onlyA, 0]  # type: ignore
        i_matched = matching_[k_matched, 0]
        j_onlyB = matching_[k_onlyB, 1]  # type: ignore
        j_matched = matching_[k_matched, 1]

        absolute_weight = A.absolute_weight + B.absolute_weight
        
        # absolute weights per point
        weights_onlyA = A.relative_weights[i_onlyA] * A.absolute_weight
        weights_onlyB = B.relative_weights[j_onlyB] * B.absolute_weight
        weights_matchedA = A.relative_weights[i_matched] * A.absolute_weight
        weights_matchedB = B.relative_weights[j_matched] * B.absolute_weight
        weights_matched = weights_matchedA + weights_matchedB

        coords = np.empty((dim, n))
        coords[:, k_onlyA] = A.coords[:, i_onlyA]
        coords[:, k_onlyB] = B.coords[:, j_onlyB]
        coords[:, k_matched] = (A.coords[:, i_matched] * weights_matchedA +
                             B.coords[:, j_matched] * weights_matchedB) / weights_matched

        relative_weights = np.empty(n)
        relative_weights[k_onlyA] = weights_onlyA / absolute_weight
        relative_weights[k_onlyB] = weights_onlyB / absolute_weight
        relative_weights[k_matched] = weights_matched / absolute_weight
        
        return cls(coords, relative_weights=relative_weights, absolute_weight=absolute_weight)
    
    def full_info(self) -> str:
        head = f'WeightedCoordinates, length {len(self)}, abs. weight {self.absolute_weight}\n'
        body = '\n'.join(f'{self.relative_weights[i]} {self.coords[:, i]}' for i in range(len(self)))
        return head + body
    
    def to_structure(self) -> Structure:
        n = len(self)
        assert self.coords.shape == (3, n)
        struct = Structure(symbol=np.full(n, 'C'), name=np.full(n, 'CA'), resn=np.full(n, 'XXX'), resi=range(1, n+1), 
                           chain=np.full(n, 'X'), auth_chain=np.full(n, 'X'), entity=np.full(n, '1'), coords=np.round(self.coords, decimals=3))
        struct.add_field('occupancy', self.relative_weights)
        return struct

def fake_read_cif(filename):  # Will work only for PyMOL-generated files (hopefully)
    struct = defaultdict(list)
    with open(filename) as f:
        for line in f:
            if line.startswith('ATOM'):
                fields = line.split()
                struct['symbol'].append(fields[2])
                struct['name'].append(fields[3])
                struct['resn'].append(fields[5])
                struct['chain'].append(fields[6])
                struct['resi'].append(int(fields[8]))
                struct['x'].append(float(fields[10]))
                struct['y'].append(float(fields[11]))
                struct['z'].append(float(fields[12]))
    for key, values in struct.items():
        struct[key] = np.array(values)
    struct['coords'] = np.stack((struct['x'], struct['y'], struct['z']))
    return Structure(struct)

def substitution_matrix(scores: Dict[Tuple[str, str], float], alphabet=None, gap_penalty=0):
    if alphabet is None:
        alphabet = sorted(set( letter for substitution, score in scores for letter in substitution ))
    letter2index = { letter: i for i, letter in enumerate(alphabet) }
    n = len(alphabet)
    matrix = np.zeros((n, n))#, dtype=int)
    for (x, y), score in scores.items():
        xi = letter2index.get(x, None)
        yi = letter2index.get(y, None)
        if xi is not None and yi is not None:
            matrix[xi, yi] = score
            matrix[yi, xi] = score
    matrix[1:, 1:] += gap_penalty
    return matrix, alphabet, letter2index

def default_substitution_matrix(match_value=10) -> np.ndarray:
    mat, alph, index = substitution_matrix(Bio.SubsMat.MatrixInfo.blosum30, alphabet=ALPHABET)
    if mat.diagonal().min() <= -match_value:
        raise Exception(f'match_value must be more than {-mat.diagonal().min()}')
    mat += match_value
    norm = 1/np.sqrt(mat.diagonal())
    mat = norm.reshape((-1,1)) * mat * norm.reshape((1,-1))
    return mat

def seq_dynprog_score_matrix(seq1, seq2, subst_matrix) -> np.ndarray:
    seq1 = np.array([ALPHABET_INDEX[s] for s in seq1])
    seq2 = np.array([ALPHABET_INDEX[s] for s in seq2])
    return lib.submatrix_int_indexing(subst_matrix, seq1, seq2)

def shape_dynprog_score_matrix(struct1: Structure, struct2: Structure, match_value=2, kmer=5) -> np.ndarray:
    try:
        raise IOError
        rmsds, _, _ = lib.read_matrix('rmsds.tsv')
        print('Warning: read rmsds from file', file=sys.stderr)
    except IOError:
        n1 = struct1.count
        n2 = struct2.count
        # r1 = np.array([struct1.x, struct1.y, struct1.z])
        # r2 = np.array([struct2.x, struct2.y, struct2.z])
        r1 = struct1.coords
        r2 = struct2.coords
        rmsds = np.full((n1, n2), float(match_value))
        if kmer % 2 != 1:
            raise ValueError('kmer must be odd number')
        else:
            incl = kmer // 2
        for i in range(incl, n1-incl):
            for j in range(incl, n2-incl):
                A = r1[:, i-incl:i+incl+1]
                B = r2[:, j-incl:j+incl+1]
                rmsd = superimpose3d.rmsd(A, B)
                rmsds[i, j] = rmsd
        # print(rmsds, file=sys.stderr)
        lib.print_matrix(rmsds, 'rmsds.tsv')
    score = (match_value - rmsds) / match_value
    score[score < 0] = 0
    return score

def sqdist_dynprog_score_matrix(struct1: Structure, struct2: Structure, match_value=10) -> np.ndarray:
    r1 = struct1.coords
    r2 = struct2.coords
    diff = r1.reshape((3, -1, 1)) - r2.reshape((3, 1, -1))
    sqdist = (diff**2).sum(0)
    return (match_value - sqdist) / match_value

def canonical_shapes(struct: Structure):
    n = struct.count
    r = struct.coords
    incl = 2
    shapes = np.empty((n-2*incl, 3, 1+2*incl))
    for i in range(incl, n-incl):
        # A = shapes[i-incl, :, :]
        # A[:, :] = r[:, i-incl:i+incl+1]
        A = shapes[i-incl, :, :] = r[:, i-incl:i+incl+1]
        c = A.sum(axis=1, keepdims=True) / 5
        A -= c
        px = A[:, 3] - A[:, 1]
        px /= np.linalg.norm(px, axis=0)  # type: ignore
        py = 2*A[:, 2] - A[:, 1] - A[:, 3]
        py -= np.sum(px*py) * px
        py /= np.linalg.norm(py, axis=0)  # type: ignore
        pz = np.cross(px, py)  # type: ignore
        R = np.array([px, py, pz])
        A[:] = R @ A
    return shapes

def shape_dynprog_score_matrix_simpler(struct1: Structure, struct2: Structure, match_value=2) -> np.ndarray:
    shapes1 = canonical_shapes(struct1).reshape((-1, 1, 3, 5))
    shapes2 = canonical_shapes(struct2).reshape((1, -1, 3, 5))
    msds = ((shapes1 - shapes2) ** 2).sum(axis=(2,3)) / 5
    rmsds = np.full((msds.shape[0]+4, msds.shape[1]+4), match_value, dtype=np.float64)
    rmsds[2:-2, 2:-2] = np.sqrt(msds)
    # lib.print_matrix(rmsds, 'rmsds.tsv')
    score = (match_value - rmsds) / match_value
    score[score < 0] = 0
    return score

def check_triangle_inequality(distance_matrix: np.ndarray) -> bool:
    m, n = distance_matrix.shape
    assert m == n
    for i, j, k in itertools.permutations(range(n), 3):
        if distance_matrix[i, j] + distance_matrix[j, k] < distance_matrix[i, k]:
            return False
    return True

def matrix_overview(matrix):
    plt.subplot(2, 2, 1)
    plt.imshow(matrix, cmap='inferno')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.plot(matrix.max(1))
    plt.plot(matrix.min(1))
    plt.subplot(2, 2, 3)
    plt.plot(matrix.max(0))
    plt.plot(matrix.min(0))
    plt.show()

def print_alignment(structA: Structure, structB: Structure, matching: List[Tuple[int, int]], filename: Path):
    NONMATCHED = lib_acyclic_clustering_simple.NONMATCHED
    out_aln = {'aligned_residues': [(structA.chain[i], int(structA.resi[i]), structB.chain[j], int(structB.resi[j])) for i, j in matching if i != NONMATCHED and j!= NONMATCHED]}
    lib.dump_json(out_aln, filename)


def eucldist_dynprog_score_matrix(struct1: Structure, struct2: Structure, match_value=10) -> np.ndarray:
    r1 = struct1.coords
    r2 = struct2.coords
    diff = r1.reshape((3, -1, 1)) - r2.reshape((3, 1, -1))
    sqdist = (diff**2).sum(0)
    eucldist = np.sqrt(sqdist)
    return (match_value - eucldist) / match_value

def dist(structfileA: Path, structfileB: Path, with_cealign=True, with_iteration=True):
    '''Calculate edit-distance of two structures. If with_cealign==False, assume that the structures are pre-aligned and skip cealign step.
    Prototype!
    '''
    if with_cealign:
        cealign_result = lib_pymol.cealign(structfileA, structfileB, fallback_to_dumb_align=True)
        R = cealign_result.rotation
        t = cealign_result.translation
    else:
        R = np.eye(3)
        t = np.zeros((3, 1))
    
    structA = lib_pymol.read_cif(structfileA)
    structB = lib_pymol.read_cif(structfileB)
    assert structA.is_alpha_trace(), f'Input structures must be alpha traces, "{structfileA}" is not.'
    assert structB.is_alpha_trace(), f'Input structures must be alpha traces, "{structfileB}" is not.'
    # TODO assert that we have 1 state, 1 chain, and every residue has exactly one C-alpha

    coordsB_original = structB.coords.copy()
    
    EPSILON = 0.1  # convergence criterion
    old_score = 0.0
    n_iters = 0    
    # with Timing('dist'):
    while True:
        structB.coords = superimpose3d.rotate_and_translate(coordsB_original, R, t)
        score_matrix = eucldist_dynprog_score_matrix(structA, structB, match_value=10)
        aln, score = lib_acyclic_clustering_simple.dynprog_align(score_matrix)
        # print('score', score, file = sys.stderr)
        if not with_iteration:
            break
        if score < old_score + EPSILON:
            break
        old_score = score
        n_iters += 1
        assert len(aln) > 0
        firsts, seconds = zip(*aln)
        R, t = superimpose3d.optimal_rotation_translation(coordsB_original[:, seconds], structA.coords[:, firsts])
    distance = 0.5 * (structA.count + structB.count) - score
    relative_distance = 2 * distance / (structA.count + structB.count)
    # print(n_iters, structA.count, structB.count, score, distance, relative_distance, file=sys.stderr)
    return distance

def bound_dist_dynprog_score_matrix(A: WeightedCoordinates, B: WeightedCoordinates, R0=10.0) -> np.ndarray:
    '''Create score matrix for dynamic programming alignment algorithm, for edit_distance_weighted'''
    return bound_dist_dynprog_score_matrix_jit(A.coords, B.coords, A.relative_weights, B.relative_weights, R0)
    # rA = A.coords
    # rB = B.coords
    # dim, m = rA.shape
    # dim, n = rB.shape
    # wA = A.relative_weights.reshape((m, 1))
    # wB = B.relative_weights.reshape((1, n))
    # diff = rA.reshape((dim, m, 1)) - rB.reshape((dim, 1, n))
    # sq_r = (diff**2).sum(0)
    # r = np.sqrt(sq_r)
    # bound_dist = 1 - np.exp(-r / R0)  # in [0, 1)
    # distance = bound_dist * np.minimum(wA, wB) + 0.5 * np.abs(wA - wB)
    # score = 0.5 * wA + 0.5 * wB - distance
    # return score

@jit(nopython=True)
def bound_dist_dynprog_score_matrix_jit(rA: np.ndarray, rB: np.ndarray, wA_: np.ndarray, wB_: np.ndarray, R0: float) -> np.ndarray:
    '''Create score matrix for dynamic programming alignment algorithm, for edit_distance_weighted'''
    dim, m = rA.shape
    dim, n = rB.shape
    wA = wA_.reshape((m, 1))
    wB = wB_.reshape((1, n))
    # diff = rA.reshape((dim, m, 1)) - rB.reshape((dim, 1, n))
    # r = np.sqrt((diff**2).sum(0))
    r = np.sqrt(((rA.reshape((dim, m, 1)) - rB.reshape((dim, 1, n)))**2).sum(0))  # ugly but a bit faster
    bound_dist = 1 - np.exp(-r / R0)  # in [0, 1)
    distance = bound_dist * np.minimum(wA, wB) + 0.5 * np.abs(wA - wB)
    score = 0.5 * wA + (0.5 * wB - distance)
    return score

def edit_distance_weighted_and_matching(A: WeightedCoordinates, B: WeightedCoordinates) -> Tuple[float, List[Tuple[int, int]]]:
    '''Calculate edit-distance of two weighted structures. Adding/removing a point costs 1/2, moving point cost depends on the distance.
    (Don't perform cealign nor iterative refinement)'''
    score_matrix = bound_dist_dynprog_score_matrix(A, B, R0=10)
    # MAX_SCORE, MAX_ELEMENTS, SCORE_TYPE = 2**16, 2**16 - 1, np.uint32
    # score_matrix = (score_matrix * MAX_SCORE).astype(SCORE_TYPE)
    matching, total_score = lib_acyclic_clustering_simple.dynprog_align(score_matrix, include_nonmatched=True)
    # total_score = total_score / MAX_SCORE
    distance = 0.5 * (A.relative_weights.sum() + B.relative_weights.sum()) - total_score
    return distance, matching

def edit_distance_weighted(A: WeightedCoordinates, B: WeightedCoordinates) -> float:
    distance, matching = edit_distance_weighted_and_matching(A, B)
    return distance

def test_edit_distance_weighted():
    structfileA = '/home/adam/Workspace/Python/OverProt/data/GuidedAcyclicClustering/trying/cyp_2/2nnj.alphas.cif'
    structfileB = '/home/adam/Workspace/Python/OverProt/data/GuidedAcyclicClustering/trying/cyp_2/1tqn.alphas.cif'
    structA = lib_pymol.read_cif(structfileA)
    coordsA = WeightedCoordinates(structA.coords)
    structB = lib_pymol.read_cif(structfileB)
    coordsB = WeightedCoordinates(structB.coords)
    distance, matching = edit_distance_weighted_and_matching(coordsA, coordsB)
    lib.log_debug('distance:', distance)
    lib.log_debug('len(matching):', len(matching))
    # for i, j in matching:
    #     lib.log_debug(i, j)
    print_alignment(structA, structB, matching, 'tmp/alignment.json')
    lib_pymol.create_alignment_session(structfileA, structfileB, 'tmp/alignment.json', 'tmp/alignment.pse')
    coordsAB = WeightedCoordinates.combine(coordsA, coordsB, matching)
    lib.log_debug('lengths A B AB:', len(coordsA), len(coordsB), len(coordsAB))
    lib.log_debug('abs. weight AB:', coordsAB.absolute_weight)
    for i in range(len(coordsAB)):
        lib.log_debug(matching[i], coordsAB.coords[:,i], coordsAB.relative_weights[i])


def make_structure_tree(structs: List[Path], show_tree=False, with_cealign=True, with_iteration=True):
    structs = list(structs)
    n_structs = len(structs)
    names = [struct.stem for struct in structs]

    for i in range(n_structs):
        try:
            s = lib_pymol.read_cif(structs[i])
        except ValueError:
            print(names[i], file=sys.stderr)
            raise
        if not s.is_alpha_trace():
            alpha_struct = structs[i].parent / f'{names[i]}.alphas.cif'
            lib_pymol.extract_alpha_trace(structs[i], alpha_struct)
            structs[i] = alpha_struct

    distance_matrix = np.zeros((n_structs, n_structs), dtype=np.float64)
    with ProgressBar(n_structs*(n_structs-1)//2, title=f'Calculating structure distance matrix for {n_structs} structures') as bar:
        for i, j in itertools.combinations(range(n_structs), 2):
            distance = dist(structs[i], structs[j], with_cealign=with_cealign, with_iteration=with_iteration)
            distance_matrix[i,j] = distance_matrix[j,i] = distance
            bar.step()

    lib.print_matrix(distance_matrix, 'tmp/distance_matrix.tsv', names, names)

    ac = lib_acyclic_clustering_simple.AcyclicClusteringSimple()
    with Timing('ac.fit'):
        ac.fit(distance_matrix, np.zeros_like(distance_matrix), type_vector=np.zeros(n_structs))
    lib.log_debug(ac.n_clusters, ac.final_members, ac.labels, ac.children, sep='\n', file=sys.stderr)

    sorted_leaves = lib_clustering.children_to_list(ac.children)
    sorted_distance_matrix = lib.submatrix_int_indexing(distance_matrix, sorted_leaves, sorted_leaves)
    sorted_names = lib.submatrix_int_indexing(names, sorted_leaves)
    lib.log_debug('sorted_leaves:', sorted_leaves, sep='\n', file=sys.stderr)
    lib.print_matrix(sorted_distance_matrix, 'tmp/distance_matrix_sorted.tsv', sorted_names, sorted_names)
    
    newi = lib_clustering.children_to_newick(ac.children, [-1], distances=ac.distances, node_names=names)
    lib.log_debug(newi)
    if show_tree:
        t = Tree(newi)
        t.show()
    plt.hist(distance_matrix.flatten(), bins=range(0,int(distance_matrix.max()+1),10))  # type: ignore
    plt.show()
    return ac.children

def make_structure_tree_with_merging(structs: List[Path], show_tree=False, progress_bar=False):
    structs = list(structs)
    n_structs = len(structs)
    assert n_structs > 0
    dirs = [struct.parent for struct in structs]
    names = [struct.stem for struct in structs]
    
    with ProgressBar(n_structs, title=f'Extracting alpha-traces for {n_structs} structures', mute = not progress_bar) as bar: 
        for i in range(n_structs):
            try:
                s = lib_pymol.read_cif(structs[i])
            except ValueError:
                print(names[i], file=sys.stderr)
                raise
            if not s.is_alpha_trace():
                alpha_struct = dirs[i] / f'{names[i]}.alphas.cif'
                lib_pymol.extract_alpha_trace(structs[i], alpha_struct)
                structs[i] = alpha_struct
            bar.step()
    
    coords_dict: Dict[int, WeightedCoordinates] = {}
    finder: NNTree = NNTree(edit_distance_weighted, with_nearest_pair_queue=True)

    insertion_order = np.random.choice(n_structs, n_structs, replace=False)

    with Timing('Calculating guiding tree'):
        with ProgressBar(n_structs, title=f'Adding {n_structs} structures to {type(finder).__name__}', mute = not progress_bar) as bar:
            for i in insertion_order:
                structfile = structs[i]
                struct = lib_pymol.read_cif(structfile)
                coords = WeightedCoordinates(struct.coords)
                coords_dict[i] = coords
                finder.insert(i, coords)
                bar.step()

        # distance_matrix = lib.matrix_from_function((n_structs, n_structs), finder._tree.get_distance, symmetric=True, diag_value=0.0)
        # lib.print_matrix(distance_matrix, 'tmp/distance_matrix.tsv', names, names)
        # return

        n_nodes = 2*n_structs - 1
        children = np.full((n_nodes, 2), lib_clustering.NO_CHILD)
        tree_distances = np.full(n_nodes, 0.0) # distances between the children of each internal node

        with ProgressBar(n_structs-1, title=f'Merging {n_structs} structures in {type(finder).__name__}', mute = not progress_bar) as bar:
            for parent in range(n_structs, n_nodes):
                left, right, distance = finder.pop_nearest_pair() 
                left, right = sorted((left, right))
                coords_i = coords_dict[left]
                coords_j = coords_dict[right]
                _, matching = edit_distance_weighted_and_matching(coords_i, coords_j)
                new_coords = WeightedCoordinates.combine(coords_i, coords_j, matching)
                coords_dict[parent] = new_coords
                coords_dict.pop(left)  # save memory
                coords_dict.pop(right)  # save memory
                finder.insert(parent, new_coords)
                children[parent, :] = left, right
                tree_distances[parent] = distance
                bar.step()
        print(finder._tree.get_statistics())

    # result_coords = lib.single(finder.items.values())
    # lib.log_debug(result_coords.full_info())
    # plt.bar(range(len(result_coords)), result_coords.relative_weights, width=1)
    # plt.show()

    # sorted_leaves = lib_clustering.children_to_list(children)
    # sorted_distance_matrix = lib.submatrix_int_indexing(distance_matrix, sorted_leaves, sorted_leaves)
    # sorted_names = lib.submatrix_int_indexing(names, sorted_leaves)
    # lib.log_debug('sorted_leaves:', sorted_leaves, sep='\n', file=sys.stderr)
    # lib.print_matrix(sorted_distance_matrix, 'tmp/distance_matrix_sorted.tsv', sorted_names, sorted_names)

    # newi = lib_clustering.children_to_newick(children, node_names=names, distances=tree_distances)
    tree = lib_clustering.PhyloTree.from_children(children, node_names=names, distances=tree_distances)
    consensus_structure = coords_dict[n_nodes-1].to_structure().to_cif()
    newick_file = dirs[0]/'guide_tree.newick'
    children_file = dirs[0]/'guide_tree.children.tsv'
    consensus_structure_file = dirs[0]/'consensus_structure.cif'
    with open(newick_file, 'w') as w:
        w.write(str(tree))
    lib.print_matrix(children, children_file)
    with open(consensus_structure_file, 'w') as w:
        w.write(consensus_structure)
    print(f'Results in {newick_file}, {children_file}, {consensus_structure_file}.')
    if show_tree:
        t = Tree(str(tree), format=1)
        t.show()
    # tree.show()#debug
    # dists = distance_matrix[np.triu_indices(n_structs, k=1)]
    # rsd = dists.std() / dists.mean()
    # print('rsd', rsd)
    # plt.hist(dists, bins=range(0, int(distance_matrix.max() + 1), 10))
    # plt.show()
    return children

def testing_score_matrices(struct1: Path, struct2: Path):
    # s1 = fake_read_cif(struct)
    s1 = lib_pymol.read_cif(struct1, only_polymer=True)
    assert s1.is_alpha_trace()
    # s1 = s1.get_alpha_trace()
    seq1 = s1.get_sequence(assume_alpha_trace=True)
    # s2 = fake_read_cif(struct2)
    s2 = lib_pymol.read_cif(struct2, only_polymer=True)
    assert s2.is_alpha_trace()
    # s2 = s2.get_alpha_trace()
    seq2 = s2.get_sequence(assume_alpha_trace=True)
    print(seq1, seq2, file=sys.stderr)

    subst_matrix = default_substitution_matrix(match_value=10)
    score_matrix_seq = seq_dynprog_score_matrix(seq1, seq2, subst_matrix)
    score_matrix_sqdist = sqdist_dynprog_score_matrix(s1, s2, match_value=64)

    # with Timing('shape_dynprog_score_matrix'):
    #     score_matrix_shape = shape_dynprog_score_matrix(s1, s2, match_value=1.5, kmer=5)
    # plt.imshow(score_matrix_shape, cmap='inferno')
    # plt.colorbar()
    # plt.show()

    with Timing('shape_dynprog_score_matrix_simpler'):
        score_matrix_shape2 = shape_dynprog_score_matrix_simpler(s1, s2, match_value=2.2)
    plt.imshow(score_matrix_shape2, cmap='inferno')  # type: ignore
    plt.colorbar()  # type: ignore
    plt.show()

    # matrix_overview(score_matrix_shape2)
    print(score_matrix_shape2.shape, file=sys.stderr)

    aln, score = lib_acyclic_clustering_simple.dynprog_align(score_matrix_seq + score_matrix_shape2)

    out_aln = {'aligned_residues': [(s1.chain[i], int(s1.resi[i]), s2.chain[j], int(s2.resi[j])) for i,j in aln]}
    lib.dump_json(out_aln, sys.stdout)


#  MAIN  #####################################################################################

def parse_args() -> Dict[str, Any]:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='Directory with sample.json and structure files', type=Path)
    parser.add_argument('--show_tree', help='Show the guide tree with ete3', action='store_true')
    args = parser.parse_args()
    return vars(args)


def main(directory: Path, show_tree: bool = False) -> Optional[int]:
    '''Foo'''
    # TODO add docstring
    samples =lib_domains.load_domain_list(directory/'sample.json')
    structure_files = [directory/f'{domain.name}.cif' for domain in samples]
    make_structure_tree_with_merging(structure_files, show_tree=show_tree)
    # testing_score_matrices(*structure_files[:2])
    return None

    


if __name__ == '__main__':
    args = parse_args()
    exit_code = main(**args)
    if exit_code is not None:
        exit(exit_code)

