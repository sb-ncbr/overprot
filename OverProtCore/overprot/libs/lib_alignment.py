import sys
from pathlib import Path
from typing import Tuple, List, Dict
import itertools
import numpy as np  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from numba import jit  # type: ignore

from ete3 import Tree  # type: ignore  # sudo apt install python3-pyqt5.qtsvg

from . import lib
from . import lib_phylo_tree
from . import lib_acyclic_clustering
from .lib_logging import Timing, ProgressBar
from .lib_structure import Structure
from . import lib_pymol
from . import lib_domains
from .lib_similarity_trees.nntree import NNTree
from .lib_cli import cli_command, run_cli_command


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
        NONMATCHED = lib_acyclic_clustering.NONMATCHED
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
    try:
        import Bio  # type: ignore
        dict_blosum30 = Bio.SubsMat.MatrixInfo.blosum30
    except:
        raise NotImplementedError('Apparently Biopython does not contain Bio.SubsMat anymore. Try getting the BLOSUM30 matrix from another source (maybe https://web.archive.org/web/19991109223934/http://www.embl-heidelberg.de/~vogt/matrices/blosum30.cmp)')
    mat, alph, index = substitution_matrix(dict_blosum30, alphabet=ALPHABET)
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

def bound_dist_dynprog_score_matrix(A: WeightedCoordinates, B: WeightedCoordinates, R0=10.0) -> np.ndarray:
    '''Create score matrix for dynamic programming alignment algorithm, for edit_distance_weighted'''
    # Original implementation:
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
    return bound_dist_dynprog_score_matrix_jit(A.coords, B.coords, A.relative_weights, B.relative_weights, R0)

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
    matching, total_score = lib_acyclic_clustering.dynprog_align(score_matrix, include_nonmatched=True)
    distance = 0.5 * (A.relative_weights.sum() + B.relative_weights.sum()) - total_score
    return distance, matching

def edit_distance_weighted(A: WeightedCoordinates, B: WeightedCoordinates) -> float:
    distance, matching = edit_distance_weighted_and_matching(A, B)
    return distance

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

        n_nodes = 2*n_structs - 1
        children = np.full((n_nodes, 2), lib_phylo_tree.NO_CHILD)
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

    tree = lib_phylo_tree.PhyloTree.from_children(children, node_names=names, distances=tree_distances)
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
    return children

# TESTING #####################################################################################

def check_triangle_inequality(distance_matrix: np.ndarray) -> bool:
    m, n = distance_matrix.shape
    assert m == n
    for i, j, k in itertools.permutations(range(n), 3):
        if distance_matrix[i, j] + distance_matrix[j, k] < distance_matrix[i, k]:
            return False
    return True

def print_alignment(structA: Structure, structB: Structure, matching: List[Tuple[int, int]], filename: Path):
    NONMATCHED = lib_acyclic_clustering.NONMATCHED
    out_aln = {'aligned_residues': [(structA.chain[i], int(structA.resi[i]), structB.chain[j], int(structB.resi[j])) for i, j in matching if i != NONMATCHED and j!= NONMATCHED]}
    lib.dump_json(out_aln, filename)

def test_edit_distance_weighted():
    structfileA = '/home/adam/Workspace/Python/OverProt/data/GuidedAcyclicClustering/trying/cyp_2/2nnj.alphas.cif'
    structfileB = '/home/adam/Workspace/Python/OverProt/data/GuidedAcyclicClustering/trying/cyp_2/1tqn.alphas.cif'
    structA = lib_pymol.read_cif(structfileA)
    coordsA = WeightedCoordinates(structA.coords)
    structB = lib_pymol.read_cif(structfileB)
    coordsB = WeightedCoordinates(structB.coords)
    distance, matching = edit_distance_weighted_and_matching(coordsA, coordsB)
    lib.log('distance:', distance)
    lib.log('len(matching):', len(matching))
    print_alignment(structA, structB, matching, 'tmp/alignment.json')
    lib_pymol.create_alignment_session(structfileA, structfileB, 'tmp/alignment.json', 'tmp/alignment.pse')
    coordsAB = WeightedCoordinates.combine(coordsA, coordsB, matching)
    lib.log('lengths A B AB:', len(coordsA), len(coordsB), len(coordsAB))
    lib.log('abs. weight AB:', coordsAB.absolute_weight)
    for i in range(len(coordsAB)):
        lib.log(matching[i], coordsAB.coords[:,i], coordsAB.relative_weights[i])

def testing_score_matrices(struct1: Path, struct2: Path):
    s1 = lib_pymol.read_cif(struct1, only_polymer=True)
    assert s1.is_alpha_trace()
    seq1 = s1.get_sequence(assume_alpha_trace=True)
    s2 = lib_pymol.read_cif(struct2, only_polymer=True)
    assert s2.is_alpha_trace()
    seq2 = s2.get_sequence(assume_alpha_trace=True)
    print(seq1, seq2, file=sys.stderr)

    subst_matrix = default_substitution_matrix(match_value=10)
    score_matrix_seq = seq_dynprog_score_matrix(seq1, seq2, subst_matrix)
    score_matrix_sqdist = sqdist_dynprog_score_matrix(s1, s2, match_value=64)

    with Timing('shape_dynprog_score_matrix_simpler'):
        score_matrix_shape2 = shape_dynprog_score_matrix_simpler(s1, s2, match_value=2.2)
    plt.imshow(score_matrix_shape2, cmap='inferno')  # type: ignore
    plt.colorbar()  # type: ignore
    plt.show()

    print(score_matrix_shape2.shape, file=sys.stderr)
    aln, score = lib_acyclic_clustering.dynprog_align(score_matrix_seq + score_matrix_shape2)
    out_aln = {'aligned_residues': [(s1.chain[i], int(s1.resi[i]), s2.chain[j], int(s2.resi[j])) for i,j in aln]}
    lib.dump_json(out_aln, sys.stdout)


#  MAIN  #####################################################################################

@cli_command()
def main(directory: Path, show_tree: bool = False) -> None:
    '''Run make_structure_tree_with_merging on all domains in directory/sample.json.
    @param `directory`  Directory with sample.json and structure files
    @param `show_tree`  Show the guide tree with ete3
    '''
    samples =lib_domains.load_domain_list(directory/'sample.json')
    structure_files = [directory/f'{domain.name}.cif' for domain in samples]
    # testing_score_matrices(*structure_files[:2])
    make_structure_tree_with_merging(structure_files, show_tree=show_tree)


if __name__ == '__main__':
    run_cli_command(main)
