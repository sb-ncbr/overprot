import json
import numpy as np
from numba import jit
from pathlib import Path
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple, Literal, NamedTuple, Final
from numpy import ndarray as Array
np.seterr(all='raise')

from overprot.libs import lib
from overprot.libs import lib_alignment
from overprot.libs import lib_pymol
from overprot.libs import lib_acyclic_clustering_simple
from overprot.libs.lib_structure import Structure
from overprot.libs.lib import Timing, ProgressBar
from overprot.libs import superimpose3d
from overprot.libs.lib_similarity_trees.nntree import NNTree
from overprot.libs.lib_similarity_trees.ghtree import GHTree
from overprot.libs.lib_similarity_trees.mtree import MTree
from overprot.libs.lib_similarity_trees.vptree import VPTree
from overprot.libs.lib_similarity_trees.mvptree import MVPTree
from overprot.libs.lib_similarity_trees.target import Target
from overprot.libs.lib_similarity_trees.target_with_vpt import TargetWithVPT
from overprot.libs.lib_similarity_trees.dummy_searcher import DummySearcher
from overprot.libs.lib_similarity_trees.bubble_tree import BubbleTree
from overprot.libs.lib_similarity_trees.laesa import LAESA



Array1D = Array2D = Array3D = Array4D = Array

DATA = Path('/home/adam/Workspace/Python/OverProt/data-ssd/tree/sample4076')
SAMPLE_JSON = DATA / 'sample.json'
STRUCTURES = DATA / 'structures_cif'
ALPHAS_CIF = DATA / 'alphas_cif'
ALPHAS_CSV = DATA / 'alphas_csv'
ALPHAS_NPY = DATA / 'alphas_npy'
RESULTS = DATA / 'results'
EPSILON = 1e-4

SHAPE_LEN = 5


@dataclass
class StructInfo(object):
    name: str
    _coords: Optional[Array2D] = None
    _shapes: Optional[Array3D] = None
    def __init__(self, name: str, init_all: bool = False) -> None:
        self.name = name
        if init_all:
            _ = self.coords
            _ = self.shapes
    @property
    def coords(self) -> Array2D:
        if self._coords is None:
            self._coords = np.load(ALPHAS_NPY/f'{self.name}.npy')
        return self._coords
    @property
    def shapes(self) -> Array3D:
        if self._shapes is None:
            self._shapes = get_shapes(self.coords)
            _n, _s, _3 = self._shapes.shape
            assert _n == self.n
            assert _s == SHAPE_LEN
            assert _3 == 3
        return self._shapes
    @property
    def n(self) -> int:
        return self.coords.shape[0]
    def __len__(self) -> int:
        return self.n
    def __str__(self) -> str:
        return f'[Structure {self.name} ({len(self)})]'
    @property
    def alphas_cif(self) -> Path:
        return ALPHAS_CIF / f'{self.name}.cif'
    @property
    def alphas_csv(self) -> Path:
        return ALPHAS_CSV / f'{self.name}.csv'
    @property
    def alphas_npy(self) -> Path:
        return ALPHAS_NPY / f'{self.name}.npy'

def get_domains(n: int, create_choice: Optional[bool] = None) -> List[str]:
    if create_choice is None:
        try:
            return get_domains(n, create_choice=False)
        except FileNotFoundError:
            return get_domains(n, create_choice=True)
    if create_choice:
        js = json.loads(SAMPLE_JSON.read_text())
        if isinstance(js, list):
            all_domains = [dom['domain'] for dom in js]
        elif isinstance(js, dict):
            all_domains = [f"{dom['domain']}/{fam}" for fam, doms in js.items() for dom in doms]
        else:
            raise TypeError
        assert n <= len(all_domains)
        choice = sorted(np.random.choice(len(all_domains), n, replace=False))
        domains_with_families = [all_domains[i] for i in choice]
        with open(DATA / f'choice_{n}.txt', 'w') as w:
            print(*domains_with_families, sep='\n', file=w)
    else:
        domains_with_families = Path.read_text(DATA/f'choice_{n}.txt').split()
    domains = [domfam.split('/')[0] for domfam in domains_with_families]
    return domains

def get_domains_and_families() -> Tuple[List[str], Dict[str, List[int]]]:
    js = json.loads(SAMPLE_JSON.read_text())
    assert isinstance(js, dict)
    families = {}
    domains = []
    i = 0
    for fam, doms in js.items():
        families[fam] = []
        for dom in doms:
            domains.append(dom['domain'])
            families[fam].append(i)
            i += 1
    return domains, families

def download_structures() -> None:
    sample = json.loads(SAMPLE_JSON.read_text())
    sample_simple = []
    for family, domains in sample.items():
        for domain in domains:
            sample_simple.append((domain['pdb'], domain['domain'], domain['chain_id'], domain['ranges']))
    (DATA/'sample.simple.json').write_text(json.dumps(sample_simple))
    STRUCTURE_CUTTER = '/home/adam/Workspace/Python/OverProt/overprot/OverProtCore/dependencies/StructureCutter/bin/Release/netcoreapp3.1/StructureCutter.dll'
    lib.run_dotnet(STRUCTURE_CUTTER, DATA/'sample.simple.json', '--sources', 'http://www.ebi.ac.uk/pdbe/entry-files/download/{pdb}_updated.cif', '--cif_outdir', STRUCTURES) 

def create_alphas() -> None:
    domains = [file.stem for file in STRUCTURES.glob('*.cif')]
    ALPHAS_CIF.mkdir(parents=True, exist_ok=True)
    ALPHAS_CSV.mkdir(parents=True, exist_ok=True)
    ALPHAS_NPY.mkdir(parents=True, exist_ok=True)
    title = f'Creating alpha traces ({len(domains)} structures)'
    with Timing(title), ProgressBar(len(domains), title=title) as bar:
        for domain in domains:
            struct: Structure = lib_pymol.read_cif(STRUCTURES / f'{domain}.cif')
            struct = struct.get_alpha_trace(remove_repeating_resi=True)
            coords = struct.coords.T  # shape (n, 3)
            struct.save_cif(ALPHAS_CIF / f'{domain}.cif')
            np.savetxt(ALPHAS_CSV / f'{domain}.csv', coords, fmt='%.3f')
            np.save(ALPHAS_NPY / f'{domain}.npy', coords)
            bar.step()

def visualize_matrix(n=10):
    mat = np.loadtxt(RESULTS / f'dist_{n}x{n}.csv')
    plt.imshow(mat)
    plt.savefig(RESULTS/f'dist_{n}x{n}.png')
    print(mat.min(), mat.max())

@jit(nopython=True)
def normalize(x: Array, axis: int) -> Array:
    norm = np.sqrt(np.sum(x**2, axis=axis))
    norm = np.expand_dims(norm, axis=axis)
    return x / norm

@jit(nopython=True)
def normalize_inplace(x: Array, axis: int) -> None:
    norm = np.sqrt(np.sum(x**2, axis=axis))
    norm = np.expand_dims(norm, axis=axis)
    x /= norm

@jit(nopython=True)
def fake_matmul(A: Array3D, B: Array3D) -> Array3D:
    '''Matrix multiplication in axes 1, 2, broadcasted along axis 0.'''
    n, a, b = A.shape
    n, b, c = B.shape
    C = np.zeros((n, a, c), dtype=A.dtype)
    for i in range(a):
        C[:, i, :] += np.sum(np.expand_dims(A[:,i,:], axis=-1) * B[:,:,:], axis=1)
    return C

@jit(nopython=True)
def get_shapes1(coords: Array2D) -> Array3D:  # (n, 3) -> (n, k, 3); k = 4
    coord_type = coords.dtype
    # TODO compare float64 (default numpy) vs float32 (coords for some reason)
    n, _3 = coords.shape
    assert _3 == 3
    assert n >= 5
    shape_length = 4
    shapes = np.empty((n, shape_length, 3), dtype=coord_type) 
    center = coords[2:n-2, :]
    q0 = coords[0:n-4, :] - center
    q1 = coords[1:n-3, :] - center
    # q2 == center - center == 0
    q3 = coords[3:n-1, :] - center
    q4 = coords[4:n, :] - center
    chi = q3 - q1
    psi = np.cross(-q1, q3)  # psi==np.cross(q2-q1, q3-q2), but q2==0
    omega = np.cross(chi, psi)
    chipsiomega = np.stack((chi, psi, omega), axis=-1)  # Transformation from x-y-z to chi-psi-omega coords
    normalize_inplace(chipsiomega, axis=1)
    quints = np.stack((q0, q1, q3, q4), axis=1)  # center point q2 is omitted because it is 0
    # Numba.jit cannot compile this: 
    # shapes[2:n-2, :, :] = centered_quints @ chipsiomega
    # Still the following loop with jit is faster than the previous line without jit
    for i in range(n-4):
        shapes[i+2, :, :] = quints[i] @ chipsiomega[i]
    # Fill missing values:
    shapes[0] = shapes[1] = shapes[2]
    shapes[-1] = shapes[-2] = shapes[-3]
    return shapes

@jit(nopython=True)
def get_shapes2(coords: Array2D) -> Array3D:  # (n, 3) -> (n, k, 3); k = 4
    coord_type = coords.dtype
    # TODO compare float64 (default numpy) vs float32 (coords for some reason)
    n, _3 = coords.shape
    assert _3 == 3
    assert n >= 5
    shape_length = 4
    shapes = np.empty((n, shape_length, 3), dtype=coord_type) 
    center = coords[2:n-2, :]
    q0 = coords[0:n-4, :] - center
    q1 = coords[1:n-3, :] - center
    # q2 == center - center == 0
    q3 = coords[3:n-1, :] - center
    q4 = coords[4:n, :] - center
    q01 = q0+q1
    q34 = q3+q4
    # chi = q3 - q1
    # psi = np.cross(-q1, q3)  # psi==np.cross(q2-q1, q3-q2), but q2==0
    chi = q34 - q01
    psi = np.cross(-q01, q34)  # psi==np.cross(q2-q1, q3-q2), but q2==0
    omega = np.cross(chi, psi)
    chipsiomega = np.stack((chi, psi, omega), axis=-1)  # Transformation from x-y-z to chi-psi-omega coords
    normalize_inplace(chipsiomega, axis=1)
    quints = np.stack((q0, q1, q3, q4), axis=1)  # center point q2 is omitted because it is 0
    # Numba.jit cannot compile this: 
    # shapes[2:n-2, :, :] = centered_quints @ chipsiomega
    # Still the following loop with jit is faster than the previous line without jit
    for i in range(n-4):
        shapes[i+2, :, :] = quints[i] @ chipsiomega[i]
    # Fill missing values:
    shapes[0] = shapes[1] = shapes[2]
    shapes[-1] = shapes[-2] = shapes[-3]
    return shapes

# @jit(nopython=True)
def get_shapes(coords: Array2D) -> Array3D:  # (n, 3) -> (n, SHAPE_LEN, 3)
    coord_type = coords.dtype
    # TODO compare float64 (default numpy) vs float32 (coords for some reason)
    n, _3 = coords.shape
    assert _3 == 3
    assert n >= 5
    shapes = np.empty((n, SHAPE_LEN, 3), dtype=coord_type) 
    q0 = coords[0:n-4, :]
    q1 = coords[1:n-3, :]
    q2 = coords[2:n-2, :]
    q3 = coords[3:n-1, :]
    q4 = coords[4:n, :]
    center = (q0+q1+q2+q3+q4)/5
    q0 = q0 - center
    q1 = q1 - center
    q2 = q2 - center
    q3 = q3 - center
    q4 = q4 - center
    quints = np.stack((q0, q1, q2, q3, q4), axis=1)
    q01 = 0.5*(q0+q1)
    q34 = 0.5*(q3+q4)
    
    chi = q34 - q01
    psi_ = q2 - 0.5 * (q1+q3)
    omega = np.cross(chi, psi_)
    psi = np.cross(omega, chi)
    
    # chi = q3 - q1
    # psi = np.cross(q2-q1, q3-q2)
    # omega = np.cross(chi, psi)
    
    chipsiomega = np.stack((chi, psi, omega), axis=-1)  # Transformation from x-y-z to chi-psi-omega coords
    normalize_inplace(chipsiomega, axis=1)
    # Numba.jit cannot compile this: 
    # shapes[2:n-2, :, :] = centered_quints @ chipsiomega
    # Still the following loop with jit is faster than the previous line without jit
    for i in range(n-4):
        shapes[i+2, :, :] = quints[i] @ chipsiomega[i]
    # Fill missing values:
    shapes[0] = shapes[1] = shapes[2]
    shapes[-1] = shapes[-2] = shapes[-3]
    return shapes

@jit(nopython=True)
def get_shapes_nice(coords: Array2D, omit_center: bool = True) -> Array3D:  # (n, 3) -> (n, k, 3); k = 5
    '''This should be more readable (but slow) implementation of get_shapes'''
    n, _3 = coords.shape
    assert _3 == 3
    assert n >= 5
    shapes = np.zeros((n, 5, 3)) 
    for i in range(2, n-2):
        chi = coords[i+1, :] - coords[i-1, :]
        psi = np.cross(coords[i, :] - coords[i-1, :], coords[i+1, :] - coords[i, :])
        assert np.abs(np.dot(chi, psi)) < EPSILON
        omega = np.cross(chi, psi)
        cpo = np.stack((chi, psi, omega), axis=-1)
        normalize_inplace(cpo, axis=0)
        shapes[i, :, :] = (coords[i-2:i+3, :] - coords[i:i+1, :]) @ cpo
    # Fill missing value:
    shapes[0] = shapes[1] = shapes[2]
    shapes[-1] = shapes[-2] = shapes[-3]
    if omit_center:
        shapes = np.stack((shapes[:,0,:], shapes[:,1,:], shapes[:,3,:], shapes[:,4,:]), axis=1)
    return shapes

def matrix_info(name: str, A: Array) -> None:
    print(name, ':', A.shape, A.dtype, 'min:', A.min(), 'max:', A.max())

def length_dist_min(domainA: Union[np.ndarray, str], domainB: Union[np.ndarray, str]) -> float:
    coordsA = np.load(ALPHAS_NPY/f'{domainA}.npy') if isinstance(domainA, str) else domainA
    coordsB = np.load(ALPHAS_NPY/f'{domainB}.npy') if isinstance(domainB, str) else domainB
    m, _3 = coordsA.shape; assert _3 == 3
    n, _3 = coordsB.shape; assert _3 == 3
    return 0.5 * abs(m - n)

def length_dist_max(domainA: Union[np.ndarray, str], domainB: Union[np.ndarray, str]) -> float:
    coordsA = np.load(ALPHAS_NPY/f'{domainA}.npy') if isinstance(domainA, str) else domainA
    coordsB = np.load(ALPHAS_NPY/f'{domainB}.npy') if isinstance(domainB, str) else domainB
    m, _3 = coordsA.shape; assert _3 == 3
    n, _3 = coordsB.shape; assert _3 == 3
    return 0.5 * (m + n)

DIST_TYPE = 'opi'
SHAPEDIST_MAX_RMSD = 7.0  #5.0
OPDIST_SCORE_TYPE = 'linear'  # 'exponential'|'linear'
OPDIST_MAX_RMSD = 15
# VERSION_NAME = '-s-v3maxrmsd7'
VERSION_NAME = '-opi10-lin15'
# VERSION_NAME = '-s+op-v3maxrmsd7-lin15'  # so far the best is s+op-v3maxrmsd7-lin15 (best in classifying against CATH)
LOWBOUND_VERSION_NAME = '-s_Opt-v3maxrmsd7'
# LOWBOUND_VERSION_NAME = '-s-v3maxrmsd7'
# best options: exp20, lin15 (based on sample885)

def op_score(domainA: StructInfo, domainB: StructInfo, rot_trans: Optional[Tuple[Array2D, Array2D]] = None) -> Tuple[Array2D, Tuple[Array2D, Array2D]]:
    if rot_trans is None:
        cealign = lib_pymol.cealign(domainA.alphas_cif, domainB.alphas_cif, fallback_to_dumb_align=True)
        rot_trans = cealign.rotation.T, cealign.translation.T  # convert matrices from column style (3, n) to row style (n, 3)
    R, t = rot_trans
    coordsA = domainA.coords
    coordsB = domainB.coords @ R + t
    r = np.sqrt(np.sum((coordsA.reshape((domainA.n, 1, 3)) - coordsB.reshape((1, domainB.n, 3)))**2, axis=2))
    if OPDIST_SCORE_TYPE == 'exponential':
        score = np.exp(-r / OPDIST_MAX_RMSD)
    elif OPDIST_SCORE_TYPE == 'linear':
        score = 1 - r / OPDIST_MAX_RMSD
        score[score < 0] = 0
    else: 
        raise AssertionError
    return score, rot_trans

def op_score_iterated(domainA: StructInfo, domainB: StructInfo, rot_trans: Optional[Tuple[Array2D, Array2D]] = None, n_iter: int = 10) -> Tuple[Array2D, Tuple[Array2D, Array2D]]:
    if rot_trans is None:
        cealign = lib_pymol.cealign(domainA.alphas_cif, domainB.alphas_cif, fallback_to_dumb_align=True)
        rot_trans = cealign.rotation.T, cealign.translation.T  # convert matrices from column style (3, n) to row style (n, 3)
    R, t = rot_trans
    i_iter = 0
    print(f'{domainA.name}-{domainB.name}')
    while True:
        coordsA = domainA.coords
        coordsB = domainB.coords @ R + t
        r = np.sqrt(np.sum((coordsA.reshape((domainA.n, 1, 3)) - coordsB.reshape((1, domainB.n, 3)))**2, axis=2))
        if OPDIST_SCORE_TYPE == 'exponential':
            score = np.exp(-r / OPDIST_MAX_RMSD)
        elif OPDIST_SCORE_TYPE == 'linear':
            score = 1 - r / OPDIST_MAX_RMSD
            score[score < 0] = 0
        else: 
            raise AssertionError
        if i_iter == n_iter:
            break
        if i_iter == 0:
            matching, total_score = lib_acyclic_clustering_simple.dynprog_align(score)
        else:
            _, total_score = lib_acyclic_clustering_simple.dynprog_align(score)
        print(f'    {i_iter}: {total_score}')
        n_matched = len(matching)
        coordsA_matched = domainA.coords[[u for u, v in matching]]
        coordsB_matched = domainB.coords[[v for u, v in matching]]
        r_uv = np.array([r[u, v] for u, v in matching])
        if OPDIST_SCORE_TYPE == 'exponential':
            MINR = 5.0
            r_uv[r_uv<MINR] = MINR
            weights = np.exp(-r_uv / OPDIST_MAX_RMSD) / r_uv
        elif OPDIST_SCORE_TYPE == 'linear':
            MINR = 5.0
            r_uv[r_uv<MINR] = MINR
            weights = 1 / (r_uv + EPSILON)
            weights[r_uv > OPDIST_MAX_RMSD] = 0.0  # TODO only calculate where <=
            if np.sum(weights) == 0:
                weights = np.ones((n_matched,))
        # weights = np.ones((n_matched,))
        # weights[r_uv > OPDIST_MAX_RMSD] = 0.0
        R, t = optimal_rot_trans(coordsA_matched, coordsB_matched, weights)
        i_iter += 1
    rot_trans = R, t
    return score, rot_trans

def optimal_rot_trans(A: Array2D, B: Array2D, weights: Optional[Array1D] = None) -> Tuple[Array2D, Array2D]:
    R_T, t_T = superimpose3d.optimal_rotation_translation(A.T, B.T, weights=weights)
    return R_T.T, t_T.T

def shape_score(domainA: StructInfo, domainB: StructInfo) -> Tuple[Array2D]:
    diff = domainA.shapes.reshape((domainA.n, 1, SHAPE_LEN, 3)) - domainB.shapes.reshape((1, domainB.n, SHAPE_LEN, 3))
    sqerror = np.sum(diff**2, axis=(2, 3))
    rmsd = np.sqrt(sqerror / SHAPE_LEN)
    score = 1 - (rmsd / SHAPEDIST_MAX_RMSD)
    score[score < 0] = 0
    return score,
   
def sop_dist(type: Literal['s', 'op', 's+op', 's*op', 's_Opt', 's_Pes'], domainA: StructInfo, domainB: StructInfo, rot_trans: Optional[Tuple[Array2D, Array2D]] = None) -> Tuple[float, Optional[Tuple[Array2D, Array2D]]]:
    if type == 's':
        score, = shape_score(domainA, domainB)
        rot_trans = None
    elif type == 'op':
        score, rot_trans = op_score(domainA, domainB, rot_trans)
    elif type == 'opi':
        score, rot_trans = op_score_iterated(domainA, domainB, rot_trans)
    elif type == 's+op':
        score_s, = shape_score(domainA, domainB)
        score_op, rot_trans = op_score(domainA, domainB, rot_trans=rot_trans)
        score = 0.5 * (score_s + score_op)
    elif type == 's_Opt':
        score_s, = shape_score(domainA, domainB)
        score = 0.5 * (score_s + 1)
    elif type == 's_Pes':
        score_s, = shape_score(domainA, domainB)
        score = 0.5 * score_s
    elif type == 's*op':
        score_s, = shape_score(domainA, domainB)
        score_op, rot_trans = op_score(domainA, domainB, rot_trans=rot_trans)
        score = score_s * score_op
    else:
        raise AssertionError
    # matrix_info('score', score)
    # plt.figure()
    # plt.hist(score.flatten())
    # plt.savefig(RESULTS / f'score_hist_{domainA}_{domainB}.png')
    # matrix_info('score', score)
    # plt.figure()
    # plt.imshow(score)
    # plt.savefig(RESULTS / f'score_{domainA}_{domainB}.png')
    # np.savetxt(RESULTS / f'score_{domainA}_{domainB}.csv', score)
    matching, total_score = lib_acyclic_clustering_simple.dynprog_align(score)
    print(f'    Score:{total_score}')
    input()
    distance = 0.5 * (domainA.n + domainB.n) - total_score
    return distance, rot_trans

def test_shape_dist():
    domains = [StructInfo(dom) for dom in '2nnj 1pq2 1og2 1tqn 1akd 6eye 1bpv 3mbt'.split()]
    domA = domains[-1]
    with Timing(f'shape_dist * {len(domains)}'):
        for domB in domains:
            sop_dist('s', domA, domB)

def test_shape():
    domA, domB = get_domains(2)
    coordsA = np.load(ALPHAS_NPY/f'{domA}.npy')
    coordsB = np.load(ALPHAS_NPY/f'{domB}.npy')
    R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    coordsA_ = coordsA @ R
    shapesA = get_shapes_nice(coordsA)
    shapesA = get_shapes(coordsA)
    shapesA_ = get_shapes(coordsA_)
    diff = shapesA - shapesA_
    error = np.sqrt(np.sum(diff**2))
    print('Diff (min, max, rse):', diff.min(), diff.max(), error)
    k = 500
    # with Timing(f'get_shapes_o=nice * {2*k}'):
    #     for i in range(k):
    #         shapesA = get_shapes_nice(coordsA)
    #         shapesB = get_shapes_nice(coordsB)
    with Timing(f'get_shapes     * {2*k}'):
        for i in range(k):
            shapesA = get_shapes(coordsA)
            shapesB = get_shapes(coordsB)
    assert error < EPSILON, error

def make_bubbles(n: int):
    distance_matrix = np.loadtxt(RESULTS / f'distance_{n}x{n}{VERSION_NAME}.csv')
    for max_radius in [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625]:
        bubbles = []
        for i in range(n):
            d, closest_bubble = min(((distance_matrix[i, b[0]], b) for b in bubbles), default=(np.inf, None))
            if d <= max_radius:
                closest_bubble.append(i)
            else:
                bubbles.append([i])
        print(max_radius, len(bubbles), max(len(bub) for bub in bubbles), sep='\t')
        # print(*[len(bub) for bub in bubbles])

def make_lengths(n: int):
    domains = [StructInfo(dom) for dom in get_domains(n)]
    lengths = [dom.n for dom in domains]
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savetxt(RESULTS / f'length_{n}.csv', lengths)

ZERO_ELEMENT = '-'

@dataclass
class DistanceCalculatorFromMatrix(object):
    distance_matrix: Array2D
    domain_index: Dict[str, int]
    domain_structures: Dict[str, StructInfo]
    distance_matrix_s_low: Optional[Array2D] = None
    def distance_function(self, a: str, b: str):
        if a == ZERO_ELEMENT:
            return 0.5 * self.domain_structures[b].n
        elif b == ZERO_ELEMENT:
            return 0.5 * self.domain_structures[a].n
        else:
            return self.distance_matrix[self.domain_index[a], self.domain_index[b]]
    def dist_low_bound_len(self, a: str, b: str):
        return 0.5*abs(self.domain_structures[a].n - self.domain_structures[a].n)
    def dist_low_bound_s(self, a: str, b: str):
        if self.distance_matrix_s_low is None:
            return 0.0
        else:
            return self.distance_matrix_s_low[self.domain_index[a], self.domain_index[b]]

@dataclass
class DistanceCalculatorFromAlignment(object):
    rotations: Array4D
    translations: Array4D
    domain_index: Dict[str, int]
    domain_structures: Dict[str, StructInfo]
    def distance_function(self, a: str, b: str):
        domA = self.domain_structures[a]
        domB = self.domain_structures[b]
        if a == ZERO_ELEMENT:
            return 0.5 * domB.n
        elif b == ZERO_ELEMENT:
            return 0.5 * domA.n
        else:
            ia = self.domain_index[a]
            ib = self.domain_index[b]
            rot = self.rotations[ia, ib]
            trans = self.translations[ia, ib]
            dist, _ = sop_dist(DIST_TYPE, domA, domB, rot_trans=(rot, trans))
            return dist

@dataclass
class DistanceCalculatorFromScratch(object):
    domain_structures: Dict[str, StructInfo]
    def distance_function(self, a: str, b: str):
        if a == ZERO_ELEMENT:
            return 0.5 * self.domain_structures[b].n
        elif b == ZERO_ELEMENT:
            return 0.5 * self.domain_structures[a].n
        else:
            dist, _ = sop_dist(DIST_TYPE, self.domain_structures[a], self.domain_structures[b])
            return dist

global distance_calculator

def global_distance_function(a: str, b: str):
    return distance_calculator.distance_function(a, b)

def make_tree(n: int, n_search: int = -1):
    if n_search == -1:
        n_search = n
    domains = []
    with ProgressBar(n, title=f'Creating {n} StructInfos') as bar, Timing(f'Creating {n} StructInfos'):
        for dom in get_domains(n):
            domains.append(StructInfo(dom, init_all=True))
            bar.step()
    # domains = [StructInfo(dom, init_all=True) for dom in get_domains(n)]
    domain_dict = {dom.name: dom for dom in domains}
    domain_index = {dom.name: i for i, dom in enumerate(domains)}
    global distance_calculator
    try:
        distances = np.loadtxt(RESULTS / f'distance_{n}x{n}{VERSION_NAME}.csv')
        distances_s_opt = np.loadtxt(RESULTS / f'distance_{n}x{n}{LOWBOUND_VERSION_NAME}.csv')
        # distance_popul = np.array([distances[i, j] for i in range(n) for j in range(i)])
        # print(f'distances: {min(distance_popul):.3f}-{max(distance_popul):.3f}, mean: {np.mean(distance_popul):.3f} median: {np.median(distance_popul):.3f}')
        distance_calculator = DistanceCalculatorFromMatrix(distances, domain_index, domain_dict, distance_matrix_s_low=distances_s_opt)
        print('Using precomputed distances.')
    except OSError:
        try:
            rotations: Array4D = np.load(RESULTS / f'rotations_{n}x{n}.npy')
            translations: Array4D = np.load(RESULTS / f'translations_{n}x{n}.npy')
            distance_calculator = DistanceCalculatorFromAlignment(rotations, translations, domain_index, domain_dict)
            print('Using precomputed alignments.')
        except OSError:
            # rotations = np.zeros((n, n, 3, 3))
            # translations = np.zeros((n, n, 1, 3))
            distance_calculator = DistanceCalculatorFromScratch(domain_dict)
            print('ALIGNING!!!')
    distance_function = global_distance_function
    n_dup = 1
    k = 3 * n_dup
    N_LOADING_PROCESSES = 1

    bulk_domains = [(f'{domain.name}', domain.name) for i in range(n_dup) for domain in np.random.permutation(domains)]
    with Timing(f'Bulk-loading {n*n_dup} domains in {N_LOADING_PROCESSES} processes'):
        # tree = MVPTree(distance_function, bulk_domains, n_processes=N_LOADING_PROCESSES)
        # tree = MVPTree(distance_function, bulk_domains, root_element=(ZERO_ELEMENT, ZERO_ELEMENT), n_processes=N_LOADING_PROCESSES, leaf_size=8)
        tree = LAESA(distance_function, bulk_domains, n_pivots=100, n_processes=N_LOADING_PROCESSES)
    # tree.save(RESULTS / f'mvptree_{n}.json', with_cache=False)
    # tree.save(RESULTS / f'mvptree_with_cache_{n}.json', with_cache=True)
    # tree = MVPTree.load(RESULTS / f'mvptree_with_cache_{n}.json', distance_function=distance_function, get_value = lambda key: key)

    print(tree.get_statistics())
    calcs_before = tree._distance_cache._calculated_distances_counter
    # return

    idx = 0
    wrongies = 0
    n_dup = 1
    times = []    
    with Timing(f'{n_dup}x searching {n_search} domains in the tree') as timing, ProgressBar(n_dup*n, title=f'{n_dup}x searching {n_search} domains in the tree', mute=False) as bar:
        for i in range(n_dup):
            domains_to_search = np.random.choice(domains, n_search, replace=False)
            for domain in np.random.permutation(domains_to_search):
                idx += 1
                real_knn = sorted((distance_function(domain.name, other.name), other.name) for other in domains)[:k]
                real_range = real_knn[-1][0]
                with Timing(f'Searching {domain}', mute=True) as timing1:
                    knn = tree.kNN_query_by_value3(domain.name, k, distance_low_bounds=[])
                    # knn = tree.kNN_query_by_value3(domain.name, k, distance_low_bounds=[], the_range=real_range+EPSILON)
                    # knn = tree.kNN_query_by_value(domain.name, k, distance_low_bounds=[distance_calculator.dist_low_bound_len, distance_calculator.dist_low_bound_s])
                times.append(timing1.time.total_seconds())
                if [dom for dist, dom in knn] != [dom for dist, dom in real_knn]:
                    wrongies += 1
                    # print('N Real kNN:', real_knn, 'Found kNN:', knn)
                bar.step()
    print('Per one:', timing.time / (n_dup*n_search))
    print('Wrongies:', wrongies)
    print(tree.get_statistics())
    calcs_after = tree._distance_cache._calculated_distances_counter
    print('Calculations:', calcs_after - calcs_before)
    # print(sorted(times))
    # matrix_info('times', times)

def make_distance_matrix_multiprocessing(n: int):
    N_PROCESSES = 6
    print('make_distance_matrix', DIST_TYPE, SHAPEDIST_MAX_RMSD, OPDIST_SCORE_TYPE, OPDIST_MAX_RMSD, VERSION_NAME, N_PROCESSES, 'processes')
    domains = []
    with ProgressBar(n, title=f'Creating {n} StructInfos') as bar, Timing(f'Creating {n} StructInfos'):
        for dom in get_domains(n):
            domains.append(StructInfo(dom, init_all=True))
            bar.step()
    domain_dict = {dom.name: dom for dom in domains}
    domain_index = {dom.name: i for i, dom in enumerate(domains)}
    
    global distance_calculator
    try:
        file = RESULTS / f'distance_{n}x{n}{VERSION_NAME}.csv'
        distances = np.loadtxt(file)
        raise Exception(f"{file} already exists. Refusing to overwrite. Please remove it manually before calculating distances again.")
    except OSError:
        try:
            rotations: Array4D = np.load(RESULTS / f'rotations_{n}x{n}.npy')
            translations: Array4D = np.load(RESULTS / f'translations_{n}x{n}.npy')
            distance_calculator = DistanceCalculatorFromAlignment(rotations, translations, domain_index, domain_dict)
            print('Using precomputed alignments.')
            aligning = False
        except OSError:
            rotations = np.zeros((n, n, 3, 3))
            translations = np.zeros((n, n, 1, 3))
            distance_calculator = DistanceCalculatorFromScratch(domain_dict)
            print('ALIGNING!!!')
            aligning = True

    aligned = False

    distance_function = global_distance_function
    distances = np.zeros((n, n))

    from multiprocessing import Pool
    pool = Pool(N_PROCESSES)
    n_calcs = n*(n+1)//2
    with lib.ProgressBar(n_calcs, title=f'Running {n_calcs} distance calculations in {N_PROCESSES} processes') as bar, Timing(f'Running {n_calcs} distance calculations in {N_PROCESSES} processes'):
        for i in range(n):
            jobs = []
            for j in range(i, n):
                job = lib.Job(name=f'{i}-{j}', func=distance_function, args=(domains[i].name, domains[j].name), kwargs={}, stdout=None, stderr=None)
                jobs.append(job)
            results = lib.run_jobs_with_multiprocessing(jobs, pool=pool)
            for result in results:
                si, sj = result.job.name.split('-')
                i, j = int(si), int(sj)
                dist = result.result
                distances[i,j] = distances[j,i] = dist
                bar.step()
    
    matrix_info('distances:', distances)
    np.savetxt(RESULTS / f'distance_{n}x{n}{VERSION_NAME}.csv', distances)
    RESULTS.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(distances.flatten(), bins=range(0, 281, 28))
    plt.savefig(RESULTS / f'distance_hist_{n}x{n}{VERSION_NAME}.png')
    plt.figure()
    plt.imshow(distances)
    plt.savefig(RESULTS / f'distance_{n}x{n}{VERSION_NAME}.png')
    if aligned:
        np.save(RESULTS / f'rotations_{n}x{n}.npy', rotations)
        np.save(RESULTS / f'translations_{n}x{n}.npy', translations)

def make_distance_matrix(n: int):
    print('make_distance_matrix', DIST_TYPE, SHAPEDIST_MAX_RMSD, OPDIST_SCORE_TYPE, OPDIST_MAX_RMSD, VERSION_NAME)
    domains = [StructInfo(domain) for domain in get_domains(n)]
    distances = np.zeros((n, n))
    try: 
        rotations: Array4D = np.load(RESULTS / f'rotations_{n}x{n}.npy')
        translations: Array4D = np.load(RESULTS / f'translations_{n}x{n}.npy')
        aligning = False
    except FileNotFoundError:
        rotations = np.zeros((n, n, 3, 3))
        translations = np.zeros((n, n, 1, 3))
        aligning = True
    aligned = False
    print('aligning:', aligning)
    with Timing(f'Calculating distances {n}x{n}') as timing, ProgressBar(n*(n+1)//2, title=f'Calculating distances {n}x{n}') as bar:
        for i in range(n):
            domA = domains[i]
            for j in range(i+1):
                domB = domains[j]
                rot_trans = None if aligning else (rotations[i, j], translations[i, j])
                dist, rot_trans = sop_dist(DIST_TYPE, domA, domB, rot_trans=rot_trans)
                distances[i,j] = distances[j, i] = dist
                if aligning and rot_trans is not None:
                    R, t = rot_trans
                    rotations[i,j] = R
                    rotations[j,i] = R.T
                    translations[i,j] = t
                    translations[j,i] = -t @ R.T
                    aligned = True
                bar.step()
    matrix_info('distances:', distances)
    RESULTS.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(distances.flatten(), bins=range(0, 281, 28))
    plt.savefig(RESULTS / f'distance_hist_{n}x{n}{VERSION_NAME}.png')
    plt.figure()
    plt.imshow(distances)
    plt.savefig(RESULTS / f'distance_{n}x{n}{VERSION_NAME}.png')
    np.savetxt(RESULTS / f'distance_{n}x{n}{VERSION_NAME}.csv', distances)
    if aligned:
        np.save(RESULTS / f'rotations_{n}x{n}.npy', rotations)
        np.save(RESULTS / f'translations_{n}x{n}.npy', translations)

def show_distances_in_families():
    domains, families = get_domains_and_families()
    n = len(domains)
    n_fams = len(families)
    print(n, 'domains,', n_fams, 'families')
    distances = np.loadtxt(RESULTS / f'distance_{n}x{n}{VERSION_NAME}.csv')
    lengths = np.loadtxt(RESULTS / f'length_{n}.csv')
    sizes = []
    diameters = []
    mean_dists = []
    median_dists = []
    mean_lengths = []
    median_lengths = []
    classes = []
    shown_families = []
    for fam, doms in families.items():
        size = len(doms)
        clas = int(fam[0])
        if size == 1 or clas > 3:
            continue
        dists = np.array([distances[i,j] for i in doms for j in doms])
        lens = np.array([lengths[i] for i in doms])
        diameter = np.max(dists)
        if fam.startswith('3.40.50.'):
            d = max((distances[i,j], domains[i], domains[j]) for i in doms for j in doms)
            print(fam, size, d)
        sizes.append(size)
        diameters.append(diameter)
        mean_dists.append(np.mean(dists))
        median_dists.append(np.median(dists))
        mean_lengths.append(np.mean(lens))
        median_lengths.append(np.median(lens))
        classes.append(clas)
        shown_families.append(fam)
        if fam == '1.10.630.10':
            print(fam, size, diameter, np.mean(dists))
    plt.figure(figsize=(10, 10))
    x = mean_lengths
    y = np.array(diameters)/np.array(mean_lengths)
    plt.scatter(x, y, c=classes, s=sizes)
    for i, fam in enumerate(shown_families):
        plt.annotate(fam, (x[i], y[i]))
    # plt.xlog(sizes, diameters, '.')
    # plt.gca().set_xscale('log')
    plt.axis(xmin=0, ymin=0)
    # plt.xlabel('Family size (#domains in sample)')
    plt.xlabel('Family mean length')
    # plt.ylabel('Family rel. mean distance')
    plt.ylabel('Family rel. diameter')
    plt.savefig(RESULTS / f'family_length_vs_reldiameter{VERSION_NAME}.png')
    plt.figure()
    ml, l = zip(*[(mean_lengths[i_fam], lengths[i_dom]) for i_fam, fam in enumerate(shown_families) for i_dom in families[fam]])
    plt.scatter(ml, l, marker='.')
    plt.xlabel('Family length')
    plt.ylabel('Length')
    plt.savefig(RESULTS / f'length_vs_family_length.png')

def test_triangle_inequality(n):
    distance = np.loadtxt(RESULTS / f'distance_{n}x{n}{VERSION_NAME}.csv')
    n, n = distance.shape
    A_through_B_to_C = distance.reshape((n, n, 1)) + distance.reshape((1, n, n))
    A_direct_to_C = distance.reshape((n, 1, n))
    mask = np.where(A_direct_to_C > EPSILON)
    ratio = A_through_B_to_C[mask] / A_direct_to_C[mask]
    # ratio = (distance.reshape((n, n, 1)) + distance.reshape((1, n, n))) / distance.reshape((n, 1, n))
    print(ratio.shape, ratio.min())
    wrong = 0
    if ratio.min() < 1.0:
        for i in range(n):
            for j in range(n):
                if distance[i,j] > EPSILON:
                    rat = (distance[i,:] + distance[:,j]) / distance[i,j]
                    if np.any(rat < 1.0):
                        for k in range(n):
                            if rat[k] < 1.0:
                                print(f'{i} - {k} - {j}:  {distance[i,k]+distance[k,j]} / {distance[i,j]} ({rat[k]})')
                                wrong += 1
    print('wrong:', wrong)
    plt.hist(ratio, bins=np.arange(0,2.5,0.1))
    plt.savefig(RESULTS / f'triangle_inequality_{n}x{n}{VERSION_NAME}.png')

def test_triangle_inequality2(n):
    distance = np.loadtxt(RESULTS / f'distance_{n}x{n}{VERSION_NAME}.csv')
    n, n = distance.shape
    wrongies = 0
    max_diff = 0.0
    with ProgressBar(n) as bar:
        for i in range(n):
            for j in range(i+1, n):
                d_direct = distance[i,j]
                d_throughs = distance[i,:] + distance[:,j]
                d_through = np.min(d_throughs)
                if d_through < d_direct:
                    wrongies += 1
                    max_diff = max(max_diff, d_direct - d_through)
                    k = np.argmin(d_throughs)
                    print(f'{i}-{k}-{j}: {distance[i,k]} + {distance[k,j]} = {d_through} >= {d_direct}')
                # print(f'{i}-{j}')
            bar.step()
    print('wrongies:', wrongies, 'max_diff:', max_diff)

def sort_sample():
    '''Sort families 1st by class, 2nd by size.'''
    js = json.loads((DATA / 'sample.json').read_text())
    fams = sorted( (fam[0], -len(doms), fam) for fam, doms in js.items() )
    result = {fam: js[fam] for _, _, fam in fams}
    (DATA / 'sample_sorted.json').write_text(json.dumps(result, indent=2))


def main() -> None:
    '''Main'''
    # download_structures()
    # create_alphas()
    n = 100 #4076
    # visualize_matrix(n)
    # test_shape()
    # test_shape_dist()
    # make_lengths(n)
    make_distance_matrix(n)
    # make_distance_matrix_multiprocessing(n)
    # show_distances_in_families()
    # make_bubbles(n)
    # make_tree(n)
    # test_triangle_inequality2(n)



if __name__ == '__main__':
    main()