import json
import numpy as np
from numba import jit
from pathlib import Path
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Literal
from numpy import ndarray as Array
np.seterr(all='raise')

from overprot.libs import lib
from overprot.libs import lib_alignment
from overprot.libs import lib_pymol
from overprot.libs import lib_acyclic_clustering_simple
from overprot.libs.lib_structure import Structure
from overprot.libs.lib import Timing, ProgressBar
from overprot.libs.lib_similarity_trees.nntree import NNTree
from overprot.libs.lib_similarity_trees.ghtree import GHTree
from overprot.libs.lib_similarity_trees.mtree import MTree

Array1D = Array2D = Array3D = Array4D = Array

DATA = Path('/home/adam/Workspace/Python/OverProt/data-ssd/tree/sample885')
# DATA = Path('/home/adam/Workspace/Python/OverProt/data-ssd/tree/sample40')
SAMPLE_JSON = DATA / 'sample.json'
STRUCTURES = DATA / 'structures_cif'
ALPHAS_CIF = DATA / 'alphas_cif'
ALPHAS_CSV = DATA / 'alphas_csv'
ALPHAS_NPY = DATA / 'alphas_npy'
RESULTS = DATA / 'results'
EPSILON = 1e-4

# SHAPE_LEN = 4
SHAPE_LEN = 5


@dataclass
class StructInfo(object):
    name: str
    _coords: Optional[Array2D] = None
    _shapes: Optional[Array3D] = None
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

def download_structures() -> None:
    sample = json.loads(SAMPLE_JSON.read_text())
    sample_simple = []
    for family, domains in sample.items():
        for domain in domains:
            sample_simple.append((domain['pdb'], domain['domain'], domain['chain_id'], domain['ranges']))
    (DATA/'sample.simple.json').write_text(json.dumps(sample_simple))
    STRUCTURE_CUTTER = '/home/adam/Workspace/Python/OverProt/overprot/OverProt/dependencies/StructureCutter/bin/Release/netcoreapp3.1/StructureCutter.dll'
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

DIST_TYPE = 's+op'
SHAPEDIST_MAX_RMSD = 7.0  #5.0
OPDIST_SCORE_TYPE = 'linear'  # 'exponential'|'linear'
OPDIST_MAX_RMSD = 15
# VERSION_NAME = '-v3maxrmsd7'
# VERSION_NAME = '-op-lin15'
VERSION_NAME = '-s+op-v3maxrmsd7-lin15'  # so far the best is s+op-v3maxrmsd7-lin15 (best in classifying against CATH)
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

def make_lengths(n: int):
    domains = [StructInfo(dom) for dom in get_domains(n)]
    lengths = [dom.n for dom in domains]
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savetxt(RESULTS / f'length_{n}.csv', lengths)

def make_tree(n: int):
    domains = [StructInfo(dom) for dom in get_domains(n)]
    index = {dom.name: i for i, dom in enumerate(domains)}
    try:
        distances = np.loadtxt(RESULTS / f'distance_{n}x{n}{VERSION_NAME}.csv')
        precomputed = 'distances'
        print('Using precomputed distances.')
    except OSError:
        try:
            rotations: Array4D = np.load(RESULTS / f'rotations_{n}x{n}.npy')
            translations: Array4D = np.load(RESULTS / f'translations_{n}x{n}.npy')
            precomputed = 'alignment'
            print('Using precomputed alignments.')
        except FileNotFoundError:
            rotations = np.zeros((n, n, 3, 3))
            translations = np.zeros((n, n, 1, 3))
            precomputed = 'nothing'
            print('ALIGNING!!!')
    if precomputed == 'distances':
        distance_function = lambda a, b: distances[index[a.name], index[b.name]]
    elif precomputed == 'alignment':
        distance_function = lambda a, b: sop_dist('s+op', a, b, rot_trans=(rotations[index[a.name], index[b.name]], translations[index[a.name], index[b.name]]))[0]
    else:
        distance_function = lambda a, b: sop_dist('s+op', a, b)[0]
    # ghtree = GHTree(distance_function)
    # ghtree2 = GHTree(distance_function)
    # mtree = MTree(distance_function)
    # nntree = NNTree(distance_function)
    n_dup = 50
    k = 3 * n_dup

    bulk_domains = [(f'{domain.name}_{i}', domain) for i in range(n_dup) for domain in np.random.permutation(domains)]
    # bulk_domains = [(f'{domain.name}_{i}', domain) for i in range(n_dup) for domain in domains]
    with Timing(f'{n_dup}x adding {n} domains to the tree') as timing, ProgressBar(n_dup*n, title=f'{n_dup}x adding {n} domains to the tree') as bar:
        for domain_key, domain in bulk_domains:
            # ghtree.insert(domain_key, domain)
            # ghtree2.insert(domain_key, domain)
            # nntree.insert(domain_key, domain)
            # mtree.insert(domain_key, domain)
            bar.step()
        # ghtreeB = GHTree(distance_function, bulk_domains)
        ghtreeBn = GHTree(distance_function, bulk_domains)
    print('Per one:', timing.time / (n_dup*n))
    # print(ghtree.get_statistics())
    # print(ghtreeB.get_statistics())
    print(ghtreeBn.get_statistics())
    calcs_before = ghtreeBn._distance_cache._calculated_distances_counter
    # print(nntree.get_statistics())
    # print(mtree.get_statistics())

    idx = 0
    wrongies = 0
    n_dup = 1
    with Timing(f'{n_dup}x searching {n} domains in the tree') as timing, ProgressBar(n_dup*n, title=f'{n_dup}x searching {n} domains in the tree', mute=False) as bar:
        for i in range(n_dup):
            for domain in np.random.permutation(domains):
                idx += 1
                # real_knn = sorted((distance_function(domain, other), f'{other.name}_{i}') for other in domains)[:k]
                # knn = ghtree.kNN_query_classical(f'{domain.name}_{i}', k, include_query=True)
                # knnB = ghtreeB.kNN_query_classical(f'{domain.name}_{i}', k, include_query=True)
                # knnB = ghtreeB.kNN_query_classical(f'{domain.name}_{i}', k, include_query=True)
                # knnBn = ghtreeBn.kNN_query_classical_by_value(domain, k)
                knnBn = ghtreeBn.kNN_query_classical_by_value_with_priority_queue(domain, k)
                # knnM = mtree.kNN_query(f'{domain.name}_{i}', k, include_query=True)
                # knnN = nntree.kNN_query_by_value(domain, k)
                # rang = knn[-1][0]
                # range_query = ghtree2.range_query(f'{domain.name}_{i}', rang)
                # if [dom for dist, dom in knnBn] != [dom for dist, dom in real_knn]:
                #     print('  Real kNN:', real_knn, 'Found kNN:', knnBn)
                #     wrongies += 1
                # if [dom for dist, dom in knn] != [dom for dist, dom in real_knn]:
                #     print('  Real kNN:', real_knn, 'Found kNN:', knn)
                # if [dom for dist, dom in knnN] != [dom for dist, dom in real_knn]:
                #     wrongies += 1
                #     # print('N Real kNN:', real_knn, 'Found kNN:', knnN)
                bar.step()
    print('Per one:', timing.time / (n_dup*n))
    print('Wrongies:', wrongies)
    # print(ghtree.get_statistics())
    # print(ghtreeB.get_statistics())
    print(ghtreeBn.get_statistics())
    calcs_after = ghtreeBn._distance_cache._calculated_distances_counter
    print('Calculations:', calcs_after - calcs_before)
    # print(nntree.get_statistics())
    # print(mtree.get_statistics())
    # print(treeB)

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
    
def test_triangle_inequality(n):
    print('picovina')
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
    n = 100  # 885
    # n = 40
    # visualize_matrix(n)
    # test_shape()
    # test_shape_dist()
    # make_lengths(n)
    make_distance_matrix(n)
    # make_tree(n)
    # test_triangle_inequality(n)



if __name__ == '__main__':
    main()