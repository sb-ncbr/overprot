import json
import numpy as np
from numba import jit
from pathlib import Path
from matplotlib import pyplot as plt
from typing import List, Optional, Union, Tuple

from overprot.libs import lib_alignment
from overprot.libs import lib_pymol
from overprot.libs import lib_acyclic_clustering_simple
from overprot.libs.lib_structure import Structure
from overprot.libs.lib import Timing, ProgressBar
from overprot.libs.lib_similarity_trees.nntree import NNTree

DATA = Path('/home/adam/Workspace/Python/OverProt/data-ssd/tree/sample885')
SAMPLE_JSON = DATA / 'sample.json'
STRUCTURES = DATA / 'structures_cif'
ALPHAS_CIF = DATA / 'alphas_cif'
ALPHAS_CSV = DATA / 'alphas_csv'
ALPHAS_NPY = DATA / 'alphas_npy'
EPSILON = 1e-4

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


def create_alphas() -> None:
    # representants = json.loads(REPRS_JSON.read_text())
    # domains = [rep['domain'] for rep in representants]
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

def cealign_try(n=10, create_choice=False):
    domains = get_domains(n, create_choice=create_choice)
    dist = np.zeros((n,n))
    with Timing(f'Cealign each-to-each ({n} structures)') as timing:
        with ProgressBar(n, title=f'Cealign each-to-each ({n} structures)') as bar:
            for i in range(n):
                for j in range(n):
                    A = domains[i]
                    B = domains[j]
                    # c = lib_pymol.cealign(ALPHAS/f'{A}.cif', ALPHAS/f'{B}.cif')
                    try:
                        d = lib_alignment.dist(ALPHAS_CIF/f'{A}.cif', ALPHAS_CIF/f'{B}.cif')
                    except lib_pymol.CmdException:
                        d = -1.0
                    dist[i,j] = d
                    # print(d)
                bar.step()

    print('Per one:', timing.time / n**2)
    np.savetxt(DATA / f'dist_{n}x{n}.csv', dist)
    np.save(DATA / f'dist_{n}x{n}.npy', dist)

def cealign_try2(n=10):
    representants = json.loads(SAMPLE_JSON.read_text())
    domains = Path.read_text(DATA/f'choice_{n}.txt').split()
    i = 156
    dist = np.zeros((n,))
    n_aln = np.zeros((n,), dtype=int)
    rmsd = np.zeros((n,))
    
    with Timing(f'Cealign 1-to-each ({n} structures)') as timing:
        with ProgressBar(n, title=f'Cealign 1-to-each ({n} structures)') as bar:
            for j in range(n):
                A = domains[i]
                B = domains[j]
                # c = lib_pymol.cealign(ALPHAS/f'{A}.cif', ALPHAS/f'{B}.cif')
                try:
                    d = lib_alignment.dist(ALPHAS_CIF/f'{A}.cif', ALPHAS_CIF/f'{B}.cif')
                    # d = lib_pymol.cealign(ALPHAS/f'{A}.cif', ALPHAS/f'{B}.cif')
                except lib_pymol.CmdException:
                    d = -1.0
                dist[j] = d
                bar.step()

    print('Per one:', timing.time / n)
    np.savetxt(DATA / f'dist_1x{n}.csv', dist)
    print(dist.min(), dist.max())

def visualize_matrix(n=10):
    mat = np.loadtxt(DATA / f'dist_{n}x{n}.csv')
    plt.imshow(mat)
    plt.savefig(DATA/f'dist_{n}x{n}.png')
    print(mat.min(), mat.max())

@jit(nopython=True)
def normalize(x: np.ndarray, axis: int) -> None:
    norm = np.sqrt(np.sum(x**2, axis=axis))
    norm = np.expand_dims(norm, axis=axis)
    return x / norm

@jit(nopython=True)
def normalize_inplace(x: np.ndarray, axis: int) -> None:
    norm = np.sqrt(np.sum(x**2, axis=axis))
    norm = np.expand_dims(norm, axis=axis)
    x /= norm

@jit(nopython=True)
def fake_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n, a, b = A.shape
    n, b, c = B.shape
    C = np.zeros((n, a, c), dtype=A.dtype)
    for i in range(a):
        # for j in range(c):
            C[:, i, :] += np.sum(np.expand_dims(A[:,i,:], axis=-1) * B[:,:,:], axis=1)
    return C

@jit(nopython=True)
def get_shapes(coords: np.ndarray) -> np.ndarray:  # (n, 3) -> (n, k, 3); k = 4
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
def get_shapes_nice(coords: np.ndarray, omit_center: bool = True) -> np.ndarray:  # (n, 3) -> (n, k, 3); k = 5
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

def matrix_info(name: str, A: np.ndarray) -> None:
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

SHAPEDIST_MAX_RMSD = 7.0  #5.0
# VERSION_NAME = '-maxrmsd7'
OPDIST_MAX_RMSD = 15.0
OPDIST_SCORE_TYPE = 'linear'  # 'exponential'|'linear'
VERSION_NAME = '-sop-maxrmsd7*lin15'
# best options: exp20, lin15 (based on sample885)

def op_score(domainA: Union[np.ndarray, str], domainB: Union[np.ndarray, str], rot_trans: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    coordsA = np.load(ALPHAS_NPY/f'{domainA}.npy') if isinstance(domainA, str) else domainA
    m, _3 = coordsA.shape; assert _3 == 3
    coordsB = np.load(ALPHAS_NPY/f'{domainB}.npy') if isinstance(domainB, str) else domainB
    n, _3 = coordsB.shape; assert _3 == 3
    if rot_trans is None:
        cealign = lib_pymol.cealign(ALPHAS_CIF/f'{domainA}.cif', ALPHAS_CIF/f'{domainB}.cif', fallback_to_dumb_align=True)
        R, t = cealign.rotation.T, cealign.translation.T  # convert matrices from column style (3, n) to row style (n, 3)
    else:
        R, t = rot_trans
    coordsB = coordsB @ R + t
    r = np.sqrt(np.sum((coordsA.reshape((m, 1, 3)) - coordsB.reshape((1, n, 3)))**2, axis=2))
    if OPDIST_SCORE_TYPE == 'exponential':
        score = np.exp(-r / OPDIST_MAX_RMSD)
    elif OPDIST_SCORE_TYPE == 'linear':
        score = 1 - r / OPDIST_MAX_RMSD
        score[score < 0] = 0
    else: 
        raise AssertionError
    # matrix_info('score', score)
    # plt.figure()
    # plt.hist(score.flatten())
    # plt.savefig(DATA / f'score_hist_{domainA}_{domainB}.png')
    # matrix_info('score', score)
    # plt.figure()
    # plt.imshow(score)
    # plt.savefig(DATA / f'score_{domainA}_{domainB}.png')
    # np.savetxt(DATA / f'score_{domainA}_{domainB}.csv', score)
    return score
    

def sop_dist(domainA: Union[np.ndarray, str], domainB: Union[np.ndarray, str], shapesA: Optional[np.ndarray] = None, shapesB: Optional[np.ndarray] = None, rot_trans: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:
    coordsA = np.load(ALPHAS_NPY/f'{domainA}.npy') if isinstance(domainA, str) else domainA
    m, _3 = coordsA.shape; assert _3 == 3
    coordsB = np.load(ALPHAS_NPY/f'{domainB}.npy') if isinstance(domainB, str) else domainB
    n, _3 = coordsB.shape; assert _3 == 3
    if rot_trans is None:
        cealign = lib_pymol.cealign(ALPHAS_CIF/f'{domainA}.cif', ALPHAS_CIF/f'{domainB}.cif', fallback_to_dumb_align=True)
        R, t = cealign.rotation.T, cealign.translation.T  # convert matrices from column style (3, n) to row style (n, 3)
    else:
        R, t = rot_trans
    coordsB = coordsB @ R + t
    r = np.sqrt(np.sum((coordsA.reshape((m, 1, 3)) - coordsB.reshape((1, n, 3)))**2, axis=2))
    if OPDIST_SCORE_TYPE == 'exponential':
        score_op = np.exp(-r / OPDIST_MAX_RMSD)
    elif OPDIST_SCORE_TYPE == 'linear':
        score_op = 1 - r / OPDIST_MAX_RMSD
        score_op[score_op < 0] = 0
    else: 
        raise AssertionError

    SHAPE_LEN = 4
    if shapesA is None:
        coordsA = np.load(ALPHAS_NPY/f'{domainA}.npy') if isinstance(domainA, str) else domainA
        m, _3 = coordsA.shape; assert _3 == 3
        shapesA = get_shapes(coordsA)
    else:
        m, _SHAPE_LEN, _3 = shapesA.shape; assert _SHAPE_LEN == SHAPE_LEN; assert _3 == 3
    if shapesB is None:
        coordsB = np.load(ALPHAS_NPY/f'{domainB}.npy') if isinstance(domainB, str) else domainB
        n, _3 = coordsB.shape; assert _3 == 3
        shapesB = get_shapes(coordsB)
    else:
        n, _SHAPE_LEN, _3 = shapesB.shape; assert _SHAPE_LEN == SHAPE_LEN; assert _3 == 3
    diff = shapesA.reshape((m, 1, SHAPE_LEN, 3)) - shapesB.reshape((1, n, SHAPE_LEN, 3))
    sqerror = np.sum(diff**2, axis=(2, 3))
    rmsd = np.sqrt(sqerror / SHAPE_LEN)
    score_s = 1 - (rmsd / SHAPEDIST_MAX_RMSD)
    score_s[score_s < 0] = 0

    # score = 0.5 * (score_s + score_op)
    score = score_s * score_op
    
    # matrix_info('score', score)
    # plt.figure()
    # plt.hist(score.flatten())
    # plt.savefig(DATA / f'score_hist_{domainA}_{domainB}.png')
    # matrix_info('score', score)
    # plt.figure()
    # plt.imshow(score)
    # plt.savefig(DATA / f'score_{domainA}_{domainB}.png')
    # np.savetxt(DATA / f'score_{domainA}_{domainB}.csv', score)
    
    matching, total_score = lib_acyclic_clustering_simple.dynprog_align(score)
    distance = 0.5 * (m + n) - total_score
    return distance, R, t

def op_dist(domainA: Union[np.ndarray, str], domainB: Union[np.ndarray, str], rot_trans: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:
    coordsA = np.load(ALPHAS_NPY/f'{domainA}.npy') if isinstance(domainA, str) else domainA
    m, _3 = coordsA.shape; assert _3 == 3
    coordsB = np.load(ALPHAS_NPY/f'{domainB}.npy') if isinstance(domainB, str) else domainB
    n, _3 = coordsB.shape; assert _3 == 3
    if rot_trans is None:
        cealign = lib_pymol.cealign(ALPHAS_CIF/f'{domainA}.cif', ALPHAS_CIF/f'{domainB}.cif', fallback_to_dumb_align=True)
        R, t = cealign.rotation.T, cealign.translation.T  # convert matrices from column style (3, n) to row style (n, 3)
    else:
        R, t = rot_trans
    coordsB = coordsB @ R + t
    r = np.sqrt(np.sum((coordsA.reshape((m, 1, 3)) - coordsB.reshape((1, n, 3)))**2, axis=2))
    if OPDIST_SCORE_TYPE == 'exponential':
        score = np.exp(-r / OPDIST_MAX_RMSD)
    elif OPDIST_SCORE_TYPE == 'linear':
        score = 1 - r / OPDIST_MAX_RMSD
        score[score < 0] = 0
    else: 
        raise AssertionError
    
    # matrix_info('score', score)
    # plt.figure()
    # plt.hist(score.flatten())
    # plt.savefig(DATA / f'score_hist_{domainA}_{domainB}.png')
    # matrix_info('score', score)
    # plt.figure()
    # plt.imshow(score)
    # plt.savefig(DATA / f'score_{domainA}_{domainB}.png')
    # np.savetxt(DATA / f'score_{domainA}_{domainB}.csv', score)
    
    matching, total_score = lib_acyclic_clustering_simple.dynprog_align(score)
    distance = 0.5 * (m + n) - total_score
    return distance, R, t

def shape_dist(domainA: Union[np.ndarray, str], domainB: Union[np.ndarray, str], shapesA: Optional[np.ndarray] = None, shapesB: Optional[np.ndarray] = None) -> float:
    SHAPE_LEN = 4
    if shapesA is None:
        coordsA = np.load(ALPHAS_NPY/f'{domainA}.npy') if isinstance(domainA, str) else domainA
        m, _3 = coordsA.shape; assert _3 == 3
        shapesA = get_shapes(coordsA)
    else:
        m, _SHAPE_LEN, _3 = shapesA.shape; assert _SHAPE_LEN == SHAPE_LEN; assert _3 == 3
    if shapesB is None:
        coordsB = np.load(ALPHAS_NPY/f'{domainB}.npy') if isinstance(domainB, str) else domainB
        n, _3 = coordsB.shape; assert _3 == 3
        shapesB = get_shapes(coordsB)
    else:
        n, _SHAPE_LEN, _3 = shapesB.shape; assert _SHAPE_LEN == SHAPE_LEN; assert _3 == 3
    diff = shapesA.reshape((m, 1, SHAPE_LEN, 3)) - shapesB.reshape((1, n, SHAPE_LEN, 3))
    sqerror = np.sum(diff**2, axis=(2, 3))
    rmsd = np.sqrt(sqerror / SHAPE_LEN)
    score = 1 - (rmsd / SHAPEDIST_MAX_RMSD)
    score[score < 0] = 0
    
    # matrix_info('score', score)
    # plt.figure()
    # plt.hist(score.flatten())
    # plt.savefig(DATA / f'score_hist_{domainA}_{domainB}.png')
    # matrix_info('score', score)
    # plt.figure()
    # plt.imshow(score)
    # plt.savefig(DATA / f'score_{domainA}_{domainB}.png')
    # np.savetxt(DATA / f'score_{domainA}_{domainB}.csv', score)
    
    matching, total_score = lib_acyclic_clustering_simple.dynprog_align(score)
    distance = 0.5 * (m + n) - total_score
    return distance

def test_shape_dist():
    domains = '2nnj 1pq2 1og2 1tqn 1akd 6eye 1bpv 3mbt'.split()
    domA = domains[-1]
    with Timing(f'shape_dist * {len(domains)}'):
        for domB in domains:
            shape_dist(domA, domB)

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
    k = 5_000
    # with Timing(f'get_shapes_o=nice * {2*k}'):
    #     for i in range(k):
    #         shapesA = get_shapes_nice(coordsA)
    #         shapesB = get_shapes_nice(coordsB)
    with Timing(f'get_shapes     * {2*k}'):
        for i in range(k):
            shapesA = get_shapes(coordsA)
            shapesB = get_shapes(coordsB)
    assert error < EPSILON, error

def make_tree(n: int):
    nntree = NNTree(shape_dist, with_nearest_pair_queue=True)
    domains = get_domains(n)
    dup = 1
    structs = {domain: np.load(ALPHAS_NPY/f'{domain}.npy') for domain in domains}
    with Timing() as timing, ProgressBar(dup*n, title=f'{dup}x adding {n} domains to NNTree') as bar:
        for i in range(dup):
            for domain in np.random.permutation(domains):
                struct = np.load(ALPHAS_NPY/f'{domain}.npy')
                # struct = structs[domain]
                nntree.insert(f'{domain}_{i}', struct)
                bar.step()
    print('Per one:', timing.time / (dup*n))
    print(nntree.get_statistics())

def make_distance_matrix(n: int):
    domains = get_domains(n)
    structs = {domain: np.load(ALPHAS_NPY/f'{domain}.npy') for domain in domains}
    shapes = {domain: get_shapes(structs[domain]) for domain in domains}
    distances = np.zeros((n, n))
    try: 
        rotations = np.load(DATA / f'rotations_{n}x{n}.npy')
        translations = np.load(DATA / f'translations_{n}x{n}.npy')
        aligning = False
    except FileNotFoundError:
        rotations = np.zeros((n, n, 3, 3))
        translations = np.zeros((n, n, 1, 3))
        aligning = True
    print('aligning:', aligning)
    with Timing(f'Calculating distances {n}x{n}') as timing, ProgressBar(n*(n+1)//2, title=f'Calculating distances {n}x{n}') as bar:
        for i in range(n):
            domA = domains[i]
            for j in range(i+1):
                domB = domains[j]
                if aligning:
                    rot_trans = None
                else:
                    rot_trans = rotations[i, j], translations[i, j]
                # dist, R, t = op_dist(domA, domB, rot_trans=rot_trans)
                dist, R, t = sop_dist(domA, domB, shapesA=shapes[domA], shapesB=shapes[domB], rot_trans=rot_trans)
                if aligning:
                    rotations[i,j] = R
                    rotations[j,i] = R.T
                    translations[i,j] = t
                    translations[j,i] = -t @ R.T
                # dist = shape_dist(structs[domA], structs[domB])
                # dist = shape_dist(structs[domA], structs[domB], shapesA=shapes[domA], shapesB=shapes[domB])
                # dist = length_dist_min(structs[domA], structs[domB])
                distances[i,j] = distances[j, i] = dist
                bar.step()
    matrix_info('distances:', distances)
    plt.figure()
    plt.hist(distances.flatten(), bins=range(0, 281, 28))
    plt.savefig(DATA / f'distance_hist_{n}x{n}{VERSION_NAME}.png')
    plt.figure()
    plt.imshow(distances)
    plt.savefig(DATA / f'distance_{n}x{n}{VERSION_NAME}.png')
    np.savetxt(DATA / f'distance_{n}x{n}{VERSION_NAME}.csv', distances)
    if aligning:
        np.save(DATA / f'rotations_{n}x{n}.npy', rotations)
        np.save(DATA / f'translations_{n}x{n}.npy', translations)
    
def sort_sample():
    '''Sort families 1st by class, 2nd by size.'''
    js = json.loads((DATA / 'sample.json').read_text())
    fams = sorted( (fam[0], -len(doms), fam) for fam, doms in js.items() )
    result = {fam: js[fam] for _, _, fam in fams}
    (DATA / 'sample_sorted.json').write_text(json.dumps(result, indent=2))


def main() -> None:
    '''Main'''
    # create_alphas()
    n = 885
    # cealign_try2(n)
    # cealign_try(n)
    # visualize_matrix(n)
    # d = shape_dist('2opkA01', '2opkA01')
    # print(d)
    # c = np.load(ALPHAS_NPY / '2opkA01.npy')
    # s = get_shapes(c)
    # np.save(DATA / 'shapes_2opkA01.npy', s)
    # print(s)
    # return
    test_shape()
    # test_shape_dist()
    make_distance_matrix(n)
    # make_tree(n)



if __name__ == '__main__':
    main()