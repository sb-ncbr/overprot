import numpy
from numpy import linalg
from typing import Tuple

def optimal_rotation(A: numpy.ndarray, B: numpy.ndarray, allow_mirror: bool=False, weights: numpy.ndarray=None) -> numpy.ndarray:
    ''' A, B - matrices 3*n, weights - vector n, result - matrix 3*3
    Find the optimal rotation matrix for 3D superimposition of A onto B,
    where columns of A, B are coordinates of corresponding points.
    If allow_mirror == True, allow also improper rotation (i.e. mirroring + rotation).
    '''
    if weights is not None:
        A = A * weights.reshape((1, -1))
    H = A @ B.transpose()
    U, S, Vh = linalg.svd(H)
    R = (U @ Vh).transpose()
    if not allow_mirror and numpy.linalg.det(R) < 0:  # type: ignore  # mypy doesn't know .det
        Vh[-1,:] = -Vh[-1,:]
        R = (U @ Vh).transpose()
    return R

def optimal_rotation_translation(A: numpy.ndarray, B: numpy.ndarray, allow_mirror: bool=False, weights: numpy.ndarray=None) -> Tuple[numpy.ndarray, numpy.ndarray]:
    '''  A, B - matrices 3*n, weights - vector n, result - (matrix 3*3, matrix 3*1)
    Find the optimal rotation matrix R and translation vector t for 3D superimposition of A onto B,
    where columns of A, B are coordinates of corresponding points.
    If allow_mirror == True, allow also improper rotation (i.e. mirroring + rotation).
    A_superimposed = R * A + t
    '''
    if weights is not None:
        sumW = weights.sum()
        cA = (A * weights).sum(axis=1, keepdims=True) / sumW
        cB = (B * weights).sum(axis=1, keepdims=True) / sumW
    else:
        cA = numpy.mean(A, axis=1, keepdims=True)
        cB = numpy.mean(B, axis=1, keepdims=True)
    R = optimal_rotation(A - cA, B - cB, allow_mirror=allow_mirror, weights=weights)
    t = numpy.matmul(R, -cA) + cB
    return R, t

def rotate_and_translate(A: numpy.ndarray, rotation: numpy.ndarray = numpy.eye(3), translation: numpy.ndarray = numpy.zeros((3,1))) -> numpy.ndarray:
    ''' Applies rotation (or improper rotation) matrix 'rotation' and translation vector 'translation'
    to a 2D object A,
    where columns of A are coordinates of individual points.
    A_result = R * A + t
    '''
    return rotation @ A + translation

def rmsd(A: numpy.ndarray, B: numpy.ndarray, superimpose: bool=True, allow_mirror: bool=False) -> float:
    ''' Calculates RMSD between 2D objects A and B.
    If superimpose == False, skips superimposition and calculates RMSD as is.
    If allow_mirror == True, allows also improper rotation (i.e. mirroring + rotation).
    '''
    if superimpose:
        R, t = optimal_rotation_translation(A, B, allow_mirror=allow_mirror)
        A = rotate_and_translate(A, R, t)
    diff = A - B
    n = A.shape[1]
    sumsq = numpy.sum(diff**2)
    return numpy.sqrt(sumsq / n)

def optimal_rotation_translation_rmsd(A: numpy.ndarray, B: numpy.ndarray, allow_mirror: bool=False) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
    ''' Works as optimal_rotation_translation() but returns a triple (rotation, translation, RMSD).
    '''
    R, t = optimal_rotation_translation(A, B, allow_mirror=allow_mirror)
    A = rotate_and_translate(A, R, t)
    RMSD = rmsd(A, B, superimpose=False)
    return R, t, RMSD

def laying_rotation_translation(A: numpy.ndarray, return_laid_coords: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray]:
    '''Return rotation and translation matrices which center A and align the PCA1, 2, 3 with axes x, y, z.
    If return_laid_coords==True, also return the transformed matrix A.
    One of 4 possible results is selected so that: 
    1) starting and ending coordinates tend to be more in front (z > 0), middle more behind (z < 0).
    2) starting coordinates tend to be more left-top (x < y), ending more right-bottom (x > y), 
    '''
    cA = numpy.mean(A, axis=1, keepdims=True)
    R = laying_rotation(A - cA)
    t = -R @ cA
    return R, t

def laying_rotation(A: numpy.ndarray) -> numpy.ndarray:
    '''Return rotation matrix which aligns the PCA1, 2, 3 with axes x, y, z.
    Centered input matrix is expected.
    One of 4 possible results is selected so that: 
    1) starting and ending coordinates tend to be more in front (z > 0), middle more behind (z < 0).
    2) starting coordinates tend to be more left-top (x < y), ending more right-bottom (x > y), 
    '''
    assert A.shape[0] == 3
    n = A.shape[1]
    U, S, Vh = numpy.linalg.svd(A)
    R = U.T
    if numpy.linalg.det(R) < 0:  # type: ignore  # avoid improper rotation (mirroring)
        R[-1,:] *= -1
    slope = numpy.linspace(-1, 1, n)
    vee_slope = v_slope(n)
    A_rot = R @ A
    front_to_front = numpy.dot(A_rot[2, :], vee_slope) > 0
    if not front_to_front:  # rotate around x
        R = numpy.diag([1, -1, -1]) @ R  # type: ignore
    A_rot = R @ A
    lefttop_to_rightbottom = numpy.dot(A_rot[0, :] - A_rot[1, :], slope) > 0
    if not lefttop_to_rightbottom:  # rotate around z
        R = numpy.diag([-1, -1, 1]) @ R  # type: ignore
    return R

def v_slope(n: int) -> numpy.ndarray:
    '''Return an array with values in V shape, i.e. starting with +x, linearly decreasing to -y in the middle, and then increasing back to +x.
    The values have mean 0 and stdev 1.'''
    n_half = (n+1)//2
    result = numpy.empty(n)
    result[:n_half] = numpy.linspace(1, -1, n_half)
    result[-n_half:] = numpy.linspace(-1, 1, n_half)
    result -= numpy.mean(result)  # type: ignore
    result /= numpy.std(result)
    return result

def dumb_align(A: numpy.ndarray, B: numpy.ndarray, allow_mirror: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
    '''From matrices A (shape 3*m) and B (shape 3*n) select submatrices A', B' (shape 3*k, k=min(m,n)) minimizing RMSD(A', B').
    Return ... rotation, translation matrix and RMSD. A is the mobile, B is the target matrix.'''
    _, m = A.shape  # mobile
    _, n = B.shape  # target
    k = min(m, n)
    min_shift, max_shift = sorted((0, n - m))
    tranformations = []
    for shift in range(min_shift, max_shift+1):
        if shift >= 0:
            rot, trans, rmsd_ = optimal_rotation_translation_rmsd(A[:, 0:k], B[:, shift:k+shift], allow_mirror=allow_mirror)
        else:
            rot, trans, rmsd_ = optimal_rotation_translation_rmsd(A[:, -shift:k-shift], B[:, 0:k], allow_mirror=allow_mirror)
        tranformations.append((rmsd_, shift, rot, trans))
    rmsd_, shift, rot, trans = min(tranformations)
    return (rot, trans, rmsd_)
